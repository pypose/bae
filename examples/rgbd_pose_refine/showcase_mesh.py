"""Identical TSDF fusion for each pose state, with GPU mesh rendering."""
from pathlib import Path
import hashlib
import numpy as np
import moderngl
import open3d as o3d
from PIL import Image
from showcase_render import homogeneous

VERT='''#version 330
in vec3 xyz;in vec3 rgb;in vec3 normal;
uniform mat4 view;uniform mat4 projection;
out vec3 color;out vec3 world;out vec3 norm;
void main(){world=xyz;color=rgb;norm=normal;gl_Position=projection*view*vec4(xyz,1);}
'''
FRAG='''#version 330
in vec3 color;in vec3 world;in vec3 norm;
uniform vec3 up;uniform float ceiling;uniform vec3 eye;uniform int cutaway;
uniform vec3 front_normal;uniform float front_offset;
out vec4 out_color;
void main(){if(dot(world,up)>ceiling || dot(world,front_normal)>front_offset)discard;
 if(cutaway==1 && dot(norm,eye-world)<0.0)discard;
 out_color=vec4(color,1);}
'''

class MeshRenderer:
    def __init__(self,data,K,cache,size=(960,720),voxel=.012):
        self.ctx=moderngl.create_standalone_context(backend='egl')
        self.size=size;self.data={k:data[k] for k in ('depth','conf','images')};self.K=K;self.voxel=voxel;self.cache=Path(cache)
        self.cache.mkdir(exist_ok=True)
        self.program=self.ctx.program(vertex_shader=VERT,fragment_shader=FRAG)
        self.fbo=self.ctx.simple_framebuffer(size,components=3)
        self.last_key=None;self.vao=None;self.buffers=[]
        self.counts=[0];self.sample_points=[]
        print('Mesh renderer:',self.ctx.info['GL_RENDERER'],flush=True)

    def mesh(self,w2c,exclude):
        w2c=homogeneous(w2c)
        key=hashlib.sha256(w2c.tobytes()+self.K.tobytes()+str((exclude,self.voxel,'conf25-components200')).encode()).hexdigest()[:20]
        if key==self.last_key:return
        path=self.cache/(key+'.ply')
        if path.exists():mesh=o3d.io.read_triangle_mesh(str(path))
        else:
            vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=self.voxel,
                 sdf_trunc=self.voxel*4,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for i,(d,rgb,k,pose) in enumerate(zip(self.data['depth'],self.data['images'],self.K,w2c)):
                if i==exclude:continue
                h,w=d.shape
                confidence=self.data['conf'][i]
                d=np.where(confidence>=np.percentile(confidence,25),d,0)
                rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rgb)),
                    o3d.geometry.Image(np.ascontiguousarray(d.astype('f4'))),
                    depth_scale=1,depth_trunc=10,convert_rgb_to_intensity=False)
                intr=o3d.camera.PinholeCameraIntrinsic(w,h,k[0,0],k[1,1],k[0,2],k[1,2])
                vol.integrate(rgbd,intr,pose)
            mesh=vol.extract_triangle_mesh();mesh.compute_vertex_normals()
            labels,counts,_=mesh.cluster_connected_triangles()
            mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)]<200)
            mesh.remove_unreferenced_vertices()
            o3d.io.write_triangle_mesh(str(path),mesh)
            print(f'Fused {len(w2c)} views: {len(mesh.vertices):,} vertices',flush=True)
        a=np.concatenate([np.asarray(mesh.vertices),np.asarray(mesh.vertex_colors),np.asarray(mesh.vertex_normals)],axis=1).astype('f4')
        idx=np.asarray(mesh.triangles,dtype='i4')
        if self.vao:self.vao.release()
        for b in self.buffers:b.release()
        vbo=self.ctx.buffer(a.tobytes());ibo=self.ctx.buffer(idx.tobytes())
        self.buffers=[vbo,ibo];self.vao=self.ctx.vertex_array(self.program,[(vbo,'3f 3f 3f','xyz','rgb','normal')],ibo)
        self.last_key=key;self.counts=[len(a)];self.sample_points=[a[::10,:3]]

    def render(self,w2c,view,fov=65,exclude=-1,up=(0,1,0),ceiling=1e6,tolerance=.025,cutaway=False,
               front_normal=(1,0,0),front_offset=1e6):
        self.mesh(w2c,exclude)
        p=self.program;width,height=self.size;fy=height/(2*np.tan(np.deg2rad(fov)/2));near,far=.03,200
        proj=np.zeros((4,4));proj[0,0]=2*fy/width;proj[1,1]=-2*fy/height
        proj[2,2]=(far+near)/(far-near);proj[2,3]=-2*far*near/(far-near);proj[3,2]=1
        for name,matrix in [('view',view),('projection',proj)]:p[name].write(np.asarray(matrix.T,'f4').tobytes())
        p['up'].value=tuple(up);p['ceiling'].value=float(ceiling);p['eye'].value=tuple(np.linalg.inv(view)[:3,3]);p['cutaway'].value=int(cutaway)
        p['front_normal'].value=tuple(front_normal);p['front_offset'].value=float(front_offset)
        self.fbo.use();self.fbo.clear(12/255,18/255,27/255,depth=1)
        self.ctx.enable(moderngl.DEPTH_TEST);self.vao.render()
        return Image.frombytes('RGB',self.size,self.fbo.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
