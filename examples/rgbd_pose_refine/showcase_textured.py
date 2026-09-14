"""Continuous DSLR-textured depth surfaces with depth-aware multiview blending.

Geometry uses DA3's unmodified depth. High-resolution undistorted photographs
supply texture only, using the same resize geometry as the prediction. Every
scene frame contributes geometry; the selected render source view is excluded.
"""
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import moderngl
from showcase_render import homogeneous,QUAD_V,QUAD_F

VERT='''#version 330
in vec3 xyz;in vec2 texcoord;in float quality;
uniform mat4 pose;uniform mat4 view;uniform mat4 projection;
out vec2 uv;out float camera_z;out float weight;
void main(){vec4 p=view*pose*vec4(xyz,1);gl_Position=projection*p;uv=texcoord;camera_z=p.z;weight=quality;}
'''
FRAG='''#version 330
in vec2 uv;in float camera_z;in float weight;
uniform int pass_id;uniform sampler2D nearest_depth;uniform sampler2D photograph;
uniform float tolerance;uniform float view_weight;
out vec4 out_color;
void main(){if(camera_z<=0.0)discard;
 if(pass_id==0){out_color=vec4(camera_z,0,0,1);return;}
 float front=texelFetch(nearest_depth,ivec2(gl_FragCoord.xy),0).r;
 if(camera_z>front+tolerance)discard;
 float w=weight*view_weight;out_color=vec4(texture(photograph,uv).rgb*w,w);}
'''

class TexturedRenderer:
    def __init__(self,data,K,cache,size=(960,720),dataset_root='/data/zitong/scannetpp_val'):
        self.ctx=moderngl.create_standalone_context(backend='egl');self.size=size
        self.program=self.ctx.program(vertex_shader=VERT,fragment_shader=FRAG)
        self.quadprog=self.ctx.program(vertex_shader=QUAD_V,fragment_shader=QUAD_F)
        buf=self.ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[1,1]],'f4').tobytes())
        self.quad=self.ctx.simple_vertex_array(self.quadprog,buf,'pos')
        self.depthtex=self.ctx.texture(size,4,dtype='f4');self.depthbuf=self.ctx.depth_renderbuffer(size)
        self.depthfbo=self.ctx.framebuffer([self.depthtex],self.depthbuf)
        self.accum=self.ctx.texture(size,4,dtype='f4');self.accumfbo=self.ctx.framebuffer([self.accum])
        self.final=self.ctx.simple_framebuffer(size,components=3)
        self.vaos=[];self.textures=[];self.counts=[]
        self.initial=np.linalg.inv(homogeneous(data['w2c']))
        cache=Path(cache);cache.mkdir(exist_ok=True)
        depths,confs,colors=data['depth'],data['conf'],data['images']
        if not (cache/'complete').exists():
            from dslr import load_dslr_scene
            rgb,_,_=load_dslr_scene(cache.parent.name,data['names'].tolist(),max_size=1512,dataset_root=dataset_root)
            h,w=depths.shape[1:]
            for i,im in enumerate(rgb):
                im=cv2.resize(im,(w*3,h*3),interpolation=cv2.INTER_AREA)
                Image.fromarray(im).save(cache/f'{i:04d}.jpg',quality=97)
            del rgb
            (cache/'complete').write_text(str(len(depths)))
        for i,(d,q,k) in enumerate(zip(depths,confs,K)):
            h,w=d.shape;yy,xx=np.mgrid[0:h:2,0:w:2];z=d[yy,xx]
            xyz=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(k).T*z[...,None]
            uv=np.stack([(xx+.5)/w,(yy+.5)/h],-1)
            quality=np.clip(q[yy,xx]/np.median(q),.2,3)
            a=np.concatenate([xyz,uv,quality[...,None]],axis=-1).astype('f4').reshape(-1,6)
            rows,cols=z.shape;index=np.arange(rows*cols).reshape(rows,cols)
            tl=index[:-1,:-1].ravel();tr=index[:-1,1:].ravel();bl=index[1:,:-1].ravel();br=index[1:,1:].ravel()
            triangles=np.concatenate([np.stack([tl,tr,bl],-1),np.stack([tr,br,bl],-1)])
            zs=z.ravel()[triangles]
            valid=(zs.min(-1)>0)&np.isfinite(zs).all(-1)&((zs.max(-1)-zs.min(-1))<zs.mean(-1)*.06)
            # Confidence is fixed per source pixel for every optimization state.
            valid&=(q[yy,xx].ravel()[triangles].min(-1)>=np.percentile(q,20))
            triangles=triangles[valid].astype('i4')
            vbo=self.ctx.buffer(a.tobytes());ibo=self.ctx.buffer(triangles.tobytes())
            self.vaos.append(self.ctx.vertex_array(self.program,[(vbo,'3f 2f 1f','xyz','texcoord','quality')],ibo))
            im=Image.open(cache/f'{i:04d}.jpg').convert('RGB');tex=self.ctx.texture(im.size,3,im.tobytes());tex.build_mipmaps()
            self.textures.append(tex);self.counts.append(len(a))
        print('Continuous textured surfaces:',self.ctx.info['GL_RENDERER'],flush=True)

    def render(self,w2c,view,fov=65,exclude=-1,**kwargs):
        c=self.ctx;p=self.program;width,height=self.size;fy=height/(2*np.tan(np.deg2rad(fov)/2));near,far=.03,200
        proj=np.zeros((4,4));proj[0,0]=2*fy/width;proj[1,1]=-2*fy/height
        proj[2,2]=(far+near)/(far-near);proj[2,3]=-2*far*near/(far-near);proj[3,2]=1
        for name,matrix in [('view',view),('projection',proj)]:p[name].write(np.asarray(matrix.T,'f4').tobytes())
        p['tolerance'].value=.035
        poses=np.linalg.inv(homogeneous(w2c))
        # Fixed source preference for the render camera, shared before/after.
        camera=np.linalg.inv(view);dist=np.linalg.norm(self.initial[:,:3,3]-camera[:3,3],axis=-1)
        direction=np.clip(self.initial[:,:3,2]@camera[:3,2],0,1)
        vw=(.1+direction**4)/(np.maximum(dist,np.median(dist)*.15)**2)
        vw/=vw.max()
        c.disable(moderngl.BLEND);c.enable(moderngl.DEPTH_TEST)
        self.depthfbo.use();self.depthfbo.clear(200,0,0,1,depth=1);p['pass_id'].value=0
        for i,vao in enumerate(self.vaos):
            if i==exclude:continue
            p['pose'].write(np.asarray(poses[i].T,'f4').tobytes());vao.render()
        self.accumfbo.use();self.accumfbo.clear(0,0,0,0)
        c.disable(moderngl.DEPTH_TEST);c.enable(moderngl.BLEND);c.blend_func=(moderngl.ONE,moderngl.ONE)
        p['pass_id'].value=1;self.depthtex.use(0);p['nearest_depth'].value=0;p['photograph'].value=1
        for i,vao in enumerate(self.vaos):
            if i==exclude:continue
            p['pose'].write(np.asarray(poses[i].T,'f4').tobytes());p['view_weight'].value=float(vw[i]);self.textures[i].use(1);vao.render()
        self.final.use();c.disable(moderngl.BLEND);self.accum.use(0);self.quadprog['accum'].value=0;self.quad.render(moderngl.TRIANGLE_STRIP)
        return Image.frombytes('RGB',self.size,self.final.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
