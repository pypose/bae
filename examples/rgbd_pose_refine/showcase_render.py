"""Headless, depth-aware surface-splat renderer and synchronized DSLR comparisons.

The same source pixels, confidence mask, depth, splat size and rendering camera
are used on both sides. Only the optimized camera poses change. No inpainting,
synthetic perturbation, neural image generation or separate before/after camera
alignment is used. Interior views exclude the selected source camera to avoid
covering alignment errors with its own image. EGL renders on the GPU.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moderngl
import imageio.v2 as imageio

BG = (12, 18, 27)
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')),size)

def homogeneous(x):
    if x.shape[-2:] == (4,4): return x.copy()
    out = np.tile(np.eye(4),(len(x),1,1));out[:,:3,:]=x
    return out

def look_at(eye, target, up):
    z = np.asarray(target)-eye;z=z/np.linalg.norm(z)
    x = np.cross(z,up);x/=np.linalg.norm(x)
    y = np.cross(z,x)
    c = np.eye(4);c[:3,:3] = np.stack([x,y,z],axis=1);c[:3,3]=eye
    return np.linalg.inv(c)


VERT = '''#version 330
in vec3 xyz; in vec3 rgb; in float radius; in float quality;
uniform mat4 pose; uniform mat4 view; uniform mat4 projection;
uniform float focal; uniform float splat_scale;
out vec3 color; out float camera_z; out float weight; out vec3 world;
void main(){
 vec4 w=pose*vec4(xyz,1); vec4 p=view*w; world=w.xyz;
 gl_Position=projection*p;
 gl_PointSize=clamp(2.0*focal*radius*splat_scale/max(p.z,0.01),1.0,48.0);
 color=rgb; camera_z=p.z; weight=quality;
}
'''
FRAG = '''#version 330
in vec3 color; in float camera_z; in float weight; in vec3 world;
uniform int pass_id; uniform sampler2D nearest_depth; uniform vec2 resolution;
uniform vec3 up; uniform float ceiling; uniform float tolerance;
layout(location=0) out vec4 out_color;
void main(){
 vec2 p=2.0*gl_PointCoord-1.0; float r2=dot(p,p);
 if(r2>1.0 || camera_z<=0.0 || dot(world,up)>ceiling) discard;
 if(pass_id==0){out_color=vec4(camera_z,0,0,1);return;}
 float front=texelFetch(nearest_depth,ivec2(gl_FragCoord.xy),0).r;
 if(camera_z>front+tolerance) discard;
 float w=exp(-2.0*r2)*weight;
 out_color=vec4(color*w,w);
}
'''
QUAD_V = '''#version 330
in vec2 pos;out vec2 uv;
void main(){uv=pos*.5+.5;gl_Position=vec4(pos,0,1);}
'''
QUAD_F = '''#version 330
in vec2 uv;uniform sampler2D accum;out vec4 color;
void main(){vec4 a=texture(accum,uv);vec3 bg=vec3(12,18,27)/255.0;
 color=vec4(a.a>0.0001?a.rgb/a.a:bg,1);}
'''


class SurfaceRenderer:
    def __init__(self,data, K, size=(960,720), stride=2, percentile=25):
        self.ctx = moderngl.create_standalone_context(backend='egl')
        self.size=size; self.K=K;self.data=data
        print('Renderer:',self.ctx.info['GL_RENDERER'],flush=True)
        self.program=self.ctx.program(vertex_shader=VERT,fragment_shader=FRAG)
        self.quadprog=self.ctx.program(vertex_shader=QUAD_V,fragment_shader=QUAD_F)
        buf=self.ctx.buffer(np.array([[-1,-1],[1,-1],[-1,1],[1,1]],'f4').tobytes())
        self.quad=self.ctx.simple_vertex_array(self.quadprog,buf,'pos')
        self.depthtex=self.ctx.texture(size,4,dtype='f4')
        self.depthbuf=self.ctx.depth_renderbuffer(size)
        self.depthfbo=self.ctx.framebuffer([self.depthtex],self.depthbuf)
        self.accum=self.ctx.texture(size,4,dtype='f4')
        self.accumfbo=self.ctx.framebuffer([self.accum])
        self.finaltex=self.ctx.texture(size,3)
        self.final=self.ctx.framebuffer([self.finaltex])
        self.vaos=[];self.counts=[];self.sample_points=[]
        depths,confs,images=data['depth'],data['conf'],data['images']
        for i,(d,q,rgb,k) in enumerate(zip(depths,confs,images,K)):
            h,w=d.shape
            yy,xx=np.mgrid[1:h-1:stride,1:w-1:stride]
            z=d[yy,xx]
            # Fixed confidence and relative depth-edge filter for all variants.
            edge=np.maximum.reduce([np.abs(z-d[yy-1,xx]),np.abs(z-d[yy+1,xx]),
                                    np.abs(z-d[yy,xx-1]),np.abs(z-d[yy,xx+1])])
            valid=np.isfinite(z)&(z>0)&(q[yy,xx]>np.percentile(q,percentile))&(edge<z*.04)
            uv=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)
            xyz=(uv@np.linalg.inv(k).T)*z[...,None]
            rad=z/k[0,0]*stride*.85
            quality=np.clip(q[yy,xx]/np.median(q),.2,3)
            a=np.concatenate([xyz,rgb[yy,xx]/255.,rad[...,None],quality[...,None]],-1)[valid].astype('f4')
            vbo=self.ctx.buffer(a.tobytes()); vao=self.ctx.vertex_array(self.program,[(vbo,'3f 3f 1f 1f','xyz','rgb','radius','quality')])
            self.vaos.append(vao);self.counts.append(len(a));self.sample_points.append(a[::max(1,len(a)//1000),:3])
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        print(f'{sum(self.counts):,} fixed surface samples from {len(self.vaos)} DSLR frames',flush=True)

    def render(self,w2c,view,fov=65,exclude=-1,up=(0,1,0),ceiling=1e6,tolerance=.025):
        c=self.ctx;p=self.program;width,height=self.size
        fy=height/(2*np.tan(np.deg2rad(fov)/2));near,far=.03,200.
        proj=np.zeros((4,4));proj[0,0]=2*fy/width;proj[1,1]=-2*fy/height
        proj[2,2]=(far+near)/(far-near);proj[2,3]=-2*far*near/(far-near);proj[3,2]=1
        for name,matrix in [('view',view),('projection',proj)]:p[name].write(np.asarray(matrix.T,'f4').tobytes())
        p['focal'].value=fy;p['splat_scale'].value=1.;p['up'].value=tuple(up);p['ceiling'].value=float(ceiling)
        p['tolerance'].value=tolerance
        poses=np.linalg.inv(homogeneous(w2c))
        c.disable(moderngl.BLEND);c.enable(moderngl.DEPTH_TEST)
        self.depthfbo.use();self.depthfbo.clear(200,0,0,1,depth=1)
        p['pass_id'].value=0
        for i,vao in enumerate(self.vaos):
            if i==exclude:continue
            p['pose'].write(np.asarray(poses[i].T,'f4').tobytes());vao.render(moderngl.POINTS)
        self.accumfbo.use();self.accumfbo.clear(0,0,0,0)
        c.disable(moderngl.DEPTH_TEST);c.enable(moderngl.BLEND);c.blend_func=(moderngl.ONE,moderngl.ONE)
        p['pass_id'].value=1;self.depthtex.use(0);p['nearest_depth'].value=0
        for i,vao in enumerate(self.vaos):
            if i==exclude:continue
            p['pose'].write(np.asarray(poses[i].T,'f4').tobytes());vao.render(moderngl.POINTS)
        self.final.use();c.disable(moderngl.BLEND);self.accum.use(0);self.quadprog['accum'].value=0
        self.quad.render(moderngl.TRIANGLE_STRIP)
        return Image.frombytes('RGB',self.size,self.final.read(components=3,alignment=1)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def layout(left,right,scene,subtitle,label,stats=None,step=None):
    w,h=left.size
    canvas=Image.new('RGB',(2*w+72,h+220),BG);d=ImageDraw.Draw(canvas)
    d.text((28,22),'bae',font=font(38,True),fill='#71efce')
    d.text((130,30),'CAMERA POSES  /  ROOM RECONSTRUCTION',font=font(20,True),fill='white')
    d.text((28,78),subtitle,font=font(17),fill='#9aaabc')
    canvas.paste(left,(24,148));canvas.paste(right,(w+48,148))
    for x,title,col in [(24,'DA3  /  INITIAL', '#ffbd8c'),(w+48,label,'#71efce')]:
        d.rounded_rectangle((x,112,x+9,132),radius=4,fill=col)
        d.text((x+18,108),title,font=font(19,True),fill=col)
    footer=f'ScanNet++  {scene}   |   Undistorted DSLR   |   Fixed depth + rendering'
    d.text((24,h+172),footer,font=font(14),fill='#9aaabc')
    if stats:d.text((w+48,h+172),stats,font=font(14),fill='#71efce')
    if step is not None:
        d.rectangle((24,h+207,24+int((2*w+24)*step),h+210),fill='#71efce')
    return canvas


def main(args):
    out=args.out/args.scene;data=np.load(out/'prediction.npz');ref=dict(np.load(out/'refined.npz'))
    w0=homogeneous(data['w2c']);K=ref['K'];w1=homogeneous(ref[args.variant])
    if args.renderer=='mesh':
        from showcase_mesh import MeshRenderer
        renderer=MeshRenderer(data,K,out/'meshes',size=(args.width,args.height),voxel=args.voxel)
    elif args.renderer=='textured':
        from showcase_textured import TexturedRenderer
        renderer=TexturedRenderer(data,K,out/'textures',size=(args.width,args.height),dataset_root=args.dataset_root)
    else:
        renderer=SurfaceRenderer(data,K,size=(args.width,args.height),stride=args.stride)
    c2w=np.linalg.inv(w0); up=-np.mean(c2w[:,:3,1],axis=0);up/=np.linalg.norm(up)
    report=json.loads((out/'metrics.json').read_text())
    dest=out/'renders';dest.mkdir(exist_ok=True)
    if args.preview:
        ids=np.linspace(0,len(w0)-1,min(24,len(w0))).astype(int)
        sheet=Image.new('RGB',(4*480,math.ceil(len(ids)/4)*390),BG);d=ImageDraw.Draw(sheet)
        for n,i in enumerate(ids):
            view=w0[i].copy()
            im=renderer.render(w1,view,exclude=-1 if args.renderer=='mesh' else int(i),fov=65)
            im.thumbnail((472,354));x=n%4*480;y=n//4*390;sheet.paste(im,(x,y))
            d.text((x+10,y+357),f'View {i} | {data["names"][i]}',font=font(16),fill='white')
        sheet.save(dest/'view_candidates.jpg',quality=95);return
    i=args.view
    view=look_at(c2w[i,:3,3],c2w[i,:3,3]+c2w[i,:3,2],up)
    kwargs=dict(exclude=-1 if args.renderer=='mesh' else i)
    if args.overview is not None:
        center=np.median(c2w[:,:3,3],axis=0)
        radius=np.percentile(np.linalg.norm(c2w[:,:3,3]-center,axis=-1),95)
        x=c2w[0,:3,0].copy();x-=up*np.dot(x,up);x/=np.linalg.norm(x);y=np.cross(up,x)
        angle=np.deg2rad(args.overview)
        eye=center+radius*1.7*(x*np.cos(angle)+y*np.sin(angle))+up*radius*1.1
        view=look_at(eye,center-up*.4,up)
        kwargs.update(up=up,ceiling=np.dot(center,up)+.7,cutaway=True,fov=60)
        if args.front_cut is not None:
            front=x*np.cos(angle)+y*np.sin(angle)
            kwargs.update(front_normal=front,front_offset=np.dot(center,front)+radius*args.front_cut)
        if args.renderer!='mesh':raise ValueError('overview needs --renderer mesh')
    a=renderer.render(w0,view,**kwargs);b=renderer.render(w1,view,**kwargs)
    a.save(dest/'before.png');b.save(dest/'after.png')
    meta=json.loads((out/'input.json').read_text())
    subtitle=f'{len(w0)} views across the full scene  /  {meta["registered_views"]} registered photographs'
    if args.variant.endswith('_raw'):
        subtitle+='  /  Raw optimizer output (before guard)'
    m0=report['variants']['da3']['auc@5'];m1=report['variants'][args.variant]['auc@5']
    stats=f'Pose AUC @ 5 deg: {100*m0:.1f} -> {100*m1:.1f}'
    poster=layout(a,b,args.scene,subtitle,args.variant.upper().replace('_',' / '),stats)
    poster.save(dest/'comparison.jpg',quality=97)
    # All formulations in one contact sheet, exact same rendering settings.
    names=['da3','bae_d','bae_e','bae_de','form_d','form_e','form_de']
    atlas=Image.new('RGB',(args.width*3,args.height*3+180),BG);d=ImageDraw.Draw(atlas)
    for n,name in enumerate([] if args.skip_atlas else names):
        if name not in ref:continue
        raw_name=name+'_raw' if name+'_raw' in ref else name
        im=renderer.render(ref[raw_name],view,**kwargs);x=(n%3)*args.width;y=(n//3)*(args.height+60)
        atlas.paste(im,(x,y+45));auc=report['variants'][raw_name]['auc@5']
        d.text((x+12,y+8),f'{raw_name}   AUC@5 {auc*100:.1f}',font=font(20,True),fill='white')
    if not args.skip_atlas:atlas.save(dest/'formulations.jpg',quality=95)
    if args.video:
        base_variant=args.variant.removesuffix('_raw')
        hist=ref.get(base_variant+'_history')
        if hist is None: raise ValueError(f'No actual optimization history for {args.variant}')
        # Include the odometry stage when animating the composed pipeline.
        if base_variant=='bae_de' and 'bae_d_history' in ref:
            hist=np.concatenate([ref['bae_d_history'],hist])
        indices=list(range(0,len(hist),args.history_stride))
        frames=[w0,*hist[indices],w1]
        frame_paths=[]
        from showcase_video import VideoPanel
        panel=VideoPanel(frames,data['gt'])
        writer=imageio.get_writer(dest/'refinement.mp4',fps=24,codec='libx264',quality=9,
                                 macro_block_size=2,ffmpeg_params=['-crf','17','-pix_fmt','yuv420p'])
        try:
            for j,w in enumerate(frames):
                b=renderer.render(w,view,**kwargs)
                path=dest/f'step_{j:03d}.webp';b.save(path,quality=92);frame_paths.append(str(path.relative_to(out)))
                label='BAE  /  INITIAL' if j==0 else 'BAE  /  RAW RESULT' if j==len(frames)-1 and args.variant.endswith('_raw') else 'BAE  /  FINAL' if j==len(frames)-1 else f'BAE  /  LM STEP {indices[j-1]+1}'
                frame=layout(a,b,args.scene,subtitle,label,stats if j==len(frames)-1 else None,j/(len(frames)-1))
                frame=panel.compose(frame,j)
                repeat=48 if j in (0,len(frames)-1) else 8
                for _ in range(repeat):writer.append_data(np.asarray(frame))
                if j%5==0: print(f'rendered optimization step {j}/{len(frames)-1}',flush=True)
        finally:writer.close()
        from showcase_manifest import write_manifest
        write_manifest(out,args,frames,frame_paths,report,meta)
    (dest/'render.json').write_text(json.dumps(dict(view=i,excluded_source_view=kwargs['exclude'],variant=args.variant,
             renderer=args.renderer,confidence_percentile=20 if args.renderer=='textured' else 25,
             depth_edge_relative_threshold=.06 if args.renderer=='textured' else .04 if args.renderer=='splat' else None,
             blending_depth_tolerance=.035 if args.renderer=='textured' else .025 if args.renderer=='splat' else None,
             voxel_length=args.voxel if args.renderer=='mesh' else None,
             overview_degrees=args.overview,front_cut_radius_fraction=args.front_cut,
             source_vertices=sum(renderer.counts),size=renderer.size),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',default='7831862f02');p.add_argument('--out',type=Path,default=Path('outputs/rgbd_showcase'))
    p.add_argument('--variant',default='bae_de');p.add_argument('--view',type=int,default=0)
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=720)
    p.add_argument('--stride',type=int,default=2);p.add_argument('--preview',action='store_true');p.add_argument('--video',action='store_true')
    p.add_argument('--renderer',choices=['splat','mesh','textured'],default='textured')
    p.add_argument('--dataset-root',default='/data/zitong/scannetpp_val')
    p.add_argument('--voxel',type=float,default=.012)
    p.add_argument('--overview',type=float)
    p.add_argument('--front-cut',type=float,help='Fixed front cutaway plane, as a fraction of camera-trajectory radius')
    p.add_argument('--history-stride',type=int,default=3)
    p.add_argument('--skip-atlas',action='store_true')
    main(p.parse_args())
