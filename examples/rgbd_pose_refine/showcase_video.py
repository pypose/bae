"""Camera trajectories and measured pose accuracy underneath the map animation."""
import numpy as np
from PIL import Image,ImageDraw
from showcase_render import BG,font,homogeneous
from showcase_manifest import camera_plot


def pose_auc(w,gt):
    w=homogeneous(w);gt=homogeneous(gt);i,j=np.triu_indices(len(w),1)
    rel=w[i]@np.linalg.inv(w)[j];truth=gt[i]@np.linalg.inv(gt)[j]
    rotation=np.swapaxes(rel[:,:3,:3],1,2)@truth[:,:3,:3]
    r=np.rad2deg(np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1)))
    a,b=rel[:,:3,3],truth[:,:3,3];a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-8)
    t=np.rad2deg(np.arccos(np.clip(np.abs((a*b).sum(1)),-1,1)));error=np.sort(np.maximum(r,t))
    e=np.r_[0,error];recall=np.r_[0,(np.arange(len(error))+1)/len(error)];n=np.searchsorted(e,5)
    return np.trapz(np.r_[recall[:n],recall[n-1]],np.r_[e[:n],5])/5*100


class VideoPanel:
    def __init__(self,states,gt):
        self.poses=camera_plot(states,gt)
        self.aucs=[pose_auc(w,gt) for w in states]

    def compose(self,poster,index):
        w,h=poster.size;out=Image.new('RGB',(w,h+180),BG);out.paste(poster,(0,0));d=ImageDraw.Draw(out)
        d.line((24,h,w-24,h),fill='#2c3948')
        d.text((24,h+10),'CAMERA CENTERS',font=font(15,True),fill='#93a5bb')
        bounds=np.array(self.poses['bounds']);lo,hi=bounds[:2],bounds[2:]
        def xy(p):return (165+(p[0]-lo[0])/(hi[0]-lo[0])*285,h+159-(p[1]-lo[1])/(hi[1]-lo[1])*143)
        for points,col in [(self.poses['gt'],'#62738a'),(self.poses['initial'],'#ffbd8c'),(self.poses['history'][index],'#71efce')]:
            coords=[xy(p) for p in points];d.line(coords,fill=col,width=1)
            for x,y in coords:d.ellipse((x-1,y-1,x+1,y+1),fill=col)
        for n,(s,col) in enumerate([('COLMAP','#62738a'),('Initial','#ffbd8c'),('Current','#71efce')]):
            d.text((26,h+46+n*27),s,font=font(14),fill=col)
        x0,x1=560,w-460;y0,y1=h+142,h+42
        lower=max(0,min(self.aucs)-3);upper=min(100,max(self.aucs)+3)
        def chart(k,val):return(x0+k/(len(self.aucs)-1)*(x1-x0),y0-(val-lower)/(upper-lower)*(y0-y1))
        d.text((x0,h+10),'POSE AUC @ 5°  /  HIGHER IS BETTER',font=font(15,True),fill='#93a5bb')
        for val in [lower,(lower+upper)/2,upper]:
            y=chart(0,val)[1];d.line((x0,y,x1,y),fill='#243142');d.text((x1+10,y-8),f'{val:.0f}',font=font(12),fill='#93a5bb')
        if index>0:d.line([chart(k,self.aucs[k]) for k in range(index+1)],fill='#71efce',width=3)
        x,y=chart(index,self.aucs[index]);d.ellipse((x-4,y-4,x+4,y+4),fill='#71efce')
        d.text((w-365,h+36),f'{self.aucs[0]:.1f} → {self.aucs[index]:.1f}',font=font(33,True),fill='#71efce')
        d.text((w-365,h+87),'Actual optimizer states',font=font(15),fill='#93a5bb')
        d.text((w-365,h+116),'Fixed depth · No pose exaggeration',font=font(14),fill='#93a5bb')
        return out
