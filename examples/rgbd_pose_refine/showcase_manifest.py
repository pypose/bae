"""Measured metadata and a fixed-alignment camera trajectory for the gallery."""
import json
from pathlib import Path
import numpy as np
from showcase_render import homogeneous


def camera_plot(states,gt):
    centers=[np.linalg.inv(homogeneous(w))[:,:3,3] for w in states]
    target=np.linalg.inv(gt)[:,:3,3]
    source=centers[0];ms,mt=source.mean(0),target.mean(0)
    a,b=source-ms,target-mt;u,s,v=np.linalg.svd(b.T@a/len(a));fix=np.eye(3)
    fix[-1,-1]=np.linalg.det(u@v);rotation=u@fix@v
    scale=np.sum(s*np.diag(fix))/np.mean(np.sum(a*a,axis=1))
    translation=mt-scale*rotation@ms
    aligned=[scale*p@rotation.T+translation for p in centers]
    _,_,vt=np.linalg.svd(target-mt,full_matrices=False);basis=vt[:2].T
    proj=lambda p:((p-mt)@basis).tolist()
    points=[proj(p) for p in aligned];truth=proj(target)
    allpts=np.concatenate([np.array(points).reshape(-1,2),truth]);lo=allpts.min(0);hi=allpts.max(0)
    span=max(hi-lo);center=(lo+hi)/2;lo=center-span*.55;hi=center+span*.55
    return dict(initial=points[0],final=points[-1],history=points,gt=truth,bounds=[*lo,*hi])


def write_manifest(out,args,states,frame_paths,report,meta):
    data=np.load(out/'prediction.npz')
    name=args.variant.removesuffix('_raw');raw=args.variant.endswith('_raw')
    accepted=report['variants'][name]['accepted'];before=report['variants']['da3']['auc@5'];after=report['variants'][args.variant]['auc@5']
    titles={'3f15a9266d':'Bookshelf office','09c1414f1b':'Living room','7831862f02':'Lounge',
            '0d2ee665be':'Home office','cc5237fd77':'Library','6115eddb86':'Hotel room',
            'c50d2d1d42':'Monitor office','38d58a7a31':'Lecture room','40aec5fffa':'Kitchenette'}
    rows=[];labels={'da3':'DA3','bae_d':'bae D','bae_e':'bae E','bae_de':'bae D+E','form_d':'Open3D D','form_e':'Open3D E','form_de':'Open3D D+E'}
    for k,label in labels.items():
        v=report['variants'].get(k);r=report['variants'].get(k+'_raw',v)
        if not v or 'auc@5' not in v:rows.append([label,'—','—','not run']);continue
        rows.append([label,f'{r["auc@5"]*100:.2f}',f'{v["auc@5"]*100:.2f}', '—' if k=='da3' else 'accept' if v['accepted'] else 'reject'])
    status=('Showing the raw optimizer result. The current photometric guard rejects this proposal and returns the initial poses.'
            if raw and not accepted else 'The common photometric guard accepts the optimized poses.' if accepted else 'The guard rejects the proposal; the returned poses equal the initialization.')
    record=dict(id=meta['scene'],title=titles.get(meta['scene'],meta['scene']),views=len(data['depth']),registered=meta['registered_views'],
        auc=f'{before*100:.1f} → {after*100:.1f}',stage=labels[name]+(' · raw result' if raw else ''),
        status=status,accepted=accepted,raw=raw,rows=rows,frames=frame_paths,
        before='renders/before.png',poster='renders/comparison.jpg',video='renders/refinement.mp4',metrics='metrics.json',
        method=('TSDF surface fusion, fixed confidence filtering and a fixed cutaway plane.' if args.overview is not None else 'Interior view of the reconstructed surface.')+' Depth stays fixed; the camera poses change.',
        provenance=f'{meta["model"]}; {meta["process_res"]}-pixel inference. Uniform sampling over the complete registered DSLR sequence, including both endpoints. Fisheye undistortion precedes inference. Both maps use the same self-calibrated intrinsics. The shown scene and viewpoint are selected for demonstration; the metrics table includes regressions.',
        poses=camera_plot(states,data['gt']))
    ambient=report.get('config',{}).get('ambient_grad') == '1'
    record['run_label']='GIANT 1.1 · ambient gradients' if ambient else 'Historical showcase run'
    record['provenance']+= (' PyPose ambient gradients enabled.' if ambient else
                           ' Historical render: this run predates the ambient-gradient correction; see the separate complete validation report for current results.')
    (out/'renders/gallery.json').write_text(json.dumps(record))
