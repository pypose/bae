"""Collect every measured scene, including regressions and unrendered scenes."""
import argparse,csv,json,shutil
from pathlib import Path


def report(roots,dest):
    dest.mkdir(parents=True,exist_ok=True);records=[];rows=[]
    keys=['da3','bae_d_raw','bae_e_raw','bae_de_raw','form_d_raw','form_e_raw','form_de_raw']
    for root in roots:
        for path in sorted(root.glob('*/metrics.json')):
            m=json.loads(path.read_text());inp=json.loads((path.parent/'input.json').read_text())
            label='GIANT-1.1' if inp['model'].endswith('DA3-GIANT-1.1') else 'NESTED-GIANT-LARGE'
            records.append(dict(input=inp,metrics=m))
            rows.append([label,m['scene'],m['views'],*[m['variants'].get(k,{}).get('auc@5') for k in keys]])
    (dest/'results.json').write_text(json.dumps(records,indent=2))
    with (dest/'results.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['model','scene','views',*keys]);writer.writerows(rows)
    md=['# Measured DSLR refinement results','',
        'Six ScanNet++ scenes were tested with the nested checkpoint, and two scene runs were repeated with the `DA3-GIANT-1.1` checkpoint used in the earlier DA3 experiments. This is a demonstration study, **not the full 50-scene validation benchmark**. Five scene/model combinations include all actual Open3D D, E and D+E references. Missing reference runs are marked —.','',
        'All numbers below are **raw optimizer AUC@5 percentages**, before the common guard. Higher is better. JSON and CSV store AUC as fractions. Every row uses identical inputs and the same scene-wide frames across methods. Frame sets differ between model runs, so the two checkpoints are not a controlled model bake-off.','',
        '| Model | Scene | Views | Initial | bae D | bae E | bae D+E | Open3D D | Open3D E | Open3D D+E |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append('| '+' | '.join([str(x) for x in r[:3]]+[f'{x*100:.2f}' if x is not None else '—' for x in r[3:]])+' |')
    candidates=[(r['metrics']['variants'][k],r['metrics']['variants'].get(k+'_raw'),r['metrics']['variants']['da3']) for r in records for k in ['bae_d','bae_e','bae_de','form_d','form_e','form_de'] if k in r['metrics']['variants']]
    rejected=sum(not a.get('accepted',True) for a,_,_ in candidates)
    improvements=sum(raw is not None and raw['auc@5']>base['auc@5'] for _,raw,base in candidates)
    md+=['','## What the results support','',
         '- The native solver improves pose accuracy on several real scene-wide DSLR predictions. In the selected bookshelf office, the nested-model result goes from **4.86 to 22.63** with bae D+E. The actual Open3D D+E reference reaches **29.45** on the same arrays. The direction agrees; the magnitude does not match.',
         '- The compact home office has a smaller native D+E gain (**44.82 to 46.71**). The apartment improves numerically (**12.18 to 24.97**) but retains conspicuous reconstruction artifacts, so it was excluded from the main visual gallery.',
         '- The lounge demonstrates that chaining is not always beneficial: native E alone improves accuracy, while D and D+E regress. Raw regressions are preserved in this report.',
         f'- The existing photometric guard rejected **{rejected}/{len(candidates)}** evaluated proposals, including **{improvements}** proposals whose raw AUC@5 improved. Thus the guarded results return the initial poses. The main gallery intentionally shows labeled raw optimizer output; it is **not** a successful demonstration of the current guard.',
         '- GIANT-1.1 does not remove the gap to the reference on the office. These measurements do not establish parity with Forms D/E across scenes.','',
         '## Coverage and memory','',
         '| Checkpoint | Scene | Selected / registered | Largest successful PyTorch allocation (GiB) | Next failed view count |',
         '|---|---|---:|---:|---:|']
    for r in records:
        inp=r['input'];n=inp['selected_views'];ok=[a for a in inp['attempts'] if a['success']];bad=[a['views'] for a in inp['attempts'] if not a['success'] and a['views']>n]
        md.append(f'| {inp["model"].split("/")[-1]} | {inp["scene"]} | {n} / {inp["registered_views"]} | {max(a["peak_allocated_gib"] for a in ok):.2f} | {min(bad) if bad else "cap reached"} |')
    md+=['','Each successful prediction attends globally across the selected scene-wide frame set. The search is bracketed to eight views; it does not claim an exact maximum. The 139-frame run shared GPU capacity with another job during its initial search. The cap remains a user setting.','',
         '## Reproducibility and limitations','',
         '- Inputs: actual undistorted DSLR frames; full selected filenames and memory attempts are in [results.json](results.json). No COLMAP conditioning, synthetic noise, inpainting, or generated room imagery.',
         '- Native D and Open3D D solve different problems; native E and Open3D E fuse different structures. All reference calls use the actual functions under `da3/`, with raw D+E composition and common post-hoc guarding as described in [SHOWCASE.md](../../SHOWCASE.md).',
         '- Pose AUC uses every unordered frame pair. It is not a ground-truth surface metric and does not guarantee a visually clean mesh. No claim of map-to-LiDAR accuracy is made.',
         '- The showcased scenes and views were selected after examining results for visual clarity. This is not an unbiased estimate of generalization. All eight tested scene/model combinations are included above.',
         '- Wall times and PyTorch allocation peaks are retained in JSON. Workloads sometimes overlapped and the formulations use different sampling/iteration schedules, so these times are not a controlled speed comparison.',
         '- Six initial scene runs plus two matching-checkpoint runs; seven example tests pass. Browser checks cover desktop/mobile layout, loaded assets, scene selection, slider state and timeline playback.',
         '', '[Portable gallery](index.html) · [Machine-readable results](results.json) · [CSV](results.csv)']
    (dest/'RESULTS.md').write_text('\n'.join(md)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path);p.add_argument('--dest',type=Path,default=Path('examples/rgbd_pose_refine/docs/showcase'));a=p.parse_args();report(a.roots,a.dest)
