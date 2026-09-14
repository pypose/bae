"""Resume a complete DSLR split evaluation, with one isolated worker per GPU.

Each scene gets one globally attended prediction and all six refinement arms.
The manifest enumerates the split before launching work; failures remain visible.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault('BAE_USE_PYPOSE_AMBIENT_GRAD', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = ['bae_d', 'bae_e', 'bae_de', 'form_d', 'form_e', 'form_de']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=Path('/data/zitong/scannetpp_val'))
    p.add_argument('--split', default='nvs_sem_val.txt')
    p.add_argument('--out', type=Path, default=Path('outputs/rgbd_validation_giant11'))
    p.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    p.add_argument('--model', default='depth-anything/DA3-GIANT-1.1')
    p.add_argument('--revision', default='72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19')
    p.add_argument('--max-views', type=int, default=192)
    p.add_argument('--initial-views', type=int, default=156)
    p.add_argument('--threads', type=int, default=1,
                   help='OpenMP threads per worker; GPU tensor operations use eight CPU threads')
    p.add_argument('--reuse-predictions', type=Path, default=Path('outputs/rgbd_showcase_giant11'))
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    split_path = args.dataset_root/'splits'/args.split
    scenes = split_path.read_text().split()
    if len(set(scenes)) != len(scenes):
        raise ValueError('Duplicate scene in split')
    for scene in scenes:
        for name in ('colmap.zip', 'resized_images.zip'):
            if not (args.dataset_root/'data'/scene/'dslr'/name).is_file():
                raise FileNotFoundError(f'{scene}: missing DSLR {name}')
    if os.environ['BAE_USE_PYPOSE_AMBIENT_GRAD'].lower() not in ('1', 'true', 'yes', 'on'):
        raise ValueError('This validation requires BAE_USE_PYPOSE_AMBIENT_GRAD=1')
    config = dict(split=str(split_path), split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                  scenes=scenes, model=args.model, revision=args.revision,
                  max_views=args.max_views, ambient_grad=True, variants=VARIANTS,
                  allocator=os.environ['PYTORCH_CUDA_ALLOC_CONF'], initial_views=args.initial_views,
                  reference_workers=8)
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config:
            raise ValueError('Existing split manifest configuration differs')
    else:
        manifest = dict(config=config, status={s: {'state': 'pending'} for s in scenes})
    lock = threading.Lock()

    def update(scene, **info):
        with lock:
            manifest['status'][scene].update(info, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            temp = manifest_path.with_suffix('.tmp')
            temp.write_text(json.dumps(manifest, indent=2))
            temp.replace(manifest_path)
            counts = {state: sum(v['state'] == state for v in manifest['status'].values())
                      for state in ('pending', 'inference', 'refinement', 'complete', 'failed')}
            print(scene, info, counts, flush=True)

    def worker(device, selected):
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
                   MKL_NUM_THREADS=str(args.threads), PYTHONUNBUFFERED='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        for scene in selected:
            out = args.out/scene
            out.mkdir(exist_ok=True)
            started = time.perf_counter()
            try:
                source = args.reuse_predictions/scene
                if not (out/'prediction.npz').exists() and (source/'input.json').exists():
                    meta = json.loads((source/'input.json').read_text())
                    if (meta['model'] == args.model and meta['checkpoint_revision'] == args.revision
                            and meta['process_res'] == 504 and meta['max_views'] == args.max_views):
                        # Immutable inference arrays may be shared. Optimizer results always stay separate.
                        os.link(source/'prediction.npz', out/'prediction.npz')
                        (out/'input.json').write_text(json.dumps(meta, indent=2))
                with (out/'run.log').open('a') as log:
                    def run(script, options):
                        cmd = [sys.executable, str(HERE/script), '--scenes', scene,
                               '--out', str(args.out), '--device', str(device), *options]
                        log.write('\nCOMMAND '+json.dumps(cmd)+'\n'); log.flush()
                        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if not (out/'prediction.npz').exists():
                        update(scene, state='inference', device=device)
                        run('showcase_prepare.py', ['--dataset-root', str(args.dataset_root),
                            '--model', args.model, '--revision', args.revision, '--max-views', str(args.max_views),
                            '--initial-views', str(args.initial_views)])
                    update(scene, state='refinement', device=device)
                    run('showcase_benchmark.py', [])
                metrics = json.loads((out/'metrics.json').read_text())
                for name in VARIANTS:
                    for key in (name, name+'_raw'):
                        if 'auc@5' not in metrics['variants'].get(key, {}):
                            raise RuntimeError(f'Missing result: {key}')
                update(scene, state='complete', seconds=time.perf_counter()-started, views=metrics['views'], error=None)
            except Exception as exc:
                update(scene, state='failed', error=f'{type(exc).__name__}: {exc}')

    # Include cached demo scenes early as a quick check of the corrected gradient mode.
    scenes = sorted(scenes, key=lambda s: (not (args.reuse_predictions/s/'prediction.npz').exists(), s))
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures = [pool.submit(worker, d, scenes[i::len(args.devices)]) for i, d in enumerate(args.devices)]
        for f in futures:
            f.result()
    if any(v['state'] != 'complete' for v in manifest['status'].values()):
        raise SystemExit('Validation incomplete; consult manifest.json and per-scene run.log; rerun to resume')


if __name__ == '__main__':
    main()
