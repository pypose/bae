"""Compare native bae odometry/structure with the actual DA3 Form D/E functions.

All variants consume identical cached DSLR predictions, depths, intrinsics,
frame selections and seed. Raw candidates and guarded outputs are both saved.
Ground truth is used exclusively for reporting pose errors.
"""
from __future__ import annotations
import os
os.environ.setdefault('BAE_USE_PYPOSE_AMBIENT_GRAD', '1')

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import cv2
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'da3'))
from geometry import as_44
from eval import pair_errors, error_auc
from photometric import refine_odometry, refine_structure
from selfcal import estimate_focal_selfcal
from guards import trust_region_accept_reject
import refine_poses as reference


def metrics(w2c, gt):
    # Keep evaluation on the input device too. This container's CPU vector
    # acos kernel segfaults for some large pair sets; CUDA avoids that kernel.
    errors = pair_errors(as_44(w2c).double(), gt.double())
    return dict(error_auc(errors, (3,5,10,20,30)), median_pair_error_deg=float(np.median(errors)),
                mean_pair_error_deg=float(errors.mean()))


def benchmark(args, scene):
    out = args.out/scene
    data = np.load(out/'prediction.npz')
    device = f'cuda:{args.device}'
    cv = lambda x: torch.as_tensor(x, device=device, dtype=torch.float64)
    w0, depth, K = cv(data['w2c']), cv(data['depth']), cv(data['K'])
    conf, images, gt = cv(data['conf']), data['images'], cv(data['gt'])
    w0 = as_44(w0)
    report_path = out/'metrics.json'
    result_path = out/'refined.npz'
    report = json.loads(report_path.read_text()) if report_path.exists() and not args.force else {}
    saved = dict(np.load(result_path)) if result_path.exists() and not args.force else {}
    if 'config' in report and report['config'] != vars_config(args):
        raise ValueError('Cached benchmark configuration differs; use --force or another output directory')
    with (out/'prediction.npz').open('rb') as stream:
        prediction_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if report.get('prediction_sha256', prediction_hash) != prediction_hash:
        raise ValueError('Prediction cache changed; use --force to recompute refinement')
    report['prediction_sha256'] = prediction_hash
    report.update(scene=scene, views=len(w0), config=vars_config(args))
    report.setdefault('variants', {})['da3'] = metrics(w0, gt)
    if 'K' in saved:
        K = cv(saved['K'])
    elif args.selfcal:
        K, info = estimate_focal_selfcal(w0, K, images)
        report['selfcal'] = info
    saved.update(K=K.cpu().numpy(), da3=w0.cpu().numpy())

    def save():
        np.savez_compressed(result_path, **saved)
        report_path.write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if hasattr(x,'tolist') else str(x)))
    save()
    for name in args.variants:
        if name in report['variants'] and name in saved:
            continue
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        history = []
        print(f'[{scene}] {name} starting', flush=True)
        base = cv(saved['bae_d_raw']) if name == 'bae_de' else cv(saved['form_d_raw']) if name == 'form_de' else w0
        try:
            if name == 'bae_d':
                candidate = refine_odometry(base, depth, K, images, n_neighbors=args.neighbors,
                        min_baseline_frac=0.1, min_rot_deg=1.0, n_samples=args.samples,
                        history=history)
            elif name in ('bae_e', 'bae_de'):
                candidate = refine_structure(base, depth, K, images, history=history,
                        max_points_per_view=args.structure_points, voxel_length=0.02)
            elif name == 'form_d':
                # Parallel independent edges, each with one OpenMP thread. Avoid
                # nesting eight edge workers inside eight OpenMP workers each.
                previous_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                try:
                    candidate = reference.refine_poses_rgbd_pgo(base, depth, conf, K, images=images,
                        n_neighbors=args.neighbors, min_baseline_frac=0.1, min_rot_deg=1.0,
                        tex_gate=0, reject_on_regression=False, odometry_workers=args.reference_workers)
                finally:
                    torch.set_num_threads(previous_threads)
            elif name in ('form_e','form_de'):
                candidate = reference.refine_poses_rgbd_colormap(base, depth, conf, K,
                        images=images, maximum_iteration=100)
                # Correct only the reference implementation's global re-anchoring:
                # w_new[0] @ correction == base[0]. Relative poses are unchanged.
                c = as_44(candidate)
                candidate = c @ torch.linalg.inv(c[:1]) @ base[:1]
            else:
                raise ValueError(name)
            candidate = as_44(candidate).detach()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            saved[name+'_raw'] = candidate.cpu().numpy()
            report['variants'][name+'_raw'] = dict(metrics(candidate, gt), seconds=seconds)
            accepted_pose, accepted = trust_region_accept_reject(w0, candidate, depth, conf, K, images,
                       trust_n_pairs=512, trust_seed=args.seed, stage_name=name)
            saved[name] = as_44(accepted_pose).cpu().numpy()
            if history:
                saved[name+'_history'] = np.stack([as_44(h).detach().cpu().numpy() for h in history])
            report['variants'][name] = dict(metrics(accepted_pose, gt), seconds=seconds,
                    accepted=bool(accepted), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    steps=len(history))
            print(name, report['variants'][name], flush=True)
        except Exception as exc:
            report['variants'][name] = dict(error=f'{type(exc).__name__}: {exc}')
            save()
            raise
        save()
        del history
        gc.collect()
        torch.cuda.empty_cache()


def vars_config(a):
    return dict(seed=a.seed, neighbors=a.neighbors, samples=a.samples,
                structure_points=a.structure_points, selfcal=a.selfcal,
                baseline_frac=0.1, min_rot_deg=1.0, common_guard_pairs=512,
                reference_d_texture_gate=0, reference_d_internal_guard=False,
                reference_e_iterations=100, chain='raw D then E; guard entire proposal',
                ambient_grad=os.environ['BAE_USE_PYPOSE_AMBIENT_GRAD'],
                reference_workers=a.reference_workers, implementation_revision=4)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', nargs='+', default=['7831862f02'])
    p.add_argument('--out', type=Path, default=Path('outputs/rgbd_showcase'))
    p.add_argument('--variants', nargs='+', default=['bae_d','bae_e','bae_de','form_d','form_e','form_de'])
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--neighbors', type=int, default=16)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--structure-points', type=int, default=5000)
    p.add_argument('--reference-workers', type=int, default=8)
    p.add_argument('--selfcal', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    torch.cuda.set_device(args.device)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    for scene in args.scenes:
        benchmark(args, scene)
