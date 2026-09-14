"""Cache one globally attended DA3 prediction from scene-wide undistorted DSLR views.

Adaptive search finds the largest successful view count within --max-views,
to --view-step precision. Every attempt spans the entire registered sequence.
COLMAP poses are saved only for evaluation, never passed to the predictor.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from dslr import list_dslr_frames, load_dslr_scene


def prepare(args):
    from depth_anything_3.api import DepthAnything3
    from huggingface_hub import try_to_load_from_cache
    torch.set_num_threads(8)
    torch.cuda.set_device(args.device)
    model = DepthAnything3.from_pretrained(args.model, revision=args.revision).to('cuda').eval()
    config_path = try_to_load_from_cache(args.model, 'config.json', revision=args.revision)
    revision = Path(config_path).parent.name if isinstance(config_path, str) else args.revision
    for scene in args.scenes:
        out = args.out / scene
        out.mkdir(parents=True, exist_ok=True)
        cache = out / 'prediction.npz'
        if cache.exists() and not args.force:
            print(f'{scene}: cached; use --force to replace', flush=True)
            continue
        all_names = list_dslr_frames(scene, n_views=100000, dataset_root=args.dataset_root)
        cap = min(args.max_views, len(all_names))
        low, high, count = 0, cap + 1, min(args.initial_views or cap, cap)
        attempts = []
        selected_names = []
        while True:
            names = [all_names[i] for i in np.linspace(0, len(all_names)-1, count).round().astype(int)]
            images, undist_K, gt = load_dslr_scene(scene, names, dataset_root=args.dataset_root,
                                                   max_size=args.process_res * 2)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            print(f'{scene}: DA3 {count}/{len(all_names)} scene-wide DSLR frames, res={args.process_res}', flush=True)
            succeeded = False
            try:
                pred = model.inference(image=list(images), process_res=args.process_res,
                                       infer_gs=False, export_dir=None)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2**30
                np.savez_compressed(cache, images=pred.processed_images, depth=pred.depth,
                                    conf=pred.conf, K=pred.intrinsics, w2c=pred.extrinsics,
                                    gt=gt, names=np.array(names), undist_K=undist_K)
                del pred
                low = count
                selected_names = names
                succeeded = True
                print(f'{scene}: success, peak {peak:.2f} GiB', flush=True)
            except torch.cuda.OutOfMemoryError:
                peak = torch.cuda.max_memory_allocated() / 2**30
                high = count
                print(f'{scene}: OOM at {count}; searching smaller global sample', flush=True)
            attempts.append(dict(views=count, success=succeeded, peak_allocated_gib=peak,
                                 seconds=time.perf_counter()-start))
            del images
            gc.collect()
            torch.cuda.empty_cache()
            meta = dict(scene=scene, model=args.model, checkpoint_revision=revision, process_res=args.process_res,
                        registered_views=len(all_names), selected_views=low,
                        selected_frame_names=selected_names,
                        sampling='linspace over all registered DSLR names, including both endpoints',
                        undistortion='COLMAP OPENCV_FISHEYE, OpenCV balance=0, before resize',
                        input_pose_or_intrinsics_conditioning=False,
                        gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''),
                        attempts=attempts, view_step=args.view_step, max_views=cap,
                        initial_views=args.initial_views)
            (out/'input.json').write_text(json.dumps(meta, indent=2))
            if low == cap or (low > 0 and high-low <= args.view_step):
                break
            if high <= 3:
                raise RuntimeError('DA3 cannot fit even a two-view scene')
            count = min(cap, low+args.view_step) if high == cap+1 and low else max(2, (low+high)//2)
        print(f'{scene}: saved {low} globally attended frames to {cache}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', nargs='+', default=['7831862f02'])
    p.add_argument('--dataset-root', default='/data/zitong/scannetpp_val')
    p.add_argument('--out', type=Path, default=Path('outputs/rgbd_showcase'))
    p.add_argument('--model', default='depth-anything/DA3NESTED-GIANT-LARGE')
    p.add_argument('--revision', help='Optional Hugging Face checkpoint commit to pin')
    p.add_argument('--process-res', type=int, default=504)
    p.add_argument('--max-views', type=int, default=192)
    p.add_argument('--initial-views', type=int, help='Start near a previously measured GPU budget, then probe upward')
    p.add_argument('--view-step', type=int, default=8)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    if args.max_views < 2 or args.view_step < 1:
        p.error('max-views must be >=2 and view-step >=1')
    if args.initial_views is not None and args.initial_views < 2:
        p.error('initial-views must be >=2')
    if args.scenes == ['all']:
        args.scenes = sorted(p.name for p in (Path(args.dataset_root)/'data').iterdir()
                             if (p/'dslr/colmap.zip').exists())
    prepare(args)
