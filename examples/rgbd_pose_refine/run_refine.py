"""CLI entry point for GPU-native RGB-D pose refinement (Forms D / E / D+E).

Mirrors `pgo.py`'s explicit-construction style: load a scene, run the fixed
pipeline (self-calibration -> texture gate -> Form D/E/D+E -> trust-region
accept/reject per stage), then report a pose-AUC before/after comparison
against the COLMAP pseudo-GT. See README.md for the algorithm and a
quickstart.

Example:
    python examples/rgbd_pose_refine/run_refine.py --scene_id 7831862f02 --form D+E
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(EXAMPLE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from covis import build_covis_graph
from dataset import DEFAULT_DATASET_ROOT, DEFAULT_FRAMES_ROOT, DEFAULT_VIDEO_ROOT, load_scene
from eval import evaluate_against_colmap
from geometry import as_44
from guards import texture_gate, trust_region_accept_reject
from photometric import refine_form_d, refine_form_e
from selfcal import estimate_focal_selfcal

DTYPE_CHOICES = {"float64": torch.float64, "float32": torch.float32}


def parse_int_list(s: str):
    return tuple(int(x) for x in s.split(","))


def build_argparser():
    p = argparse.ArgumentParser(description="GPU-native RGB-D pose refinement (bae)")
    p.add_argument("--scene_id", type=str, required=True)
    p.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--frames_root", type=str, default=DEFAULT_FRAMES_ROOT)
    p.add_argument("--video_root", type=str, default=DEFAULT_VIDEO_ROOT)
    p.add_argument("--cache_root", type=str, default=None)
    p.add_argument("--max_views", type=int, default=60)

    p.add_argument("--form", type=str, default="D+E", choices=["D", "E", "D+E"])
    p.add_argument("--selfcal", dest="selfcal", action="store_true")
    p.add_argument("--no-selfcal", dest="selfcal", action="store_false")
    p.set_defaults(selfcal=True)

    p.add_argument("--tex-gate", type=float, default=0.003,
                    help="Gradient-magnitude threshold. NOTE: resolution-dependent -- "
                         "recalibrated for native ScanNet++ iPhone resolution (1920x1440); "
                         "da3/refine_poses.py's 0.008 default was calibrated at DA3's much "
                         "smaller model input resolution and is too strict here.")
    p.add_argument("--tex-mode", type=str, default="covis", choices=["covis", "global"])
    p.add_argument("--reject-on-regression", dest="reject_on_regression", action="store_true")
    p.add_argument("--no-reject-on-regression", dest="reject_on_regression", action="store_false")
    p.set_defaults(reject_on_regression=True)
    p.add_argument("--trust-pairs", type=str, default="wide", choices=["wide"])
    p.add_argument("--trust-n-pairs", type=int, default=4096)
    p.add_argument("--trust-min-rot", type=float, default=20.0)
    p.add_argument("--trust-seed", type=int, default=0)

    p.add_argument("--pyramid-levels", type=int, default=3)
    p.add_argument("--n-relin", type=str, default="3,2,1",
                    help="comma-separated relinearizations per pyramid level, coarse->fine")
    p.add_argument("--n-inner", type=int, default=5, help="bae LM steps per relinearization")
    p.add_argument("--huber-delta", type=float, default=1.5)
    p.add_argument("--certain-boost", type=float, default=3.0)
    p.add_argument("--n-neighbors", type=int, default=16)
    p.add_argument("--min-baseline-frac", type=float, default=0.0)
    p.add_argument("--min-rot-deg", type=float, default=0.0)
    p.add_argument("--n-samples", type=int, default=2048)

    p.add_argument("--voxel-length", type=float, default=0.02)
    p.add_argument("--form-e-outer", type=int, default=3)
    p.add_argument("--form-e-inner", type=int, default=5)
    p.add_argument("--form-e-pixel-stride", type=int, default=4)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float64", choices=tuple(DTYPE_CHOICES.keys()))
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    dtype = DTYPE_CHOICES[args.dtype]
    n_relin = parse_int_list(args.n_relin)
    if len(n_relin) != args.pyramid_levels:
        raise ValueError("--n-relin must have exactly --pyramid-levels entries")

    print(f"[run_refine] loading scene {args.scene_id} (max_views={args.max_views})")
    scene = load_scene(args.scene_id, dataset_root=args.dataset_root,
                        frames_root=args.frames_root, video_root=args.video_root,
                        cache_root=args.cache_root, max_views=args.max_views)
    images = scene["images"]
    depth = torch.from_numpy(scene["depth"]).to(args.device, dtype)
    K = torch.from_numpy(scene["K_init"]).to(args.device, dtype)
    w2c_init = torch.from_numpy(scene["w2c_init"]).to(args.device, dtype)
    N = images.shape[0]
    print(f"[run_refine] {N} views, image {images.shape[1]}x{images.shape[2]}")

    auc_pre = evaluate_against_colmap(w2c_init, scene["frame_names"], scene["gt_w2c"])
    print(f"[run_refine] pre-refinement AUC: {auc_pre}")

    K_use = K
    if args.selfcal:
        K_use, info = estimate_focal_selfcal(w2c_init, K, images)
    else:
        print("[selfcal] disabled by --no-selfcal")

    pairs = build_covis_graph(w2c_init, n_neighbors=args.n_neighbors,
                               min_baseline_frac=args.min_baseline_frac,
                               min_rot_deg=args.min_rot_deg)
    conf = None  # ScanNet++ iPhone depth has no confidence channel
    ok = texture_gate(images, w2c_init, depth, conf, K_use, pairs,
                       tex_gate=args.tex_gate, mode=args.tex_mode)
    if not ok:
        print("[run_refine] scene failed the texture gate; leaving poses unchanged")
        w2c_final = w2c_init
    else:
        w2c_cur = w2c_init
        for stage in args.form.split("+"):
            if stage == "D":
                w2c_next = refine_form_d(
                    w2c_cur, depth, K_use, images, n_neighbors=args.n_neighbors,
                    min_baseline_frac=args.min_baseline_frac, min_rot_deg=args.min_rot_deg,
                    n_samples=args.n_samples, pyramid_levels=args.pyramid_levels,
                    n_relin=n_relin, n_inner=args.n_inner, huber_delta=args.huber_delta,
                    certain_boost=args.certain_boost, dtype=dtype)
            elif stage == "E":
                w2c_next = refine_form_e(
                    w2c_cur, depth, K_use, images, voxel_length=args.voxel_length,
                    pixel_stride=args.form_e_pixel_stride, n_outer=args.form_e_outer,
                    n_inner=args.form_e_inner, huber_delta=args.huber_delta, dtype=dtype)
            else:
                raise ValueError(f"unknown form stage {stage!r}")

            if args.reject_on_regression:
                w2c_next, accepted = trust_region_accept_reject(
                    as_44(w2c_cur), as_44(w2c_next), depth, conf, K_use, images,
                    trust_pairs=args.trust_pairs, trust_n_pairs=args.trust_n_pairs,
                    trust_min_rot=args.trust_min_rot, trust_seed=args.trust_seed,
                    stage_name=stage)
            w2c_cur = w2c_next
        w2c_final = w2c_cur

    auc_post = evaluate_against_colmap(w2c_final, scene["frame_names"], scene["gt_w2c"])
    print(f"[run_refine] post-refinement ({args.form}) AUC: {auc_post}")
    for k in sorted(set(auc_pre) & set(auc_post)):
        print(f"[run_refine] lift {k}: {auc_post[k] - auc_pre[k]:+.4f}")

    out_path = args.out or str(EXAMPLE_DIR / "save" / f"{args.scene_id}_{args.form.replace('+', 'plus')}.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"w2c": as_44(w2c_final).detach().cpu(), "K": K_use.detach().cpu(),
                "frame_names": scene["frame_names"], "args": vars(args),
                "auc_pre": auc_pre, "auc_post": auc_post}, out_path)
    print(f"[run_refine] saved refined poses to {out_path}")


if __name__ == "__main__":
    main()
