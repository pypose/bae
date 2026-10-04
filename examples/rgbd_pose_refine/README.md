# RGB-D Pose Refinement

A `bae` example: refine a set of noisy camera poses for an RGB-D scene by
directly minimizing photometric (image-intensity) disagreement between
views, solved as a single joint sparse Levenberg–Marquardt problem on GPU.

[![Real DSLR reconstruction before and after raw pose refinement](docs/showcase/3f15a9266d/comparison.jpg)](docs/showcase/index.html)

**[Interactive DSLR showcase](docs/showcase/index.html)** ·
**[33-second room refinement film](docs/showcase/room_refinement.mp4)** ·
**[Full validation results](docs/validation/RESULTS.md)**

**[Full GIANT-1.1 validation with PyPose ambient gradients](VALIDATION.md)**
provides the complete split comparison. Three new gallery scenes use these
corrected runs: a monitor office, lecture room and kitchenette, each with 156
undistorted DSLR views sampled across the entire scene. The gallery includes
3D camera tours, detail sliders and matched D / E / D+E renders. Their raw
AUC@5 improves from 37.5 → 69.0, 56.9 → 73.8 and 45.9 → 65.2 respectively.

The approved bookshelf image above retains its original 144-view nested-model
run (AUC@5 4.9 → 22.6), labeled historical in the gallery. All selected examples
show raw refinement; the current photometric guard rejects those proposals.
Raw and returned results are reported separately.
See [SHOWCASE.md](SHOWCASE.md) for installation, reproduction, rendering details,
and the comparison with the real DA3 reference implementations.

## What this does

Given a set of RGB-D views with approximate initial poses (e.g. from visual-
inertial odometry, or any other rough estimator), this example refines those
poses in two stages, both implemented as native `bae` sparse-LM problems
(no external solver, no CPU fallback):

1. **Odometry** — treats every pair of nearby views as a photometric
   constraint (does the warped image from view *i* match view *j*?) and
   solves for *all* views' poses jointly in one sparse system, instead of
   pairwise-then-averaging.
2. **Structure** — fuses the current views into a single point cloud, then
   refines every pose against that shared structure.

An optional, pose/depth-free self-calibration step corrects camera intrinsics
before refinement. The iPhone CLI also checks texture and rejects each stage
if it worsens a held-out photometric score. That score is a proxy: acceptance
does not guarantee better ground-truth pose accuracy.

Everything (correspondence search, structure fusion, visibility resolution)
is implemented as batched GPU tensor ops — see the module docstrings in
`photometric.py`, `covis.py`, and `guards.py` for exactly how each step is
vectorized. See [SHOWCASE.md](SHOWCASE.md) for rendering real DSLR scenes.

## Install

```bash
pip install --no-build-isolation -v -e .        # base bae install, from repo root
pip install opencv-python lz4                    # this example's extra deps
```

The DSLR showcase dependencies and rendering commands are documented in
[SHOWCASE.md](SHOWCASE.md).

## Quickstart

Choose the path matching your inputs. All commands run from the repository root.

| Entry point | Inputs | Refinement and guard behavior |
|---|---|---|
| `run_refine.py` | iPhone RGB-D, LiDAR depth, ARKit initial poses | Texture gate; guard after each stage; structure starts from the accepted odometry result |
| `benchmark/showcase_prepare.py` → `benchmark/showcase_benchmark.py` | DSLR photographs → cached DA3 depth and poses | Raw D then E; guard the complete proposal; save both raw and guarded results |
| `space/` (optional hosted app) | Uploaded photographs or cached examples | Display raw D/E history and report a guard verdict using the plotted held-out scores |

The images above come from the **DSLR benchmark**, whose defaults use 512
samples per pair and baseline/rotation gates of 0.1/1°. The iPhone CLI uses
2,048 samples and gates of 0/0°. Both call the same refinement functions.

```bash
# Sanity-check the math/API first (fast, no dataset needed):
pytest examples/rgbd_pose_refine/tests/

# iPhone RGB-D refinement; replace the paths with your dataset locations.
python examples/rgbd_pose_refine/run_refine.py \
  --scene_id 7831862f02 --stage odometry+structure \
  --dataset_root /path/to/scannetpp \
  --frames_root /path/to/iphone_frames --video_root /path/to/iphone_videos

# BAE-only DSLR refinement, after preparing prediction.npz for this scene.
python examples/rgbd_pose_refine/benchmark/showcase_benchmark.py \
  --scenes 7831862f02 --variants bae_d bae_e bae_de
```

See [SHOWCASE.md](SHOWCASE.md) for DSLR prediction preparation and rendering,
or [VALIDATION.md](VALIDATION.md) for the full split protocol. Cached-prediction
BAE runs need no reference implementation; requesting `form_d`, `form_e`, or
`form_de` requires the separate `da3/refine_poses.py` checkout and dependencies.

The iPhone CLI prints a self-calibration line, a texture-gate verdict, per-step LM
loss, an accept/reject verdict per stage, and a pose-accuracy comparison
before/after, then saves the refined poses to `examples/rgbd_pose_refine/save/`.

## Dataset

The iPhone CLI expects ScanNet++ RGB-D captures (real LiDAR depth + IMU-based
initial pose/intrinsics). See `dataset.py` for the exact expected layout and
`--dataset_root`/`--frames_root`/`--video_root` to point at your own copy.
The DSLR benchmark uses `dslr.py` to read undistorted photographs and COLMAP
evaluation poses; DA3 supplies its initial depth, intrinsics, and poses.

## Files

| File | Role |
|---|---|
| `geometry.py` | Camera projection/back-projection/image-sampling primitives |
| `covis.py` | Which view pairs are close enough to constrain each other |
| `selfcal.py` | Pose/depth-free intrinsics self-calibration |
| `guards.py` | Texture gate + trust-region accept/reject |
| `photometric.py` | The two refinement stages (odometry, structure) |
| `dataset.py` | ScanNet++ iPhone data loader |
| `dslr.py` | ScanNet++ DSLR loader and undistortion |
| `eval.py` | Pose-accuracy evaluation against ground truth |
| `tests/_scene_fixture.py` | Textured-plane fixture for numerical correctness tests |
| `run_refine.py` | iPhone CLI entry point |
| [`benchmark/`](benchmark/README.md) | Prediction preparation, comparisons, split validation, reports |
| [`visualization/`](visualization/README.md) | Rendering, videos, tours, and gallery generation |
| `space/` | Optional hosted app and its deployment scripts; vendor copies are generated |
| `docs/` | Published galleries and validation artifacts; not imported by the solver |

## iPhone CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--scene_id` | required | ScanNet++ scene id |
| `--max_views` | 60 | frame subsample size |
| `--stage` | `odometry+structure` | `odometry`, `structure`, or both in sequence |
| `--selfcal` / `--no-selfcal` | on | intrinsics self-calibration |
| `--tex-gate` | 0.003 | texture-gate threshold (resolution-dependent, see `guards.py`) |
| `--reject-on-regression` | on | trust-region accept/reject per stage |
| `--pyramid-levels`, `--n-relin`, `--n-inner` | 3, `3,2,1`, 5 | coarse-to-fine schedule |
| `--device`, `--dtype` | `cuda`, `float64` | |
| `--out` | auto | output path |

Run `--help` for the full list (covisibility gating, structure-stage
sub-parameters, etc.).

## Notes on scope

Dense photometric refinement has a couple of known failure modes worth
knowing about before pointing this at a new dataset: it can degrade poses on
near-static/low-parallax view sets (an "aperture problem"-like ambiguity —
mitigate with `--min-baseline-frac`/`--min-rot-deg`), and the texture-gate
threshold is tuned for a specific image resolution. Both are documented
where they're handled, in `guards.py` and `covis.py`.
