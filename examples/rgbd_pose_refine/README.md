# DSLR RGB-D pose refinement

Refine DA3 camera poses from ScanNet++ DSLR photographs using `bae` on CUDA.
DA3 supplies depth, intrinsics, and initial poses. COLMAP camera calibration is
used to undistort input images; COLMAP poses are used only for evaluation.

The native solver has two stages:

- **D (odometry):** jointly optimize camera poses over photometric constraints
  between covisible views.
- **E (structure):** fuse a fixed point structure, then refine camera poses
  against it, rebuilding the structure between outer iterations.

`bae_de` chains raw D into E. Each proposal is scored against the initialization
using a held-out photometric guard. Both raw and guarded results are saved;
photometric acceptance does not guarantee better ground-truth pose accuracy.

## Run

Commands run from the repository root. For BAE-only refinement of cached
predictions, install `bae` and OpenCV:

```bash
python -m pip install --no-build-isolation -v -e .
python -m pip install opencv-python-headless
python -m pytest examples/rgbd_pose_refine/tests/ -q

python examples/rgbd_pose_refine/benchmark/showcase_benchmark.py \
  --scenes 7831862f02 --variants bae_d bae_e bae_de
```

The benchmark reads `outputs/rgbd_showcase/<scene>/prediction.npz`. To create
that cache, first install the DA3 dependencies in [SHOWCASE.md](SHOWCASE.md):

```bash
python examples/rgbd_pose_refine/benchmark/showcase_prepare.py \
  --scenes 7831862f02 --dataset-root /path/to/scannetpp \
  --out outputs/rgbd_showcase --device 0
```

The dataset must contain:

```text
<dataset-root>/data/<scene>/dslr/colmap.zip
<dataset-root>/data/<scene>/dslr/resized_images.zip
```

The COLMAP archive contains `colmap/cameras.txt` and `colmap/images.txt`.
Preparation writes `prediction.npz` and `input.json`. Refinement writes
`refined.npz` (poses and optimization histories) and `metrics.json` (AUC,
timings, and guard verdicts) beside the predictions. Outputs stay under the
ignored `outputs/` directory.

## Options

| Benchmark flag | Default | Purpose |
|---|---|---|
| `--scenes` | `7831862f02` | Scene IDs with cached predictions |
| `--out` | `outputs/rgbd_showcase` | Prediction/refinement cache root |
| `--variants` | All six methods | Use `bae_d bae_e bae_de` for BAE-only runs |
| `--selfcal` / `--no-selfcal` | on | Intrinsics self-calibration |
| `--neighbors` | 16 | Covisibility neighbors |
| `--samples` | 512 | Odometry pixels sampled per pair |
| `--structure-points` | 5000 | Maximum structure points per view |
| `--device` | 0 | CUDA device index |
| `--force` | off | Recompute cached refinement |

`form_d`, `form_e`, and `form_de` compare against the separate DA3/Open3D
reference implementation. They require `da3/refine_poses.py`; BAE-only runs do
not import it. See [SHOWCASE.md](SHOWCASE.md) for the comparison protocol and
[VALIDATION.md](VALIDATION.md) for full-split evaluation.

## Code map

| Files | Purpose |
|---|---|
| `photometric.py` | BAE residuals and refinement stages |
| `geometry.py`, `covis.py` | Camera geometry and view pairing |
| `guards.py`, `selfcal.py` | Held-out scoring and intrinsics self-calibration |
| `dslr.py`, `eval.py` | DSLR loading and pose accuracy |
| `benchmark/` | Prediction preparation, refinement, and split evaluation |
| `tests/` | Synthetic convergence, derivatives, scoring, and loading checks |
