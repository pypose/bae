# RGB-D Pose Refinement (Forms D / E / D+E) — GPU-native, `bae`-powered

Dense photometric camera-pose refinement for feed-forward 3D reconstruction
models, reimplemented from scratch as a fully GPU-vectorized `bae` example.

## 1. Motivation

Earlier experiments in this repo's sibling workspace (`da3/refine_poses.py`)
tried several hand-crafted bundle-adjustment-style pose-refinement strategies
for feed-forward reconstruction models (DepthAnything3, VGGT-Omega) — sparse
landmark BA, DUSt3R-style depth-consistency alignment, classical feature-track
BA (Forms A/B/C in that file). None of them reliably improved pose accuracy.

What *did* work was reimplementing two of Open3D's own CPU tutorials —
**RGB-D Odometry** and **Color Map Optimization** — as `refine_poses_rgbd_pgo`
(**Form D**) and `refine_poses_rgbd_colormap` (**Form E**), which call
Open3D's C++ pipeline directly. Ablations in `da3/FINDINGS_vggt_refine_bug.md`
show the gain is purely **photometric** (dense per-pixel image-intensity
matching, not depth as a residual signal — depth is only needed to know
*where* to sample). This example reimplements that proven algorithm natively
in `bae` — no Open3D dependency at runtime, single joint sparse
Levenberg–Marquardt solve on GPU instead of Open3D's CPU per-pair loop — and
runs it on real ScanNet++ iPhone RGB-D data instead of the sibling project's
DA3/VGGT-Omega predictions.

## 2. Algorithm

### Why this isn't a literal Open3D port

`bae`'s sparse-Jacobian tracer vmaps **every** tensor argument to a `@psjac`
function over dim 0 unconditionally (`bae/autograd/graph.py::_vmap_in_dims`,
confirmed by reading the source) — there is no way to pass a shared/broadcast
full image into a traced residual. So `sample_at`/`grid_sample` never appears
inside a `@psjac` function here. Instead, both residuals use
**inverse-compositional first-order photometric linearization**: once per
outer relinearization, under `torch.no_grad()`, sample the target intensity
and its image gradient at the current warp location; the `@psjac` residual
itself is then pure SE3/pinhole tensor algebra —
`r = I_i - (I_j + gx_j*du + gy_j*dv)` — which `jacrev`/`vmap` can differentiate
freely. This is mathematically what dense visual-odometry Gauss-Newton
already computes (image gradient × projection Jacobian), so it's a faithful
vectorization, not an approximation invented to dodge the framework.

### Form D — joint sparse-LM dense photometric odometry (`photometric.py`)

Open3D's RGB-D odometry tutorial runs a **per-pair** Gauss-Newton solve, then
feeds relative poses + information matrices into a **separate** global
pose-graph solver. `bae`'s strength is solving one large joint sparse system,
so Form D collapses both stages into **one joint gauge-fixed sparse LM
problem**: all views' poses (view 0 fixed as the gauge anchor,
`pp.SE3`/`trim_SE3_grad`, exactly `pgo.py`'s `PoseGraphFixedFirst` pattern) are
optimized together against **all** covisible-pair pixel correspondences at
once, coarse-to-fine over an image pyramid (mirroring Open3D's
`[20,10,5]`-iteration default). Adjacent-frame ("certain") edges get an extra
weight boost, approximating Open3D's information-matrix-weighted PGO with a
scalar per-row weight (a documented simplification — see §7).

### Form E — joint photometric BA against a fixed fused structure

No differentiable GPU mesh renderer exists in this stack, so Open3D's
"TSDF-fuse → render → refine extrinsics" becomes: fuse a voxel-downsampled
colored point cloud from all views at the current poses (fixed, non-optimized
buffer), then find each view's visible structure points via a GPU **z-buffer**
(project + sort-by-depth + keep nearest per pixel bin — respects occlusion,
unlike a 3D-KNN which would happily match through a wall), then jointly
refine all poses (structure held fixed) against that snapshot. Rebuilt once
per outer iteration.

### Form D+E and the mandatory self-calibration pre-step

`--form D+E` runs D to convergence, then feeds its (trust-region-accepted)
output into E. Before any of D/E/D+E, `selfcal.py` runs a **mandatory-by-default**
pose/depth-free focal self-calibration (`estimate_focal_selfcal`, ported from
`refine_poses.py`): for each covisible pair, estimate the fundamental matrix F
via RANSAC on SIFT matches (no poses, no depth), then grid-search a shared
focal multiplier minimizing Sturm's criterion `(s1-s2)/(s1+s2)` on
`E(m)=K(m)^T F K(m)`'s singular values. **Why mandatory**: Forms D/E unproject
every pixel through K, so a systematically wrong focal bends every ray and
corrupts the photometric optimum *regardless of pose*. The historical finding
that "D+E underperforms D alone" was traced to exactly this — VGGT-Omega's
intrinsics head regresses only 2 scalars (`fov_h`, `fov_w`) with a hard-forced
central principal point and under-predicts FOV by 11–21% on wide-FOV scenes;
under oracle intrinsics, D+E actually *beats* D alone. Self-calibration is
cheap (pure CPU RANSAC+SVD) and a near no-op when intrinsics are already
accurate, so it runs by default regardless of backbone (`--no-selfcal` to
disable).

### Texture gate and trust-region accept/reject (`guards.py`)

Two guards not in Open3D's tutorials, added because unguarded photometric
refinement regresses on a meaningful fraction of real scenes: a **texture
gate** skips refinement outright on scenes below a gradient-magnitude
threshold (flat regions carry no photometric signal), and a **trust-region
accept/reject** scores a fixed, independently-sampled (`wide_pairs`, *not*
the optimizer's own covisibility graph — that would be circular) photometric
objective before vs. after each stage and reverts on regression. Both apply
independently around Form D and Form E in a chain, so an E-caused regression
reverts to D's output, not all the way to the raw input.

## 3. Installation

Base `bae` install (repo root): `pip install --no-build-isolation -v -e .`

This example's extra dependencies:
```
pip install opencv-python lz4
```
(`opencv-python` for SIFT/RANSAC self-calibration; `lz4` for decoding
ScanNet++'s iPhone depth format.)

## 4. Dataset

Expects the raw official ScanNet++ download at `/data/zitong/scannetpp_val`
(**not** the DA3-BENCH preprocessed copy `da3/`'s own dataset code uses — that
has a different `merge_dslr_iphone` layout not present in this raw copy),
using the **iPhone** modality (real LiDAR depth + ARKit pose/intrinsics,
unlike DSLR which needs rendered depth from a mesh):

- `<scene>/iphone/colmap.zip` → COLMAP-refined poses/intrinsics, used as
  **pseudo-ground-truth** for the pose-AUC evaluation.
- `<scene>/iphone/depth.zip` → `depth.bin`, a sequence of 4-byte-length-prefixed,
  **LZ4-block-compressed** (raw-deflate zlib fallback) 192×256 uint16-mm depth
  frames — the official ScanNet++ toolkit format
  (`scannetpp/iphone/prepare_iphone_data.py::extract_depth`), decoded by
  streaming directly out of the zip (`dataset.py::iter_depth_frames`) without
  ever writing the ~575MB `depth.bin` to disk.
- `<scene>/iphone/pose_intrinsic_imu.zip` → per-frame ARKit camera-to-world
  pose + intrinsics, used as the **initial** (noisy) pose/K guess that Form
  D/E/self-cal refine. ARKit's right-up-back convention is converted to
  OpenCV's right-down-forward convention via a `diag(1,-1,-1,1)`
  right-multiplication (`dataset.py::arkit_to_opencv_pose`) — empirically
  verified against COLMAP pseudo-GT (`dataset.py::verify_arkit_convention`):
  0.35° mean relative-rotation disagreement on a real scene, confirming the
  flip is correct.
- RGB: already-extracted at `/data/zitong/scannetpp_iphone_frames/<scene_id>/frame_%06d.jpg`
  (no extraction needed), with a full-video `.mkv` fallback at
  `/data/zitong/scannetpp_iphone_256/<scene_id>.mkv`.

The two small zips (colmap, pose_intrinsic_imu) are extracted to a scratch
directory, parsed, and deleted; `depth.zip` is streamed and never extracted.
Decoded results are cached to one `<cache_root>/<scene_id>_mv<N>.pt` file per
scene+view-count so repeat runs skip re-decoding.

## 5. Quickstart

```bash
python examples/rgbd_pose_refine/tests/test_synthetic_convergence.py   # or: pytest examples/rgbd_pose_refine/tests/
python examples/rgbd_pose_refine/run_refine.py --scene_id 7831862f02 --form D+E
```

Expected console output: a self-calibration line (`[selfcal] ...`), a
texture-gate verdict, per-relinearization `bae` LM loss traces, a
trust-region accept/reject verdict per stage, and a final pose-AUC
before/after comparison. Output is saved to
`examples/rgbd_pose_refine/save/<scene_id>_<form>.pt`.

## 6. CLI reference (`run_refine.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--scene_id` | required | ScanNet++ scene id, e.g. `7831862f02` |
| `--max_views` | 60 | frame subsample size (linspace across the pre-extracted set) |
| `--form` | `D+E` | `D`, `E`, or `D+E` |
| `--selfcal` / `--no-selfcal` | on | mandatory-by-default focal self-calibration |
| `--tex-gate` | 0.003 | gradient-magnitude threshold (resolution-dependent, see §7) |
| `--tex-mode` | `covis` | `covis` (correspondence-restricted) or `global` |
| `--reject-on-regression` | on | trust-region accept/reject per stage |
| `--pyramid-levels`, `--n-relin`, `--n-inner` | 3, `3,2,1`, 5 | coarse-to-fine schedule |
| `--huber-delta` | 1.5 | IRLS robust weight (folded into the residual, see §7) |
| `--n-neighbors`, `--min-baseline-frac`, `--min-rot-deg` | 16, 0, 0 | covisibility graph gating |
| `--device`, `--dtype` | `cuda`, `float64` | |
| `--out` | auto | output `.pt` path |

Saved output: `{"w2c": (N,4,4), "K": (N,3,3), "frame_names": [...], "args": {...}, "auc_pre": {...}, "auc_post": {...}}`.

## 7. Known risks / simplifications (read before extending)

- **Image sampling outside all `@psjac` traces** (§2) is a faithful
  vectorization of dense visual odometry, not an approximation — but it does
  mean the linearization is only valid within roughly half a pixel-intensity
  "cycle" of the true warp per relinearization; `n_relin`/`pyramid_levels`
  control how often it's refreshed.
- **Vectorization boundary.** Correspondence building (`build_photo_correspondences`),
  structure fusion (`fuse_structure`), structure visibility
  (`zbuffer_visible_all_views`), and covisibility-graph neighbor selection
  (`build_covis_graph`) all batch across pairs/views/points in single tensor
  ops — no Python loop over pairs or views. The one place a small loop
  remains is the final image-intensity/gradient lookup, grouped by the
  (≤N-views) unique source/target images actually present: `F.grid_sample`
  pairs input batch row *b* with grid row *b*, so it cannot itself gather
  from a different source image per row — this is an intrinsic limitation
  of `grid_sample`'s batching model, not a missed vectorization opportunity,
  and its loop bound is the view count, never the (much larger) pair or
  sample count. `estimate_focal_selfcal`'s SIFT/RANSAC loops are similarly
  bounded by image/pair count but are OpenCV-bound (no batched tensor API
  exists for them), so they're a different kind of exception.
- **Scalar certain/uncertain edge weight** (Form D, §2) approximates Open3D's
  anisotropic 6×6 per-edge information matrix with an isotropic scalar boost.
  Form A/B in `da3/refine_poses.py` use the same order of approximation.
- **Form E's z-buffer visibility** is a GPU rasterization substitute for a
  differentiable mesh renderer, not a renderer itself; pixel-bin granularity
  (`--form-e-pixel-stride`) trades correspondence density against aliasing at
  structure silhouettes.
- **Small-baseline / near-static view sets can alias.** Dense photometric BA
  has a well-known degeneracy on low-parallax view pairs (an "aperture
  problem"-like ambiguity): if consecutive sampled frames have almost no
  camera motion, the photometric objective can decrease smoothly while poses
  slide into a geometrically wrong but photometrically near-equivalent
  configuration. This was directly observed while validating this example: a
  handful of temporally-adjacent, near-static ScanNet++ iPhone frames (real
  camera-center motion of a few millimeters between "consecutive" samples)
  caused Form D to *degrade* pose-AUC even as its own photometric loss
  decreased normally — not a Jacobian/API bug (the synthetic tests, which use
  a deliberately larger baseline, converge to near machine precision — see
  §8), but a property of the objective itself on degenerate inputs. Mitigate
  with `--min-baseline-frac`/`--min-rot-deg` (covisibility-graph gating,
  ported from `da3/refine_poses.py`'s own documented fix for the same issue)
  and by choosing `--max_views` samples with genuine parallax.
- **ARKit's raw VIO poses are a harder initialization regime than the
  feed-forward predictions Form D/E were originally validated against.**
  DA3/VGGT-Omega produce single-shot, globally-consistent-if-imperfect poses
  over a small, well-distributed view set (~60 views). ARKit integrates
  frame-to-frame visual-inertial odometry over a long, uncapped video with no
  loop closure, so sampling frames spread across a whole scan can carry
  substantial accumulated drift the dense photometric objective alone isn't
  designed to fully correct (Form D's own docstring already describes a
  "small convergence basin"). The texture gate and trust-region guard exist
  precisely to detect this and decline to refine rather than apply a bad
  update — during validation, both guards were observed correctly triggering
  (texture-gate skip on a low-texture scene; trust-region rejection on a
  scene with meaningful pre-existing drift) rather than silently degrading
  poses.
- **`--tex-gate` default (0.003) is resolution-dependent and was recalibrated
  for this example.** `da3/refine_poses.py`'s original `tex_gate=0.008` was
  calibrated at DA3's much smaller model-input resolution; the same physical
  scene measured at ScanNet++'s native 1920×1440 iPhone resolution shows
  substantially smaller pixel-to-pixel gradients (an edge's intensity change
  is spread over more pixels), so the original threshold was found, during
  validation, to reject essentially every real scene tested. 0.003 was set
  from empirical measurement across ~30 real ScanNet++ iPhone scenes
  (observed range ≈0.002–0.0075 at native resolution) — recalibrate further
  if you resize `images` before feeding them into this pipeline.

## 8. Validation

`tests/test_synthetic_convergence.py` (no dataset dependency, a few seconds,
CUDA required — `bae`'s sparse LM path uses CUDA-only sparse ops):

1. A smoke test asserting `photo_residual`'s `bae`-traced Jacobian w.r.t.
   `pose_i`/`pose_j` is nonzero and finite (catches the classic
   "accidentally detached tensor" bug when porting hand-written autograd code
   into `@psjac`).
2. A standalone check for Form E's `structure_residual` against a
   finite-difference Jacobian, independent of `bae`'s tracer entirely
   (isolates "is the math right" from "is `bae`'s tracer right").
3. An end-to-end synthetic scene (6 cameras, a textured slanted-plane-style
   frontal scene with deliberately large-period texture to avoid the aliasing
   failure mode in §7) where Form D recovers a known, deliberately perturbed
   ground-truth pose set: mean relative-rotation error 1.23° → 0.01°,
   mean camera-center error 0.080 → 0.002 in this repo's own test run.

Run this before trusting any real-data run — if it fails, the bug is in the
`bae` Jacobian/API wiring, not the dataset.

Additionally, `dataset.py::verify_arkit_convention(scene_dir)` checks the
ARKit→OpenCV pose conversion against COLMAP pseudo-GT on a handful of real
frames; run it once against a new dataset copy before trusting the convention
broadly.
