# Scene-wide DSLR showcase

The gallery now includes three new scenes from the complete validation run with
corrected PyPose ambient gradients. The approved bookshelf office and home office
retain their original nested-model runs and are labeled historical. See
[VALIDATION.md](VALIDATION.md) for the complete 50-scene evaluation.

Open [the offline gallery](docs/showcase/index.html), or serve it from the repository root:

```bash
python -m http.server 8765 --directory examples/rgbd_pose_refine/docs/showcase
# http://localhost:8765
```

The gallery provides a before/after slider, a timeline of actual optimizer states,
a synchronized camera-center plot, MP4s with camera trajectories and pose AUC,
and raw-versus-guarded Form D / E / D+E comparisons. It works without a backend,
a GPU, or an Internet connection once rendered.

## New room tours

Watch the [33-second, 1080p film](docs/showcase/room_refinement.mp4). Each chapter
combines a synchronized 3D camera move, actual optimizer states, and a second
viewpoint focused on room details. The gallery adds a separate detail slider and
six matched renders for native and Open3D D / E / D+E.

| Scene | What to inspect | DSLR views | Raw AUC@5, initial → bae D+E |
| --- | --- | ---: | ---: |
| [Monitor office](docs/showcase/c50d2d1d42/comparison.jpg) | Monitor bezels, keyboard and desk edges | 156 / 327 | 37.46 → 68.99 |
| [Lecture room](docs/showcase/38d58a7a31/detail_comparison.jpg) | Long table edges and window frames | 156 / 410 | 56.91 → 73.78 |
| [Kitchenette](docs/showcase/40aec5fffa/comparison.jpg) | Cabinet handles, seams and red door frame | 156 / 219 | 45.85 → 65.18 |

All three use DA3-GIANT-1.1, 504-pixel inference, and the existing validation
caches with `BAE_USE_PYPOSE_AMBIENT_GRAD=1`. The scenes were selected after a
50-scene photograph survey and paired reconstruction previews of five candidate
scenes, with 20 viewpoints per candidate. This is a curated demonstration;
the complete validation report includes every scene and regressions.

The presentation camera moves sideways by at most 3.5% of the median source
depth while looking at a fixed target. Both maps share the exact camera path.
Optimizer poses are never interpolated or exaggerated. The selected source
photograph is excluded from the geometry for each viewpoint. Missing surfaces
and residual DA3 depth artifacts remain visible. The portable `story.json`
records every moving render camera, selected solver state and excluded view.

Rebuild these artifacts from the completed validation caches:

```bash
python examples/rgbd_pose_refine/showcase_render.py --out outputs/rgbd_validation_giant11 \
  --scene c50d2d1d42 --variant bae_de_raw --view 40 --width 1200 --height 900 --video --history-stride 1
python examples/rgbd_pose_refine/showcase_render.py --out outputs/rgbd_validation_giant11 \
  --scene 38d58a7a31 --variant bae_de_raw --view 16 --width 1200 --height 900 --video --history-stride 1
python examples/rgbd_pose_refine/showcase_render.py --out outputs/rgbd_validation_giant11 \
  --scene 40aec5fffa --variant bae_de_raw --view 48 --width 1200 --height 900 --video --history-stride 1
python examples/rgbd_pose_refine/showcase_stories.py
python examples/rgbd_pose_refine/showcase_gallery.py \
  outputs/rgbd_validation_giant11/c50d2d1d42 \
  outputs/rgbd_validation_giant11/38d58a7a31 \
  outputs/rgbd_validation_giant11/40aec5fffa \
  outputs/rgbd_showcase/3f15a9266d outputs/rgbd_showcase/0d2ee665be
```

Viewpoints and captions are stored in `showcase_stories.json`. The original
bookshelf assets are preserved; only the shared gallery presentation changes.

An [exterior lounge comparison](docs/showcase/lounge_overview.jpg) is also
available. Its ceiling and front cutaway planes are identical on both sides.

**These demonstrations show raw optimizer outputs.** On the selected scenes,
pose accuracy improves but the existing wide-pair photometric guard rejects the
updates. The gallery explicitly reports this; running the guarded pipeline
returns the initial poses. See [the measured report](docs/showcase/RESULTS.md).
This is evidence of solver progress and a guard limitation, not evidence that
the default guarded pipeline is ready to deploy.

## Inputs and comparison protocol

- Actual ScanNet++ DSLR photographs, undistorted with each scene's COLMAP
  `OPENCV_FISHEYE` calibration (`balance=0`) **before** DA3 inference.
- Uniform samples across **all registered image names**, including both
  endpoints. No sliding windows or fabricated pose perturbations.
- One globally attended DA3 forward pass per scene. Search starts at the requested
  maximum and bisects on CUDA OOM until the largest successful count is bracketed
  to eight frames. The cap and resolution remain explicit quality/budget choices;
  this is not a claim of an exact hardware maximum. Each scene's `input.json`
  records the actual frame names, attempts, memory peaks and checkpoint.
- 144 views fit the initial nested-model scenes at 504-pixel inference on a 24 GB
  RTX 4090; 150 failed. `DA3-GIANT-1.1` fit 156 and failed at 162. Another concurrent
  GPU workload can change the usable budget. Run inference on an otherwise idle GPU.
- The benchmark caches DA3 RGB, confidence, depth, extrinsics and intrinsics in
  `prediction.npz`. All methods use those exact arrays, the same self-calibrated
  intrinsics, graph gating and random seed. Ground truth is never an optimizer
  input. Native optimization uses 512 gradient-biased samples per graph edge,
  keeping the global camera set while limiting Jacobian memory.
- Native D is joint sparse photometric LM; reference D calls the real
  `da3/refine_poses.py::refine_poses_rgbd_pgo` (dense pairwise Open3D odometry,
  then PGO). These objectives are related, not numerically equivalent.
- Native E uses a voxel-averaged point structure rebuilt three times; reference E
  uses a TSDF mesh and 100 Open3D rigid color-map iterations. D+E chains raw D into
  E, then the **whole proposal** is checked against the initialization. Reference
  D's internal guard/texture skip is disabled for this explicit comparison;
  all candidates receive the same external 512-wide-pair guard. This differs
  from the reference CLI's default covisibility guard.
- Reference E's output is re-anchored to input camera zero with
  `w @ inv(w[0]) @ initial[0]`. This corrects a global gauge discrepancy in the
  reference helper and leaves relative-pose AUC unchanged.
- Raw proposals and returned poses are both recorded, including regressions.
  Relative-pose AUC uses every unordered camera pair, with the same sign-ambiguous
  translation-direction convention as the DA3 evaluation. AUC is reported as a
  percentage in the gallery and a fraction in JSON.

## Rendering

The primary renderer uses continuous triangles from the original DA3 depth,
textured with 1512-pixel **undistorted source DSLR photographs**. It draws all
scene frames, excluding the selected render-source frame so its own photograph
cannot simply cover alignment errors. Color blending preferences are computed
from the initial poses and rendering camera and are fixed across optimization.
Confidence thresholds, depth-edge rejection and depth blending tolerances are
also fixed. The virtual camera, depth maps, intrinsics and source pixels are
identical before and after. Only optimized poses change.

`--renderer mesh` provides confidence-filtered TSDF surfaces; `--overview ANGLE`
adds an exterior cutaway using a fixed ceiling plane and backface removal.
`--front-cut 0.6` optionally removes the foreground with another fixed plane.
`--renderer splat` is useful for faster geometric diagnostics. Render provenance
is saved in `renders/render.json`. The gallery shows selected attractive views;
failed/visually poor scenes remain in the quantitative report. Bright magenta
areas are anonymization already present in the supplied DSLR imagery.

Videos include a camera-center projection and AUC trace. The camera plot aligns
initial centers to COLMAP once with Sim(3), then uses that **same** alignment for
every optimizer state. Motion is not exaggerated. `--history-stride 1` renders
every stored step; larger values select actual states without interpolation.

## Reproduce

The environment was installed and exercised with CUDA PyTorch 2.7.0a0 (the
container's existing build), Warp 1.17.0, Open3D 0.19.0 and the following PyPose
revision. The released PyPose wheel demanded `bae==0.2`, so the compatible git
revision is necessary for this checkout's `bae==0.2.5`.

```bash
apt-get update
apt-get install -y libegl1 libgl1 libgl1-mesa-dri fonts-dejavu-core
python -m pip install --no-deps git+https://github.com/pypose/pypose.git@afedf8a61927825bb9835e238217dfc36c80cc0e
python -m pip install 'warp-lang>=1.13.0' 'nvidia-cudss-cu12<0.9'
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-build-isolation -v -e .
python -m pip install -r examples/rgbd_pose_refine/requirements-showcase.txt
python -m pip install --no-deps moviepy==1.0.3 -e da3
# This container already provides cv2 4.10.0; otherwise install opencv-python-headless.
```

`xformers` and `gsplat` are not needed for this path: DA3 uses PyTorch SDPA,
and the showcase uses EGL triangle rendering. The legacy MoviePy dependency
is only imported by DA3's unused export modules; MP4 encoding here uses imageio.

```bash
# Prepare on an idle GPU. --scenes all processes all mounted DSLR scenes.
OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python examples/rgbd_pose_refine/showcase_prepare.py \
  --scenes 3f15a9266d 0d2ee665be --max-views 192 --device 0

# All native and actual Open3D reference formulations, with cached progress.
OMP_NUM_THREADS=8 python examples/rgbd_pose_refine/showcase_benchmark.py \
  --scenes 3f15a9266d 0d2ee665be --device 0

# Render the office, including every optimizer state and raw formulation atlas.
OMP_NUM_THREADS=8 python examples/rgbd_pose_refine/showcase_render.py \
  --scene 3f15a9266d --variant bae_de_raw --view 68 \
  --renderer textured --video --history-stride 1
OMP_NUM_THREADS=8 python examples/rgbd_pose_refine/showcase_render.py \
  --scene 0d2ee665be --variant bae_de_raw --view 66 \
  --renderer textured --video --history-stride 1

python examples/rgbd_pose_refine/showcase_gallery.py \
  outputs/rgbd_showcase/3f15a9266d outputs/rgbd_showcase/0d2ee665be

# Re-run the checkpoint used in the earlier DA3 experiments in a separate cache.
python examples/rgbd_pose_refine/showcase_prepare.py \
  --scenes 7831862f02 3f15a9266d --model depth-anything/DA3-GIANT-1.1 \
  --out outputs/rgbd_showcase_giant11 --max-views 192
python examples/rgbd_pose_refine/showcase_benchmark.py \
  --scenes 7831862f02 3f15a9266d --out outputs/rgbd_showcase_giant11

python -m pytest examples/rgbd_pose_refine/tests/ -q
```

Run `showcase_render.py --preview` to create a contact sheet of candidate views.
Re-use an output directory only for its original scene/model/resolution; `--force`
replaces its cache. Benchmark configuration mismatches fail explicitly. Predictions,
meshes and high-resolution texture caches stay under ignored `outputs/`; the
portable rendered gallery is under `docs/showcase/`.

## Changes to the native example

- IRLS recomputes weights each step. Both stages now refresh LM's cached loss
  for the new weights, avoiding comparisons between different objectives.
- The signed image gradient now actually uses the central difference documented
  by the helper, with one-sided border differences.
- Structure visibility projects bounded batches of eight cameras, preserving
  exactly the same per-view z-buffer result while avoiding giant all-view
  intermediate tensors. The solve still optimizes every camera jointly.
- Structure now records optional per-step pose history, like odometry.

The tests cover synthetic convergence, analytic residual derivatives, batched
visibility equivalence, image-gradient finite differences, and refreshing a
stale LM loss after IRLS changes. The CPU `acos` kernel in this NVIDIA container
segfaulted on some larger evaluation arrays; the showcase keeps pose evaluation
on CUDA and independently computes video AUC with NumPy.
