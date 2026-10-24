# DSLR reproduction and comparison protocol

See [README.md](README.md) for BAE-only refinement of cached predictions.
Generating predictions also requires Depth Anything 3. The experiments used a
local checkout at `da3/`; reference comparisons additionally need the custom
`refine_poses.py` in that checkout. Those reference scripts are not bundled
with this example, and a stock DA3 install does not supply them.

## Recorded environment

The experiments used CUDA PyTorch 2.7.0a0, Warp 1.17.0, and Open3D 0.19.0.
The PyPose revision below avoids the released wheel's `bae==0.2` dependency:

```bash
python -m pip install --no-deps git+https://github.com/pypose/pypose.git@afedf8a61927825bb9835e238217dfc36c80cc0e
python -m pip install 'warp-lang>=1.13.0' 'nvidia-cudss-cu12<0.9'
python -m pip install --no-build-isolation -v -e .
python -m pip install opencv-python-headless
python -m pip install -r examples/rgbd_pose_refine/requirements.txt
python -m pip install --no-deps moviepy==1.0.3 -e da3
```

`requirements.txt` supplies DA3 inference/reference dependencies. MoviePy is
needed by that checkout's eager export imports even though this example does
not export videos. Cached-prediction BAE runs need only the base installation
and OpenCV described in the README.

## Inputs

`benchmark/showcase_prepare.py` undistorts DSLR images using COLMAP's fisheye
calibration before inference. It samples registered image names uniformly
across the full scene, including both endpoints, and runs one globally
attended DA3 prediction. Its view-count search backs off on CUDA OOM; it does
not split scenes into windows or condition DA3 on COLMAP poses.

`input.json` records the checkpoint revision, selected frame names, resolution,
memory attempts, and successful view count. Reuse a cache only for the same
scene/model/resolution; choose another `--out` or use `--force` to replace it.

## Comparisons

- Every method consumes identical cached RGB, depth, confidence, intrinsics,
  and initial poses. Self-calibration and the random seed are shared.
- Native D solves one joint sparse photometric LM problem. Reference D calls
  Open3D pairwise RGB-D odometry followed by pose-graph optimization.
- Native E uses a voxel-averaged point structure rebuilt three times.
  Reference E uses a TSDF mesh and 100 rigid color-map iterations.
- D+E chains raw D into E. The whole proposal is compared with the initial
  poses using a common 512-pair photometric guard. Reference D's internal
  texture gate and guard are disabled for this comparison.
- Reference E is re-anchored to camera zero with
  `w @ inv(w[0]) @ initial[0]`; relative-pose accuracy is unchanged.
- AUC uses every unordered camera pair and sign-ambiguous translation
  direction. JSON values are fractions. Raw regressions and rejected proposals
  are retained; the formulations are not numerically equivalent.

```bash
# BAE only, using cached predictions:
python examples/rgbd_pose_refine/benchmark/showcase_benchmark.py \
  --scenes 7831862f02 --variants bae_d bae_e bae_de

# All six methods, when the reference checkout is available:
python examples/rgbd_pose_refine/benchmark/showcase_benchmark.py \
  --scenes 7831862f02
```

The source example contains no HTML gallery, rendering application, or generated
media. Numerical outputs and reports are generated locally under `outputs/`.
