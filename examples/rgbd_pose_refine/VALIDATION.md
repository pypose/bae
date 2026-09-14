# Complete DSLR validation

Install dependencies as described in [SHOWCASE.md](SHOWCASE.md). The split
runner uses the Hugging Face cache offline. On a fresh machine, cache the pinned
checkpoint once before starting it:

```bash
python -c "from depth_anything_3.api import DepthAnything3; DepthAnything3.from_pretrained('depth-anything/DA3-GIANT-1.1', revision='72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19')"
```

Run all 50 scenes from the mounted `nvs_sem_val.txt` split:

```bash
python examples/rgbd_pose_refine/validate_split.py
python examples/rgbd_pose_refine/validation_report.py
```

The runner uses both 24 GB RTX 4090 GPUs, with one isolated scene worker per GPU.
It pins `DA3-GIANT-1.1` to commit
`72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19`. Each scene's input spans the complete
registered DSLR sequence. Images are undistorted before inference. The global
attention view search starts near the measured 156-frame budget, probes upward
to a cap of 192, and bisects on OOM with eight-frame precision. No sliding
windows are used. Cached inputs retain their original search attempts.
The runner enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, as in the
showcase commands, to avoid allocator fragmentation shrinking the usable budget.

`outputs/rgbd_validation_giant11/manifest.json` lists every expected scene and
its state. Each scene has its own `run.log`, prediction, input metadata, raw
and guarded optimizer poses, and metrics. Rerunning the command resumes the
same configuration; it does not count failures as successful scenes.

The [validation report](docs/validation/RESULTS.md) uses the original DA3
evaluator to rescore every method on identical unordered camera pairs. It
reports both macro AUC (equal scene weight) and pooled AUC (equal pair weight),
per-scene results, native/reference gaps, and differences between the example
and DA3 metric implementations. Raw and guarded results are kept separate.

## Gradient correctness

The example enables this before importing bae:

```python
import os
os.environ.setdefault('BAE_USE_PYPOSE_AMBIENT_GRAD', '1')
```

`photometric.py` also installs the enabled patch when bae was imported earlier.
The validation runner rejects an explicitly disabled setting. This is required
for correct derivatives of composed PyPose Lie operations and their conversion
to the optimizer's local SE(3) update. The original visual demonstration was
created without this setting and retains its original nested-model checkpoint
and inputs. Its AUCs describe that run; use the GIANT-1.1 validation report for
complete split claims. The corrected gradient setting alone left native D's
AUC essentially unchanged on the two cached GIANT-1.1 control scenes.

Checks include nonidentity composed-pose sparse Jacobians against finite
differences, synthetic RGBD convergence, Lie-operation ambient gradients, and
three PGO convergence problems with final costs calibrated against Ceres:

```bash
python -m pytest -q examples/rgbd_pose_refine/tests \
  tests/autograd/test_pypose_ambient_grad.py \
  tests/autograd/test_pgo_convergence.py
```

## What can be compared

All six methods use identical cached DA3 images, depths, confidence, predicted
poses, self-calibrated intrinsics, and frame selections. The benchmark follows
the [showcase comparison protocol](SHOWCASE.md): raw D, raw E, and raw D then E,
each followed by the same wide-pair photometric guard. Reference D calls the
actual Open3D pairwise RGBD odometry and PGO code in `da3/refine_poses.py`.
Reference E calls its TSDF/color-map implementation. Native D instead solves
a joint photometric problem; native E uses a voxel-averaged point structure.
Correct gradients alone do not make these formulations numerically equivalent.

The optional `odometry_workers` argument in the reference evaluates independent
pairwise odometry problems concurrently and collects them in the original edge
order. Its default remains one worker. This validation uses eight edge workers,
with one OpenMP thread each; global PGO and all numerical settings are unchanged.
Serial control results are retained in `outputs/rgbd_reference_serial_control`.

The saved historical `da3/da3_pose_auc_nvs_test_GIANT11.json` is from
`nvs_test.txt`, which has **zero scenes in common** with the mounted validation
split. It also used up to 627 frames per scene. Its absolute AUC is not a
matching target for this dataset and GPU budget. Fresh reference runs here
provide the comparison on identical inputs.
