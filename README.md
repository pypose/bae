<picture>
  <img src="https://raw.githubusercontent.com/sair-lab/bae/93eb45121965719c1abbed9437b4832f669b8c87/assets/github-banner.svg" alt="bundle adjustment in the eager-mode" width="100%" />
</picture>

<p align="center">
  <a><b>IEEE Transactions on Robotics (T-RO)</b>, 2026</a>
</p>

<p align="center">
  <a href="https://github.com/zitongzhan">Zitong Zhan</a>, <a href="https://www.linkedin.com/in/huan-xu-999700169/?locale=en_US">Huan Xu</a>, Zihang Fang, <a href="https://www.linkedin.com/in/william-xp-wei/">Xinpeng Wei</a>, <a href="https://theairlab.org/team/yaoyuh/">Yaoyu Hu</a>, and <a href="https://sairlab.org">Chen Wang</a>
</p>

<p align="center">
  <a href="https://pypose.org/bae/">🌐 Project Page</a> | <a href="https://arxiv.org/abs/2409.12190">📄 PDF</a>
</p>

> Basic functionalities in `bae` have been supported by [LM](https://pypose.org/docs/main/generated/pypose.optim.LevenbergMarquardt/#pypose.optim.LevenbergMarquardt) in [PyPose](https://github.com/pypose/pypose) as a sparse backend starting from [v0.9.5](https://pypi.org/project/pypose/). Please refer to [this example](https://github.com/pypose/pypose/tree/main/examples/module/ba) and docs of [psjac](https://pypose.org/docs/main/generated/pypose.autograd.function.parallel_for_sparse_jacobian/#pypose.autograd.function.parallel_for_sparse_jacobian) for shared APIs used by both libraries.

`bae` is a PyTorch-based library supporting **exact** 2nd-order optimization techniques. The library provides efficient implementations for sparse optimization problems in robotics, particularly Bundle Adjustment (BA) and Pose Graph Optimization (PGO).

### Bundle Adjustment

<table>
  <tr>
    <td align="center" width="33%">
      <p align="center" width="100%">
        <img src="https://github.com/pypose/bae/blob/product-page/docs/assets/1c4b893630%20reconstruction_playback.gif?raw=true" alt="bundle adjustment example" width="100%" />
      </p>
    </td>
    <td align="center" width="33%">
      <p align="center" width="100%">
        <img src="https://github.com/pypose/bae/blob/product-page/docs/assets/bonsai%20playback%20optimized.gif?raw=true" alt="Bonsai bundle adjustment example" width="100%" />
      </p>
    </td>
    <td align="center" width="33%">
      <p align="center" width="100%">
        <img src="https://github.com/pypose/bae/blob/product-page/docs/assets/kitchen%20reconstruction_playback_0.95_output.gif?raw=true" alt="Kitchen bundle adjustment example" width="100%" />
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">Indoor</td>
    <td align="center">Bonsai</td>
    <td align="center">Kitchen</td>
  </tr>
</table>

<p align="center"><sub><code>bae</code> powering BA and global positioning in downstream system, <a href="https://github.com/cre185/InstantSfM">InstantSfM</a>.</sub></p>

### Pose Graph Optimization

<table>
  <tr>
    <td align="center" width="33%">
      <img src="https://github.com/sair-lab/bae/blob/product-page/docs/assets/sphere_bignoise_vertex3.gif?raw=true" alt="Sphere big-noise optimization" width="100%" />
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/sair-lab/bae/blob/product-page/docs/assets/grid3D.gif?raw=true" alt="3D grid optimization" width="100%" />
    </td>
    <td align="center" width="33%">
      <img src="https://github.com/sair-lab/bae/blob/product-page/docs/assets/sphere_g2o.gif?raw=true" alt="Sphere g2o optimization" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center">Sphere Big Noise</td>
    <td align="center">Grid3D</td>
    <td align="center">Sphere (g2o)</td>
  </tr>
</table>

### RGB-D Pose Refinement

<table>
  <tr>
    <td align="center" width="33%">
      <img src="doc/assets/monitor_office.gif" alt="Monitor and desk point clouds aligning during camera pose refinement" width="100%" />
    </td>
    <td align="center" width="33%">
      <img src="doc/assets/lecture_room.gif" alt="Lecture room table edges aligning during camera pose refinement" width="100%" />
    </td>
    <td align="center" width="33%">
      <img src="doc/assets/kitchenette.gif" alt="Kitchen cabinet and door point clouds aligning during camera pose refinement" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center">Monitor Office</td>
    <td align="center">Lecture Room</td>
    <td align="center">Kitchenette</td>
  </tr>
</table>

<p align="center"><sub>bae refines camera poses predicted by Depth Anything 3 (DA3), bringing feed-forward 3D reconstructions into sharper alignment.</sub></p>

## News

- 2026-03-22: Added [skills](.agent/skills) for coding agents to write custom compute graphs.
- 2025-12-12: Added a VGGT integration example.

## Features

- **Sparse Block Matrix Operations**: Optimized implementations of sparse matrix operations for large-scale optimization
- **CUDA Acceleration**: Custom CUDA kernels for high-performance sparse linear algebra
- **PyTorch Integration**: Seamlessly integrates with PyTorch's automatic differentiation framework
- **Levenberg-Marquardt Optimizer**: The vanilla LM algorithm for non-linear least squares problems
- **Schur Complement Optimizer**: Solve a Schur-reduce linear system for [optimized memory consumption](https://github.com/pypose/bae/blob/release/doc/memory_performance.md) 

### Future Plan
- [x] Reduce runtime overhead using CUDA graph; compile fwd & backward with `torch.compile()` reducing latency from 10ms to 2.2ms (added in [PR #41](https://github.com/pypose/bae/pull/41))
- [ ] Reduce kernel launching cost in PCG solver by kernel fusion
- [ ] Distributed Tensor (DTensor) and FSDP support for multi-GPU and distributed optimization (WIP in `distributed_dt`)
- [x] Schur complement (added in [PR #35](https://github.com/pypose/bae/pull/35))
- [ ] Add Apple Silicon GPU support, [PyTorch PR WIP](https://github.com/pytorch/pytorch/pull/177757)
- [ ] **Community-requested features:** Need support for a new framework, hardware backend, or more efficient memory allocation? Open an issue and tell us about your use case; we’d love to hear what you need.


## Installation

### Prerequisites

- CUDA toolkit (tested with CUDA 12.x)
- PyTorch (2.0+)
- (Optional) [CUDSS](https://developer.nvidia.com/cudss) (CUDA Sparse Solver library)

### User Setup Instructions
```
python -m pip install git+https://github.com/pypose/bae.git
```

### Developer Setup Instructions

1. (Optional) Install CUDSS with pip package manager.
   - cuDSS 0.8.x uses the new datatype API; older releases retain the legacy API at compile time.
   - For CUDA 12.x, install `nvidia-cudss-cu12`. We verified `nvidia-cudss-cu12==0.6.0.5` and `nvidia-cudss-cu12==0.7.1.6` work with `bae`:
   ```bash
   pip install "nvidia-cudss-cu12<0.9"
   ```
   - For CUDA 13.x, install `nvidia-cudss-cu13<0.9`:
   ```bash
   pip install "nvidia-cudss-cu13<0.9"
   ```

2. Install PyPose:
   ```bash
   pip install git+https://github.com/pypose/pypose.git
   ```
3. Clone this repository:
   ```bash
   git clone https://github.com/zitongzhan/bae.git
   cd bae
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Install the package in development mode:
   ```bash
   python -m pip install --no-build-isolation -v -e .  # following https://github.com/pytorch/pytorch
   ```

### `torch.compile` with `LieTensor`

Enable the `LieTensor` TorchDynamo compatibility shim before importing either
`pypose` or `bae`:

```bash
export BAE_USE_PYPOSE_TORCH_COMPILE=1
```

The existing ambient-gradient implementation automatically enables the same
shim, so `BAE_USE_PYPOSE_AMBIENT_GRAD=1` is sufficient when ambient gradients
are required. The shim can also be installed explicitly before compiling a
model or residual function:

```python
import torch

from bae.utils.pypose_compile import install_pypose_torch_compile_monkeypatch

install_pypose_torch_compile_monkeypatch()
compiled_model = torch.compile(model, fullgraph=True)
```

With `fullgraph=True`, indexed `sjac=True` parameters retain their sparse
Jacobian dependency trace without forcing the gathered camera and point blocks
to escape the compiled graph. This gives Inductor the opportunity to load
permuted rows directly inside fused kernels. Inductor will decide the most efficient way, 
whether to materialize standalone indexed tensors when the
gathered values have multiple downstream consumers, as can happen during
Jacobian computation. `fullgraph=True` guarantees graph capture but does not ensure 
particular kernel-fusion or buffer-allocation strategy.

PyTorch cannot currently represent a sparse BSR tensor as a FakeTensor/AOT
graph output. To compile the residual and sparse-Jacobian traversal together,
return its dense component tensors from the compiled function and materialize
the BSR wrapper immediately afterward:

```python
from bae.autograd.graph import (
    jacobian_components,
    materialize_jacobian_components,
)

def residual_and_jacobian(observations, camera_indices, point_indices):
    residual = model(observations, camera_indices, point_indices)
    components = jacobian_components(
        residual, (model.pose, model.points)
    )
    return residual, components

compiled = torch.compile(residual_and_jacobian, fullgraph=True)
residual, components = compiled(observations, camera_indices, point_indices)
jacobians = materialize_jacobian_components(components)
```

The component traversal and Jacobian values are compiled. For one component per
parameter, as in the BA residual above, BSR materialization is an eager,
zero-copy wrapper operation. Graphs with multiple contributions to the same
parameter additionally combine those sparse components after materialization.

## Agent Skills

This repo includes skills in [.agent/skills](.agent/skills):

<!-- - [`bae-codebase`](.agent/skills/bae-codebase/SKILL.md): general guidance for working in this repository -->
- [`bae-compute-graph`](.agent/skills/bae-compute-graph/SKILL.md): guidance for defining BAL/PGO and more complex compute graphs

<!-- Use `bae-compute-graph` for most changes, and add `bae-compute-graph` when working on residual definitions or Jacobian structure. -->

## Example Usage

### Bundle Adjustment

Bundle Adjustment optimizes camera poses and 3D point positions to minimize reprojection error. The following example shows how to perform BA using `bae`:

```python
import torch
import pypose as pp
from pypose.autograd.function import psjac
from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.utils.pysolvers import PCG


class Reproj(torch.nn.Module):
    def __init__(self, camera_params, points):
        super().__init__()
        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points = pp.Parameter(points, sjac=True)
        self.pose.trim_SE3_grad = True
    
    # Define the projection residual with structured Jacobian support
    @psjac
    def project(points, camera_params):
        projection = pp.SE3(camera_params[..., :7]).Act(points)
        projection = -projection[..., :2] / projection[..., [2]]

        f = camera_params[..., [-3]]
        k1 = camera_params[..., [-2]]
        k2 = camera_params[..., [-1]]

        n = torch.sum(projection**2, axis=-1, keepdim=True)
        r = 1 + k1 * n + k2 * n**2
        return projection * r * f

    def forward(self, observes, cidx, pidx):
        points_proj = Reproj.project(self.points[pidx], self.pose[cidx])
        return points_proj - observes


# Load a problem from the BAL dataset
dataset = get_problem("problem-49-7776-pre", "ladybug", use_quat=True)
dataset = {k: v.to('cuda') for k, v in dataset.items() if isinstance(v, torch.Tensor)}

# Prepare input for the optimization
input = {
    "observes": dataset['points_2d'],
    "cidx": dataset['camera_index_of_observations'],
    "pidx": dataset['point_index_of_observations'],
}

# Initialize model with camera parameters and 3D points
model = Reproj(
    dataset['camera_params'].clone(),
    dataset['points_3d'].clone(),
).to('cuda')

# Configure optimizer
strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
solver = PCG(tol=1e-4, maxiter=250)
optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

# Run optimization for multiple iterations
for idx in range(20):
    loss = optimizer.step(input)
    print(f'Iteration {idx}, loss: {loss.item()}')
```

### Integration with VGGT

`bae` is used as an optional Bundle Adjustment backend in [our VGGT fork](https://github.com/zitongzhan/vggt) (Visual Geometry Grounded Transformer) to refine the camera poses, intrinsics, and 3D points predicted by VGGT before exporting a COLMAP reconstruction.

After installing `bae`, you can run VGGT's COLMAP export with BA enabled and `bae` selected as the solver:

```bash
python demo_colmap.py --scene_dir /path/to/scene --use_ba --implementation bae  # optional: --shared_camera
```

This command invokes `prepare_bae(...)` inside `vggt/demo_colmap.py`, which wraps VGGT tracks and predictions into `bae.optim.LM` and updates `extrinsic`, `intrinsic`, and `points_3d` in place before writing `scene_dir/sparse/` in COLMAP format.

## Citation

If you use `bae` in your research, please cite:

```bibtex
@article{zhan2026bundle,
  title = {Bundle Adjustment in the Eager Mode},
  author = {Zhan, Zitong and Xu, Huan and Fang, Zihang and Wei, Xinpeng and Hu, Yaoyu and Wang, Chen},
  journal = {IEEE Transactions on Robotics},
  year = {2026},
  url = {https://arxiv.org/abs/2409.12190}
}
```

## Acknowledgements

The implementation draws inspiration from:
- [PyPose](https://github.com/pypose/pypose) for SE(3) pose representations
- GTSAM for reprojection jacobian concepts
- Ceres for manifold parameter update
