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
