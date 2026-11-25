from __future__ import annotations

import os
import warnings
from importlib import import_module

import torch
from torch.utils._pytree import tree_flatten, tree_map


_ENV_FLAG = "BAE_USE_PYPOSE_TORCH_COMPILE"
_PATCH_INSTALLED = False


def pypose_torch_compile_enabled() -> bool:
    value = os.environ.get(_ENV_FLAG, "")
    return value.lower() in {"1", "true", "yes", "on"}


def install_pypose_torch_compile_monkeypatch() -> bool:
    """Make PyPose's ``LieTensor`` traceable by TorchDynamo.

    PyPose upcasts a ``LieTensor`` to ``Tensor`` with ``Tensor.as_subclass``.
    TorchDynamo cannot trace that direction of ``as_subclass``, while the
    equivalent ``Tensor(self)`` alias is supported and keeps autograd history.
    PyPose's ``__torch_function__`` also introspects method-wrapper objects with
    ``hasattr``, which is another graph break. Its ``vec2skew`` helper creates a
    leaf tensor with a dynamic ``requires_grad`` argument, which Dynamo also
    rejects; the replacement uses an equivalent constant zero tensor.

    The matmul special case works around Dynamo routing the ``@`` bytecode to
    ``Tensor.matmul`` instead of ``LieTensor.__matmul__``. Eager PyPose defines
    ``@`` as Lie group multiplication (or action), so preserving that dispatch
    is required for compiled code to have the same semantics.
    """
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return False

    import pypose.lietensor.lietensor as lt
    import pypose.lietensor.operation as op

    basics = import_module("pypose.lietensor.basics")

    def _dynamo_compatible_tensor(self):
        return torch.Tensor(self)

    def _dynamo_compatible_vec2skew(input):
        value = input.tensor() if isinstance(input, lt.LieTensor) else input
        assert value.shape[-1] == 3, "Last dim should be 3"
        x, y, z = value.unbind(dim=-1)
        zero = torch.zeros_like(x)
        return torch.stack(
            (
                torch.stack((zero, -z, y), dim=-1),
                torch.stack((z, zero, -x), dim=-1),
                torch.stack((-y, x, zero), dim=-1),
            ),
            dim=-2,
        )

    @classmethod
    def _dynamo_compatible_torch_function(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        # Dynamo lowers ``lhs @ rhs`` through Tensor.matmul for a Tensor
        # subclass, bypassing LieTensor.__matmul__.
        if func is torch.Tensor.matmul:
            lhs, rhs = args[:2]
            return lhs.ltype.Mul(lhs, rhs)

        tensor_types = tuple(
            torch.Tensor
            if issubclass(tensor_type, lt.LieTensor)
            else tensor_type
            for tensor_type in types
        )
        data = torch.Tensor.__torch_function__(func, tensor_types, args, kwargs)

        # Access __name__ directly. Dynamo cannot trace hasattr() on the method
        # wrappers used by Tensor properties such as shape.
        if data is not None and func.__name__ in lt.HANDLED_FUNCTIONS:
            flat_inputs, _ = tree_flatten((args, kwargs))
            ltype = next(
                arg.ltype
                for arg in flat_inputs
                if isinstance(arg, lt.LieTensor)
            )

            def wrap(tensor):
                if isinstance(tensor, torch.Tensor) and not isinstance(tensor, cls):
                    lie_tensor = torch.Tensor.as_subclass(tensor, lt.LieTensor)
                    lie_tensor.ltype = ltype
                    if lie_tensor.shape[-1:] != lie_tensor.ltype.dimension:
                        link = "https://pypose.org/docs/main/generated/pypose.LieTensor"
                        warnings.warn(
                            f"Tensor Shape Invalid by calling {func}, go to {link}",
                            stacklevel=2,
                        )
                    return lie_tensor
                return tensor

            return tree_map(wrap, data)
        return data

    lt.LieTensor.tensor = _dynamo_compatible_tensor
    lt.LieTensor.__torch_function__ = _dynamo_compatible_torch_function
    # operation.py and lietensor.py import vec2skew directly, so update all
    # bound references as well as the defining module.
    basics.vec2skew = _dynamo_compatible_vec2skew
    op.vec2skew = _dynamo_compatible_vec2skew
    lt.vec2skew = _dynamo_compatible_vec2skew

    _PATCH_INSTALLED = True
    return True


def maybe_install_pypose_torch_compile_monkeypatch() -> bool:
    if not pypose_torch_compile_enabled():
        return False
    return install_pypose_torch_compile_monkeypatch()
