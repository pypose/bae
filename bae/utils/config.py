"""Runtime environment flags."""

import os


def pypose_ambient_grad_enabled() -> bool:
    value = os.environ.get("BAE_USE_PYPOSE_AMBIENT_GRAD", "")
    return value.lower() in {"1", "true", "yes", "on"}


def pypose_torch_compile_enabled() -> bool:
    value = os.environ.get("BAE_USE_PYPOSE_TORCH_COMPILE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def jacobian_tuple_enabled() -> bool:
    """Whether Jacobians use tuple-backed BSR tensor storage."""
    value = os.environ.get("BAE_JACOBIAN_TUPLE", "")
    return value.lower() in {"1", "true", "yes", "on"}
