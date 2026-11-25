import os

import torch

_ENV_FLAG = "BAE_USE_PYPOSE_AMBIENT_GRAD"
_PATCH_INSTALLED = False


def pypose_ambient_grad_enabled() -> bool:
    value = os.environ.get(_ENV_FLAG, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _pm(input: torch.Tensor) -> torch.Tensor:
    return torch.sign(torch.sign(input) * 2 + 1)


def _so3_act_components(qx, qy, qz, qw, px, py, pz):
    # Expand p + 2 * qw * (qv x p) + 2 * qv x (qv x p)
    # coordinate-wise. This is valid without assuming a unit quaternion and
    # avoids reduction/cross operators which otherwise block pointwise fusion.
    return (
        (1.0 - 2.0 * qy * qy - 2.0 * qz * qz) * px
        + (2.0 * qx * qy - 2.0 * qw * qz) * py
        + (2.0 * qx * qz + 2.0 * qw * qy) * pz,
        (2.0 * qx * qy + 2.0 * qw * qz) * px
        + (1.0 - 2.0 * qx * qx - 2.0 * qz * qz) * py
        + (2.0 * qy * qz - 2.0 * qw * qx) * pz,
        (2.0 * qx * qz - 2.0 * qw * qy) * px
        + (2.0 * qy * qz + 2.0 * qw * qx) * py
        + (1.0 - 2.0 * qx * qx - 2.0 * qy * qy) * pz,
    )


def _so3_act_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    quaternion = X.unbind(dim=-1)
    point = p.unbind(dim=-1)
    return torch.stack(_so3_act_components(*quaternion, *point), dim=-1)


def _se3_act_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    tx, ty, tz, qx, qy, qz, qw = X.unbind(dim=-1)
    px, py, pz = p.unbind(dim=-1)
    rx, ry, rz = _so3_act_components(qx, qy, qz, qw, px, py, pz)
    return torch.stack((tx + rx, ty + ry, tz + rz), dim=-1)


def _so3_mul_forward(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = X.unbind(dim=-1)
    x2, y2, z2, w2 = Y.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _se3_mul_forward(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    t = X[..., :3] + _so3_act_forward(X[..., 3:], Y[..., :3])
    q = _so3_mul_forward(X[..., 3:], Y[..., 3:])
    return torch.cat((t, q), dim=-1)


def _so3_inv_forward(X: torch.Tensor) -> torch.Tensor:
    return torch.cat((-X[..., :3], X[..., 3:]), dim=-1)


def _se3_inv_forward(X: torch.Tensor) -> torch.Tensor:
    q_inv = _so3_inv_forward(X[..., 3:])
    t_inv = -_so3_act_forward(q_inv, X[..., :3])
    return torch.cat((t_inv, q_inv), dim=-1)


def _so3_log_forward(input: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(input.dtype).eps
    v, w = input[..., :3], input[..., 3:]
    v_norm = torch.norm(v, 2, dim=-1, keepdim=True)
    w_abs = torch.abs(w)
    v_larger_than_eps = v_norm > eps
    w_larger_than_eps = w_abs > eps
    # Autograd differentiates inactive branches too: masking or nan_to_num
    # after a division by zero still lets NaNs enter the backward pass.
    # Protect denominators before evaluating any branch (identity: v=0;
    # half turn: w=0), preserving the existing principal-Log convention.
    safe_v_norm = torch.where(v_larger_than_eps, v_norm, torch.ones_like(v_norm))
    safe_w = torch.where(w_larger_than_eps, w, torch.ones_like(w))
    regular = 2.0 * torch.atan(v_norm / safe_w) / safe_v_norm
    half_turn = _pm(w) * torch.pi / safe_v_norm
    small = 2.0 * (1.0 / safe_w - v_norm * v_norm / (3 * safe_w**3))
    factor = torch.where(
        v_larger_than_eps,
        torch.where(w_larger_than_eps, regular, half_turn),
        small,
    )
    return factor * v


def _so3_exp_forward(input: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(input, dim=-1, keepdim=True)
    theta2 = theta.square()
    theta4 = theta2.square()
    nonzero = theta > torch.finfo(theta.dtype).eps
    safe_theta = torch.where(nonzero, theta, torch.ones_like(theta))
    imaginary_factor = torch.where(
        nonzero,
        torch.sin(0.5 * safe_theta) / safe_theta,
        0.5 - theta2 / 48.0 + theta4 / 3840.0,
    )
    real_factor = torch.where(
        nonzero,
        torch.cos(0.5 * theta),
        1.0 - theta2 / 8.0 + theta4 / 384.0,
    )
    return torch.cat((input * imaginary_factor, real_factor), dim=-1)


def _vec2skew(input: torch.Tensor) -> torch.Tensor:
    x, y, z = input.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def _so3_Jl(input: torch.Tensor) -> torch.Tensor:
    skew = _vec2skew(input)
    theta = torch.linalg.norm(input, dim=-1, keepdim=True).unsqueeze(-1)
    theta2 = theta.square()
    identity = torch.eye(3, device=input.device, dtype=input.dtype)
    identity = identity.expand(input.shape[:-1] + (3, 3))
    nonzero = theta > torch.finfo(theta.dtype).eps
    safe_theta = torch.where(nonzero, theta, torch.ones_like(theta))
    safe_theta2 = safe_theta.square()
    coefficient1 = torch.where(
        nonzero,
        (1.0 - safe_theta.cos()) / safe_theta2,
        0.5 - theta2 / 24.0,
    )
    coefficient2 = torch.where(
        nonzero,
        (safe_theta - safe_theta.sin()) / (safe_theta * safe_theta2),
        1.0 / 6.0 - theta2 / 120.0,
    )
    return identity + coefficient1 * skew + coefficient2 * (skew @ skew)


def _so3_Jl_inv(input: torch.Tensor) -> torch.Tensor:
    skew = _vec2skew(input)
    theta = torch.linalg.norm(input, dim=-1, keepdim=True).unsqueeze(-1)
    identity = torch.eye(3, device=input.device, dtype=input.dtype)
    identity = identity.expand(input.shape[:-1] + (3, 3))
    nonzero = theta > torch.finfo(theta.dtype).eps
    safe_theta = torch.where(nonzero, theta, torch.ones_like(theta))
    half_safe_theta = 0.5 * safe_theta
    coefficient = torch.where(
        nonzero,
        (
            1.0
            - half_safe_theta * half_safe_theta.cos() / half_safe_theta.sin()
        )
        / safe_theta.square(),
        torch.full_like(theta, 1.0 / 12.0),
    )
    return identity - 0.5 * skew + coefficient * (skew @ skew)


def _se3_exp_forward(input: torch.Tensor) -> torch.Tensor:
    translation = (
        _so3_Jl(input[..., 3:]) @ input[..., :3].unsqueeze(-1)
    ).squeeze(-1)
    rotation = _so3_exp_forward(input[..., 3:])
    return torch.cat((translation, rotation), dim=-1)


def _se3_log_forward(input: torch.Tensor) -> torch.Tensor:
    phi = _so3_log_forward(input[..., 3:])
    Jl_inv = _so3_Jl_inv(phi)
    tau = (Jl_inv @ input[..., :3].unsqueeze(-1)).squeeze(-1)
    return torch.cat([tau, phi], dim=-1)


def _so3_act4_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.cat((_so3_act_forward(X, p[..., :3]), p[..., 3:]), dim=-1)


def _se3_act4_forward(X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    t = _so3_act_forward(X[..., 3:], p[..., :3]) + X[..., :3] * p[..., 3:]
    return torch.cat((t, p[..., 3:]), dim=-1)


def install_pypose_ambient_grad_monkeypatch() -> bool:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return False

    from .pypose_compile import install_pypose_torch_compile_monkeypatch

    # The ambient-gradient implementations are plain PyTorch and intended to
    # be compiled. Install the semantics-preserving LieTensor compatibility
    # shim as part of this patch so the two features remain composable.
    install_pypose_torch_compile_monkeypatch()

    import pypose.lietensor.lietensor as lt
    import pypose.lietensor.operation as op

    def _ambient_so3_log(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_so3_log_forward(X), ltype=lt.so3_type)

    def _ambient_se3_log(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_se3_log_forward(X), ltype=lt.se3_type)

    def _ambient_so3_exp(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_so3_exp_forward(X), ltype=lt.SO3_type)

    def _ambient_se3_exp(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_se3_exp_forward(X), ltype=lt.SE3_type)

    def _ambient_so3_act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1] == 3 or p.shape[-1] == 4, "Invalid Tensor Dimension"
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        inputs, out_shape = op.broadcast_inputs(X, p)
        if p.shape[-1] == 3:
            out = _so3_act_forward(*inputs)
        else:
            out = _so3_act4_forward(*inputs)
        dim = -1 if out.nelement() != 0 else p.shape[-1]
        return out.view(out_shape + (dim,))

    def _ambient_se3_act(self, X, p):
        assert not self.on_manifold and isinstance(p, torch.Tensor)
        assert p.shape[-1] == 3 or p.shape[-1] == 4, "Invalid Tensor Dimension"
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        inputs, out_shape = op.broadcast_inputs(X, p)
        if p.shape[-1] == 3:
            out = _se3_act_forward(*inputs)
        else:
            out = _se3_act4_forward(*inputs)
        dim = -1 if out.nelement() != 0 else p.shape[-1]
        return out.view(out_shape + (dim,))

    def _ambient_so3_mul(self, X, Y):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        if not self.on_manifold and isinstance(Y, lt.LieTensor) and not Y.ltype.on_manifold:
            inputs, out_shape = op.broadcast_inputs(X, Y.tensor())
            out = _so3_mul_forward(*inputs)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            return lt.LieTensor(out.view(out_shape + (dim,)), ltype=lt.SO3_type)
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        if self.on_manifold:
            return lt.LieTensor(torch.mul(X, Y), ltype=lt.SO3_type)
        raise NotImplementedError("Invalid __mul__ operation")

    def _ambient_so3_inv(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_so3_inv_forward(X), ltype=lt.SO3_type)

    def _ambient_se3_mul(self, X, Y):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        if not self.on_manifold and isinstance(Y, lt.LieTensor) and not Y.ltype.on_manifold:
            Y = Y.tensor() if hasattr(Y, "ltype") else Y
            inputs, out_shape = op.broadcast_inputs(X, Y)
            out = _se3_mul_forward(*inputs)
            dim = -1 if out.nelement() != 0 else X.shape[-1]
            return lt.LieTensor(out.view(out_shape + (dim,)), ltype=lt.SE3_type)
        if not self.on_manifold and isinstance(Y, torch.Tensor):
            return self.Act(X, Y)
        if self.on_manifold:
            return lt.LieTensor(torch.mul(X, Y), ltype=lt.SE3_type)
        raise NotImplementedError("Invalid __mul__ operation")

    def _ambient_se3_inv(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        return lt.LieTensor(_se3_inv_forward(X), ltype=lt.SE3_type)

    def _ambient_se3_translation(self, X):
        X = X.tensor() if isinstance(X, lt.LieTensor) else X
        # PyPose's custom translation backward returns a padded tangent gradient.
        # Keep ambient derivatives here; BAE applies the retraction Jacobian once
        # at the parameter boundary, after differentiating the full residual.
        return X[..., :3]

    lt.SO3Type.Log = _ambient_so3_log
    lt.SE3Type.Log = _ambient_se3_log
    lt.so3Type.Exp = _ambient_so3_exp
    lt.se3Type.Exp = _ambient_se3_exp
    lt.SO3Type.Act = _ambient_so3_act
    lt.SE3Type.Act = _ambient_se3_act
    lt.SO3Type.Mul = _ambient_so3_mul
    lt.SE3Type.Mul = _ambient_se3_mul
    lt.SO3Type.Inv = _ambient_so3_inv
    lt.SE3Type.Inv = _ambient_se3_inv
    lt.SE3Type.translation = _ambient_se3_translation

    _PATCH_INSTALLED = True
    return True


def maybe_install_pypose_ambient_grad_monkeypatch() -> bool:
    if not pypose_ambient_grad_enabled():
        return False
    return install_pypose_ambient_grad_monkeypatch()
