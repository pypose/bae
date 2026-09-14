"""Pure-tensor pinhole camera math shared by the odometry and structure
refinement stages.

Kept dependency-free (no `bae`/`pypose` import) so it can be unit-tested in
isolation and reused by both the correspondence-building code (plain tensors,
outside any `@psjac` trace) and the synthetic test.

Conventions:
  - Poses are 4x4 homogeneous matrices, OpenCV +z-forward.
  - Pixel coordinates are (x=u=col, y=v=row), pixel CENTERS at integer+0.5.
  - Intrinsics K are 3x3 pinhole matrices in pixel units.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def as_44(ext: torch.Tensor) -> torch.Tensor:
    """(...,3,4) or (...,4,4) -> (...,4,4) homogeneous."""
    if ext.shape[-2:] == (4, 4):
        return ext
    bottom = torch.zeros_like(ext[..., :1, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat([ext, bottom], dim=-2)


def affine_inv(T: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of (...,4,4) SE3 (w2c<->c2w)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3:] = -Rt @ t
    out[..., 3, 3] = 1.0
    return out


def _det3x3(R: torch.Tensor) -> torch.Tensor:
    """Closed-form determinant of (...,3,3), computed elementwise.

    Deliberately NOT `torch.linalg.det`/`torch.det`: on this environment
    those dispatch to a JIT-compiled CUDA reduction kernel that fails to
    build (`nvrtc: error: failed to open libnvrtc-builtins.so.13.0`) --  a
    CUDA-toolkit/PyTorch NVRTC version mismatch on the host, not something
    fixable from application code. A hand-rolled 3x3 cofactor-expansion
    determinant is pure elementwise arithmetic, needs no JIT kernel, and is
    exact for the only shape this function ever sees.
    """
    a, b, c = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    d, e, f = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    g, h, i = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def orthonormalize(T: torch.Tensor) -> torch.Tensor:
    """Re-project the rotation block of (...,4,4) onto SO(3) via SVD, removing
    numerical drift accumulated across many pose updates (otherwise
    `pp.mat2SE3` rejects a "not orthogonal" matrix)."""
    out = T.clone()
    R = T[..., :3, :3]
    U, _, Vh = torch.linalg.svd(R.double())
    Rn = U @ Vh
    det = _det3x3(Rn)
    U2 = U.clone()
    U2[..., :, -1] = U2[..., :, -1] * det[..., None]
    Rn = U2 @ Vh
    out[..., :3, :3] = Rn.to(T)
    return out


def unproject_pixels(uv: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pixels -> camera-space 3D points.

    uv: (...,2) pixel coords (x,y). depth: (...,) z. K: (...,3,3) broadcastable.
    Returns (...,3) camera-space points.
    """
    dt = torch.promote_types(torch.promote_types(uv.dtype, depth.dtype), K.dtype)
    uv = uv.to(dt); depth = depth.to(dt); K = K.to(dt)
    ones = torch.ones_like(uv[..., :1])
    homog = torch.cat([uv, ones], dim=-1)                       # (...,3)
    Kinv = torch.linalg.inv(K)                                  # (...,3,3)
    dirs = torch.einsum("...ij,...j->...i", Kinv, homog)        # (...,3)
    return dirs * depth[..., None]


def cam_to_world(Xc: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Camera-space (...,3) -> world (...,3) given c2w (...,4,4)."""
    dt = torch.promote_types(Xc.dtype, c2w.dtype); Xc = Xc.to(dt); c2w = c2w.to(dt)
    homog = torch.cat([Xc, torch.ones_like(Xc[..., :1])], dim=-1)
    Xw = torch.einsum("...ij,...j->...i", c2w, homog)
    return Xw[..., :3]


def world_to_cam(Xw: torch.Tensor, w2c: torch.Tensor) -> torch.Tensor:
    """World (...,3) -> camera-space (...,3) given w2c (...,4,4)."""
    dt = torch.promote_types(Xw.dtype, w2c.dtype); Xw = Xw.to(dt); w2c = w2c.to(dt)
    homog = torch.cat([Xw, torch.ones_like(Xw[..., :1])], dim=-1)
    Xc = torch.einsum("...ij,...j->...i", w2c, homog)
    return Xc[..., :3]


def project_to_pixels(Xc: torch.Tensor, K: torch.Tensor):
    """Camera-space (...,3) -> (pixels (...,2), z (...,)). z is the camera depth."""
    dt = torch.promote_types(Xc.dtype, K.dtype); Xc = Xc.to(dt); K = K.to(dt)
    z = Xc[..., 2]
    uvw = torch.einsum("...ij,...j->...i", K, Xc)               # (...,3)
    uv = uvw[..., :2] / uvw[..., 2:3].clamp(min=1e-8)
    return uv, z


def sample_at(img: torch.Tensor, uv: torch.Tensor, H: int, W: int, mode: str = "bilinear"):
    """Sample (H,W) or (H,W,C) map at (M,2) pixel coords (x,y) via grid_sample.

    Pixel centers at integer+0.5 (align_corners=False maps that convention to
    the [-1,1] grid). Returns (M,) or (M,C).

    NEVER call this inside a `@psjac`-decorated function: `bae`'s tracer vmaps
    every tensor argument over dim 0 unconditionally
    (`bae/autograd/graph.py:_vmap_in_dims`), so a whole (H,W) image passed
    into a traced function would be re-materialized per observation row. Use
    it only under `torch.no_grad()` to build the fixed linearization point
    (see `photometric.py`).
    """
    if img.dim() == 2:
        img = img[..., None]
    C = img.shape[-1]
    chw = img.permute(2, 0, 1)[None]                            # (1,C,H,W)
    uv = uv.to(chw.dtype)
    gx = uv[:, 0] / W * 2 - 1
    gy = uv[:, 1] / H * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None, None]            # (1,1,M,2)
    out = F.grid_sample(chw, grid, mode=mode, align_corners=False,
                        padding_mode="border")                  # (1,C,1,M)
    out = out[0, :, 0, :].transpose(0, 1)                       # (M,C)
    return out[:, 0] if C == 1 else out


def image_gradients(gray: torch.Tensor):
    """Signed central-difference image gradients.

    gray: (N,H,W) intensity in [0,1]. Returns (gx, gy), each (N,H,W): gx is
    d(intensity)/d(x=col), gy is d(intensity)/d(y=row). Edge pixels get a
    one-sided (forward) difference so the map stays defined everywhere;
    downstream code should avoid sampling exactly at the last row/col of
    gradient-dependent residuals if that matters (in practice sampled pixels
    are interior, since covisible-pair correspondences rarely land on the
    image border).

    Returns signed partials rather than just the gradient magnitude, since
    the odometry stage's first-order photometric linearization needs signed
    gx, gy (see `grad_mag` below for the magnitude-only variant used for
    texture-biased pixel sampling).
    """
    gx = torch.zeros_like(gray)
    gy = torch.zeros_like(gray)
    gx[:, :, 1:-1] = (gray[:, :, 2:] - gray[:, :, :-2]) * 0.5
    gx[:, :, 0] = gray[:, :, 1] - gray[:, :, 0]
    gx[:, :, -1] = gray[:, :, -1] - gray[:, :, -2]
    gy[:, 1:-1, :] = (gray[:, 2:, :] - gray[:, :-2, :]) * 0.5
    gy[:, 0, :] = gray[:, 1, :] - gray[:, 0, :]
    gy[:, -1, :] = gray[:, -1, :] - gray[:, -2, :]
    return gx, gy


def grad_mag(gray: torch.Tensor) -> torch.Tensor:
    """Mean-abs-gradient magnitude map (0.5*(|gx|+|gy|)), used for both the
    texture gate and texture-biased pixel sampling."""
    gx, gy = image_gradients(gray)
    return 0.5 * (gx.abs() + gy.abs())
