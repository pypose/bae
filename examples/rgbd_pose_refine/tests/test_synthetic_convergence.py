"""Fast, dataset-free sanity tests for Form D / Form E's bae residuals.

Run this BEFORE trusting a real ScanNet++ run: if it fails, the bug is in the
bae Jacobian/API wiring, not the dataset. Modeled on
`tests/autograd/test_pgo_convergence.py`'s "converges to a known result"
pattern, adapted to this example's photometric residual (there is no
independent Ceres-style reference here, so we pin to "recovers a known
synthetic ground truth" instead of a magic reference cost).
"""
import os
import sys
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_DIR.parents[1]
for p in (str(REPO_ROOT), str(EXAMPLE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pypose as pp
from pypose.autograd.function import psjac

from bae.autograd.graph import jacobian

from geometry import as_44, affine_inv, unproject_pixels, project_to_pixels
from guards import rel_rot_deg
from photometric import photo_residual, structure_residual, refine_form_d

CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="bae's sparse LM path requires CUDA")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64


def _make_synthetic_scene(n_views=6, H=96, W=96, fov_deg=50.0, plane_z=5.0,
                           baseline=0.3, rot_perturb_deg=1.5, trans_perturb=0.05,
                           seed=0, device=DEVICE):
    """N cameras on a small dolly rig facing a smooth-textured frontal plane.

    Returns (images uint8 (N,H,W,3), depth (N,H,W), K (3,3), w2c_gt (N,4,4),
    w2c_init (N,4,4) with a known perturbation on views 1..N-1).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    fx = fy = W / (2.0 * torch.tan(torch.tensor(fov_deg * torch.pi / 360.0)))
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=DTYPE, device=device)

    # GT extrinsics: small lateral dolly + small random rotation per view.
    t_gt = torch.stack([torch.tensor([i * baseline, 0.0, 0.0]) for i in range(n_views)]).to(DTYPE)
    rvecs = torch.randn(n_views, 3, generator=g).to(DTYPE) * (2.0 * torch.pi / 180.0)
    rvecs[0] = 0.0
    R_gt = pp.so3(rvecs).Exp().matrix()
    c2w_gt = torch.eye(4, dtype=DTYPE).repeat(n_views, 1, 1)
    c2w_gt[:, :3, :3] = R_gt
    c2w_gt[:, :3, 3] = t_gt
    c2w_gt = c2w_gt.to(device)

    # Pixel grid, pixel centers.
    ys, xs = torch.meshgrid(torch.arange(H, dtype=DTYPE, device=device),
                             torch.arange(W, dtype=DTYPE, device=device), indexing="ij")
    uv = torch.stack([xs + 0.5, ys + 0.5], dim=-1).reshape(-1, 2)   # (HW,2)
    dirs_cam = unproject_pixels(uv, torch.ones(uv.shape[0], dtype=DTYPE, device=device), K)  # z=1 ray

    images = torch.zeros(n_views, H, W, 3, dtype=torch.uint8, device=device)
    depth = torch.zeros(n_views, H, W, dtype=DTYPE, device=device)
    for i in range(n_views):
        Rw = c2w_gt[i, :3, :3]
        ow = c2w_gt[i, :3, 3]
        dirs_w = torch.einsum("ij,mj->mi", Rw, dirs_cam)           # (HW,3)
        t = (plane_z - ow[2]) / dirs_w[:, 2].clamp(min=1e-6)
        Xw = ow[None, :] + t[:, None] * dirs_w
        # Period (3.0) chosen well above the world-space displacement a
        # perturbed pose can induce at this depth/rotation scale, so the
        # optimizer cannot alias onto a neighboring texture cycle.
        period = 3.0
        tex = (0.5 + 0.25 * torch.sin(2 * torch.pi * Xw[:, 0] / period)
               + 0.25 * torch.sin(2 * torch.pi * Xw[:, 1] / period))
        tex = tex.clamp(0.0, 1.0)
        img = (tex * 255.0).round().to(torch.uint8)
        images[i] = img.reshape(H, W)[..., None].expand(H, W, 3)
        depth[i] = t.reshape(H, W)

    w2c_gt = affine_inv(c2w_gt)

    # Perturb views 1..N-1's initial guess.
    d_rvec = torch.randn(n_views, 3, generator=g).to(DTYPE) * (rot_perturb_deg * torch.pi / 180.0)
    d_t = torch.randn(n_views, 3, generator=g).to(DTYPE) * trans_perturb
    d_rvec[0] = 0.0
    d_t[0] = 0.0
    dR = pp.so3(d_rvec).Exp().matrix().to(device)
    c2w_init = c2w_gt.clone()
    c2w_init[:, :3, :3] = c2w_gt[:, :3, :3] @ dR
    c2w_init[:, :3, 3] = c2w_gt[:, :3, 3] + d_t.to(device)
    w2c_init = affine_inv(c2w_init)

    return images, depth, K, w2c_gt, w2c_init


@CUDA_ONLY
def test_photo_residual_jacobian_nonzero():
    """Smoke test: `photo_residual`'s bae Jacobian w.r.t. pose_i/pose_j must
    be nonzero and finite -- catches the classic 'accidentally detached
    tensor' bug when porting hand-written autograd code into `@psjac`."""
    device = DEVICE
    M = 32
    pose_i = pp.Parameter(pp.identity_SE3(M, dtype=DTYPE, device=device).tensor(), sjac=True)
    pose_j = pp.Parameter(pp.identity_SE3(M, dtype=DTYPE, device=device).tensor(), sjac=True)
    pose_j.data[:, 0] += 0.5  # nontrivial baseline so the Jacobian isn't degenerate

    rd_i_cam = torch.randn(M, 3, dtype=DTYPE, device=device) + torch.tensor([0., 0., 5.], device=device)
    K = torch.tensor([[100., 0, 48.], [0, 100., 48.], [0, 0, 1.]], dtype=DTYPE, device=device)
    K_j = K[None].expand(M, -1, -1)
    I_i0 = torch.rand(M, dtype=DTYPE, device=device)
    I_j0 = torch.rand(M, dtype=DTYPE, device=device)
    gx_j0 = torch.randn(M, dtype=DTYPE, device=device) * 0.1
    gy_j0 = torch.randn(M, dtype=DTYPE, device=device) * 0.1
    uv_j0 = torch.rand(M, 2, dtype=DTYPE, device=device) * 96
    wgt = torch.ones(M, 1, dtype=DTYPE, device=device)

    R = photo_residual(pose_i, pose_j, rd_i_cam, K_j, I_i0, I_j0, gx_j0, gy_j0, uv_j0, wgt)
    Js = jacobian(R, [pose_i, pose_j])
    assert len(Js) == 2
    for J in Js:
        assert J is not None
        Jd = J.to_dense() if J.layout != torch.strided else J
        assert torch.isfinite(Jd.to_dense() if hasattr(Jd, "to_dense") else Jd).all()
        assert (Jd.to_dense() if hasattr(Jd, "to_dense") else Jd).abs().sum() > 0


@CUDA_ONLY
def test_structure_residual_matches_finite_difference():
    """Standalone check for Form E's residual, independent of bae's tracer:
    compare the analytic (autograd) Jacobian w.r.t. the pose's tangent
    against a central finite difference. Isolates 'is the math right' from
    'is bae's tracer right'."""
    device = DEVICE
    pose_v = pp.SE3(pp.identity_SE3(1, dtype=DTYPE, device=device).tensor())
    Xw_pt = torch.tensor([[0.2, -0.1, 5.0]], dtype=DTYPE, device=device)
    K_v = torch.tensor([[[100., 0, 48.], [0, 100., 48.], [0, 0, 1.]]], dtype=DTYPE, device=device)
    C0 = torch.tensor([0.7], dtype=DTYPE, device=device)
    uv0, _ = project_to_pixels(pose_v.Inv().Act(Xw_pt), K_v)
    I_v0 = torch.tensor([0.6], dtype=DTYPE, device=device)
    gx_v0 = torch.tensor([0.05], dtype=DTYPE, device=device)
    gy_v0 = torch.tensor([-0.03], dtype=DTYPE, device=device)
    wgt = torch.ones(1, 1, dtype=DTYPE, device=device)

    def f(tangent):
        p = (pp.se3(tangent).Exp() * pose_v).tensor()
        return structure_residual(p, Xw_pt, K_v, C0, I_v0, gx_v0, gy_v0, uv0, wgt)[0, 0]

    tangent0 = torch.zeros(6, dtype=DTYPE, device=device, requires_grad=True)
    analytic = torch.autograd.grad(f(tangent0), tangent0)[0]

    eps = 1e-6
    fd = torch.zeros(6, dtype=DTYPE, device=device)
    for k in range(6):
        d = torch.zeros(6, dtype=DTYPE, device=device)
        d[k] = eps
        fd[k] = (f(d) - f(-d)) / (2 * eps)

    assert torch.allclose(analytic, fd, atol=1e-4, rtol=1e-3), (analytic, fd)


@CUDA_ONLY
def test_form_d_recovers_perturbed_poses():
    """End-to-end: Form D on a small synthetic textured-plane scene should
    recover the known ground-truth poses from a perturbed initial guess."""
    images, depth, K, w2c_gt, w2c_init = _make_synthetic_scene()
    N = images.shape[0]
    K_all = K[None].expand(N, -1, -1).contiguous()

    err_rot_before = rel_rot_deg(w2c_init, torch.arange(1, N), torch.zeros(N - 1, dtype=torch.long))
    w2c_ref = refine_form_d(w2c_init, depth, K_all, images,
                             n_neighbors=N - 1, pyramid_levels=1, n_relin=(4,),
                             n_inner=8, huber_delta=0.0, dtype=DTYPE, verbose=True)
    w2c_ref = as_44(w2c_ref)

    idx_i = torch.arange(1, N)
    idx_j = torch.zeros(N - 1, dtype=torch.long)
    err_rot_after = rel_rot_deg(w2c_ref, idx_i, idx_j)
    err_rot_gt = rel_rot_deg(w2c_gt, idx_i, idx_j)

    # Recovered relative rotation (view i vs fixed view 0) should match GT's
    # relative rotation much more closely than the perturbed initial guess did.
    before_err = (err_rot_before - err_rot_gt).abs().mean().item()
    after_err = (err_rot_after - err_rot_gt).abs().mean().item()
    print(f"[test] mean rel-rot error vs GT: before={before_err:.4f} deg, after={after_err:.4f} deg")
    assert after_err < before_err * 0.3
    assert after_err < 1.0

    t_before = (affine_inv(w2c_init)[1:, :3, 3] - affine_inv(w2c_gt)[1:, :3, 3]).norm(dim=-1).mean().item()
    t_after = (affine_inv(w2c_ref)[1:, :3, 3] - affine_inv(w2c_gt)[1:, :3, 3]).norm(dim=-1).mean().item()
    print(f"[test] mean camera-center error vs GT: before={t_before:.4f}, after={t_after:.4f}")
    assert t_after < t_before * 0.3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s", "-v"]))
