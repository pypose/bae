"""bae-native, GPU-vectorized dense photometric pose refinement: two
complementary stages reimplementing Open3D's RGB-D Odometry and Color Map
Optimization tutorials natively on GPU.

Design note (see README for the full argument): `bae`'s sparse-Jacobian tracer
vmaps every tensor argument to a `@psjac` function over dim 0 unconditionally
(`bae/autograd/graph.py::_vmap_in_dims`), so a full (H,W) image can never be
passed into a traced residual. Both residuals below use inverse-compositional
first-order photometric linearization instead: `sample_at`/`grid_sample` is
called only under `torch.no_grad()`, once per outer relinearization, to build
a fixed (intensity, gradient, pixel) linearization point; the `@psjac`
residual itself is pure SE3/pinhole tensor algebra.

The odometry stage collapses Open3D's two-stage pipeline (per-pair
Gauss-Newton odometry, then a separate pose-graph solve) into a single
joint gauge-fixed sparse LM problem over all views and all covisible-pair
pixel correspondences at once -- this is what "fully vectorized" means in
`bae`'s idiom (see README).
"""
from __future__ import annotations

import os
os.environ.setdefault('BAE_USE_PYPOSE_AMBIENT_GRAD', '1')

import numpy as np
import pypose as pp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pypose.autograd.function import psjac

from bae.optim import LM
from bae.utils.pysolvers import PCG
from bae.utils.pypose_ambient_grad import maybe_install_pypose_ambient_grad_monkeypatch

# Also support importing this example after another module already imported bae.
maybe_install_pypose_ambient_grad_monkeypatch()

from covis import build_covis_graph
from geometry import (as_44, affine_inv, orthonormalize, unproject_pixels,
                       project_to_pixels, sample_at, image_gradients, grad_mag)


def to_gray(images) -> torch.Tensor:
    """(N,H,W,3) uint8 (numpy or tensor) -> (N,H,W) float tensor in [0,1]."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    return images.float().mean(-1) / 255.0


# --------------------------------------------------------------------------- #
# Odometry stage: joint sparse-LM photometric refinement over covisible pairs
# --------------------------------------------------------------------------- #
@psjac
def photo_residual(pose_i, pose_j, rd_i_cam, K_j, I_i0, I_j0, gx_j0, gy_j0, uv_j0, wgt):
    """First-order photometric residual for one covisible-pair pixel row.

    pose_i, pose_j: (M,7) gathered c2w SE3 rows (the only traced/free inputs).
    rd_i_cam: (M,3) camera-space point in view i (= unproject_pixels(uv_i,
        d_i, K_i)), CONSTANT this relinearization.
    K_j: (M,3,3) target intrinsics.
    I_i0, I_j0, gx_j0, gy_j0, uv_j0: sampled ONCE per relinearization at the
        current warp (no_grad, see `build_photo_correspondences`).
    wgt: (M,1) IRLS * certain-edge weight.
    """
    Xw = pp.SE3(pose_i).Act(rd_i_cam)                 # world point (pose_i free)
    Xc_j = pp.SE3(pose_j).Inv().Act(Xw)                # into cam j (pose_j free)
    uv_j, _ = project_to_pixels(Xc_j, K_j)
    duv = uv_j - uv_j0
    I_j_lin = I_j0 + gx_j0 * duv[..., 0] + gy_j0 * duv[..., 1]
    return (I_i0 - I_j_lin)[..., None] * wgt


class RGBDPhotoModel(nn.Module):
    """View 0 pose FIXED (gauge anchor). Free: poses 1..N-1."""

    def __init__(self, c2w_se3):
        super().__init__()
        self.fixed0 = c2w_se3[:1].clone()
        self.pose_rest = pp.Parameter(c2w_se3[1:].clone(), sjac=True)
        self.pose_rest.trim_SE3_grad = True

    def poses_all(self):
        return torch.cat([self.fixed0, self.pose_rest], dim=0)

    def forward(self, i_idx, j_idx, rd_i_cam, K_j, I_i0, I_j0, gx_j0, gy_j0, uv_j0, wgt):
        poses = self.poses_all()
        return photo_residual(poses[i_idx], poses[j_idx], rd_i_cam, K_j,
                               I_i0, I_j0, gx_j0, gy_j0, uv_j0, wgt)


def build_photo_correspondences(c2w_se3, pairs, gray, depth, K, gx_all, gy_all,
                                 n_samples=2048, z_eps=1e-3, certain_boost=3.0,
                                 max_depth_diff=None):
    """Build one relinearization's fixed correspondence set, entirely under
    `no_grad` (the only place `sample_at`/`grid_sample` is called).

    Fully vectorized across pairs -- no Python loop over `pairs.tolist()`:
      1. ONE batched `torch.multinomial` call (batch=N views, not pairs)
         draws a per-view candidate pixel pool, biased toward image-gradient
         magnitude (flat regions carry no photometric signal). Views sharing
         a source view in several pairs reuse the SAME pool for that view --
         a deliberate simplification (the source view's texture distribution
         doesn't depend on which target view j it's being compared against).
      2. Every pair's pool is gathered via advanced indexing and flattened to
         (P*n_samples,...), then unprojected/warped/projected in one batched
         call -- pure tensor algebra, no loop.
      3. Only the final image-intensity/gradient sampling groups rows by the
         (<=N) unique source/target views actually present, one `sample_at`
         call per group: `F.grid_sample` pairs input batch row b with grid
         row b, so it cannot itself gather from a different source image per
         row -- this is the only place a loop remains, and its bound is the
         number of views, not the number of pairs or samples.

    c2w_se3: (N,7) CURRENT poses (plain tensor, detached).
    Returns a dict of flat (M,...) tensors, or None if too few survive.
    """
    device = depth.device
    N, H, W = depth.shape
    if pairs is None or pairs.numel() == 0:
        return None
    dtype = depth.dtype
    gmag_all = grad_mag(gray)

    with torch.no_grad():
        # ---- Step 1: one vectorized per-view candidate pool ----
        valid = depth > z_eps
        w_flat = torch.where(valid, gmag_all, torch.zeros_like(gmag_all)).reshape(N, -1).clamp(min=1e-6)
        n_pool = min(n_samples, w_flat.shape[1])
        pool_pix = torch.multinomial(w_flat, n_pool, replacement=False)    # (N, n_pool)
        py = (pool_pix // W).to(dtype)
        px = (pool_pix % W).to(dtype)
        uv_pool = torch.stack([px + 0.5, py + 0.5], dim=-1)                 # (N, n_pool, 2)
        d_pool = torch.gather(depth.reshape(N, -1), 1, pool_pix)           # (N, n_pool)

        # ---- Step 2: gather + flatten across ALL pairs at once, then one
        # batched unproject/warp/project call -- no loop. ----
        i_idx0, j_idx0 = pairs[:, 0], pairs[:, 1]                          # (P,)
        P = i_idx0.shape[0]
        uv_i = uv_pool[i_idx0].reshape(P * n_pool, 2)
        d_i = d_pool[i_idx0].reshape(P * n_pool)
        i_flat = i_idx0.repeat_interleave(n_pool)
        j_flat = j_idx0.repeat_interleave(n_pool)

        rd_i_cam = unproject_pixels(uv_i, d_i, K[i_flat])
        Xw = pp.SE3(c2w_se3[i_flat]).Act(rd_i_cam)
        Xc_j = pp.SE3(c2w_se3[j_flat]).Inv().Act(Xw)
        uv_j0, z_j = project_to_pixels(Xc_j, K[j_flat])
        u, v = uv_j0[:, 0], uv_j0[:, 1]
        m = (d_i > z_eps) & (z_j > z_eps) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if int(m.sum()) < 8:
            return None
        rd_i_cam, uv_i, uv_j0 = rd_i_cam[m], uv_i[m], uv_j0[m]
        i_flat, j_flat = i_flat[m], j_flat[m]
        z_j = z_j[m]
        M = rd_i_cam.shape[0]

        # ---- Step 3: image sampling, grouped by the <=N unique views
        # actually present (bounded by view count, not pair/sample count). ----
        I_i0 = torch.empty(M, dtype=dtype, device=device)
        I_j0 = torch.empty(M, dtype=dtype, device=device)
        gx_j0 = torch.empty(M, dtype=dtype, device=device)
        gy_j0 = torch.empty(M, dtype=dtype, device=device)
        target_depth = torch.empty(M, dtype=dtype, device=device) if max_depth_diff is not None else None
        for vv in torch.unique(i_flat):
            sel = i_flat == vv
            I_i0[sel] = sample_at(gray[vv], uv_i[sel], H, W)
        for vv in torch.unique(j_flat):
            sel = j_flat == vv
            I_j0[sel] = sample_at(gray[vv], uv_j0[sel], H, W)
            gx_j0[sel] = sample_at(gx_all[vv], uv_j0[sel], H, W)
            gy_j0[sel] = sample_at(gy_all[vv], uv_j0[sel], H, W)
            if target_depth is not None:
                target_depth[sel] = sample_at(depth[vv], uv_j0[sel], H, W)

        if target_depth is not None:
            visible = (target_depth > z_eps) & ((target_depth - z_j).abs() <= max_depth_diff)
            if int(visible.sum()) < 8:
                return None
            i_flat, j_flat = i_flat[visible], j_flat[visible]
            rd_i_cam, uv_j0 = rd_i_cam[visible], uv_j0[visible]
            I_i0, I_j0 = I_i0[visible], I_j0[visible]
            gx_j0, gy_j0 = gx_j0[visible], gy_j0[visible]

    cat = {"i_idx": i_flat, "j_idx": j_flat, "rd_i_cam": rd_i_cam, "I_i0": I_i0,
           "I_j0": I_j0, "gx_j0": gx_j0, "gy_j0": gy_j0, "uv_j0": uv_j0}
    cat["K_j"] = K[cat["j_idx"]]
    certain = (cat["j_idx"] - cat["i_idx"]).abs() == 1
    edge_w = torch.ones(cat["i_idx"].shape[0], dtype=dtype, device=device)
    edge_w[certain] = certain_boost
    cat["edge_w"] = edge_w
    return cat


def _downsample_gray(gray: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(gray[:, None], kernel_size=2, stride=2)[:, 0]


def _downsample_depth(depth: torch.Tensor, z_eps: float = 1e-3) -> torch.Tensor:
    valid = (depth > z_eps).to(depth.dtype)
    xs = F.avg_pool2d((depth * valid)[:, None], kernel_size=2, stride=2)[:, 0]
    vs = F.avg_pool2d(valid[:, None], kernel_size=2, stride=2)[:, 0]
    out = xs / vs.clamp(min=1e-6)
    return torch.where(vs > 1e-6, out, torch.zeros_like(out))


def build_pyramid(gray: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, n_levels: int = 3):
    """Returns a list ordered COARSE -> FINE: [(gray_l, depth_l, K_l, H_l, W_l), ...].
    Intrinsics are scaled by simple multiplication (no sub-pixel pixel-center
    correction)."""
    levels = []
    g, d, k = gray, depth, K.clone()
    for _ in range(n_levels):
        H, W = g.shape[-2:]
        levels.append((g, d, k, H, W))
        g = _downsample_gray(g)
        d = _downsample_depth(d)
        k = k.clone()
        k[:, 0, :] *= 0.5
        k[:, 1, :] *= 0.5
    levels.reverse()
    return levels


def refine_odometry(w2c_init, depth, K, images, *, n_neighbors=16,
                   min_baseline_frac=0.0, min_rot_deg=0.0, n_samples=2048,
                   pyramid_levels=3, n_relin=(3, 2, 1), n_inner=5,
                   huber_delta=1.5, certain_boost=3.0, z_eps=1e-3,
                   dtype=torch.float64, verbose=True, history=None,
                   max_depth_diff=None):
    """Joint gauge-fixed sparse-LM dense photometric refinement over
    covisible-view pairs, coarse-to-fine over `pyramid_levels` image pyramid
    levels.

    w2c_init: (N,3,4) or (N,4,4). depth: (N,H,W) meters. K: (N,3,3).
    images: (N,H,W,3) uint8. Returns refined w2c (N,3,4) on depth's device.

    history: optional list; if given, the current w2c is appended to it after
    every inner LM step (purely for diagnostics/visualization, e.g.
    `showcase_render.py` -- has no effect on the optimization itself).
    """
    assert len(n_relin) == pyramid_levels, "n_relin must have one entry per pyramid level"
    device = depth.device
    depth = depth.to(dtype)
    K = K.to(dtype)
    gray = to_gray(images).to(device=device, dtype=dtype)
    cur_w2c = orthonormalize(as_44(w2c_init).to(dtype))

    pyramid = build_pyramid(gray, depth, K, n_levels=pyramid_levels)

    for lvl, (gray_l, depth_l, K_l, H_l, W_l) in enumerate(pyramid):
        gx_all, gy_all = image_gradients(gray_l)
        for relin in range(n_relin[lvl]):
            pairs = build_covis_graph(cur_w2c, n_neighbors=n_neighbors,
                                       min_baseline_frac=min_baseline_frac,
                                       min_rot_deg=min_rot_deg)
            c2w_se3 = pp.mat2SE3(affine_inv(cur_w2c)[:, :3, :], check=False).tensor().to(dtype)
            corr = build_photo_correspondences(
                c2w_se3.detach(), pairs, gray_l, depth_l, K_l, gx_all, gy_all,
                n_samples=n_samples, z_eps=z_eps, certain_boost=certain_boost,
                max_depth_diff=max_depth_diff)
            if corr is None or corr["i_idx"].numel() < 8:
                if verbose:
                    print(f"[odometry] level {lvl} relin {relin}: too few correspondences, stop level")
                break

            model = RGBDPhotoModel(c2w_se3.clone()).to(device)
            strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5 ** 4)
            solver = PCG(tol=1e-4, maxiter=250)
            opt = LM(model, strategy=strategy, solver=solver, reject=30)

            inp = {
                "i_idx": corr["i_idx"].int(), "j_idx": corr["j_idx"].int(),
                "rd_i_cam": corr["rd_i_cam"], "K_j": corr["K_j"],
                "I_i0": corr["I_i0"], "I_j0": corr["I_j0"],
                "gx_j0": corr["gx_j0"], "gy_j0": corr["gy_j0"],
                "uv_j0": corr["uv_j0"], "wgt": corr["edge_w"][:, None].clone(),
            }
            base_w = corr["edge_w"][:, None]
            for it in range(n_inner):
                if huber_delta and huber_delta > 0:
                    with torch.no_grad():
                        probe = {k: v for k, v in inp.items() if k != "wgt"}
                        r = model.forward(**probe, wgt=base_w)
                        rn = (r.reshape(-1).abs() / base_w[:, 0].clamp(min=1e-6))
                        delta = huber_delta * rn.median().clamp(min=1e-9)
                        hw = (delta / rn.clamp(min=delta)).sqrt()[:, None]
                        inp["wgt"] = base_w * hw
                # IRLS changes the objective between steps. LM caches its last
                # loss; comparing a new weighted loss to that stale value can
                # reject every update even at infinite damping.
                opt.loss = opt.model.loss(inp, None)
                loss = opt.step(inp)
                if history is not None:
                    full_it = torch.cat([model.fixed0, model.pose_rest.data], dim=0)
                    history.append(orthonormalize(affine_inv(pp.SE3(full_it).matrix())).to(dtype))
            if verbose:
                print(f"[odometry] level {lvl} relin {relin}: "
                      f"{corr['i_idx'].numel()} corr, final loss {loss.item():.6f}", flush=True)

            full = torch.cat([model.fixed0, model.pose_rest.data], dim=0)
            opt_c2w = pp.SE3(full).matrix()
            cur_w2c = orthonormalize(affine_inv(opt_c2w)).to(dtype)

    return cur_w2c[:, :3, :]


# --------------------------------------------------------------------------- #
# Structure stage: joint photometric refinement against a fixed fused structure
# --------------------------------------------------------------------------- #
@psjac
def structure_residual(pose_v, Xw_pt, K_v, C0, I_v0, gx_v0, gy_v0, uv_v0, wgt):
    """Simpler than the odometry stage's residual: structure (`Xw_pt`) is a
    fixed buffer, not a parameter -- only the observing view's pose is free."""
    Xc = pp.SE3(pose_v).Inv().Act(Xw_pt)
    uv_v, _ = project_to_pixels(Xc, K_v)
    duv = uv_v - uv_v0
    I_v_lin = I_v0 + gx_v0 * duv[..., 0] + gy_v0 * duv[..., 1]
    return ((C0 - I_v_lin)[..., None]) * wgt


class StructureModel(nn.Module):
    def __init__(self, c2w_se3):
        super().__init__()
        self.fixed0 = c2w_se3[:1].clone()
        self.pose_rest = pp.Parameter(c2w_se3[1:].clone(), sjac=True)
        self.pose_rest.trim_SE3_grad = True

    def poses_all(self):
        return torch.cat([self.fixed0, self.pose_rest], dim=0)

    def forward(self, v_idx, Xw_pt, K_v, C0, I_v0, gx_v0, gy_v0, uv_v0, wgt):
        poses = self.poses_all()
        return structure_residual(poses[v_idx], Xw_pt, K_v, C0, I_v0, gx_v0, gy_v0, uv_v0, wgt)


def fuse_structure(c2w_se3, depth, gray, K, voxel_length=0.02,
                    max_points_per_view=20000, z_eps=1e-3):
    """Unproject every view's valid-depth pixels to world space at the
    CURRENT poses, then voxel-downsample (scatter-mean position + color).
    Fixed, non-differentiable snapshot -- the GPU analogue of Open3D's
    TSDF-fused mesh, without needing a mesh/marching-cubes implementation.

    Fully vectorized -- no Python loop over views: ONE global boolean-mask
    `nonzero()` across all (view,row,col) replaces the per-view loop; the
    per-view point budget becomes a single global random subsample of size
    `N * max_points_per_view` (a deliberate simplification -- some views may
    end up contributing more or fewer points than the cap, unlike an exact
    per-view quota, but every view's valid pixels are drawn from with equal
    probability).

    Returns (X_struct (P,3), C_struct (P,)), or (None, None)."""
    device = depth.device
    N, H, W = depth.shape
    valid = depth > z_eps
    view_idx, py, px = valid.nonzero(as_tuple=True)
    if view_idx.numel() == 0:
        return None, None
    budget = max_points_per_view * N
    if view_idx.numel() > budget:
        perm = torch.randperm(view_idx.numel(), device=device)[:budget]
        view_idx, py, px = view_idx[perm], py[perm], px[perm]

    uv = torch.stack([px.to(depth.dtype) + 0.5, py.to(depth.dtype) + 0.5], dim=-1)
    d = depth[view_idx, py, px]
    Xc = unproject_pixels(uv, d, K[view_idx])
    Xw = pp.SE3(c2w_se3[view_idx]).Act(Xc)
    col = gray[view_idx, py, px]

    vox = torch.floor(Xw / voxel_length).long()
    _, inverse = torch.unique(vox, dim=0, return_inverse=True)
    P = int(inverse.max().item()) + 1
    sum_pos = torch.zeros(P, 3, dtype=Xw.dtype, device=device).index_add_(0, inverse, Xw)
    sum_col = torch.zeros(P, dtype=col.dtype, device=device).index_add_(0, inverse, col)
    cnt = torch.zeros(P, dtype=Xw.dtype, device=device).index_add_(0, inverse, torch.ones_like(col))
    X_struct = sum_pos / cnt[:, None]
    C_struct = sum_col / cnt
    return X_struct, C_struct


def zbuffer_visible_all_views(X_struct, c2w_se3, K, H, W, z_eps=1e-3, stride=4,
                              view_batch_size=8):
    """GPU z-buffer visibility, vectorized across ALL views at once: project
    every structure point into every view in one batched call, then resolve
    per-(view,pixel-bin) occlusion with a SINGLE `scatter_reduce_` over a
    combined (view,bin) key space -- replacing what would otherwise be a
    Python loop over views (the previous per-view `zbuffer_visible` function
    this generalizes).

    Chosen over a 3D-KNN "nearest structure point" because it respects
    visibility along the viewing ray (a KNN would happily pick a point behind
    a wall if it's Euclidean-closest). Bins by `stride` via floor-division
    (denser than only keeping exact multiples of stride). Ties (rare, float
    z equality) are not specially broken -- an occasional duplicated row only
    mildly reweights the least-squares residual.

    Returns a dict of flat (M,) tensors {v_idx, struct_idx, uv0}, or None.
    """
    device = X_struct.device
    N = c2w_se3.shape[0]
    # Bound temporary N*P projection tensors for scene-wide captures. Each
    # view has an independent z-buffer, so batching preserves exact visibility.
    if view_batch_size is not None and N > view_batch_size:
        chunks = []
        for start in range(0, N, view_batch_size):
            vis = zbuffer_visible_all_views(
                X_struct, c2w_se3[start:start + view_batch_size],
                K[start:start + view_batch_size], H, W, z_eps, stride,
                view_batch_size=None)
            if vis is not None:
                vis["v_idx"] += start
                chunks.append(vis)
        return ({key: torch.cat([c[key] for c in chunks]) for key in chunks[0]}
                if chunks else None)
    P = X_struct.shape[0]
    Xw = X_struct[None].expand(N, P, 3)
    Xc = pp.SE3(c2w_se3)[:, None].Inv().Act(Xw)                     # (N,P,3)
    K_exp = K[:, None].expand(N, P, 3, 3)
    uv, z = project_to_pixels(Xc, K_exp)                             # (N,P,2), (N,P)
    u, v = uv[..., 0], uv[..., 1]
    valid = (z > z_eps) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not bool(valid.any()):
        return None

    v_idx, p_idx = valid.nonzero(as_tuple=True)
    uv_m = uv[v_idx, p_idx]
    z_m = z[v_idx, p_idx]
    px = uv_m[:, 0].long().clamp(0, W - 1)
    py = uv_m[:, 1].long().clamp(0, H - 1)
    Wb = (W + stride - 1) // stride
    Hb = (H + stride - 1) // stride
    bin_id = (py // stride) * Wb + (px // stride)
    key = v_idx * (Hb * Wb) + bin_id

    min_z = torch.full((N * Hb * Wb,), float("inf"), dtype=z_m.dtype, device=device)
    min_z.scatter_reduce_(0, key, z_m, reduce="amin", include_self=True)
    is_min = z_m <= (min_z[key] + 1e-6)

    return {"v_idx": v_idx[is_min], "struct_idx": p_idx[is_min], "uv0": uv_m[is_min]}


def refine_structure(w2c_init, depth, K, images, *, voxel_length=0.02,
                   max_points_per_view=20000, pixel_stride=4, n_outer=3,
                   n_inner=5, huber_delta=1.5, z_eps=1e-3,
                   dtype=torch.float64, verbose=True, history=None):
    """Joint photometric refinement against a fixed fused structure, rebuilt
    once per outer iteration."""
    device = depth.device
    N, H, W = depth.shape
    depth = depth.to(dtype)
    K = K.to(dtype)
    gray = to_gray(images).to(device=device, dtype=dtype)
    gx_all, gy_all = image_gradients(gray)
    cur_w2c = orthonormalize(as_44(w2c_init).to(dtype))

    for outer in range(n_outer):
        c2w_se3 = pp.mat2SE3(affine_inv(cur_w2c)[:, :3, :], check=False).tensor().to(dtype)
        with torch.no_grad():
            X_struct, C_struct = fuse_structure(
                c2w_se3.detach(), depth, gray, K, voxel_length, max_points_per_view, z_eps)
        if X_struct is None or X_struct.shape[0] < 8:
            if verbose:
                print(f"[structure] outer {outer}: fused structure too small, stop")
            break

        with torch.no_grad():
            vis = zbuffer_visible_all_views(X_struct, c2w_se3.detach(), K, H, W, z_eps, pixel_stride)
        if vis is None or vis["v_idx"].numel() == 0:
            if verbose:
                print(f"[structure] outer {outer}: no visible structure correspondences, stop")
            break

        v_idx, struct_idx, uv0 = vis["v_idx"], vis["struct_idx"], vis["uv0"]
        M = v_idx.shape[0]
        I_v0 = torch.empty(M, dtype=dtype, device=device)
        gx_v0 = torch.empty(M, dtype=dtype, device=device)
        gy_v0 = torch.empty(M, dtype=dtype, device=device)
        # Only the image-sampling step needs grouping, bounded by the <=N
        # unique views present (F.grid_sample pairs input row b with grid
        # row b, so it cannot gather from a different source image per row).
        with torch.no_grad():
            for vv in torch.unique(v_idx):
                sel = v_idx == vv
                I_v0[sel] = sample_at(gray[vv], uv0[sel], H, W)
                gx_v0[sel] = sample_at(gx_all[vv], uv0[sel], H, W)
                gy_v0[sel] = sample_at(gy_all[vv], uv0[sel], H, W)

        cat = {"v_idx": v_idx, "Xw_pt": X_struct[struct_idx], "C0": C_struct[struct_idx],
               "I_v0": I_v0, "gx_v0": gx_v0, "gy_v0": gy_v0, "uv_v0": uv0}
        cat["K_v"] = K[cat["v_idx"]]

        model = StructureModel(c2w_se3.clone()).to(device)
        strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5 ** 4)
        solver = PCG(tol=1e-4, maxiter=250)
        opt = LM(model, strategy=strategy, solver=solver, reject=30)

        ones = torch.ones(cat["v_idx"].shape[0], 1, dtype=dtype, device=device)
        inp = {"v_idx": cat["v_idx"].int(), "Xw_pt": cat["Xw_pt"], "K_v": cat["K_v"],
               "C0": cat["C0"], "I_v0": cat["I_v0"], "gx_v0": cat["gx_v0"],
               "gy_v0": cat["gy_v0"], "uv_v0": cat["uv_v0"], "wgt": ones.clone()}
        for it in range(n_inner):
            if huber_delta and huber_delta > 0:
                with torch.no_grad():
                    probe = {k: v for k, v in inp.items() if k != "wgt"}
                    r = model.forward(**probe, wgt=ones)
                    rn = r.reshape(-1).abs()
                    delta = huber_delta * rn.median().clamp(min=1e-9)
                    hw = (delta / rn.clamp(min=delta)).sqrt()[:, None]
                    inp["wgt"] = hw
            opt.loss = opt.model.loss(inp, None)
            loss = opt.step(inp)
            if history is not None:
                with torch.no_grad():
                    full_it = torch.cat([model.fixed0, model.pose_rest.data], dim=0)
                    history.append(orthonormalize(affine_inv(pp.SE3(full_it).matrix())).to(dtype))
        if verbose:
            print(f"[structure] outer {outer}: {cat['v_idx'].numel()} corr, "
                  f"final loss {loss.item():.6f}", flush=True)

        full = torch.cat([model.fixed0, model.pose_rest.data], dim=0)
        opt_c2w = pp.SE3(full).matrix()
        cur_w2c = orthonormalize(affine_inv(opt_c2w)).to(dtype)

    return cur_w2c[:, :3, :]
