"""Depth-based correspondence induction, texture gate, and trust-region
accept/reject guards. Ported from `da3/refine_poses.py`.

These two guards are NOT part of Open3D's own RGB-D odometry / color-map
tutorials -- they were added empirically after finding that unguarded
photometric refinement regresses on a meaningful fraction of real scenes
(textureless rooms, already-good poses with a small basin). Both hook into
`run_refine.py` around the Form D/E optimization, not inside it.
"""
from __future__ import annotations

import numpy as np
import torch

from geometry import as_44, affine_inv, unproject_pixels, cam_to_world, world_to_cam, \
    project_to_pixels, sample_at, grad_mag


# --------------------------------------------------------------------------- #
# Depth-induced correspondences (used by the trust-region pixel set, and by
# Form D's coarse initial pair filtering -- NOT by the photometric residual
# itself, which uses first-order linearization around a warp built with the
# CURRENT pose estimate, see photometric.py).
# --------------------------------------------------------------------------- #
def conf_thresholds(conf: torch.Tensor, percentile: float = 40.0):
    """Per-view confidence threshold = `percentile`-th percentile of each
    view's conf map. conf: (N,H,W). Returns (N,)."""
    flat = conf.reshape(conf.shape[0], -1)
    return torch.quantile(flat.float(), percentile / 100.0, dim=1)


def induce_correspondences(w2c, depth, conf, K, pairs, n_samples=2048,
                            conf_percentile=40.0, z_eps=1e-3):
    """For each pair (i->j): sample valid (optionally confident) pixels in i,
    unproject via depth_i, project into j; keep matches that are in-frustum,
    in front of j, and where view-j's measured depth is valid (+confident).

    `conf` may be None (ScanNet++ iPhone depth has no confidence channel;
    every valid-depth pixel is trusted equally).

    Fully vectorized across pairs -- no Python loop over `pairs.tolist()`,
    mirroring `photometric.py::build_photo_correspondences`'s design: one
    batched per-view candidate pool (weighted by confidence, or uniform when
    `conf` is None), one flattened batched unproject/warp/project call across
    all pairs, and a final `sample_at` step (for the target-view depth/conf)
    grouped by the (<=N) unique target views present -- `F.grid_sample` pairs
    input batch row b with grid row b, so it can't gather from a different
    source image per row in one call; this is the same intrinsic boundary
    documented in `photometric.py`, not a missed vectorization.

    Returns a dict of flat tensors over all kept correspondences (i_idx,
    j_idx, uv_i, d_i, uv_j, d_j, z_j, w), or None if too few survive.
    """
    device = depth.device
    N, H, W = depth.shape
    if pairs is None or pairs.numel() == 0:
        return None
    dtype = depth.dtype
    w2c44 = as_44(w2c)
    c2w = affine_inv(w2c44)
    ci_all = conf if conf is not None else torch.ones_like(depth)
    cthr = conf_thresholds(conf, conf_percentile) if conf is not None else None

    valid = depth > z_eps
    if cthr is not None:
        valid = valid & (ci_all >= cthr[:, None, None])
    w_flat = torch.where(valid, ci_all, torch.zeros_like(ci_all)).reshape(N, -1).clamp(min=1e-6)
    n_pool = min(n_samples, w_flat.shape[1])
    pool_pix = torch.multinomial(w_flat, n_pool, replacement=False)        # (N, n_pool)
    py = (pool_pix // W).to(dtype)
    px = (pool_pix % W).to(dtype)
    uv_pool = torch.stack([px + 0.5, py + 0.5], dim=-1)                    # (N, n_pool, 2)
    d_pool = torch.gather(depth.reshape(N, -1), 1, pool_pix)
    ci_pool = torch.gather(ci_all.reshape(N, -1).to(dtype), 1, pool_pix)

    i_idx0, j_idx0 = pairs[:, 0], pairs[:, 1]
    P = i_idx0.shape[0]
    uv_i = uv_pool[i_idx0].reshape(P * n_pool, 2)
    d_i = d_pool[i_idx0].reshape(P * n_pool)
    ci_i = ci_pool[i_idx0].reshape(P * n_pool)
    i_flat = i_idx0.repeat_interleave(n_pool)
    j_flat = j_idx0.repeat_interleave(n_pool)

    Xc_i = unproject_pixels(uv_i, d_i, K[i_flat])
    Xw = cam_to_world(Xc_i, c2w[i_flat])
    Xc_j = world_to_cam(Xw, w2c44[j_flat])
    uv_j, z_j = project_to_pixels(Xc_j, K[j_flat])
    u, v = uv_j[:, 0], uv_j[:, 1]
    m = (d_i > z_eps) & (z_j > z_eps) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if int(m.sum()) == 0:
        return None
    uv_i, d_i, ci_i, uv_j, z_j = uv_i[m], d_i[m], ci_i[m], uv_j[m], z_j[m]
    i_flat, j_flat = i_flat[m], j_flat[m]
    M = uv_i.shape[0]

    d_j = torch.empty(M, dtype=dtype, device=device)
    cj = torch.empty(M, dtype=dtype, device=device)
    for vv in torch.unique(j_flat):
        sel = j_flat == vv
        d_j[sel] = sample_at(depth[vv], uv_j[sel], H, W)
        cj[sel] = sample_at(ci_all[vv], uv_j[sel], H, W)

    m2 = d_j > z_eps
    if cthr is not None:
        m2 = m2 & (cj >= cthr[j_flat])
    if int(m2.sum()) == 0:
        return None

    return {
        "i_idx": i_flat[m2], "j_idx": j_flat[m2],
        "uv_i": uv_i[m2], "d_i": d_i[m2],
        "uv_j": uv_j[m2], "d_j": d_j[m2], "z_j": z_j[m2],
        "w": torch.minimum(ci_i[m2], cj[m2]),
    }


# --------------------------------------------------------------------------- #
# Texture gate: skip refinement outright on textureless scenes.
# --------------------------------------------------------------------------- #
def scene_texture(images) -> float:
    """Mean abs image-gradient over views (intensity in [0,1]); low => textureless.
    images: (N,H,W,3) uint8 numpy."""
    g = images.astype(np.float32).mean(-1) / 255.0
    return float(0.5 * (np.abs(np.diff(g, axis=2)).mean()
                        + np.abs(np.diff(g, axis=1)).mean()))


def covis_texture(images, w2c, depth, conf, K, pairs, H, W, device,
                   conf_percentile=40.0, n_samples=1024) -> float:
    """Texture measured ONLY in the co-visible regions odometry actually uses
    (unlike `scene_texture`'s whole-image mean, which is dragged down by
    textureless regions the optimizer never samples)."""
    gray = torch.from_numpy(images.astype(np.float32).mean(-1) / 255.0).to(device)
    gmag = grad_mag(gray)
    corr = induce_correspondences(w2c, depth, conf, K, pairs, n_samples=n_samples,
                                   conf_percentile=conf_percentile)
    if corr is None or corr["i_idx"].numel() == 0:
        return float("nan")
    vals = torch.empty(corr["i_idx"].numel(), device=device, dtype=gmag.dtype)
    for vv in torch.unique(corr["i_idx"]):
        sel = corr["i_idx"] == vv
        vals[sel] = sample_at(gmag[vv], corr["uv_i"][sel], H, W)
    return float(vals.mean())


def texture_gate(images, w2c, depth, conf, K, pairs, *, tex_gate=0.003,
                  mode="covis", verbose=True) -> bool:
    """Returns True if the scene passes the texture gate (refinement should
    proceed), False if it should be skipped."""
    device = depth.device
    H, W = depth.shape[-2:]
    if mode == "covis":
        tex = covis_texture(images, w2c, depth, conf, K, pairs, H, W, device)
    else:
        tex = scene_texture(images)
    ok = np.isfinite(tex) and tex >= tex_gate
    if verbose:
        print(f"[texture-gate] mode={mode} tex={tex:.5f} threshold={tex_gate:.5f} "
              f"-> {'PASS' if ok else 'SKIP (textureless)'}", flush=True)
    return ok


# --------------------------------------------------------------------------- #
# Trust-region accept/reject.
# --------------------------------------------------------------------------- #
def rel_rot_deg(w2c, i_idx, j_idx):
    """Relative rotation angle (deg) between view pairs under the GIVEN poses."""
    w2c44 = as_44(w2c)
    Ri = w2c44[i_idx, :3, :3]
    Rj = w2c44[j_idx, :3, :3]
    Rrel = Rj @ Ri.transpose(-1, -2)
    tr = Rrel[..., 0, 0] + Rrel[..., 1, 1] + Rrel[..., 2, 2]
    cos = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def wide_pairs(w2c, n_pairs=4096, min_rot_deg=20.0, seed=0):
    """Random view pairs with a genuine baseline, for measuring the trust
    region. Deliberately NOT the covisibility kNN graph used by the
    optimizer itself -- scoring the guard on the optimizer's own neighbor
    list is circular (near-duplicate pairs are trivially satisfied by almost
    any pose set), so this samples independently and drops near-duplicates
    outright rather than backfilling them."""
    w2c44 = as_44(w2c)
    N = w2c44.shape[0]
    if N < 2:
        return torch.zeros((0, 2), dtype=torch.long, device=w2c44.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = min(max(n_pairs * 8, 64), N * (N - 1) // 2)
    ii = torch.randint(0, N, (m,), generator=g)
    jj = torch.randint(0, N, (m,), generator=g)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    if ii.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=w2c44.device)
    lo = torch.minimum(ii, jj)
    hi = torch.maximum(ii, jj)
    _, uniq = np.unique((lo * N + hi).numpy(), return_index=True)
    lo, hi = lo[uniq], hi[uniq]
    if min_rot_deg > 0:
        rr = rel_rot_deg(w2c44.detach().cpu(), lo, hi)
        sel = rr >= min_rot_deg
        lo, hi = lo[sel], hi[sel]
    if lo.numel() > n_pairs:
        lo, hi = lo[:n_pairs], hi[:n_pairs]
    return torch.stack([lo, hi], dim=-1).to(w2c44.device)


def photo_resid_penalized(w2c, src, K, gray, H, W, z_eps=1e-3):
    """Photometric residual with a CONSTANT denominator, comparable across
    pose sets with different in-frustum support: every source pixel counts,
    in-frustum pixels contribute |I_i - I_j|, out-of-frustum pixels
    contribute the maximum possible 1.0 (so a pose set can't lower its own
    score by pushing pixels out of view). Returns (score, support_fraction).
    """
    w2c44 = as_44(w2c).double()
    c2w = affine_inv(w2c44)
    uv_i, d_i = src["uv_i"], src["d_i"]
    i_idx, j_idx = src["i_idx"], src["j_idx"]
    n_tot = int(uv_i.shape[0])
    if n_tot == 0:
        return float("nan"), 0.0
    Xc_i = unproject_pixels(uv_i, d_i, K[i_idx])
    Xw = cam_to_world(Xc_i, c2w[i_idx])
    Xc_j = world_to_cam(Xw, w2c44[j_idx])
    uv_j, z_j = project_to_pixels(Xc_j, K[j_idx])
    u, v = uv_j[:, 0], uv_j[:, 1]
    m = ((z_j > z_eps) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
         & torch.isfinite(u) & torch.isfinite(v))
    n_in = int(m.sum())
    if n_in == 0:
        return 1.0, 0.0
    ii, jj = i_idx[m], j_idx[m]
    uvi_m, uvj_m = uv_i[m], uv_j[m]
    Ii = torch.empty(n_in, device=gray.device, dtype=gray.dtype)
    Ij = torch.empty_like(Ii)
    for vv in torch.unique(ii):
        sel = ii == vv
        Ii[sel] = sample_at(gray[vv], uvi_m[sel], H, W)
    for vv in torch.unique(jj):
        sel = jj == vv
        Ij[sel] = sample_at(gray[vv], uvj_m[sel], H, W)
    score = (float((Ii - Ij).abs().sum()) + 1.0 * (n_tot - n_in)) / n_tot
    return score, n_in / n_tot


def trust_region_accept_reject(w2c_before, w2c_after, depth, conf, K, images,
                                *, trust_pairs="wide", trust_n_pairs=4096,
                                trust_min_rot=20.0, trust_seed=0,
                                stage_name="stage", verbose=True):
    """Score a FIXED held-out photometric objective before vs after a
    refinement stage; reject (revert to `w2c_before`) if the score got worse.

    `trust_pairs="wide"` (default) is recommended over `"covis"`: the
    optimizer's own covisibility neighbor list is circular for this purpose
    (measured: on 7/7 destroyed scenes the covis residual improved while
    accuracy fell -- the guard would have accepted every one).
    """
    device = depth.device
    H, W = depth.shape[-2:]
    gray = torch.from_numpy(images.astype(np.float32).mean(-1) / 255.0).to(device).double()
    Kd = K.double()

    if trust_pairs == "wide":
        pairs = wide_pairs(w2c_before, n_pairs=trust_n_pairs,
                            min_rot_deg=trust_min_rot, seed=trust_seed)
    else:
        pairs = None  # caller passes a covis graph via src below in that case

    src = induce_correspondences(w2c_before, depth, conf, Kd, pairs,
                                  n_samples=2048) if pairs is not None else None
    if src is None or src["i_idx"].numel() == 0:
        if verbose:
            print(f"[trust-region:{stage_name}] no held-out pixel set, accepting by default")
        return w2c_after, True

    score_before, sup_before = photo_resid_penalized(w2c_before, src, Kd, gray, H, W)
    score_after, sup_after = photo_resid_penalized(w2c_after, src, Kd, gray, H, W)
    accept = score_after <= score_before
    if verbose:
        verdict = "ACCEPT" if accept else "REJECT (regression)"
        print(f"[trust-region:{stage_name}] before={score_before:.5f} (support "
              f"{sup_before:.2f}) after={score_after:.5f} (support {sup_after:.2f}) "
              f"-> {verdict}", flush=True)
    return (w2c_after if accept else w2c_before), accept
