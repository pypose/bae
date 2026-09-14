"""Pose/depth-free focal self-calibration.

Runs once per scene, on CPU, before any GPU optimization. This is a mandatory
default pre-step for both refinement stages (see README): dense photometric
refinement unprojects every pixel through K, so a systematically wrong focal
length bends every ray and corrupts the photometric optimum regardless of
pose. This estimator conditions on neither the input poses nor the input
depth (both of which the refinement is trying to fix), so it cannot be
circular the way a depth- or pose-conditioned calibration would be. It is a
near no-op when the input intrinsics are already accurate.
"""
from __future__ import annotations

import numpy as np
import torch

from covis import build_covis_graph
from geometry import as_44


def _sift_detect(images: np.ndarray, max_feats: int = 1024):
    """images: (N,H,W,3) uint8 numpy. Returns list of (kpts (Ki,2) xy, desc (Ki,128))."""
    import cv2
    sift = cv2.SIFT_create(nfeatures=max_feats)
    out = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None or len(kp) == 0:
            out.append((np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)))
        else:
            out.append((np.array([k.pt for k in kp], np.float32), des.astype(np.float32)))
    return out


def _match_pair_sift(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = 0.8):
    """Mutual + Lowe-ratio match on SIFT descriptors. Returns (idx_a, idx_b)."""
    import cv2
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros(0, int), np.zeros(0, int)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_match(d1, d2):
        good = {}
        for m in bf.knnMatch(d1, d2, k=2):
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good[m[0].queryIdx] = m[0].trainIdx
        return good

    ab = ratio_match(desc_a, desc_b)
    ba = ratio_match(desc_b, desc_a)
    ia, ib = [], []
    for qa, tb in ab.items():
        if ba.get(tb, -1) == qa:
            ia.append(qa); ib.append(tb)
    return np.array(ia, int), np.array(ib, int)


def estimate_focal_selfcal(w2c: torch.Tensor, K: torch.Tensor, images,
                            *, n_neighbors: int = 8, max_feats: int = 1024,
                            lo: float = 0.5, hi: float = 1.8,
                            n_coarse: int = 53, n_refine: int = 25,
                            ransac_px: float = 1.0, min_inliers: int = 40,
                            min_pairs_used: int = 6, verbose: bool = True):
    """Self-calibrate the focal, pose-free and depth-free, from fundamental
    matrices. For each covisible pair, estimate F via RANSAC on SIFT matches
    (no poses, no depth, no K), then use Sturm's criterion: a valid essential
    matrix E(m) = K(m)^T F K(m) has singular values (s,s,0), so the multiplier
    m minimizing (s1-s2)/(s1+s2), aggregated by MEDIAN over pairs, recovers
    the true shared focal multiplier without any ground truth.

    w2c: (N,4,4) or (N,3,4), used ONLY to pick which pairs to match (via
    covisibility), never in the cost itself. K: (N,3,3) initial intrinsics.
    images: (N,H,W,3) uint8 (numpy or torch).

    Returns (K_corrected (N,3,3) on K's device/dtype, info dict).
    """
    import cv2

    device = K.device
    w2c44 = as_44(w2c).double()
    Kd = K.double()
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    images = np.ascontiguousarray(images.astype(np.uint8))

    pairs = build_covis_graph(w2c44, n_neighbors=n_neighbors)
    seen, cand = set(), []
    for (i, j) in pairs.tolist():
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            cand.append(key)

    feats = _sift_detect(images, max_feats)

    Fs = []
    for (a, b) in cand:
        ia, ib = _match_pair_sift(feats[a][1], feats[b][1])
        if len(ia) < min_inliers:
            continue
        pa = feats[a][0][ia].astype(np.float64)
        pb = feats[b][0][ib].astype(np.float64)
        F = mask = None
        for method in (cv2.USAC_MAGSAC, cv2.FM_RANSAC):
            try:
                F, mask = cv2.findFundamentalMat(
                    pa, pb, method=method, ransacReprojThreshold=ransac_px,
                    confidence=0.9999, maxIters=10000)
            except cv2.error:
                F = mask = None
            if F is not None and getattr(F, "shape", None) == (3, 3):
                break
        if F is None or F.shape != (3, 3) or mask is None:
            continue
        if int(mask.sum()) < min_inliers:
            continue
        Fs.append(torch.from_numpy(np.asarray(F, dtype=np.float64)).to(device))

    if len(Fs) < min_pairs_used:
        if verbose:
            print(f"[selfcal] only {len(Fs)} usable fundamental matrices "
                  f"(need {min_pairs_used}); leaving K unchanged")
        return K, {"multiplier": 1.0, "ok": False, "n_pairs_used": len(Fs)}
    Fst = torch.stack(Fs)                                       # (Q,3,3)

    fx0 = float(Kd[:, 0, 0].mean())
    fy0 = float(Kd[:, 1, 1].mean())
    cx = float(Kd[0, 0, 2])
    cy = float(Kd[0, 1, 2])

    def score(m):
        Km = torch.eye(3, dtype=torch.float64, device=device)
        Km[0, 0] = fx0 * m
        Km[1, 1] = fy0 * m
        Km[0, 2] = cx
        Km[1, 2] = cy
        E = Km.T[None] @ Fst @ Km[None]                         # (Q,3,3)
        s = torch.linalg.svdvals(E)                             # (Q,3) descending
        return float(((s[:, 0] - s[:, 1])
                      / (s[:, 0] + s[:, 1]).clamp(min=1e-12)).median())

    grid = np.linspace(lo, hi, n_coarse)
    scores = [score(float(x)) for x in grid]
    b = int(np.nanargmin(scores))
    step = (hi - lo) / (n_coarse - 1)
    g2 = np.linspace(max(lo, grid[b] - step), min(hi, grid[b] + step), n_refine)
    s2 = [score(float(x)) for x in g2]
    m_best = float(g2[int(np.nanargmin(s2))])
    at1 = score(1.0)
    if verbose:
        print(f"[selfcal] multiplier {m_best:.4f} "
              f"(E-constraint {min(s2):.5f} vs {at1:.5f} at m=1.0), "
              f"fx {fx0:.1f} -> {fx0 * m_best:.1f}, "
              f"{len(Fs)}/{len(cand)} pairs usable", flush=True)
    Kout = Kd.clone()
    Kout[:, 0, 0] = Kd[:, 0, 0] * m_best
    Kout[:, 1, 1] = Kd[:, 1, 1] * m_best
    return Kout.to(K.dtype), {"multiplier": m_best, "ok": True,
                               "n_pairs_used": len(Fs), "n_pairs_cand": len(cand),
                               "resid_at_1": at1, "resid_best": float(min(s2))}
