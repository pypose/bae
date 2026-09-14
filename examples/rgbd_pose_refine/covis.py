"""Covisibility graph construction, ported verbatim (logic-preserving) from
`da3/refine_poses.py::build_covis_graph`."""
from __future__ import annotations

import numpy as np
import torch

from geometry import as_44, affine_inv


def build_covis_graph(w2c: torch.Tensor, n_neighbors: int = 16,
                       dir_thresh_deg: float = 75.0,
                       min_baseline_frac: float = 0.0,
                       min_rot_deg: float = 0.0) -> torch.Tensor:
    """Directed covisible pairs (i->j) by camera-center kNN gated by
    view-direction agreement. Guarantees connectivity (index-neighbor
    fallback) so every view appears in >=1 pair (an unreferenced view would
    have an empty bae parameter column, which is a solver failure).

    min_baseline_frac / min_rot_deg reject pairs whose camera centers are
    closer than `min_baseline_frac` x (median pairwise camera distance), or
    whose relative rotation is under `min_rot_deg`. Both default to 0 (off).
    At high view counts, unfiltered kNN picks near-duplicate frames whose
    photometric alignment is close to degenerate (many pose configurations
    satisfy it about equally), so gating on baseline/rotation keeps the graph
    made of pairs that actually constrain relative pose.

    Returns LongTensor (P,2) of (i,j) pairs.
    """
    N = len(w2c)
    c2w = affine_inv(as_44(w2c))
    centers = c2w[:, :3, 3]                                     # (N,3)
    fwd = c2w[:, :3, 2]                                         # +z forward (N,3)
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    dist = torch.cdist(centers, centers)                       # (N,N)
    dist.fill_diagonal_(float("inf"))
    cos = fwd @ fwd.T
    dir_ok = cos > np.cos(np.deg2rad(dir_thresh_deg))

    if min_baseline_frac > 0:
        finite = dist[torch.isfinite(dist)]
        if finite.numel():
            thr = float(finite.median()) * min_baseline_frac
            dir_ok = dir_ok & (dist >= thr)
    if min_rot_deg > 0:
        R = as_44(w2c)[:, :3, :3]
        rel = torch.einsum("nij,mkj->nmik", R, R)               # R_i @ R_j^T
        tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1.0, 3.0)
        ang = torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))
        dir_ok = dir_ok & (ang >= np.deg2rad(min_rot_deg))

    # Vectorized top-k selection (one GPU op, not a per-candidate `.item()`
    # sync): mask disallowed entries to +inf, then topk picks the k nearest
    # among the dir_ok=True (and baseline/rot-gated) candidates in one call.
    k = min(n_neighbors, N - 1)
    masked_dist = dist.masked_fill(~dir_ok, float("inf"))
    topk_dist, topk_idx = torch.topk(masked_dist, k, dim=1, largest=False)
    valid_topk = torch.isfinite(topk_dist)                     # (N,k)

    # ONE bulk device->host transfer (not O(N*k) `.item()` calls); the
    # remaining per-row loop is pure Python set bookkeeping (no GPU ops) to
    # assemble the variable-length, deduplicated, symmetric edge list and
    # apply the fixed index-neighbor fallback for under-connected rows.
    topk_idx_np = topk_idx.cpu().numpy()
    valid_np = valid_topk.cpu().numpy()

    pairs = set()
    for i in range(N):
        picked = topk_idx_np[i, valid_np[i]].tolist()
        if len(picked) < 2:                                    # fallback: nearest by index
            picked = sorted({(i - 1) % N, (i + 1) % N} | set(picked) - {i})
        for j in picked:
            if j != i:
                pairs.add((i, j))
                pairs.add((j, i))                              # symmetric
    return torch.tensor(sorted(pairs), dtype=torch.long, device=w2c.device)
