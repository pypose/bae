"""Pose-AUC evaluation against the COLMAP pseudo-GT.

For each unordered pair of frames (i, j), compares the *relative* pose
predicted between them against the relative pose from ground truth: rotation
error is the geodesic angle between the two relative rotations, translation
error is the angle between the two relative-translation *directions* (no
magnitude comparison, since a monocular/RGB-D reconstruction's absolute scale
is unrecoverable without an external reference). Both are gauge-invariant (a
shared global transform on all predicted poses cancels in the relative pose)
and scale-invariant, so no Sim(3) alignment to ground truth is needed. This
is the standard relative-pose-error protocol used in the wide-baseline
pose-estimation literature (SuperGlue, LoFTR, and follow-on work).
"""
from __future__ import annotations

import numpy as np
import torch

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz


def error_auc(errors: np.ndarray, thresholds=(5, 10, 20, 30)) -> dict:
    """Trapezoidal area under the recall-vs-error-threshold curve, normalized
    by each threshold -- the standard pose-AUC metric."""
    errors = np.asarray(errors, dtype=np.float64)
    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]

    aucs = {}
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1] if last > 0 else 0.0]
        e = np.r_[errors[:last], t]
        aucs[f"auc@{t}"] = float(_trapz(r, x=e) / t)
    return aucs


def pair_errors(pred_w2c: torch.Tensor, gt_w2c: torch.Tensor) -> np.ndarray:
    """All-pairs max(rotation_err_deg, translation_err_deg) between predicted
    and ground-truth poses for one scene. pred_w2c, gt_w2c: (N,4,4)."""
    n = pred_w2c.shape[0]
    c2w_pred = torch.linalg.inv(pred_w2c)
    c2w_gt = torch.linalg.inv(gt_w2c)
    ii, jj = torch.combinations(torch.arange(n), r=2).unbind(-1)

    rel_pred = torch.linalg.inv(c2w_pred[ii]) @ c2w_pred[jj]     # T_i^-1 @ T_j, predicted
    rel_gt = torch.linalg.inv(c2w_gt[ii]) @ c2w_gt[jj]

    R_err = rel_pred[:, :3, :3].transpose(-1, -2) @ rel_gt[:, :3, :3]
    tr = R_err.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1.0, 3.0)
    rot_err_deg = torch.rad2deg(torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))

    t_pred = rel_pred[:, :3, 3]
    t_gt = rel_gt[:, :3, 3]
    n_pred = t_pred / t_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    n_gt = t_gt / t_gt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_t = (n_pred * n_gt).sum(-1).clamp(-1.0, 1.0).abs()       # direction-only, sign-ambiguous
    t_err_deg = torch.rad2deg(torch.arccos(cos_t))

    return torch.maximum(rot_err_deg, t_err_deg).cpu().numpy()


def evaluate_against_colmap(w2c_pred: torch.Tensor, frame_names, gt_w2c: dict,
                             thresholds=(5, 10, 20, 30)) -> dict:
    """w2c_pred: (N,4,4) or (N,3,4) predicted poses, aligned index-for-index
    with `frame_names`. Only frames present in `gt_w2c` (COLMAP-registered)
    are scored. Returns the `error_auc` dict, or {} if fewer than 2 frames
    overlap."""
    from geometry import as_44

    keep = [i for i, n in enumerate(frame_names) if n in gt_w2c]
    if len(keep) < 2:
        return {}
    w2c_pred = as_44(w2c_pred)[keep].detach().cpu().double()
    gt = torch.stack([torch.from_numpy(gt_w2c[frame_names[i]]) for i in keep]).double()
    errs = pair_errors(w2c_pred, gt)
    return error_auc(errs, thresholds)
