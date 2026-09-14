"""Pose-AUC evaluation against the COLMAP pseudo-GT, reusing DA3's tested
relative-pose-error machinery (`da3/eval_pose_auc_scannetpp.py`) rather than
reimplementing it, so numbers are directly comparable to the historical
Form D/E/D+E results in `da3/FINDINGS_vggt_refine_bug.md`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz

_DA3_SRC = str(Path(__file__).resolve().parents[2] / "da3" / "src")
if _DA3_SRC not in sys.path:
    sys.path.insert(0, _DA3_SRC)

from depth_anything_3.bench.utils import se3_to_relative_pose_error  # noqa: E402


def error_auc(errors: np.ndarray, thresholds=(5, 10, 20, 30)) -> dict:
    """Trapezoidal AUC of the recall curve. Verbatim port of
    `da3/eval_pose_auc_scannetpp.py::error_auc` (itself a port of iMatching's
    `ext/aspanformer/src/utils/metrics.py::error_auc`)."""
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
    """All-pairs max(R_err, t_err) in degrees, ported from
    `da3/eval_pose_auc_scannetpp.py::pair_errors`."""
    n = len(pred_w2c)
    pred = torch.linalg.inv(pred_w2c)
    gt = torch.linalg.inv(gt_w2c)
    r_err, t_err = se3_to_relative_pose_error(pred, gt, n)
    r_err = r_err.reshape(-1).cpu().numpy()
    t_err = t_err.reshape(-1).cpu().numpy()
    return np.maximum(r_err, t_err)


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
