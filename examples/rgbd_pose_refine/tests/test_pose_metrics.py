"""Rendering and benchmarking must agree on the reported pose AUC."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import error_auc, pair_errors, pair_errors_numpy
from _scene_fixture import make_synthetic_scene


def test_numpy_and_torch_pose_metrics_agree():
    _, _, _, gt, initial = make_synthetic_scene(n_views=4, H=8, W=8)
    numpy_errors = pair_errors_numpy(initial.numpy(), gt.numpy())
    torch_errors = pair_errors(initial, gt)
    np.testing.assert_allclose(numpy_errors, torch_errors, rtol=1e-8, atol=1e-7)
    numpy_auc, torch_auc = error_auc(numpy_errors), error_auc(torch_errors)
    for key in numpy_auc:
        np.testing.assert_allclose(numpy_auc[key], torch_auc[key], rtol=1e-8, atol=1e-7)


def test_auc_includes_partial_threshold_interval():
    # At threshold 5: trapezoids contribute 1/6 + 1/2 + 2, divided by 5.
    assert np.isclose(error_auc(np.array([6., 2., 1.]), (5,))["auc@5"], 8 / 15)
