"""Check the pose-AUC integration convention used by the benchmark."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import error_auc


def test_auc_includes_partial_threshold_interval():
    # At threshold 5: trapezoids contribute 1/6 + 1/2 + 2, divided by 5.
    assert np.isclose(error_auc(np.array([6., 2., 1.]), (5,))["auc@5"], 8 / 15)
