"""The plotted history and acceptance decision must use the same pixel set."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import guards
from _scene_fixture import make_synthetic_scene


def scene():
    images, depth, K, gt, _ = make_synthetic_scene(n_views=3, H=24, W=24)
    return images.numpy(), depth, K.repeat(3, 1, 1), gt


def test_history_reuses_pixels_and_rejects_regression(monkeypatch):
    images, depth, K, initial = scene()
    candidate = initial.clone()
    candidate[1:, 0, 3] += 100  # Move the observed points out of the target images.
    calls = []
    original = guards.induce_correspondences

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(guards, "induce_correspondences", record)
    scores = guards.score_pose_history([initial, candidate, initial, candidate],
                                      depth, None, K, images, min_rot_deg=0,
                                      n_samples=64)
    assert len(calls) == 1
    assert scores.shape == (4, 2)
    np.testing.assert_array_equal(scores[0], scores[2])
    np.testing.assert_array_equal(scores[1], scores[3])
    assert scores[-1, 0] > scores[0, 0]
    assert not guards.accept_held_out_scores(scores, verbose=False)
    assert guards.accept_held_out_scores(scores[::-1], verbose=False)


def test_guard_matches_direct_scoring_and_returns_original_on_rejection():
    images, depth, K, initial = scene()
    candidate = initial.clone()
    candidate[1:, 0, 3] += 100
    torch.manual_seed(19)
    scores = guards.score_pose_history([initial, candidate], depth, None, K, images,
                                      n_pairs=32, min_rot_deg=0, seed=7)
    torch.manual_seed(19)
    returned, accepted = guards.trust_region_accept_reject(
        initial, candidate, depth, None, K, images,
        trust_n_pairs=32, trust_min_rot=0, trust_seed=7, verbose=False)
    assert accepted == guards.accept_held_out_scores(scores, verbose=False)
    assert not accepted
    assert returned is initial


def test_no_held_out_pixels_preserves_accept_by_default():
    images, depth, K, initial = scene()
    scores = guards.score_pose_history([initial, initial], depth, None, K, images,
                                      min_rot_deg=181)
    assert scores is None
    assert guards.accept_held_out_scores(scores, verbose=False)


def test_unsupported_pair_mode_is_not_silently_accepted():
    images, depth, K, initial = scene()
    with pytest.raises(TypeError, match="trust_pairs"):
        guards.trust_region_accept_reject(initial, initial, depth, None, K, images,
                                         trust_pairs="covis", verbose=False)
