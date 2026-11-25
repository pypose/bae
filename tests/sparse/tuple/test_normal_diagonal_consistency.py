import pypose as pp
import pytest
import torch
import torch.nn as nn

from bae.optim import LM
from bae.utils.linear_operator import NormalMatVec


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS is unavailable'))])
def test_lm_reuses_one_diagonal_reduction_for_small_damping(device, monkeypatch):
    """Reduction-order noise must not turn positive damping into a subtraction."""

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.variable = pp.Parameter(torch.zeros(2, 1, device=device), sjac=True)

        def forward(self, target, index):
            return self.variable[index] * 10000.0 - target

    reductions = 0
    normal_diagonal = NormalMatVec._compute_diag

    def count_reductions(jacobian):
        nonlocal reductions
        reductions += 1
        return normal_diagonal(jacobian)

    monkeypatch.setattr(NormalMatVec, "_compute_diag", staticmethod(count_reductions))

    class InspectSolver(nn.Module):
        def forward(self, operator, rhs):
            expected = torch.full_like(rhs.flatten(), 1e8) * (1 + 1e-6)
            torch.testing.assert_close(operator.diagonal(), expected, rtol=0, atol=0)
            return torch.zeros_like(rhs)

    model = Residual()
    optimizer = LM(model, solver=InspectSolver(), matrix_free_normal=True,
                   strategy=pp.optim.strategy.Constant(damping=1e-6))
    loss = optimizer.step({"target": torch.ones(2, 1, device=device),
                           "index": torch.arange(2, device=device)})
    assert reductions == 1
    assert torch.isfinite(loss)


def test_sub_epsilon_float32_damping_remains_positive():
    jacobian = torch.tensor([[1e5, 1e5]], dtype=torch.float32)
    operator = NormalMatVec(jacobian, damping=1e-8)
    # In this null direction J.T @ J contributes zero, so the result directly
    # exposes the positive damping that a rounded (1 + damping) would lose.
    vector = torch.tensor([1., -1.])
    torch.testing.assert_close(operator @ vector, torch.tensor([100., -100.]))


@pytest.mark.parametrize("raw", [
    torch.ones(2, 1),
    torch.ones(3),
    torch.ones(2, dtype=torch.float64),
    torch.ones(2, device="meta"),
])
def test_supplied_raw_diagonal_is_validated(raw):
    with pytest.raises(ValueError, match="raw_diag"):
        NormalMatVec(torch.eye(2), raw_diag=raw)
