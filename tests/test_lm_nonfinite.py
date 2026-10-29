import pytest
import torch
import pypose as pp
from pypose.autograd.function import psjac

from bae.optim.optimizer import LM, Schur
from bae.utils.pysolvers import PCG


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class TrialModel(torch.nn.Module):
    def __init__(self, kind="sqrt"):
        super().__init__()
        self.kind = kind
        self.x = pp.Parameter(torch.full((2, 2), 0.0 if kind in ("exp", "uphill") else 1.0,
                                           device="cuda", dtype=torch.float64), sjac=True)
        self.y = pp.Parameter(torch.tensor([[0.2, 0.3], [0.4, 0.5]], device="cuda", dtype=torch.float64), sjac=True)

    @psjac
    def sqrt_residual(x, y):
        return torch.cat((x.sqrt() - 0.1, y), dim=-1)

    @psjac
    def exp_residual(x, y):
        return torch.cat((x.exp() - 1000.0, y), dim=-1)

    @psjac
    def uphill_residual(x, y):
        return torch.cat((x.exp() - 10.0, y), dim=-1)

    @psjac
    def linear_residual(x, y):
        return torch.cat((x, y), dim=-1)

    def forward(self):
        return getattr(TrialModel, self.kind + "_residual")(self.x, self.y)


@pytest.mark.parametrize("optimizer_cls", [LM, Schur])
@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("kind", ["sqrt", "exp"])
@pytest.mark.parametrize("reject", [0, 2])
def test_nonfinite_trials_restore_state(optimizer_cls, matrix_free, kind, reject):
    model = TrialModel(kind)
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    optimizer = optimizer_cls(model, solver=PCG(tol=1e-12, maxiter=20),
                              strategy=pp.optim.strategy.Constant(damping=1e-6),
                              reject=reject, matrix_free_normal=matrix_free)
    initial_loss = optimizer.model.loss((), None).detach().clone()
    for _ in range(2):
        loss = optimizer.step(())
        assert torch.isfinite(loss)
        assert optimizer.reject_count == reject
        for parameter, original, value in zip(model.parameters(), params, before):
            assert parameter is original
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
        torch.testing.assert_close(loss, initial_loss, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.model.loss((), None), loss, rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_cls", [LM, Schur])
@pytest.mark.parametrize("matrix_free", [False, True])
def test_finite_trials_still_accepted(optimizer_cls, matrix_free):
    model = TrialModel("linear")
    before = [p.detach().clone() for p in model.parameters()]
    damping = 0.1
    optimizer = optimizer_cls(model, solver=PCG(tol=1e-12, maxiter=20),
                              strategy=pp.optim.strategy.Constant(damping=damping),
                              matrix_free_normal=matrix_free)
    loss = optimizer.step(())
    for parameter, value in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, value * damping / (1 + damping))
    assert optimizer.reject_count == 0
    torch.testing.assert_close(loss, optimizer.model.loss((), None))


@pytest.mark.parametrize("optimizer_cls", [LM, Schur])
@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("reject", [0, 2])
def test_finite_uphill_trial_at_budget_is_preserved(optimizer_cls, matrix_free, reject):
    model = TrialModel("uphill")
    optimizer = optimizer_cls(model, solver=PCG(tol=1e-12, maxiter=20),
                              strategy=pp.optim.strategy.Constant(damping=1e-6),
                              reject=reject, matrix_free_normal=matrix_free)
    initial_loss = optimizer.model.loss((), None).detach().clone()
    loss = optimizer.step(())
    assert torch.isfinite(loss)
    assert loss > initial_loss
    assert optimizer.reject_count == reject
    torch.testing.assert_close(loss, optimizer.model.loss((), None))


class NonfiniteSolver(torch.nn.Module):
    def forward(self, A, b):
        return torch.full_like(b, float("nan"))


@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("reject", [0, 2])
def test_nonfinite_update_restores_parameters(matrix_free, reject):
    model = TrialModel("linear")
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = LM(model, solver=NonfiniteSolver(),
                   strategy=pp.optim.strategy.Constant(damping=1e-6),
                   reject=reject, matrix_free_normal=matrix_free)
    initial_loss = optimizer.model.loss((), None).detach().clone()
    loss = optimizer.step(())
    for parameter, value in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    assert optimizer.reject_count == reject
    torch.testing.assert_close(loss, initial_loss, rtol=0, atol=0)
