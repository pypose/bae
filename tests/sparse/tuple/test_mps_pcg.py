"""Regression checks for the single-RHS MPS CG reduction path."""

import pytest
import torch
from pypose.optim.solver import CG
from torch.utils._python_dispatch import TorchDispatchMode

from bae.utils.linear_operator import DiagonalMatVec, _MatVecOperator
from bae.utils.pysolvers import PCG


MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


def _system(device="cpu"):
    factor = torch.tensor([[2., 0., 1., 0.], [1., 3., 0., 0.],
                           [0., 1., 2., 1.], [0., 0., 1., 2.]])
    matrix = factor.mT @ factor + torch.eye(4)
    rhs = torch.tensor([[1.], [-2.], [3.], [0.5]])
    return matrix.to(device), rhs.to(device)


class _CountingMatrix(_MatVecOperator):
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape, self.device, self.dtype = matrix.shape, matrix.device, matrix.dtype
        self.calls = []

    def matvec(self, x, out=None):
        self.calls.append(None if out is None else out.data_ptr())
        return torch.matmul(self.matrix, x, out=out)


@pytest.mark.parametrize("maxiter", [0, 1, 2, None])
def test_reduction_recurrence_matches_pypose_and_mutates_warm_start(maxiter):
    matrix, rhs = _system()
    initial = torch.full_like(rhs, 0.125)
    reference = CG(maxiter=maxiter, tol=1e-6)(
        matrix, rhs, x=initial.clone(), M=torch.diag(matrix.diagonal().reciprocal()))
    op = _CountingMatrix(matrix)
    preconditioner = _CountingMatrix(torch.diag(matrix.diagonal().reciprocal()))
    result = PCG(maxiter=maxiter, tol=1e-6)._forward_mps(op, rhs, initial, preconditioner)
    assert result is initial
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
    # A nonzero initial guess consumes one residual evaluation before CG.
    if maxiter is not None:
        assert len(op.calls) == maxiter + 1
        assert len(preconditioner.calls) == maxiter
    else:
        torch.testing.assert_close(result.double(), torch.linalg.solve(matrix.double(), rhs.double()),
                                   rtol=2e-5, atol=2e-6)
        assert 0 < len(preconditioner.calls) < rhs.numel() * 10
    # The recurrence supplies and reuses output buffers for A and custom M.
    assert all(pointer is not None for pointer in op.calls[1:] + preconditioner.calls)
    assert len(set(op.calls[1:])) <= 1
    assert len(set(preconditioner.calls)) <= 1


def test_zero_rhs_and_exact_warm_start_do_not_iterate():
    matrix = torch.diag(torch.tensor([1., 2., 4., 8.]))
    op, preconditioner = _CountingMatrix(matrix), _CountingMatrix(torch.eye(4))
    solver = PCG()
    zero, initial = torch.zeros(4, 1), torch.ones(4, 1)
    assert solver._forward_mps(op, zero, initial, preconditioner) is zero
    assert torch.equal(initial, torch.ones_like(initial))
    assert op.calls == preconditioner.calls == []
    rhs = matrix @ initial
    assert solver._forward_mps(op, rhs, initial, preconditioner) is initial
    assert len(op.calls) == 1
    assert preconditioner.calls == []


class _NoMatrixInnerProducts(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.mm.default, torch.ops.aten.mm.out,
                    torch.ops.aten.bmm.default, torch.ops.aten.bmm.out):
            left, right = args[:2]
            assert not (left.shape[-2] == 1 and right.shape[-1] == 1), \
                "CG scalar products must use reductions, not 1xN @ Nx1"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_other_backends_still_delegate_to_pypose(device, dtype, monkeypatch):
    matrix, rhs = _system(device)
    matrix, rhs = matrix.to(dtype), rhs.to(dtype).squeeze(-1)
    calls = []

    def delegated(self, A, b, x, M):
        calls.append((A, b, M))
        return b.clone()

    monkeypatch.setattr(CG, "forward", delegated)
    result = PCG()(matrix, rhs)
    assert len(calls) == 1 and calls[0][0] is matrix
    assert calls[0][1].shape == (4, 1)
    assert isinstance(calls[0][2], DiagonalMatVec)
    torch.testing.assert_close(result, rhs)


@MPS
@pytest.mark.parametrize("form", ["half", "multiple_rhs", "batched"])
def test_other_mps_forms_still_delegate_to_pypose(form, monkeypatch):
    matrix, rhs = _system("mps")
    if form == "half":
        matrix, rhs = matrix.half(), rhs.half()
    elif form == "batched":
        matrix, rhs = matrix.expand(2, 4, 4), rhs.expand(2, 4, 1)
    else:
        rhs = rhs.expand(4, 2)
    calls = []

    def delegated(self, A, b, x, M):
        calls.append(b.shape)
        return b.clone()

    monkeypatch.setattr(CG, "forward", delegated)
    result = PCG()(matrix, rhs, M=torch.eye(4, device="mps"))
    assert calls == [rhs.shape]
    torch.testing.assert_close(result, rhs)


@MPS
@pytest.mark.parametrize("vector", [False, True])
def test_mps_dense_solve_normalizes_rhs_and_uses_reductions(vector):
    matrix, rhs = _system("mps")
    if vector:
        rhs = rhs.squeeze(-1)
    initial = torch.full_like(rhs, 0.125)
    expected = torch.linalg.solve(matrix.cpu().double(), rhs.cpu().double())
    with _NoMatrixInnerProducts():
        result = PCG(tol=1e-6)(matrix, rhs, x=initial)
    assert result.shape == rhs.shape and result.data_ptr() == initial.data_ptr()
    torch.testing.assert_close(result.cpu().double(), expected, rtol=2e-5, atol=2e-6)
