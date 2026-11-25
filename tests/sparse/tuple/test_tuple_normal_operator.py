import pytest
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import ModuleType
from unittest.mock import Mock
import torch
import torch.nn as nn
import pypose as pp

from bae.optim import LM
from bae.sparse import BSRTensor
from bae.utils.linear_operator import NormalMatVec
from bae.utils.pysolvers import PCG


DEVICES = ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS is unavailable'))]


def _cat_blocks(blocks, sizes):
    return torch.cat([
        BSRTensor(*block, size=size, is_coalesced=True)
        for block, size in zip(blocks, sizes)
    ], dim=1)


def _same_components(actual, expected):
    return all(a is b for a, b in zip(actual, expected))


def test_matrix_free_imports_without_optional_backends():
    script = '''
import importlib.abc
import sys

attempts = []
class BlockCuDSS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bae.sparse.solve":
            attempts.append(fullname)
            raise ImportError("CuDSS import intercepted by test")

sys.modules["warp"] = None  # Simulate an installation without Warp.
sys.meta_path.insert(0, BlockCuDSS())
from bae.optim import LM
from bae.utils.pysolvers import PCG, CuDSS
PCG()
assert attempts == ["bae.sparse.solve"], attempts
assert "bae.sparse.spgemm" not in sys.modules
assert "bae.sparse.conversion" not in sys.modules
if sys.platform == "darwin":
    assert "triton" not in sys.modules
try:
    CuDSS()
except ImportError as error:
    assert "Use `PCG(...)` instead" in str(error)
    assert "intercepted by test" in str(error.__cause__)
else:
    raise AssertionError("CuDSS should fail when its extension is unavailable")
assert attempts == ["bae.sparse.solve"], attempts
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_lm_constructor_preserves_explicit_matrix_free_setting(monkeypatch):
    spgemm = ModuleType('bae.sparse.spgemm')
    spgemm.CuSparse = Mock()
    monkeypatch.setitem(sys.modules, 'bae.sparse.spgemm', spgemm)
    model = nn.Linear(1, 1)
    optimizer = LM(model, solver=PCG())
    assert optimizer.matrix_free_normal is False
    assert optimizer.tuple_mode is False
    assert optimizer.mm is spgemm.CuSparse.return_value
    spgemm.CuSparse.assert_called_once_with()

    # Matrix-free construction must work when the CUDA extension cannot import.
    monkeypatch.setitem(sys.modules, 'bae.sparse.spgemm', None)
    optimizer = LM(model, solver=PCG(), matrix_free_normal=True)
    assert optimizer.matrix_free_normal is True
    assert optimizer.tuple_mode is False
    spgemm.CuSparse.assert_called_once_with()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS is unavailable')
@pytest.mark.parametrize('requires_grad', [True, False])
def test_lm_constructor_checks_all_mps_parameters(requires_grad):
    model = nn.Module()
    model.first = nn.Parameter(torch.ones(1))
    model.second = nn.Parameter(torch.ones(1, device='mps'), requires_grad=requires_grad)
    with pytest.raises(ValueError, match='matrix_free_normal=True'):
        LM(model, solver=PCG())
    optimizer = LM(model, solver=PCG(), matrix_free_normal=True)
    assert optimizer.matrix_free_normal is True
    assert optimizer.tuple_mode is True


def _block_components(block_width, device):
    # One empty block row, repeated column references across rows, and an
    # unobserved trailing block column. Different widths cannot be concatenated
    # into a single BSR matrix without changing its blocking.
    crow = torch.tensor([0, 2, 2, 3], dtype=torch.int32)
    col = torch.tensor([0, 1, 0], dtype=torch.int32)
    values = torch.arange(1, 1 + 3 * 2 * block_width, dtype=torch.float32)
    values = values.reshape(3, 2, block_width) / 10
    dense = torch.zeros(6, 3 * block_width)
    for row in range(3):
        for entry in range(crow[row], crow[row + 1]):
            column = col[entry] * block_width
            dense[2 * row:2 * row + 2, column:column + block_width] += values[entry]
    return tuple(t.to(device) for t in (crow, col, values)), dense


@pytest.mark.parametrize('device', DEVICES)
def test_normal_diagonal_replacement_and_repeated_damping(device):
    block, dense = _block_components(2, device)
    jacobian = _cat_blocks([block], [(6, 6)])
    raw = dense.square().sum(0)
    replacement = raw.clamp(min=0.25, max=2.0)
    op = NormalMatVec(jacobian, diag=replacement.to(device))
    x = torch.arange(6, dtype=torch.float32)
    scale = 1.0
    for damping in (0.0, 0.1, 0.7):
        scale *= 1.0 + damping
        op.set_damping(scale - 1.0)
        expected = dense.mT @ dense
        expected.diagonal().copy_(scale * replacement)
        out = torch.empty(6, device=device)
        assert torch.matmul(op, x.to(device), out=out) is out
        torch.testing.assert_close(out.cpu(), expected @ x, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(op.diagonal().cpu(), expected.diagonal())


@pytest.mark.parametrize('device', DEVICES)
def test_pcg_tuple_normal_without_sparse_constructors(device, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('tuple solver must not construct native sparse tensors')

    for name in ('sparse_coo_tensor', 'sparse_csr_tensor', 'sparse_csc_tensor',
                 'sparse_bsr_tensor', 'sparse_bsc_tensor'):
        monkeypatch.setattr(torch, name, forbidden)
    block, dense = _block_components(2, device)
    jacobian = _cat_blocks([block], [(6, 6)])
    diagonal = dense.square().sum(0).clamp(min=0.1)
    op = NormalMatVec(jacobian, damping=0.5, diag=diagonal.to(device))
    expected_matrix = dense.mT @ dense
    expected_matrix.diagonal().copy_(diagonal * 1.5)
    rhs = torch.arange(1, 7, dtype=torch.float32)
    solver = PCG(tol=1e-5, maxiter=100)
    for vector in (True, False):
        b = rhs if vector else rhs[:, None]
        initial = torch.full_like(b, 0.1, device=device)
        actual = solver(op, b.to(device), x=initial)
        expected = torch.linalg.solve(expected_matrix, b)
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
        assert actual.shape == b.shape
    zero = torch.zeros(6, device=device)
    assert torch.equal(solver(op, zero), zero)
    torch.testing.assert_close(op.diagonal().cpu(), diagonal * 1.5)


@pytest.mark.parametrize('device', DEVICES)
def test_tuple_operator_reuses_outputs_and_prepares_transposes_once(device, monkeypatch):
    from bae.sparse import tuple_ops

    a, adense = _block_components(3, device)
    b, bdense = _block_components(2, device)
    dense = torch.cat([adense, bdense], dim=1)
    op = _cat_blocks([a, b], [(6, 9), (6, 6)])
    normal = NormalMatVec(op, damping=0.25)
    original = tuple_ops.bsr_transposed
    prepared = []

    def track_transpose(block, **kwargs):
        prepared.append(block)
        return original(block, **kwargs)

    monkeypatch.setattr(tuple_ops, 'bsr_transposed', track_transpose)
    x = torch.arange(15, dtype=torch.float32, device=device)[:, None] / 10
    residual = torch.arange(6, dtype=torch.float32, device=device)[:, None] / 5
    forward_out = torch.empty(6, 1, device=device)
    transpose_out = torch.empty(15, 1, device=device)
    normal_out = torch.empty_like(x)
    pointers = [out.data_ptr() for out in (forward_out, transpose_out, normal_out)]
    with torch.no_grad():
        for scale in (1., 2., 0.5):
            current_x = x * scale
            assert torch.matmul(op, current_x, out=forward_out) is forward_out
            assert torch.matmul(op.mT, residual, out=transpose_out) is transpose_out
            assert torch.matmul(normal, current_x, out=normal_out) is normal_out
            assert [out.data_ptr() for out in (forward_out, transpose_out, normal_out)] == pointers
            expected_normal = dense.mT @ (dense @ current_x.cpu())
            expected_normal += 0.25 * dense.square().sum(0)[:, None] * current_x.cpu()
            torch.testing.assert_close(forward_out.cpu(), dense @ current_x.cpu(), rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(transpose_out.cpu(), dense.mT @ residual.cpu(), rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(normal_out.cpu(), expected_normal, rtol=2e-5, atol=2e-5)
    assert len(prepared) == 2
    assert _same_components(prepared[0], a) and _same_components(prepared[1], b)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
@pytest.mark.parametrize('camera_count,block_shape,expected_tile', [
    (1023, (2, 9), 0),
    (1024, (2, 9), -1),
    (1024, (3, 9), 0),
    (1024, (2, 8), 0),
])
def test_cached_camera_transpose_dispatch(camera_count, block_shape, expected_tile, monkeypatch):
    from bae.sparse import tuple_ops

    br, bc = block_shape
    values = torch.arange(1., 1 + br * bc).reshape(1, br, bc)
    matrix = BSRTensor(torch.tensor([0, 1], device='mps'),
                       torch.tensor([0], device='mps'), values.to('mps'),
                       size=(br, camera_count * bc), is_coalesced=True)
    tiles = []
    original = tuple_ops.bsr_mv

    def record(*args, **kwargs):
        tiles.append(kwargs['tile_size'])
        return original(*args, **kwargs)

    monkeypatch.setattr(tuple_ops, 'bsr_mv', record)
    result = matrix.mT @ torch.ones(br, device='mps')
    expected = torch.zeros(camera_count, bc)
    expected[0] = values[0].sum(0)
    torch.testing.assert_close(result.cpu(), expected.flatten())
    assert tiles == [expected_tile]


def test_normal_matvec_concurrent_calls_keep_intermediates_independent(monkeypatch):
    block = (
        torch.arange(5, dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        torch.ones(4, 1, 1),
    )
    jacobian = _cat_blocks([block], [(4, 4)])
    normal = NormalMatVec(jacobian)
    # Warm all caches before coordinating simultaneous calls. With a shared
    # intermediate, both calls would read the last-written Jx after the barrier.
    with torch.no_grad():
        normal @ torch.zeros(4)
    completed_forward = Barrier(2)
    matvec = jacobian.matvec

    def coordinated_forward(x, out=None):
        result = matvec(x, out=out)
        completed_forward.wait(timeout=10)
        return result

    monkeypatch.setattr(jacobian, 'matvec', coordinated_forward)

    def evaluate(x):
        with torch.no_grad():
            return normal @ x

    inputs = (torch.arange(1., 5.), torch.arange(10., 14.))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(evaluate, x) for x in inputs]
        outputs = [future.result(timeout=15) for future in futures]
    for output, expected in zip(outputs, inputs):
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize('device', DEVICES)
def test_normal_output_can_alias_input_with_nonzero_damping(device):
    block, dense = _block_components(2, device)
    normal = NormalMatVec(_cat_blocks([block], [(6, 6)]), damping=0.5)
    x = torch.arange(6, dtype=torch.float32, device=device)
    expected = dense.mT @ (dense @ x.cpu()) + 0.5 * dense.square().sum(0) * x.cpu()
    with torch.no_grad():
        assert torch.matmul(normal, x, out=x) is x
    torch.testing.assert_close(x.cpu(), expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('unused_shape', [(2, 1), (), (0,)], ids=['matrix', 'scalar', 'empty_vector'])
def test_lm_rejects_disconnected_parameters_before_solving(device, unused_shape):
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.unused = pp.Parameter(torch.full(unused_shape, 3., device=device), sjac=True)
            self.variable = pp.Parameter(torch.zeros(2, 1, device=device), sjac=True)

        def forward(self, target, index):
            return self.variable[index] * 2 - target

    model = Residual()
    solver = Mock()
    optimizer = LM(model, solver=solver, matrix_free_normal=True)
    data = {'target': torch.tensor([[2.], [4.]], device=device),
            'index': torch.tensor([0, 1], device=device)}
    with pytest.raises(ValueError, match='All optimized parameters must contribute to the residual'):
        optimizer.step(data)
    solver.assert_not_called()
    torch.testing.assert_close(model.variable.detach(), torch.zeros(2, 1, device=device))


@pytest.mark.parametrize('device', DEVICES)
def test_synthetic_ba_lm_reduces_reprojection_error(device):
    from ba_example import Residual, project

    torch.manual_seed(5)
    cameras = torch.zeros(3, 10)
    cameras[:, 6] = 1
    cameras[:, 7] = 300
    cameras[:, 0] = torch.tensor([-0.2, 0., 0.2])
    points = torch.randn(8, 3) * 0.3
    points[:, 2] -= 3
    cidx = torch.arange(3).repeat_interleave(8)
    pidx = torch.arange(8).repeat(3)
    observes = project(points[pidx], cameras[cidx]).detach()
    cameras[:, :3] += torch.randn(3, 3) * 0.01
    points += torch.randn_like(points) * 0.01
    model = Residual(cameras.to(device), points.to(device))
    data = {'observes': observes.to(device), 'cidx': cidx.to(device), 'pidx': pidx.to(device)}
    optimizer = LM(model, solver=PCG(tol=1e-5, maxiter=150), reject=10, matrix_free_normal=True,
                   strategy=pp.optim.strategy.TrustRegion(radius=100))
    initial = optimizer.model.loss(data, None).item()
    for _ in range(3):
        loss = optimizer.step(data)
    assert torch.isfinite(loss)
    assert loss.item() < initial * 0.1


@pytest.mark.parametrize('device', DEVICES)
def test_lm_rejected_step_keeps_cumulative_damping(device, monkeypatch):
    from bae.optim import optimizer as optimizer_module
    from pypose.autograd.function import psjac

    trace_name = 'jacobian_components' if device == 'mps' else 'jacobian'
    other_name = 'jacobian' if device == 'mps' else 'jacobian_components'
    trace = Mock(wraps=getattr(optimizer_module, trace_name))
    monkeypatch.setattr(optimizer_module, trace_name, trace)
    monkeypatch.setattr(optimizer_module, other_name, Mock(side_effect=AssertionError('Wrong Jacobian path')))
    if device == 'mps':
        for name in ('sparse_coo_tensor', 'sparse_csr_tensor', 'sparse_csc_tensor',
                     'sparse_bsr_tensor', 'sparse_bsc_tensor'):
            monkeypatch.setattr(torch, name, Mock(side_effect=AssertionError('Native sparse constructor')))

    @psjac
    def square(value):
        return value.square()

    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.variable = pp.Parameter(torch.ones(1, 1, device=device), sjac=True)

        def forward(self, target, index):
            return square(self.variable[index]) - target

    model = Residual()
    diagonals, parameter_states = [], []

    class TrialSolver(nn.Module):
        def forward(self, operator, rhs):
            parameter_states.append(torch.Tensor(model.variable).clone())
            diagonals.append(operator.diagonal().clone())
            # The first step predicts improvement but increases nonlinear loss.
            # Both increments are exactly reversible in float32.
            return torch.full_like(rhs, (0.75, 0.25)[len(diagonals) - 1])

    optimizer = LM(model, solver=TrialSolver(), reject=2, matrix_free_normal=True,
                   strategy=pp.optim.strategy.TrustRegion(radius=10))
    loss = optimizer.step({'target': torch.tensor([[2.]], device=device),
                           'index': torch.tensor([0], device=device)})

    trace.assert_called_once()
    assert optimizer.reject_count == 1
    assert len(diagonals) == 2
    for state in parameter_states:
        torch.testing.assert_close(state.cpu(), torch.ones(1, 1))
    torch.testing.assert_close(model.variable.detach().cpu(), torch.tensor([[1.25]]))
    torch.testing.assert_close(loss.cpu(), torch.tensor(0.19140625))
    torch.testing.assert_close(diagonals[0].cpu(), torch.tensor([4 * 1.1]))
    torch.testing.assert_close(diagonals[1].cpu(), torch.tensor([4 * 1.1 * 1.2]))
