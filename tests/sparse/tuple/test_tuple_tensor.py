"""Tensor dispatch for component-backed BSR matrices and mixed-width Jacobians."""

import pytest
import torch

from bae.sparse import BSRTensor


DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable")),
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable"))]


def _matrix(device, width=3, index_dtype=torch.int64, duplicates=False):
    # Internal empty rows and trailing unused column blocks must survive every
    # wrapper operation. Optional duplicates are deliberately non-adjacent.
    crow = torch.tensor([0, 3, 3, 4] if duplicates else [0, 2, 2, 3],
                        dtype=index_dtype, device=device)
    col = torch.tensor([1, 0, 1, 0] if duplicates else [0, 1, 0],
                       dtype=index_dtype, device=device)
    values = torch.arange(col.numel() * 2 * width, dtype=torch.float32,
                          device=device).reshape(-1, 2, width) / 7 - 1
    return BSRTensor(crow, col, values, size=(6, 3 * width),
                     is_coalesced=not duplicates)


def _dense_reference(matrix):
    crow, col, values = (component.detach().cpu().clone() for component in matrix.components)
    result = torch.zeros(matrix.shape, dtype=values.dtype)
    br, bc = values.shape[-2:]
    for row in range(crow.numel() - 1):
        for entry in range(int(crow[row]), int(crow[row + 1])):
            column = int(col[entry])
            result[row * br:(row + 1) * br, column * bc:(column + 1) * bc] += values[entry]
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_tensor_metadata_and_addition_coalesce_before_squaring(device, index_dtype, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("component-backed matrices must not construct native sparse tensors")

    for constructor in ("sparse_bsr_tensor", "sparse_bsc_tensor", "sparse_csr_tensor",
                        "sparse_csc_tensor", "sparse_coo_tensor"):
        monkeypatch.setattr(torch, constructor, forbidden)
    left = _matrix(device, index_dtype=index_dtype, duplicates=True)
    right = _matrix(device, index_dtype=index_dtype)
    before = [component.clone() for component in left.components + right.components]
    left_dense, right_dense = _dense_reference(left), _dense_reference(right)
    assert isinstance(left, torch.Tensor)
    assert left.shape == left.size() == (6, 9)
    assert left.ndim == 2 and left.numel() == 54
    assert left.device.type == device and left.dtype == torch.float32
    assert left.crow_indices() is left.components[0]
    assert left.col_indices() is left.components[1]
    assert left.values() is left.components[2]
    for result, expected in ((left + right, left_dense + right_dense),
                             (torch.add(left, right, alpha=-0.5), left_dense - 0.5 * right_dense)):
        assert isinstance(result, BSRTensor) and result.shape == left.shape
        assert result.values().shape[0] == 3
        torch.testing.assert_close(result.to_dense().cpu(), expected)
        torch.testing.assert_close(result.normal_diagonal().cpu(), expected.square().sum(0))
    torch.testing.assert_close(left.normal_diagonal().cpu(), left_dense.square().sum(0))
    for component, original in zip(left.components + right.components, before):
        torch.testing.assert_close(component, original)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("columns", [None, 0, 1, 3])
def test_cat_with_different_block_widths_uses_tensor_products(device, columns):
    camera, point = _matrix(device, 3), _matrix(device, 2)
    unused = BSRTensor(
        torch.zeros(4, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.int64, device=device),
        camera.values().new_empty((0, 2, 2)), size=(6, 4), is_coalesced=True,
    )
    jacobian = torch.cat([camera, unused, point], dim=-1)
    expected = torch.cat([_dense_reference(camera), torch.zeros(6, 4), _dense_reference(point)], dim=1)
    assert isinstance(jacobian, torch.Tensor) and jacobian.shape == (6, 19)
    for actual, dense in ((jacobian, expected), (jacobian.mT, expected.mT),
                          (torch.cat([camera, camera], dim=0), torch.cat([_dense_reference(camera)] * 2))):
        shape = (dense.shape[1],) if columns is None else (dense.shape[1], columns)
        vector = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape) / 13
        out = torch.empty((dense.shape[0], *shape[1:]), device=device)
        assert torch.matmul(actual, vector.to(device), out=out) is out
        torch.testing.assert_close(out.cpu(), dense @ vector, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close((actual @ vector.to(device)).cpu(), dense @ vector,
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual.to_dense().cpu(), dense)
        torch.testing.assert_close(actual.normal_diagonal().cpu(), dense.square().sum(0))
    torch.testing.assert_close(jacobian.T.to_dense().cpu(), expected.T)
    torch.testing.assert_close(jacobian.transpose(0, 1).to_dense().cpu(), expected.T)
    for _ in range(2):
        torch.testing.assert_close(torch.transpose(jacobian, 0, 1).to_dense().cpu(), expected.T)
        torch.testing.assert_close(torch.t(jacobian).to_dense().cpu(), expected.T)


@pytest.mark.parametrize("device", DEVICES)
def test_transpose_cache_tracks_value_and_index_mutation(device):
    matrix = _matrix(device)
    transpose = matrix.mT
    residual = torch.linspace(-1, 1, 6, device=device)
    for update in (lambda: None, lambda: matrix.values().mul_(2),
                   lambda: matrix.col_indices().zero_()):
        update()
        dense = _dense_reference(matrix)
        torch.testing.assert_close((transpose @ residual).cpu(), dense.T @ residual.cpu(),
                                   rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(matrix.normal_diagonal().cpu(), dense.square().sum(0))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("component", ["values", "indices"])
def test_detached_alias_mutations_invalidate_transpose_and_coalescing(device, component):
    matrix = _matrix(device)
    transpose = matrix.mT
    residual = torch.linspace(-1, 1, 6, device=device)
    # Populate the transpose before creating the detached alias. Mutations via
    # that alias must invalidate both stored values and canonical-index state.
    transpose @ residual
    alias = matrix.detach()
    if component == "values":
        alias.values().add_(1)
    else:
        alias.col_indices().zero_()
    dense = _dense_reference(matrix)
    for view in (matrix, alias):
        torch.testing.assert_close(view.normal_diagonal().cpu(), dense.square().sum(0))
    torch.testing.assert_close((transpose @ residual).cpu(), dense.T @ residual.cpu(),
                               rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_copy_detach_and_transfer_preserve_sparse_storage(device):
    matrix = _matrix(device, index_dtype=torch.int32)
    for original in (matrix, torch.cat([matrix, matrix], dim=1)):
        dense = original.to_dense().cpu()
        copied, detached = original.clone(), original.detach()
        assert isinstance(copied, torch.Tensor) and isinstance(detached, torch.Tensor)
        torch.testing.assert_close(copied.to_dense().cpu(), dense)
        torch.testing.assert_close(detached.to_dense().cpu(), dense)
        assert not detached.requires_grad
        # Native MPS tensors require transfer before a float64 conversion.
        converted = original.to(device="cpu").to(dtype=torch.float64)
        assert converted.dtype == torch.float64 and converted.device.type == "cpu"
        torch.testing.assert_close(converted.to_dense(), dense.double())
    detached = matrix.detach()
    clone = matrix.clone()
    assert all(a.data_ptr() == b.data_ptr() for a, b in zip(detached.components, matrix.components))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(clone.components, matrix.components))
    converted = matrix.to(device="cpu").to(dtype=torch.float64)
    assert converted.crow_indices().dtype == converted.col_indices().dtype == torch.int32
    assert isinstance(matrix.data, BSRTensor)


@pytest.mark.parametrize("device", DEVICES)
def test_matmul_output_can_alias_input(device):
    matrix = _matrix(device, width=2)
    dense = _dense_reference(matrix)
    for actual, reference in ((matrix, dense), (matrix.mT, dense.T),
                              (torch.cat([matrix], dim=1), dense)):
        vector = torch.linspace(-1, 1, 6, device=device)
        expected = reference @ vector.cpu()
        assert torch.matmul(actual, vector, out=vector) is vector
        torch.testing.assert_close(vector.cpu(), expected, rtol=2e-5, atol=2e-5)


def test_unsupported_tensor_operations_fail_without_densifying():
    matrix = _matrix("cpu")
    with pytest.raises((NotImplementedError, TypeError)):
        torch.sin(matrix)
    with pytest.raises((ValueError, NotImplementedError, TypeError)):
        matrix + torch.ones(matrix.shape)
    with pytest.raises((ValueError, RuntimeError)):
        torch.cat([matrix, _matrix("cpu", width=2)], dim=0)
