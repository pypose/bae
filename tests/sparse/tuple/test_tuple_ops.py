import pytest
import torch

from bae.sparse.tuple_ops import _tile_dispatch, bsr_diagonal, bsr_mv, bsr_transposed


@pytest.fixture(params=["cpu", "mps", "cuda"])
def device(request):
    name = request.param
    if name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return name


def matrix(block, index_dtype, device, empty=False):
    br, bc = block
    crow = torch.tensor([0, 2, 2, 3, 3], dtype=index_dtype)
    col = torch.tensor([0, 2, 2], dtype=index_dtype)
    generator = torch.Generator().manual_seed(9)
    values = torch.randn(3, br, bc, generator=generator)
    if empty:
        crow.zero_()
        col, values = col[:0], values[:0]
    dense = torch.zeros(4 * br, 5 * bc)
    for row in range(4):
        for k in range(int(crow[row]), int(crow[row + 1])):
            column = int(col[k])
            dense[row * br:(row + 1) * br, column * bc:(column + 1) * bc] += values[k]
    return tuple(t.to(device) for t in (crow, col, values)), dense


@pytest.mark.parametrize("block,index_dtype,empty", [
    ((1, 1), torch.int64, False),
    ((2, 9), torch.int32, False),
    ((2, 3), torch.int64, False),
    ((5, 11), torch.int32, False),
    ((3, 2), torch.int64, True),
])
def test_tuple_products_match_dense(device, block, index_dtype, empty, monkeypatch):
    A, dense = matrix(block, index_dtype, device, empty)
    size = tuple(dense.shape)
    originals = tuple(t.clone() for t in A)

    def no_sparse(*args, **kwargs):
        raise AssertionError("Tuple operations must not construct native sparse tensors")

    for constructor in ("sparse_bsr_tensor", "sparse_csr_tensor", "sparse_coo_tensor"):
        monkeypatch.setattr(torch, constructor, no_sparse)
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(size[1], generator=generator)
    r = torch.randn(size[0], generator=generator)
    # Strided inputs ensure dispatch normalizes storage correctly.
    x_storage = torch.stack((x, x), dim=-1).to(device)
    x_device = x_storage[:, 0]
    forward = bsr_mv(A, x_device, size=size)
    backward = bsr_mv(A, r.to(device)[:, None], size=size, transpose=True)
    torch.testing.assert_close(forward.cpu(), dense @ x, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(backward.cpu(), (dense.T @ r)[:, None], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        bsr_diagonal(A, size=size).cpu(), dense.square().sum(0), rtol=1e-4, atol=1e-5,
    )
    torch.testing.assert_close(
        (forward * r.to(device)).sum(), (x_device * backward[:, 0]).sum(),
        rtol=1e-4, atol=1e-5,
    )
    for actual, original in zip(A, originals):
        torch.testing.assert_close(actual, original)
    torch.testing.assert_close(x_storage[:, 0].cpu(), x)


def test_tuple_product_duplicate_coordinates(device):
    # Matvec accepts duplicate blocks, though diagonal extraction requires
    # canonical input (the Jacobian materializer coalesces first).
    A = (
        torch.tensor([0, 2], device=device),
        torch.tensor([0, 0], device=device),
        torch.tensor([[[2.0, 3.0]], [[-1.0, 4.0]]], device=device),
    )
    x = torch.tensor([5.0, 2.0], device=device)
    torch.testing.assert_close(bsr_mv(A, x, size=(1, 2)), torch.tensor([19.0], device=device))
    torch.testing.assert_close(
        bsr_mv(A, torch.tensor([2.0], device=device), size=(1, 2), transpose=True),
        torch.tensor([2.0, 14.0], device=device),
    )


def test_tuple_validation():
    A, dense = matrix((2, 3), torch.int64, "cpu")
    with pytest.raises(ValueError):
        bsr_mv(A, torch.zeros(dense.shape[1], 2), size=tuple(dense.shape))
    with pytest.raises(ValueError, match="Expected x"):
        bsr_mv(A, torch.zeros(1), size=tuple(dense.shape))
    with pytest.raises(ValueError, match="divisible"):
        bsr_mv(A, torch.zeros(1), size=(7, 15))
    with pytest.raises(TypeError, match="same int32 or int64"):
        bsr_mv((A[0].int(), A[1], A[2]), torch.zeros(15), size=(8, 15))


# The affine, aliasing, and zero-coefficient contracts below follow
# NVIDIA Warp release-1.19 warp/_src/sparse.py:bsr_mv.
_TILES = [-1, 33, 64]


@pytest.mark.parametrize("tile_size", _TILES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("transpose", [False, True])
def test_warp_affine_product_updates_and_returns_y(device, tile_size, index_dtype, transpose):
    A, dense = matrix((2, 3), index_dtype, device)
    operator = dense.T if transpose else dense
    x_cpu = torch.linspace(-1.5, 2., operator.shape[1])
    y_cpu = torch.linspace(3., -2., operator.shape[0])
    x = x_cpu.to(device)
    y = y_cpu.to(device).clone()
    if transpose:
        x, y = x[:, None], y[:, None]
    expected = -1.75 * (operator @ x_cpu) + 0.25 * y_cpu
    originals = tuple(part.clone() for part in A)

    actual = bsr_mv(A, x, y, -1.75, 0.25, transpose, None, tile_size, size=tuple(dense.shape))

    assert actual is y
    torch.testing.assert_close(actual.reshape(-1).cpu(), expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x.reshape(-1).cpu(), x_cpu)
    for part, original in zip(A, originals):
        torch.testing.assert_close(part, original)


@pytest.mark.parametrize("transpose", [False, True])
def test_warp_missing_y_ignores_beta(device, transpose):
    A, dense = matrix((2, 3), torch.int32, device)
    operator = dense.T if transpose else dense
    x = torch.linspace(-1., 1., operator.shape[1], device=device)
    actual = bsr_mv(A, x, alpha=0.5, beta=float("nan"), transpose=transpose,
                    size=tuple(dense.shape))
    torch.testing.assert_close(actual.cpu(), 0.5 * (operator @ x.cpu()), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("tile_size", [-1, 64])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("alpha,beta", [(0., -2.), (0., 0.), (1.5, 0.)])
def test_warp_zero_coefficients_do_not_read_nan_operands(device, tile_size, transpose, alpha, beta):
    A, dense = matrix((2, 3), torch.int64, device)
    operator = dense.T if transpose else dense
    x = torch.arange(operator.shape[1], dtype=torch.float32, device=device)
    y = torch.arange(operator.shape[0], dtype=torch.float32, device=device)
    expected = (alpha * (operator @ x.cpu())) + beta * y.cpu()
    if alpha == 0:
        A[2].fill_(float("nan"))
        x.fill_(float("nan"))
    if beta == 0:
        y.fill_(float("nan"))

    actual = bsr_mv(A, x, y=y, alpha=alpha, beta=beta, transpose=transpose,
                    tile_size=tile_size, size=tuple(dense.shape))

    assert actual is y
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("zero_coefficient", ["alpha", "beta"])
def test_warp_coefficients_are_cast_before_zero_checks(device, transpose, zero_coefficient):
    A, dense = matrix((2, 3), torch.int32, device)
    operator = dense.T if transpose else dense
    x = torch.ones(operator.shape[1], device=device)
    y = torch.ones(operator.shape[0], device=device)
    alpha, beta = 1., 2.
    if zero_coefficient == "alpha":
        alpha = 1e-50  # Nonzero Python float, zero in A's float32 scalar type.
        x.fill_(float("nan"))
        A[2].fill_(float("nan"))
        expected = torch.full((operator.shape[0],), 2.)
    else:
        beta = 1e-50
        y.fill_(float("nan"))
        expected = operator @ x.cpu()
    actual = bsr_mv(A, x, y=y, alpha=alpha, beta=beta, transpose=transpose,
                    size=tuple(dense.shape))
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


def _square_with_rectangular_blocks(device, index_dtype=torch.int64):
    crow = torch.tensor([0, 2, 3, 4], dtype=index_dtype)
    columns = torch.tensor([0, 1, 1, 0], dtype=index_dtype)
    values = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) / 10 - 1
    dense = torch.zeros(6, 6)
    for row in range(3):
        for block in range(int(crow[row]), int(crow[row + 1])):
            column = int(columns[block])
            dense[row * 2:(row + 1) * 2, column * 3:(column + 1) * 3] = values[block]
    return tuple(part.to(device) for part in (crow, columns, values)), dense


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("use_work_buffer", [False, True])
def test_warp_x_y_alias_uses_saved_input(device, transpose, use_work_buffer):
    A, dense = _square_with_rectangular_blocks(device)
    original = torch.linspace(-1., 1., 6)
    xy = original.to(device).clone()
    work_buffer = torch.full((8,), 123., device=device) if use_work_buffer else None
    expected = 1.5 * ((dense.T if transpose else dense) @ original) - 0.5 * original

    actual = bsr_mv(A, xy, xy, alpha=1.5, beta=-0.5, transpose=transpose,
                    work_buffer=work_buffer, size=(6, 6))

    assert actual is xy
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
    if use_work_buffer:
        torch.testing.assert_close(work_buffer[:6].cpu(), original)
        torch.testing.assert_close(work_buffer[6:].cpu(), torch.full((2,), 123.))


@pytest.mark.parametrize("transpose", [False, True])
def test_warp_work_buffer_is_unused_without_alias(device, transpose):
    A, dense = _square_with_rectangular_blocks(device)
    x, y = torch.ones(6, device=device), torch.empty(6, device=device)
    # Deliberately insufficient and of the wrong type: unused buffers are ignored.
    work_buffer = torch.tensor([7], dtype=torch.int32, device=device)
    actual = bsr_mv(A, x, y=y, transpose=transpose, work_buffer=work_buffer, size=(6, 6))
    torch.testing.assert_close(actual.cpu(), (dense.T if transpose else dense) @ torch.ones(6))
    torch.testing.assert_close(work_buffer.cpu(), torch.tensor([7], dtype=torch.int32))


@pytest.mark.parametrize("transpose", [False, True])
def test_warp_block_shaped_vectors(device, transpose):
    A, dense = matrix((2, 3), torch.int64, device)
    operator = dense.T if transpose else dense
    input_shape = (4, 2) if transpose else (5, 3)
    output_shape = (5, 3) if transpose else (4, 2)
    x = torch.arange(operator.shape[1], dtype=torch.float32, device=device).reshape(input_shape)
    expected = (operator @ x.reshape(-1).cpu()).reshape(output_shape)

    allocated = bsr_mv(A, x, transpose=transpose, size=tuple(dense.shape))
    assert allocated.shape == output_shape
    torch.testing.assert_close(allocated.cpu(), expected, rtol=1e-4, atol=1e-5)
    y = torch.ones(output_shape, device=device)
    actual = bsr_mv(A, x, y=y, beta=2., transpose=transpose, size=tuple(dense.shape))
    assert actual is y
    torch.testing.assert_close(actual.cpu(), expected + 2., rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("transpose", [False, True])
def test_warp_empty_matrix_only_scales_y(device, transpose):
    A, dense = matrix((2, 3), torch.int32, device, empty=True)
    operator = dense.T if transpose else dense
    x = torch.full((operator.shape[1],), float("nan"), device=device)
    y = torch.arange(operator.shape[0], dtype=torch.float32, device=device)
    expected = -0.25 * y.cpu()
    actual = bsr_mv(A, x, y=y, alpha=3., beta=-0.25, transpose=transpose,
                    size=tuple(dense.shape))
    assert actual is y
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("tile_size", [-1, 64])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("empty,alpha", [(False, float("nan")), (True, float("inf"))])
def test_warp_empty_rows_with_nonfinite_alpha(device, tile_size, transpose, empty, alpha):
    A, dense = matrix((2, 3), torch.int32, device, empty=empty)
    operator = dense.T if transpose else dense
    x = torch.ones(operator.shape[1], device=device)
    y = torch.arange(operator.shape[0], dtype=torch.float32, device=device)
    empty_outputs = (operator == 0).all(dim=1)
    if transpose:
        # Warp never scatter-adds into unreferenced columns, so alpha is not
        # applied to those outputs even when it is NaN or infinity.
        expected = 2 * y.cpu()[empty_outputs]
    else:
        # Forward scalar/tiled kernels scale the zero row sum by alpha.
        expected = torch.full((int(empty_outputs.sum()),), float("nan"))
    actual = bsr_mv(A, x, y=y, alpha=alpha, beta=2., transpose=transpose,
                    tile_size=tile_size, size=tuple(dense.shape))
    torch.testing.assert_close(actual.cpu()[empty_outputs], expected, equal_nan=True)


@pytest.mark.parametrize("size", [(0, 6), (4, 0), (0, 0)])
@pytest.mark.parametrize("transpose", [False, True])
def test_warp_zero_dimensions(device, size, transpose):
    A = (
        torch.zeros(size[0] // 2 + 1, dtype=torch.int64, device=device),
        torch.empty(0, dtype=torch.int64, device=device),
        torch.empty(0, 2, 3, device=device),
    )
    input_size, output_size = size if transpose else size[::-1]
    x = torch.full((input_size,), float("nan"), device=device)
    y = torch.arange(output_size, dtype=torch.float32, device=device)
    expected = 2 * y.cpu()
    actual = bsr_mv(A, x, y=y, beta=2., transpose=transpose, size=size)
    assert actual is y
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("tile_size", _TILES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_warp_long_rows_and_empty_rows(device, tile_size, index_dtype):
    # 129 * 3 scalar entries per populated subrow exceeds Warp's auto-tile
    # threshold, including the empty middle row in the average.
    crow = torch.tensor([0, 129, 129, 258], dtype=index_dtype, device=device)
    columns = torch.arange(129, dtype=index_dtype, device=device).repeat(2)
    values = torch.linspace(-0.1, 0.2, 258 * 2 * 3, device=device).reshape(258, 2, 3)
    x = torch.linspace(-0.5, 0.5, 129 * 3, device=device)
    dense = torch.zeros(6, 129 * 3)
    dense[:2] = values[:129].cpu().permute(1, 0, 2).reshape(2, -1)
    dense[4:] = values[129:].cpu().permute(1, 0, 2).reshape(2, -1)
    y = torch.arange(6, dtype=torch.float32, device=device)
    expected = 0.75 * (dense @ x.cpu()) - 2 * y.cpu()
    actual = bsr_mv((crow, columns, values), x, y=y, alpha=0.75, beta=-2.,
                    tile_size=tile_size, size=tuple(dense.shape))
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=1e-5)


def _check_affine_product_with_long_mixed_and_empty_rows(device, block_shape, index_dtype, tile_size):
    block_rows, block_cols = block_shape
    crow = torch.tensor([0, 129, 130, 130, 259, 259], dtype=index_dtype)
    columns = torch.cat([torch.arange(129, dtype=index_dtype),
                         torch.tensor([7], dtype=index_dtype),
                         torch.arange(129, dtype=index_dtype)])
    generator = torch.Generator().manual_seed(19)
    values = torch.randn(259, block_rows, block_cols, generator=generator) / 10
    size = (5 * block_rows, 129 * block_cols)
    dense = _dense_from_tuple((crow, columns, values.double()), size)
    x_cpu = torch.linspace(-0.75, 1.25, size[1])
    y_cpu = torch.linspace(2., -1., size[0])
    expected = -1.75 * (dense @ x_cpu.double()) + 0.25 * y_cpu.double()
    A = tuple(part.to(device) for part in (crow, columns, values))
    x, y = x_cpu.to(device), y_cpu.to(device).clone()

    actual = bsr_mv(A, x, y=y, alpha=-1.75, beta=0.25,
                    tile_size=tile_size, size=size)

    assert actual is y
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=2e-4, atol=1e-5)
    torch.testing.assert_close(x.cpu(), x_cpu)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal kernels require MPS")
@pytest.mark.parametrize("block_rows", [2, 3])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_scalar_camera_blocks_with_long_mixed_and_empty_rows(block_rows, index_dtype):
    # br=2 exercises the camera specialization; br=3 must retain the general
    # row path. The long rows would otherwise select Warp's tiled kernel.
    _check_affine_product_with_long_mixed_and_empty_rows("mps", (block_rows, 9), index_dtype, -1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal kernels require MPS")
@pytest.mark.parametrize("block_shape,tile_size", [
    ((3, 2), -1),
    ((9, 2), -1),
    ((9, 2), 64),
    ((9, 2), 32),  # Other tile sizes keep their general block-dimension path.
    ((8, 2), 64),  # Other block heights keep the general tiled kernel.
])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_cached_transpose_block_kernels_with_long_mixed_and_empty_rows(block_shape, tile_size, index_dtype):
    # These are ordinary forward products of pretransposed BSR storage, not
    # the atomic transpose kernel. Each case includes two long rows, one short
    # row, an internal empty row, and an empty trailing row.
    _check_affine_product_with_long_mixed_and_empty_rows("mps", block_shape, index_dtype, tile_size)


def test_warp_output_and_alias_work_buffer_validation(device):
    A, _ = _square_with_rectangular_blocks(device)
    x = torch.ones(6, device=device)
    with pytest.raises(ValueError):
        bsr_mv(A, x, y=torch.zeros(5, device=device), size=(6, 6))
    with pytest.raises((TypeError, ValueError)):
        bsr_mv(A, x, y=torch.zeros(6, dtype=torch.int32, device=device), size=(6, 6))
    with pytest.raises(ValueError):
        bsr_mv(A, x, y=torch.zeros(6, 2, device=device)[:, 0], size=(6, 6))
    with pytest.raises(ValueError):
        bsr_mv(A, x, y=x, work_buffer=torch.empty(5, device=device), size=(6, 6))
    with pytest.raises((TypeError, ValueError)):
        bsr_mv(A, x, y=x, work_buffer=torch.empty(6, dtype=torch.int32, device=device), size=(6, 6))


def _dense_from_tuple(A, size):
    crow, columns, values = (part.cpu() for part in A)
    result = torch.zeros(size, dtype=values.dtype)
    br, bc = values.shape[-2:]
    for row in range(crow.numel() - 1):
        for block in range(int(crow[row]), int(crow[row + 1])):
            column = int(columns[block])
            result[row * br:(row + 1) * br, column * bc:(column + 1) * bc] += values[block]
    return result


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_bsr_transposed_preserves_full_shape_and_owns_values(device, index_dtype, empty):
    A, dense = matrix((2, 3), index_dtype, device, empty=empty)
    size = tuple(dense.shape)
    originals = tuple(part.clone() for part in A)
    transposed = bsr_transposed(A, size=size)

    assert transposed[0].shape == (6,)  # Includes the two trailing empty column blocks.
    assert transposed[2].shape[-2:] == (3, 2)
    assert transposed[0].dtype == transposed[1].dtype == index_dtype
    torch.testing.assert_close(_dense_from_tuple(transposed, size[::-1]), dense.T)
    restored = bsr_transposed(transposed, size=size[::-1])
    assert restored[0].shape == A[0].shape
    torch.testing.assert_close(_dense_from_tuple(restored, size), dense)

    residual = torch.linspace(-1., 1., size[0], device=device)
    torch.testing.assert_close(
        bsr_mv(transposed, residual, size=size[::-1]).cpu(),
        dense.T @ residual.cpu(), rtol=1e-4, atol=1e-5,
    )
    for actual, original in zip(A, originals):
        torch.testing.assert_close(actual, original)
    A[2].add_(7.)
    torch.testing.assert_close(_dense_from_tuple(transposed, size[::-1]), dense.T)


@pytest.mark.parametrize(
    "nnz,br,bc,rows,device,tile,expected",
    [
        (128, 2, 1, 2, "mps", 0, (False, 64)),
        (129, 2, 1, 2, "mps", 0, (True, 64)),
        (128, 2, 1, 2, "cuda", 0, (False, 64)),
        (129, 2, 1, 2, "cuda", 0, (True, 64)),
        (129, 2, 1, 2, "cpu", 0, (False, 64)),
        (42, 2, 3, 2, "mps", 0, (False, 64)),
        (43, 2, 3, 2, "mps", 0, (True, 64)),
        (0, 2, 3, 0, "mps", 0, (False, 64)),
        (0, 2, 3, 4, "cpu", 32, (True, 32)),
        (1000, 2, 3, 4, "mps", -7, (False, -7)),
    ],
)
def test_warp_tile_selection_threshold(nnz, br, bc, rows, device, tile, expected):
    # Equality must stay scalar: Warp uses a strict greater-than threshold.
    assert _tile_dispatch(nnz, br, bc, rows, device, tile) == expected
