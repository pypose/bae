"""BSR operations on ordinary tensor tuples, without native sparse tensors."""

from functools import lru_cache
from importlib.resources import files
from ctypes import c_float
import math

import torch


def _validate(A, size):
    if not isinstance(A, tuple) or len(A) != 3:
        raise TypeError("A must be a (crow_indices, col_indices, values) tuple")
    crow, col, values = A
    if not all(isinstance(t, torch.Tensor) for t in A):
        raise TypeError("All BSR components must be tensors")
    if crow.ndim != 1 or col.ndim != 1 or values.ndim != 3:
        raise ValueError("Expected 1-D crow/col and values shaped (nnz, block_rows, block_cols)")
    if crow.dtype not in (torch.int32, torch.int64) or col.dtype != crow.dtype:
        raise TypeError("crow and col must have the same int32 or int64 dtype")
    if crow.device != values.device or col.device != values.device:
        raise ValueError("All BSR components must reside on the same device")
    if len(size) != 2 or any(not isinstance(d, int) or d < 0 for d in size):
        raise ValueError("size must contain two nonnegative integer dimensions")
    br, bc = values.shape[-2:]
    if br == 0 or bc == 0 or size[0] % br or size[1] % bc:
        raise ValueError("Matrix dimensions must be divisible by positive block dimensions")
    if crow.numel() != size[0] // br + 1 or col.numel() != values.shape[0]:
        raise ValueError("BSR component shapes do not match size")
    if values.dtype not in (torch.float32, torch.float64):
        raise TypeError("Tuple BSR operations support float32 and float64 values")
    if values.device.type == "mps" and values.dtype != torch.float32:
        raise TypeError("MPS tuple BSR operations require float32 values")
    return crow, col, values


@lru_cache(maxsize=1)
def _metal_library():
    if not hasattr(torch.mps, "load_metallib"):
        raise RuntimeError("The MPS backend requires PyTorch 2.13 or newer")
    resource = files("bae.sparse").joinpath("bsr_mps.metallib")
    if not resource.is_file():
        raise RuntimeError(
            "The BAE Metal library is missing. Install the Xcode Metal Toolchain, "
            "then rebuild BAE with `python -m pip install --no-build-isolation -e .`."
        )
    # Bytes also work for packages accessed through an importlib resource loader.
    return torch.mps.load_metallib(resource.read_bytes())


_SPECIALIZED_BLOCK_DIMS = (1, 2, 3, 4, 6, 7, 9, 10, 12, 16)
_SPECIALIZED_TILE_SIZES = (32, 64, 128, 256)


def _tile_dispatch(nnz, br, bc, scalar_rows, device_type, tile_size):
    """Warp 1.19's exact long-row heuristic, with Metal as a GPU target."""
    if not isinstance(tile_size, int):
        raise TypeError("tile_size must be an integer")
    if tile_size > 0:
        return True, tile_size
    if tile_size < 0:
        return False, tile_size
    tile_size = 64
    return (
        device_type in ("cuda", "mps")
        and nnz * br * bc > 2 * tile_size * scalar_rows,
        tile_size,
    )


def _scale_output(y, beta):
    # Warp's transpose path zeroes before accumulation, skips beta==1, and
    # launches a separate scale kernel for any other beta.
    if beta == 0.0:
        y.zero_()
    elif beta != 1.0:
        if y.device.type == "mps" and y.numel():
            kernel = _metal_library().bsr_scale
            kernel(beta, y, threads=y.numel())
        else:
            y.mul_(beta)


def _check_vector(vector, length, values, name):
    if vector.ndim not in (1, 2):
        raise ValueError(f"{name} must be a vector, single-column tensor, or 2-D block-vector array")
    if vector.numel() != length:
        raise ValueError(f"Expected {name} with {length} scalar elements, got {vector.numel()}")
    if vector.device != values.device or vector.dtype != values.dtype:
        raise ValueError(f"{name} and BSR values must have the same device and dtype")


def bsr_mv(A, x, y=None, alpha=1.0, beta=0.0, transpose=False,
           work_buffer=None, tile_size=0, *, size):
    """Compute ``y := alpha * A @ x + beta * y`` following Warp 1.19.

    ``size`` is the full scalar shape, including unused column blocks. ``A``
    must describe a valid BSR matrix. Transpose uses ``A.T`` and atomic sums,
    whose floating-point summation order is nondeterministic on MPS.

    A supplied contiguous ``y`` is mutated and returned. Otherwise an output
    is allocated and beta is ignored. Vectors, single-column tensors, and 2-D
    block-vector arrays are accepted. Strided ``x`` is copied as a PyTorch
    convenience; output/work-buffer arrays must be contiguous. Aliasing x/y
    uses temporary storage (or the supplied work_buffer).

    Positive tile_size forces lane-strided tiled reduction, negative disables
    it, and zero uses Warp's default 64-lane GPU heuristic. Transpose ignores
    tile_size. alpha==0 does not read matrix/vector data; beta==0 does not read
    old output data. The source port is based on:
    https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L5051-L5172
    """
    crow, col, values = _validate(A, size)
    input_length, output_length = size if transpose else size[::-1]
    _check_vector(x, input_length, values, "x")
    br, bc = values.shape[-2:]
    output_block = bc if transpose else br
    if y is None:
        shape = (output_length,)
        if x.ndim == 2:
            shape = (output_length, 1) if x.shape[1] == 1 else (output_length // output_block, output_block)
        y = values.new_empty(shape)
        beta = 0.0
    _check_vector(y, output_length, values, "y")
    if not y.is_contiguous():
        raise ValueError("y must be contiguous")
    # Warp casts scalars to A.scalar_type before making zero/one decisions.
    scalar = (lambda value: c_float(float(value)).value) if values.dtype == torch.float32 else float
    alpha, beta = scalar(alpha), scalar(beta)

    if not transpose:
        use_tiles, tile_size = _tile_dispatch(values.shape[0], br, bc, size[0], values.device.type, tile_size)
    else:
        use_tiles = False

    # Copy only when the input will be consumed. Detect shared storage rather
    # than just equal starting pointers so overlapping PyTorch views work too.
    if alpha != 0.0 and x.numel() and torch._C._overlaps(x, y):
        if work_buffer is None:
            work_buffer = torch.empty_like(x, memory_format=torch.contiguous_format)
        if work_buffer.numel() < x.numel():
            raise ValueError(f"Work buffer size is insufficient; needs at least {x.numel()} elements")
        if work_buffer.device != values.device or work_buffer.dtype != values.dtype:
            raise ValueError("Work buffer must have the same device and dtype as y")
        if not work_buffer.is_contiguous():
            raise ValueError("Work buffer must be contiguous")
        if torch._C._overlaps(work_buffer, x) or torch._C._overlaps(work_buffer, y):
            raise ValueError("Work buffer must not alias x or y")
        saved = work_buffer.reshape(-1)[:x.numel()]
        saved.copy_(x.reshape(-1))
        x = saved

    if output_length == 0:
        return y
    if alpha == 0.0 or values.shape[0] == 0:
        _scale_output(y, beta)
        if not transpose and alpha != 0.0 and not math.isfinite(alpha):
            # Warp's forward row kernel multiplies even an empty row's zero
            # sum by alpha. Transpose has no contributing blocks in this case.
            y.add_(alpha * 0.0)
        return y

    nrows = size[0] // br
    vector = x.reshape(-1).contiguous()
    output = y.view(-1)
    if values.device.type == "mps":
        lib = _metal_library()
        suffix = "i32" if crow.dtype == torch.int32 else "i64"
        args = (crow.contiguous(), col.contiguous(), values.contiguous(), vector,
                output, nrows, br, bc, alpha, beta)
        if transpose:
            _scale_output(output, beta)
            specialization = f"_r{br}" if br in _SPECIALIZED_BLOCK_DIMS else ""
            kernel = getattr(lib, "bsr_mv_transpose_" + suffix + specialization)
            # Warp dim=(nnz, block_cols) has the LAST dimension fastest.
            kernel(*args, threads=values.shape[0] * bc,
                   group_size=min(256, kernel.max_threads_per_threadgroup))
        elif use_tiles:
            # Cached BA camera transposes can make both dimensions constant;
            # retain the same Warp lane sequence and collective reduction.
            if (br, bc) == (9, 2) and tile_size == 64:
                specialization = "_r9_c2_t64"
            else:
                specialization = f"_t{tile_size}" if tile_size in _SPECIALIZED_TILE_SIZES else ""
            kernel = getattr(lib, "bsr_mv_tiled_" + suffix + specialization)
            if tile_size > min(1024, kernel.max_threads_per_threadgroup):
                raise ValueError("tile_size exceeds the Metal kernel's threadgroup limit")
            kernel(*args, tile_size, threads=output_length * tile_size, group_size=tile_size)
        else:
            # BA forward blocks and cached point transposes have fixed shapes.
            # Specializing dimensions removes runtime row-decoding division,
            # while retaining Warp's scalar row loop and accumulation order.
            if br == 2 and bc in (3, 9):
                specialization = f"_r2_c{bc}"
            elif (br, bc) == (3, 2):
                specialization = "_r3_c2"
            else:
                specialization = f"_c{bc}" if bc in _SPECIALIZED_BLOCK_DIMS else ""
            kernel = getattr(lib, "bsr_mv_" + suffix + specialization)
            kernel(*args, threads=output_length,
                   group_size=min(256, kernel.max_threads_per_threadgroup))
    else:
        # Dense-tensor fallback has the same affine/output semantics without
        # constructing native sparse tensors (CPU/CUDA tuple support).
        _scale_output(output, beta)
        rows = torch.repeat_interleave(
            torch.arange(nrows, device=values.device),
            (crow[1:] - crow[:-1]).long(), output_size=values.shape[0],
        )
        if transpose:
            gathered = vector.reshape(nrows, br)[rows]
            products = torch.bmm(values.transpose(1, 2), gathered.unsqueeze(-1)).squeeze(-1)
            output.reshape(-1, bc).index_add_(0, col.long(), products, alpha=alpha)
        else:
            gathered = vector.reshape(-1, bc)[col.long()]
            products = torch.bmm(values, gathered.unsqueeze(-1)).squeeze(-1)
            sums = values.new_zeros((nrows, br))
            sums.index_add_(0, rows, products)
            # Warp scales the completed row sum, including zero empty rows.
            output.add_(sums.reshape(-1), alpha=alpha)
    return y


def bsr_transposed(A, *, size):
    """Build a transposed BSR tuple, preserving the full matrix dimensions.

    As with Warp's bsr_transposed + bsr_mv, assembling this once allows repeated
    transpose products to use row reduction instead of contended atomic sums.
    The returned values own their storage; rebuild after changing A.
    PyTorch implementation of the semantics documented in:
    https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L2889-L3148
    """
    crow, col, values = _validate(A, size)
    nnz = values.shape[0]
    rows = torch.searchsorted(
        crow[1:], torch.arange(nnz, device=crow.device, dtype=crow.dtype), right=True,
    )
    order = torch.argsort(col, stable=True)
    sorted_col = col[order]
    boundaries = torch.arange(size[1] // values.shape[-1] + 1,
                              device=col.device, dtype=col.dtype)
    transposed_crow = torch.searchsorted(sorted_col, boundaries).to(crow.dtype)
    transposed_col = rows[order].to(col.dtype)
    transposed_values = values[order].transpose(-1, -2).contiguous()
    return transposed_crow, transposed_col, transposed_values


def bsr_diagonal(A, *, size):
    """Return diag(A.T @ A); duplicate block coordinates must be coalesced."""
    _, col, values = _validate(A, size)
    bc = values.shape[-1]
    diagonal = values.new_zeros((size[1] // bc, bc))
    diagonal.index_add_(0, col.long(), values.square().sum(dim=-2))
    return diagonal.reshape(-1)
