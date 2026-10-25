
# https://nvidia.github.io/warp/modules/sparse.html
import warp as wp
from warp import sparse as wps
from warp.optim import linear as wpol
import torch
wp.init()


def torchbsr2wp(tbsr):
    assert tbsr.layout == torch.sparse_bsr
    block_type = wp.types.matrix(shape=tuple(tbsr.values().shape[-2:]), dtype=wp.dtype_from_torch(tbsr.dtype))
    bsr = wps.bsr_matrix_t(block_type)()
    bsr.nrow = int(tbsr.shape[0] // block_type._shape_[0])
    bsr.ncol = int(tbsr.shape[1] // block_type._shape_[1])
    bsr.nnz = int(tbsr.values().shape[0])
    bsr.columns = wp.from_torch(tbsr.col_indices().to(torch.int32))
    bsr.values = wp.from_torch(tbsr.values().contiguous(), dtype=block_type)
    bsr.offsets = wp.from_torch(tbsr.crow_indices().to(torch.int32))
    return bsr

def block_diag_wp(blocks):
    """Warp BSR matrix with the (n, b, b) torch tensor `blocks` on its diagonal."""
    n, bm, bn = blocks.shape
    idx = torch.arange(n + 1, dtype=torch.int32, device=blocks.device)
    return torchbsr2wp(torch.sparse_bsr_tensor(idx, idx[:-1], blocks.contiguous(), size=(n * bm, n * bn)))

def wp2torchbsr(bsr):
    """
    Convert a Warp BSR matrix back into a PyTorch sparse BSR tensor.

    Args:
        bsr (warp.bsr_matrix_t): A Warp BSR matrix.

    Returns:
        torch.Tensor: A PyTorch sparse BSR tensor with the same data.
    """

    # Extract block dimensions from the values array:
    # Typically: bsr.values has shape (nnz, block_rows, block_cols)
    # so we take the last two dimensions as the block shape.
    block_rows, block_cols = bsr.block_shape[-2:]

    # Total matrix dimensions:
    nrows = bsr.nrow * block_rows
    ncols = bsr.ncol * block_cols

    # Convert Warp arrays back to PyTorch tensors
    crow = wp.to_torch(bsr.offsets)            # shape [bsr.nrow + 1]
    col  = wp.to_torch(bsr.columns)            # shape [bsr.nnz]
    vals = wp.to_torch(bsr.values)             # shape [bsr.nnz, block_rows, block_cols]

    tbsr = torch.sparse_bsr_tensor(
        crow,
        col,
        vals,
        size=(nrows, ncols),
        dtype=vals.dtype,
        device=vals.device
    )

    return tbsr

def format_vec_for_bsr(tvec, block_shape):
    y_vec_len = block_shape[1]
    y_dtype = wp.types.vector(length=y_vec_len, dtype=wp.dtype_from_torch(tvec.dtype))
    if tvec.ndim == 1 and tvec.shape[-1] != y_vec_len:
        tvec = tvec.reshape(-1, y_vec_len)
    vwp = wp.from_torch(tvec, dtype=y_dtype)
    return vwp


def _validate_bsr_add(input, other):
    if input.layout != torch.sparse_bsr or other.layout != torch.sparse_bsr:
        raise ValueError("Both input and other must be sparse BSR tensors.")
    if input.shape != other.shape:
        raise ValueError(f"Expected matching sparse BSR shapes, got {input.shape} and {other.shape}.")
    blocksize = tuple(input.values().shape[-2:])
    if tuple(other.values().shape[-2:]) != blocksize:
        raise ValueError(
            f"Expected matching sparse BSR block shapes, got {blocksize} and {tuple(other.values().shape[-2:])}."
        )
    return blocksize


def _sparse_csr_add_cuda(input, other, alpha=1.0):
    _validate_bsr_add(input, other)
    # bsr_axpy mutates its destination argument, so keep the ATen add contract
    # by operating on a cloned destination.
    input_dst = torch.sparse_bsr_tensor(
        crow_indices=input.crow_indices().clone(),
        col_indices=input.col_indices().clone(),
        values=input.values().clone(),
        size=input.shape,
        device=input.device,
        dtype=input.dtype,
    )
    input_wp = torchbsr2wp(input_dst)
    other_wp = torchbsr2wp(other)
    res = wp2torchbsr(wps.bsr_axpy(other_wp, input_wp, alpha=alpha))

    # Warp can over-allocate `col` / `values`; trim using nnz from crow[-1].
    nnz = int(res.crow_indices()[-1].item()) if res.crow_indices().numel() > 0 else 0
    if res.col_indices().numel() != nnz or res.values().shape[0] != nnz:
        res = torch.sparse_bsr_tensor(
            crow_indices=res.crow_indices(),
            col_indices=res.col_indices()[:nnz].contiguous(),
            values=res.values()[:nnz].contiguous(),
            size=res.shape,
            device=res.device,
            dtype=res.dtype,
        )

    return res


def _sparse_csr_add_cpu(input, other, alpha=1.0):
    blocksize = _validate_bsr_add(input, other)
    result = torch.add(input.to_sparse_coo(), other.to_sparse_coo(), alpha=alpha).coalesce()
    return result.to_sparse_bsr(blocksize)

from torch.library import Library
sparse_lib = Library('aten', 'IMPL')
sparse_lib.impl('add.Tensor', _sparse_csr_add_cuda, 'SparseCsrCUDA')
sparse_lib.impl('add.Tensor', _sparse_csr_add_cpu, 'SparseCsrCPU')
# this will, however, invalidate add_sparse_csr


if __name__ == '__main__':
    crow_indices = torch.tensor([0, 2, 4], dtype=torch.int32)
    col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    values = torch.tensor([[[0, 1, 2], [6, 7, 8]],
                        [[3, 4, 5], [9, 10, 11]],
                        [[12, 13, 14], [18, 19, 20]],
                        [[15, 16, 17], [21, 22, 23]]])

    bsr = torch.sparse_bsr_tensor(crow_indices, col_indices, values, dtype=torch.float32).to('cuda')
    print(bsr.to_dense())
    a = torchbsr2wp(bsr)
    print(a.transpose() @ a)
    print(wp2torchbsr(a).to_dense())
    # print(wpol.preconditioner(a))
