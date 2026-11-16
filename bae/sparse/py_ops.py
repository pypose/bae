import warnings
import torch
from torch.library import Library
from typing import Optional, Callable

import torch
from torch.utils._triton import has_triton
from .spgemm import convert_indices_from_csr_to_coo

USE_TRITON = True

if not USE_TRITON:
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    @triton.jit
    def add_kernel(
        crow,
        col,
        out_ptr,
        nnz,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        js = block_start + tl.arange(0, BLOCK_SIZE)
        mask = js < nnz
        colj = tl.load(col + js, mask=mask)
        output = (tl.load(crow+colj) <= js) & (tl.load(crow+colj+1) > js)
        tl.store(out_ptr + js, output, mask=mask)

    def diagonal_op_triton_(x, op: Optional[Callable]=None):
        nnz = x.col_indices().shape[-1]
        crow = x.crow_indices()
        col = x.col_indices()
        diag_mask = torch.zeros_like(col)
        grid = lambda meta: (triton.cdiv(nnz, meta["BLOCK_SIZE"]),)
        add_kernel[grid](crow, col, diag_mask, nnz, BLOCK_SIZE=4)
        
        return diag_mask
    @triton.jit
    def decompress_crow_kernel(
        crow_ptr,  # Pointer to the compressed row pointers
        row_indices_ptr,  # Pointer to the output row indices
        num_rows: tl.constexpr,  # Number of rows in the matrix
        num_nonzeros,  # Number of non-zero elements in the matrix
        BLOCK_SIZE: tl.constexpr,  # Block size for parallelism
    ):
        # Get the program ID
        pid = tl.program_id(0)
        
        # Compute the range of non-zero elements this block will handle
        block_start = pid * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, num_nonzeros)
        
        j = 0
        # Loop over the non-zero elements in the current block
        for i in range(block_start, block_end):
            while j < num_rows:
                if i >= tl.load(crow_ptr + j) and i < tl.load(crow_ptr + j + 1):
                    tl.store(row_indices_ptr + i, j)
                else:
                    j += 1
        
    @triton.jit
    def aggr_kernel(
        crow,
        source,
        out_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        row = tl.program_id(axis=0)
        row_start = tl.load(crow + row)
        row_end = tl.load(crow + row + 1)
        js = tl.arange(row_start, row_end)
        valj = tl.load(source + js)
        output = tl.sum(valj)
        tl.store(out_ptr + row, output)
def jtj_diag(Jt):
    block_shape = Jt.values().shape[-2:]
    col_indices = Jt.col_indices()
    nnz = col_indices.shape[-1]
    crow_indices = Jt.crow_indices()
    srows = crow_indices.shape[-1] - 1
    values = Jt.values()
    values = values.flatten(start_dim=0, end_dim=1)
    dotp = torch.linalg.vecdot(values, values)
    cooa = convert_indices_from_csr_to_coo(crow_indices.contiguous(), col_indices.contiguous(), crow_indices.dtype == torch.int32, False)
    row_indices = cooa[0]
    if block_shape[0] == 1:
        ...
    else:
        row_indices = row_indices * block_shape[0]
        row_indices = row_indices.unsqueeze(-1)
        offsets = torch.arange(0, block_shape[0], device=row_indices.device, dtype=row_indices.dtype).unsqueeze(0)
        row_indices = row_indices + offsets
        row_indices = row_indices.flatten()
    diag_values = torch.zeros(srows * block_shape[0], device=values.device, dtype=values.dtype)
    diag_values.scatter_add_(0, row_indices.to(torch.int64), dotp)
    return diag_values

def diagonal_op_(input, offset: int=0, op: Optional[Callable]=None):
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    m, n = input.shape[-2], input.shape[-1]
    if bsr_values.ndim > 1:
        dm, dn = (bsr_values.shape[-2], bsr_values.shape[-1])
    else:
        dm, dn = 1, 1
    sm, sn = m // dm, n // dn

    #simple case(block is square and offset is 0)
    if dm == dn and offset == 0:
        if not USE_TRITON:
            dummy_val = torch.zeros(bsr_values.shape[0], device='cpu')
            dummy = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                            col_indices=col_indices.to('cpu'),
                                            values=dummy_val)
            dummy_coo = dummy.to_sparse(layout=torch.sparse_coo).coalesce()

            indices = dummy_coo.indices().to(input.device)
            diag_mask = (indices[0] == indices[1])
        else:
            diag_mask = diagonal_op_triton_(input)
        diag_indices = diag_mask.nonzero().squeeze(-1)
        if bsr_values.ndim > 1:
            block_diags = bsr_values.diagonal(dim1=-2, dim2=-1)
        else:
            block_diags = bsr_values
        values = block_diags[diag_indices]
        n_diag_blocks = sm if sm < sn else sn
        if diag_indices.shape[-1] == n_diag_blocks:
            results = values
        else:
            results_shape = (n_diag_blocks, dm)
            results = torch.zeros(results_shape, dtype=values.dtype, device=values.device)
            results[indices[0, diag_indices]] = values
            assert op is None, "op is not supported for diagonal that has empty values."
        if bsr_values.ndim > 1:
            results = torch.flatten(results, start_dim=-2, end_dim=-1)
        # apply the inplace op
        if op is not None:
            results = op(results)
            block_diags[diag_indices] = results.view(n_diag_blocks, dm) if bsr_values.ndim > 1 else results
        return results
    else:
        raise NotImplementedError('Only square block and offset 0 is supported.')


def spdiags_(diagonal, offset, shape, layout):
    """
    Creates a sparse 2D tensor by placing the values from rows of diagonals along specified diagonals of the output
    tensor.

    Args:
        diagonal (Tensor): Matrix storing diagonals row-wise.
        offset (int): The diagonal in the output tensor corresponding to the main diagonal of the input tensor.
        shape (Tuple[int, int]): The shape of the output tensor.
        layout (str): The layout of the output tensor.
    """
    crow_indices = torch.arange(0, shape[0] + 1, device=diagonal.device, dtype=torch.int32)
    col_indices = torch.arange(0, shape[1], device=diagonal.device, dtype=torch.int32)
    return torch.sparse_csr_tensor(crow_indices, col_indices, diagonal)


def compact_indices(input):
    """
    Returns the CSR/BSR tensor with int32 indices if the nnz and both dimensions fit in int32,
    otherwise the input unchanged. SpMV/SpGEMM are memory bound and int64 indices are a large
    share of the bytes read.
    """
    if input.layout not in (torch.sparse_csr, torch.sparse_bsr):
        return input
    if input.col_indices().dtype == torch.int32:
        return input
    if max(input._nnz(), *input.shape) > torch.iinfo(torch.int32).max:
        return input
    ctor = torch.sparse_csr_tensor if input.layout == torch.sparse_csr else torch.sparse_bsr_tensor
    return ctor(input.crow_indices().to(torch.int32), input.col_indices().to(torch.int32),
                input.values(), size=input.shape)


def inv_op(input):
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    inv_values = torch.linalg.inv(bsr_values)

    return torch.sparse_bsr_tensor(
        crow_indices=crow_indices, 
        col_indices=col_indices, 
        values=inv_values,
        size=input.shape, 
        dtype=input.dtype, )

def to_cooh(input):
    crow_indices = input.crow_indices() # b + 1 dimensional
    col_indices = input.col_indices() # b + 1 dimensional
    bsr_values = input.values() # 1 + 2 dimensional
    block_shape = bsr_values.shape[-2:]
    
    dummy_csr = torch.sparse_csr_tensor(crow_indices=crow_indices.to('cpu'),
                                        col_indices=col_indices.to('cpu'),
                                        values=torch.zeros(bsr_values.shape[0], device='cpu'))
    dummy_coo = dummy_csr.to_sparse(layout=torch.sparse_coo).coalesce()
    indices = dummy_coo.indices().to(input.device)
    input_cooh = torch.sparse_coo_tensor(indices, bsr_values, [input.shape[0] // block_shape[0], input.shape[1] // block_shape[1], block_shape[0], block_shape[1]])
    return input_cooh

def to_bsr(input):
    dummy_coo = torch.sparse_coo_tensor(input.indices(), torch.zeros(input.values().shape[0], device=input.device), input.shape[:2])
    dummy_coo = dummy_coo.coalesce()
    dummy_csr = dummy_coo.to_sparse(layout=torch.sparse_csr)
    crow_indices = dummy_csr.crow_indices()
    col_indices = dummy_csr.col_indices()
    values = input.values()
    block_shape = input.shape[-2:]
    return torch.sparse_bsr_tensor(crow_indices, col_indices, values, [input.shape[0] * block_shape[0], input.shape[1] * block_shape[1]])

def add_op(input1, input2):
    input1_cooh = to_cooh(input1)
    input2_cooh = to_cooh(input2)
    input1_cooh = input1_cooh.coalesce()
    input2_cooh = input2_cooh.coalesce()
    result = input1_cooh + input2_cooh
    result = result.coalesce()
    result = to_bsr(result)
    return result

def bsr2bsc(J):
    block_shape = J.values().shape[-2:]
    dummy_shape = [J.shape[0] // block_shape[0], J.shape[1] // block_shape[1]]
    dummy_csr = torch.sparse_csr_tensor(
        crow_indices=J.crow_indices(),
        col_indices=J.col_indices(),
        values=torch.arange(J.col_indices().shape[0], device=J.device, dtype=torch.int32),
        size=dummy_shape,
    )
    dummy_csc = dummy_csr.to_sparse_csc()
    J_bsc = torch.sparse_bsc_tensor(
        ccol_indices=dummy_csc.ccol_indices(),
        row_indices=dummy_csc.row_indices(),
        values=J.values()[dummy_csc.values()],
        size=J.shape,
    )
    return J_bsc

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sparse_lib = Library('aten', 'IMPL')
    sparse_lib.impl('diagonal', diagonal_op_, 'SparseCsrCPU')
    sparse_lib.impl('diagonal', diagonal_op_, 'SparseCsrCUDA')
