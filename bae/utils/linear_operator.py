from __future__ import annotations

from typing import Optional, Union, Tuple

import torch
from torch import Tensor


class _MatVecOperator:
    ndim = 2

    def __call__(self, x):
        return self.matvec(x)

    def __matmul__(self, x):
        return self.matvec(x)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.matmul and len(args) >= 2 and isinstance(args[0], cls):
            out = kwargs.get("out")
            return args[0].matvec(args[1], out=out[0] if isinstance(out, tuple) else out)
        return NotImplemented


class DiagonalMatVec(_MatVecOperator):
    """Apply a diagonal matrix without constructing a sparse tensor."""

    def __init__(self, diagonal):
        if diagonal.ndim != 1:
            raise ValueError("diagonal must be one-dimensional")
        self.values = diagonal
        self.device, self.dtype = diagonal.device, diagonal.dtype
        self.shape = (diagonal.numel(), diagonal.numel())

    def matvec(self, x, out=None):
        if x.ndim not in (1, 2) or x.shape[0] != self.shape[1]:
            raise ValueError("x must be a vector or matrix with compatible leading dimension")
        diagonal = self.values if x.ndim == 1 else self.values[:, None]
        return torch.mul(diagonal, x, out=out) if out is not None else diagonal * x

    def diagonal(self):
        return self.values


class NormalMatVec(_MatVecOperator):
    r"""Matrix-free normal-equation linear operator.

    Given a Jacobian J (dense, CSR/COO, or BSR), this operator represents:

        A x = J^T (J x) + damping * diag(J^T J) * x

    where diag(J^T J) is computed as the column-wise sum of squares of J.
    Supplying ``diag`` replaces the normal matrix's diagonal before damping,
    matching LM's explicit clamp-and-damp behavior.
    ``raw_diag`` can reuse a previously computed diagonal. The replacement and
    correction must use the same reduction result: separate parallel sums can
    differ by more than a small damping term in float32.
    """

    def __init__(
        self,
        J: Tensor,
        damping: Union[float, Tensor] = 0.0,
        diag: Optional[Tensor] = None,
        *,
        raw_diag: Optional[Tensor] = None,
    ):
        if not isinstance(J, Tensor):
            raise TypeError("J must be a torch.Tensor")
        if J.ndim != 2:
            raise ValueError("J must be 2-D")

        self.J: Tensor = J
        self._Jt: Optional[Tensor] = None

        self.device = J.device
        self.dtype = J.dtype

        self.shape: Tuple[int, int] = (J.shape[1], J.shape[1])
        self.ndim: int = 2

        self._raw_diag = self._compute_diag(J) if raw_diag is None else raw_diag
        if self._raw_diag.ndim != 1 or self._raw_diag.numel() != J.shape[1]:
            raise ValueError("raw_diag must be 1-D with length equal to J.shape[1]")
        if self._raw_diag.device != self.device or self._raw_diag.dtype != self.dtype:
            raise ValueError("raw_diag must have the same device and dtype as J")
        self._diag: Tensor = diag if diag is not None else self._raw_diag
        if self._diag.ndim != 1 or self._diag.numel() != J.shape[1]:
            raise ValueError("diag must be 1-D with length equal to J.shape[1]")
        if self._diag.device != self.device or self._diag.dtype != self.dtype:
            raise ValueError("diag must have the same device and dtype as J")

        self.set_damping(damping)

    def set_damping(self, damping: Union[float, Tensor]) -> None:
        if isinstance(damping, Tensor):
            if damping.numel() != 1:
                raise ValueError("damping tensor must be scalar")
            self.damping = damping.to(device=self.device, dtype=self.dtype)
        else:
            self.damping = float(damping)
        self._effective_diag = self._diag * (1.0 + self.damping)
        # Keep small positive damping even when 1 + damping rounds to 1 in
        # float32. Subtract only the intended diagonal replacement.
        self._correction = (self._diag - self._raw_diag) + self.damping * self._diag

    def diagonal(self) -> Tensor:
        return self._effective_diag

    def matvec(self, x: Tensor, out: Optional[Tensor] = None) -> Tensor:
        if x.ndim not in (1, 2) or x.shape[0] != self.shape[1]:
            raise ValueError("x must be a vector or matrix with compatible leading dimension")
        if out is not None:
            if out.shape != x.shape or out.device != self.device or out.dtype != self.dtype:
                raise ValueError("out must have the result's shape, device, and dtype")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            if torch._C._overlaps(x, out):
                x = x.clone()
        x2d = x.unsqueeze(-1) if x.ndim == 1 else x

        if x2d.device != self.device or x2d.dtype != self.dtype:
            x2d = x2d.to(device=self.device, dtype=self.dtype)

        # Both native and tuple-backed Jacobians use ordinary tensor dispatch.
        y = self.J @ x2d
        output = None if out is None else out.reshape(x2d.shape)
        z = torch.matmul(self._get_Jt(), y, out=output)
        z.addcmul_(self._correction.unsqueeze(-1), x2d)

        if out is not None:
            return out
        return z.squeeze(-1) if x.ndim == 1 else z

    def _get_Jt(self):
        if self._Jt is None:
            Jt = self.J.mT
            if Jt.layout == torch.sparse_csc:
                Jt = Jt.to_sparse_csr()
            elif Jt.layout == torch.sparse_bsc:
                bs = Jt.values().shape[-2:]
                Jt = Jt.to_sparse_bsr(blocksize=bs)
            self._Jt = Jt
        return self._Jt

    @staticmethod
    def _compute_diag(J: Tensor) -> Tensor:
        if hasattr(J, 'normal_diagonal'):
            return J.normal_diagonal()
        if J.layout == torch.strided:
            return J.square().sum(dim=0)

        if J.layout == torch.sparse_bsr:
            values = J.values()
            dm, dn = values.shape[-2], values.shape[-1]
            col_blocks = J.col_indices()
            contrib = values.square().sum(dim=-2)  # (nnz_blocks, dn)
            offsets = torch.arange(dn, device=contrib.device, dtype=col_blocks.dtype)
            cols = (col_blocks[:, None] * dn + offsets[None, :]).reshape(-1).to(torch.int64)
            contrib_flat = contrib.reshape(-1)
            diag = torch.zeros(J.shape[1], device=contrib.device, dtype=contrib.dtype)
            diag.scatter_add_(0, cols, contrib_flat)
            return diag

        if J.layout == torch.sparse_csr:
            values = J.values()
            col = J.col_indices().to(torch.int64)
            v2 = values.square()
            if v2.ndim > 1:
                v2 = v2.reshape(v2.shape[0], -1).sum(dim=-1)
            diag = torch.zeros(J.shape[1], device=values.device, dtype=v2.dtype)
            diag.scatter_add_(0, col, v2)
            return diag

        if J.layout == torch.sparse_coo:
            Jc = J.coalesce()
            col = Jc.indices()[1].to(torch.int64)
            v2 = Jc.values().square()
            if v2.ndim > 1:
                v2 = v2.reshape(v2.shape[0], -1).sum(dim=-1)
            diag = torch.zeros(J.shape[1], device=Jc.device, dtype=v2.dtype)
            diag.scatter_add_(0, col, v2)
            return diag

        raise NotImplementedError(f"Unsupported J layout: {J.layout}")

