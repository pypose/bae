from __future__ import annotations

from typing import Optional, Union, Any, Tuple

import torch
from torch import Tensor


class NormalMatVec:
    r"""Matrix-free normal-equation linear operator.

    Given a Jacobian J (dense, CSR/COO, or BSR), this operator represents:

        A x = J^T (J x) + damping * diag(J^T J) * x

    where diag(J^T J) is computed as the column-wise sum of squares of J.
    """

    def __init__(
        self,
        J: Tensor,
        damping: Union[float, Tensor] = 0.0,
        diag: Optional[Tensor] = None,
    ):
        if not torch.is_tensor(J):
            raise TypeError("J must be a torch.Tensor")
        if J.ndim != 2:
            raise ValueError("J must be 2-D")

        self.J: Tensor = J
        self._Jt: Optional[Tensor] = None

        self.device = J.device
        self.dtype = J.dtype
        self.layout = J.layout if J.layout != torch.sparse_coo else torch.sparse_csr

        self.shape: Tuple[int, int] = (J.shape[1], J.shape[1])
        self.ndim: int = 2

        self._diag: Tensor = diag if diag is not None else self._compute_diag(J)
        if self._diag.ndim != 1 or self._diag.numel() != J.shape[1]:
            raise ValueError("diag must be 1-D with length equal to J.shape[1]")

        self.set_damping(damping)

    def set_damping(self, damping: Union[float, Tensor]) -> None:
        # Precompute damping * diag once here so matvec needs no host sync or
        # host-to-device copy of the damping scalar.
        if isinstance(damping, Tensor):
            if damping.numel() != 1:
                raise ValueError("damping tensor must be scalar")
            self.damping = damping.to(device=self.device, dtype=self.dtype).reshape(())
            self._damp_diag = (self.damping * self._diag).unsqueeze(-1)
        else:
            self.damping = float(damping)
            self._damp_diag = (
                None if self.damping == 0.0 else (self.damping * self._diag).unsqueeze(-1)
            )

    def diagonal(self) -> Tensor:
        # Always a new tensor: callers (e.g. PCG) modify the result in place.
        if self._damp_diag is None:
            return self._diag.clone()
        return self._diag + self._damp_diag.squeeze(-1)

    def matvec(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x2d = x.unsqueeze(-1)
        elif x.ndim == 2:
            x2d = x
        else:
            raise ValueError("x must be 1-D or 2-D")

        if x2d.device != self.device or x2d.dtype != self.dtype:
            x2d = x2d.to(device=self.device, dtype=self.dtype)

        y = self.J @ x2d
        Jt = self._get_Jt()
        z = Jt @ y

        if self._damp_diag is not None:
            z.addcmul_(self._damp_diag, x2d)

        return z.squeeze(-1) if x.ndim == 1 else z

    def __call__(self, x: Tensor) -> Tensor:
        return self.matvec(x)

    def __matmul__(self, x: Tensor) -> Tensor:
        return self.matvec(x)

    @classmethod
    def __torch_function__(  # type: ignore[override]
        cls, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Optional[dict] = None
    ):
        if kwargs is None:
            kwargs = {}
        if func is torch.matmul and len(args) >= 2:
            A, B = args[0], args[1]
            if isinstance(A, cls):
                out = kwargs.get("out", None)
                result = A.matvec(B)
                if out is not None:
                    out_tensor = out[0] if isinstance(out, tuple) else out
                    out_tensor.copy_(result)
                    return out_tensor
                return result
        return NotImplemented

    def _get_Jt(self) -> Tensor:
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

