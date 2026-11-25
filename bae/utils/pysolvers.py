from typing import Optional
import torch
from torch import Tensor
from pypose.optim.solver import CG
from .linear_operator import DiagonalMatVec
from ..sparse import py_ops as _py_ops  # Registers native sparse diagonal dispatch.

try:
    from bae.sparse.solve import CuDirectSparseSolver as CuDSS
except Exception as e:
    _cudss_import_error = e

    class CuDSS(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CuDSS solver is unavailable because `bae.sparse.solve` failed to import. "
                "This is commonly caused by a CUDA/cuDSS/cuBLAS mismatch (e.g. missing "
                "`libcublas.so.13`). Use `PCG(...)` instead, or install a cuDSS build that "
                "matches your CUDA toolkit."
            ) from _cudss_import_error


class PCG(CG):
    """Diagonally preconditioned CG with scalar reductions for MPS solves."""

    def __init__(self, maxiter=None, tol=1e-5):
        super().__init__(maxiter, tol)
    def forward(self, A, b, x=None, M=None) -> torch.Tensor:
        was_vector = b.dim() == 1
        if was_vector:
            b = b[..., None]
            if x is not None and x.dim() == 1:
                x = x[..., None]
        if M is None:
            l_diag = A.diagonal().clone()
            l_diag[l_diag.abs() < 1e-6] = 1e-6
            M = DiagonalMatVec(l_diag.reciprocal())

        if (b.device.type == 'mps' and b.dtype == torch.float32
                and A.ndim == 2 and b.ndim == 2 and b.shape[-1] == 1):
            res = self._forward_mps(A, b, x, M)
        else:
            res = super().forward(A, b, x, M)
        if was_vector:
            res = res.squeeze(-1)
        return res

    def _forward_mps(self, A, b, x, M):
        # TODO: Upstream these scalar reductions to PyPose CG, then remove this
        # duplicate loop and delegate to super().forward on MPS as well.
        # Preserve PyPose CG's recurrence, stopping checks, output-buffer use,
        # and in-place warm-start updates. For a real single-column RHS, its
        # two 1xN @ Nx1 products are scalar reductions. The equivalent
        # multiply-and-sum avoids the slow MPS matrix-multiply dispatch.
        if x is None:
            x = torch.zeros_like(b)
        elif x.shape != b.shape:
            raise ValueError("x must have the same shape as b")

        bnrm2 = torch.linalg.norm(b, dim=0)
        if (bnrm2 == 0).all():
            return b
        atol = self.tol * bnrm2
        maxiter = b.shape[-2] * 10 if self.maxiter is None else self.maxiter
        r = b - A @ x if x.any() else b.clone()
        q, z = torch.empty_like(b), torch.empty_like(b)
        rho_prev, p = None, None

        for iteration in range(maxiter):
            if (torch.linalg.norm(r, dim=0) < atol).all():
                return x

            torch.matmul(M, r, out=z)
            rho_cur = (r * z).sum(dim=0, keepdim=True)
            if iteration:
                p.mul_(rho_cur / rho_prev).add_(z)
            else:
                p = z.clone()

            torch.matmul(A, p, out=q)
            alpha = rho_cur / (p * q).sum(dim=0, keepdim=True)
            x += alpha * p
            r -= alpha * q
            rho_prev = rho_cur

        return x

class SciPySpSolver(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, A, b):
        import scipy.sparse.linalg as spla
        import scipy.sparse as sp
        import numpy as np
        if A.layout != torch.sparse_csr:
            A = A.to_sparse_coo().to_sparse_csr()
        A_csr = sp.csr_matrix((A.values().cpu().numpy(), 
                                   A.col_indices().cpu().numpy(),
                                   A.crow_indices().cpu().numpy()),
                                  shape=A.shape)
        b = b.cpu().numpy()
        x = spla.spsolve(A_csr, b, use_umfpack=False)
        assert not np.isnan(x).any()
        # a_err = np.linalg.norm(A_csr @ x - b)
        # r_err = a_err / np.linalg.norm(b)
        # print(f"Linear Solver Error: {a_err}, relative error: {r_err}")
        return torch.from_numpy(x).to(A.device)


# cuda graph version of the solver
class CG_(torch.nn.Module):
    r'''The batched linear solver with conjugate gradient method.

    .. math::
        \mathbf{A}_i \bm{x}_i = \mathbf{b}_i,

    where :math:`\mathbf{A}_i \in \mathbb{C}^{M \times N}` and :math:`\bm{b}_i \in
    \mathbb{C}^{M \times 1}` are the :math:`i`-th item of batched linear equations.

    This function is a 1:1 replica of `scipy.sparse.linalg.cg <https://docs.scipy.org/doc
    /scipy/reference/generated/scipy.sparse.linalg.cg.html>`_.
    The solution is consistent with the scipy version up to numerical precision.
    Variable names are kept the same as the scipy version for easy reference.
    We recommend using only non-batched or batch size 1 input for this solver, as
    the batched version was not appeared in the original scipy version. When handling
    sparse matrices, the batched computation may introduce additional overhead.

    Examples:
        >>> # dense example
        >>> import pypose.optim.solver as ppos
        >>> A = torch.tensor([[0.1802967, 0.3151198, 0.4548111, 0.3860016, 0.2870615],
                              [0.3151198, 1.4575327, 1.5533425, 1.0540756, 1.0795838],
                              [0.4548111, 1.5533425, 2.3674474, 1.1222278, 1.2365348],
                              [0.3860016, 1.0540756, 1.1222278, 1.3748058, 1.2223261],
                              [0.2870615, 1.0795838, 1.2365348, 1.2223261, 1.2577004]])
        >>> b = torch.tensor([[ 2.64306851],
                              [-0.03593633],
                              [ 0.73612658],
                              [ 0.51501254],
                              [-0.26689271]])
        >>> solver = ppos.CG()
        >>> x = solver(A, b)
        tensor([[246.4098],
                [ 22.6997],
                [-56.9239],
                [-161.7914],
                [137.2683]])

        >>> # sparse csr example
        >>> import pypose.optim.solver as ppos
        >>> crow_indices = torch.tensor([0, 2, 4])
        >>> col_indices = torch.tensor([0, 1, 0, 1])
        >>> values = torch.tensor([1, 2, 3, 4], dtype=torch.float)
        >>> A = torch.sparse_csr_tensor(crow_indices, col_indices, values)
        >>> A.to_dense()  # visualize
        tensor([[1., 2.],
                [3., 4.]])
        >>> b = torch.tensor([[1.], [2.]])
        >>> solver = ppos.CG()
        >>> x = solver(A, b)
        tensor([-4.4052e-05,  5.0003e-01])

    '''
    def __init__(self, maxiter=None, tol=1e-5):
        super().__init__()
        self.maxiter, self.tol = maxiter, tol
        self.graph_first_iter = None
        self.graph_subsequent_iter = None
        self.static_A_shape, self.static_b_shape, self.static_M_is_none, self.static_device = \
            None, None, None, None
        # Tensors for graph capture/replay
        self.static_A, self.static_b, self.static_M = None, None, None
        self.static_x, self.static_r, self.static_p, self.static_q, self.static_z = \
            None, None, None, None, None
        self.static_rho_prev, self.static_rho_cur = None, None

    def forward(self, A: torch.Tensor, b: Tensor, x: Optional[Tensor]=None,
                M: Optional[torch.Tensor]=None) -> Tensor:
        '''
        Args:
            A (Tensor): the input tensor. It is assumed to be a symmetric
                positive-definite matrix. Layout is allowed to be COO, CSR, BSR, or dense.
            b (Tensor): the tensor on the right hand side. Layout could be sparse or dense
                but is only allowed to be a type that is compatible with the layout of A.
                In other words, `A @ b` operation must be supported by the layout of A.
            x (Tensor, optional): the initial guess for the solution. Default: ``None``.
            M (Tensor, optional): the preconditioner for A. Layout is allowed to be COO,
                CSR, BSR, or dense. Default: ``None``.

        Return:
            Tensor: the solved tensor. Layout is the same as the layout of b.
        '''
        if A.ndim == b.ndim + 1:
            b = b.unsqueeze(-1)
        else:
            assert A.ndim == b.ndim, \
                'The number of dimensions of A and b must be the same or one more than b'

        if x is None:
            x = torch.zeros_like(b)

        bnrm2 = torch.linalg.norm(b, dim=0)
        if (bnrm2 == 0).all():
            return b
        atol = self.tol * bnrm2
        n = b.shape[-2]

        if self.maxiter is None:
            maxiter = n * 10
        else:
            maxiter = self.maxiter

        # Determine if CUDA graph can be used and if re-capture is needed
        use_cuda_graph = A.is_cuda
        
        if use_cuda_graph:
            re_capture_graph = (self.graph_first_iter is None or \
                                self.static_A_shape != A.shape or \
                                self.static_b_shape != b.shape or \
                                self.static_M_is_none != (M is None) or \
                                self.static_device != A.device)

            if re_capture_graph:
                # Allocate static tensors and capture new graphs
                self.static_A = A.clone()
                self.static_b = b.clone()
                self.static_x = x.clone() # Initial x
                self.static_r = b - A @ x # Initial r
                self.static_p = torch.zeros_like(b) # Will be updated
                self.static_q = torch.empty_like(b)
                self.static_z = torch.empty_like(b)

                # Initialize rho_prev and rho_cur with shape [1, 1]
                self.static_rho_prev = torch.zeros(1, 1, device=A.device)
                self.static_rho_cur = torch.zeros(1, 1, device=A.device)

                self.static_M_is_none = (M is None)
                self.static_device = A.device
                self.static_A_shape = A.shape
                self.static_b_shape = b.shape

                if M is not None:
                    self.static_M = M.clone()
                else:
                    self.static_M = None

                # Capture first iteration graph
                self.graph_first_iter = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(self.graph_first_iter):
                    # Operations for first iteration
                    if not self.static_M_is_none:
                        torch.matmul(self.static_M, self.static_r, out=self.static_z)
                    else:
                        self.static_z.copy_(self.static_r) # z = r.clone()
                    self.static_rho_cur.copy_(torch.matmul(self.static_r.mT, self.static_z))
                    self.static_p.copy_(self.static_z) # p = z.clone()
                    torch.matmul(self.static_A, self.static_p, out=self.static_q)
                    alpha = self.static_rho_cur / torch.matmul(self.static_p.mT, self.static_q)
                    self.static_x.add_(alpha * self.static_p)
                    self.static_r.sub_(alpha * self.static_q)
                    self.static_rho_prev.copy_(self.static_rho_cur)

                # Capture subsequent iteration graph
                self.graph_subsequent_iter = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(self.graph_subsequent_iter):
                    # Operations for subsequent iterations
                    if not self.static_M_is_none:
                        torch.matmul(self.static_M, self.static_r, out=self.static_z)
                    else:
                        self.static_z.copy_(self.static_r) # z = r.clone()
                    self.static_rho_cur.copy_(torch.matmul(self.static_r.mT, self.static_z))
                    beta = self.static_rho_cur / self.static_rho_prev
                    self.static_p.mul_(beta).add_(self.static_z)
                    torch.matmul(self.static_A, self.static_p, out=self.static_q)
                    alpha = self.static_rho_cur / torch.matmul(self.static_p.mT, self.static_q)
                    self.static_x.add_(alpha * self.static_p)
                    self.static_r.sub_(alpha * self.static_q)
                    self.static_rho_prev.copy_(self.static_rho_cur)

            # Now run the loop using the (newly captured or existing) graphs
            self.static_A.copy_(A)
            self.static_b.copy_(b)
            self.static_x.copy_(x)
            self.static_r.copy_(b - A @ x) # Initial r
            if M is not None:
                self.static_M.copy_(M)

            # First iteration
            self.graph_first_iter.replay()
            if (torch.linalg.norm(self.static_r, dim=0) < atol).all():
                return self.static_x

            # Subsequent iterations
            for iteration in range(1, maxiter):
                self.graph_subsequent_iter.replay()
                if (torch.linalg.norm(self.static_r, dim=0) < atol).all():
                    return self.static_x
            return self.static_x

        else: # A is not on CUDA, or other conditions not met for graph, run original Python loop
            r = b - A @ x if x.any() else b.clone()
            rho_prev, p = None, None

            q = torch.empty_like(b)
            if M is not None:
                z = torch.empty_like(b)
            else:
                z = r.clone()

            for iteration in range(maxiter):
                if (torch.linalg.norm(r, dim=0) < atol).all():
                    return x
                
                if M is not None:
                    torch.matmul(M, r, out=z)
                rho_cur = torch.matmul(r.mT, z)
                if iteration > 0:
                    beta = rho_cur / rho_prev
                    p.mul_(beta).add_(z)
                else:  # First spin
                    p = z.clone()

                torch.matmul(A, p, out=q)
                alpha = rho_cur / torch.matmul(p.mT, q)
                x += alpha * p
                r -= alpha * q
                rho_prev = rho_cur

            return x
