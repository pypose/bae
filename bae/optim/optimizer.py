from functools import partial
import torch
from pypose.optim import LevenbergMarquardt as ppLM
import pypose as pp

from ..autograd.graph import jacobian, jacobian_components, materialize_jacobian_components
from ..autograd.function import TrackingTensor
from ..sparse.py_ops import diagonal_op_, inv_op
from ..utils.linear_operator import NormalMatVec
from ..utils.parameter import parameter_update_shape

try:
    import warp as wp
    from warp import sparse
    from warp.optim import linear
    from bae.sparse.warp_wrappers import format_vec_for_bsr, torchbsr2wp, wp2torchbsr
except ImportError:
    pass


class LM(ppLM):
    def __init__(self, *args, matrix_free_normal: bool = False, **kwargs):
        self.matrix_free_normal = matrix_free_normal
        super(LM, self).__init__(*args, **kwargs)
        self.tuple_mode = any(param.device.type == 'mps'
                              for pg in self.param_groups for param in pg['params'])
        if self.tuple_mode and not self.matrix_free_normal:
            raise ValueError("MPS optimization requires matrix_free_normal=True.")
        if not self.matrix_free_normal:
            from ..sparse.spgemm import CuSparse
            self.mm = CuSparse()

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            weight = self.weight if weight is None else weight
            R = list(self.model(input))
            R = R[0]
            if self.tuple_mode:
                J = materialize_jacobian_components(
                    jacobian_components(R, pg['params']), as_tuple=True,
                )
            else:
                J = jacobian(R, pg['params'])
            if len(J) != len(pg['params']) or any(block is None for block in J):
                raise ValueError("All optimized parameters must contribute to the residual.")
            if isinstance(R, TrackingTensor):
                R = R.tensor()
            if self.tuple_mode:
                J = torch.cat(J, dim=1)
            else:
                J = torch.cat([j.to_sparse_coo() for j in J], dim=-1).to_sparse_csr()

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            self.reject_count = 0

            if self.matrix_free_normal:
                # for A(x) = J.T @ (J @ x) + (s * diag - raw_diag) * x
                # can we avoid keeping both diag and raw_diag? 
                raw_diag = NormalMatVec._compute_diag(J)
                diag = raw_diag.clamp(min=pg['min'], max=pg['max'])
                A = NormalMatVec(J, damping=0.0, diag=diag, raw_diag=raw_diag)
                rhs = -(A._get_Jt() @ R.view(-1, 1))
                diag_scale = 1.0
            else:
                J_T = J.mT.to_sparse_csr()
                rhs = -J_T @ R.view(-1, 1)
                A = self.mm(J_T, J)
                del J_T
                diagonal_op_(A, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))

            while self.last <= self.loss:
                if self.matrix_free_normal:
                    diag_scale *= 1.0 + pg['damping']
                    A.set_damping(diag_scale - 1.0)
                else:
                    diagonal_op_(A, op=partial(torch.mul, other=1+pg['damping']))
                try:
                    D = self.solver(A, rhs)
                except Exception as e:
                    print(e, "\nLinear solver failed. Breaking optimization step...")
                    break
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])
                self.strategy.update(pg, last=self.last, loss=self.loss, J=J, D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
        return self.loss

    def update_parameter(self, params, step):
        with torch.no_grad():
            numels = []
            for param in params:
                if param.requires_grad:
                    numels.append(torch.Size(parameter_update_shape(param)).numel())
            steps = step.split(numels)
            for (param, d) in zip(params, steps):
                if param.requires_grad:
                    step_view = d.view(parameter_update_shape(param))
                    if getattr(param, 'trim_SE3_grad', False):
                        # Update the base Tensor instead of assigning through a
                        # TrackingTensor/LieTensor view. Mixed-subclass
                        # __setitem__ dispatch is not traceable by Dynamo.
                        param_tensor = torch.Tensor(param)
                        updated_pose = (
                            pp.se3(step_view[..., :6]).Exp()
                            * pp.SE3(param_tensor[..., :7])
                        )
                        param_tensor[..., :7].copy_(updated_pose.tensor())
                        if param.shape[-1] > 7:
                            param_tensor[..., 7:].add_(step_view[..., 6:])
                    else:
                        param.add_(step_view)


class Schur(LM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tuple_mode:
            raise NotImplementedError("Schur requires native sparse tensors; use LM with PCG for tuple/MPS optimization.")

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            self.reject_count = 0
            weight = self.weight if weight is None else weight
            R = self.model(input, target)[0]
            J = jacobian(R, pg['params'])

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            J0wp = torchbsr2wp(J[0])
            J1wp = torchbsr2wp(J[1])
            J0twp = sparse.bsr_transposed(J0wp)
            J1twp = sparse.bsr_transposed(J1wp)
            U = sparse.bsr_mm(J0twp, J0wp)
            V = sparse.bsr_mm(J1twp, J1wp)

            if self.matrix_free_normal:
                del J0twp, J1twp
            else:
                W = sparse.bsr_mm(J0twp, J1wp)
                Wt = sparse.bsr_transposed(W)
                del J0twp, J1twp

            Upt = wp2torchbsr(U)
            Vpt = wp2torchbsr(V)
            diagonal_op_(Upt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            diagonal_op_(Vpt, op=partial(torch.clamp_, min=pg['min'], max=pg['max']))
            R_flat = R.reshape(-1).contiguous()
            Rwp = format_vec_for_bsr(R_flat, (J0wp.block_shape[1], J0wp.block_shape[0]))
            Ic = sparse.bsr_mv(J0wp, Rwp, alpha=-1.0, transpose=True)
            Ip = sparse.bsr_mv(J1wp, Rwp, alpha=-1.0, transpose=True)
            rhs_c = wp.empty_like(Ic)
            rhs_p = wp.empty_like(Ip)
            scratch_pts2 = wp.empty_like(Ip)

            if self.matrix_free_normal:
                scratch_obs = wp.empty_like(Rwp)
                scratch_pts = wp.empty_like(Ip)

            solver_tol = getattr(self.solver, "tol", None)
            solver_maxiter = getattr(self.solver, "maxiter", 0) or 0

            while self.last <= self.loss:
                damp = partial(torch.mul, other=1+pg['damping'])
                diagonal_op_(Upt, op=damp)
                diagonal_op_(Vpt, op=damp)

                V_i = torchbsr2wp(inv_op(Vpt))

                if self.matrix_free_normal:
                    def schur_matvec(x, y, z, alpha, beta, _V_i=V_i):
                        sparse.bsr_mv(J0wp, x, y=scratch_obs, beta=0.0)
                        sparse.bsr_mv(J1wp, scratch_obs, y=scratch_pts, beta=0.0, transpose=True)
                        sparse.bsr_mv(_V_i, scratch_pts, y=scratch_pts2, beta=0.0)
                        sparse.bsr_mv(J1wp, scratch_pts2, y=scratch_obs, beta=0.0)
                        if z.ptr != y.ptr and beta != 0.0:
                            wp.copy(src=y, dest=z)
                        sparse.bsr_mv(J0wp, scratch_obs, y=z, alpha=-alpha, beta=beta, transpose=True)
                        sparse.bsr_mv(U, x, y=z, alpha=alpha, beta=1.0)

                    schur_op = linear.LinearOperator(
                        shape=U.shape, dtype=U.values.dtype, device=U.device,
                        matvec=schur_matvec,
                    )
                    schur_M = linear.preconditioner(U)

                    wp.copy(src=Ic, dest=rhs_c)
                    sparse.bsr_mv(V_i, Ip, y=scratch_pts2, beta=0.0)
                    sparse.bsr_mv(J1wp, scratch_pts2, y=scratch_obs, beta=0.0)
                    sparse.bsr_mv(J0wp, scratch_obs, y=rhs_c, alpha=-1.0, beta=1.0, transpose=True)
                else:
                    WV_i = sparse.bsr_mm(W, V_i)
                    WVi_Wt = sparse.bsr_mm(WV_i, Wt)
                    U_clone_torch = torch.sparse_bsr_tensor(
                        crow_indices=Upt.crow_indices().clone(),
                        col_indices=Upt.col_indices().clone(),
                        values=Upt.values().clone(),
                        size=Upt.shape, device=Upt.device, dtype=Upt.dtype,
                    )
                    schur_op = sparse.bsr_axpy(WVi_Wt, torchbsr2wp(U_clone_torch), alpha=-1.0)
                    schur_M = linear.preconditioner(schur_op)
                    wp.copy(src=Ic, dest=rhs_c)
                    sparse.bsr_mv(V_i, Ip, y=scratch_pts2, beta=0.0)
                    sparse.bsr_mv(W, scratch_pts2, y=rhs_c, alpha=-1.0, beta=1.0)

                D_c = wp.zeros_like(rhs_c)
                linear.cg(
                    A=schur_op,
                    b=rhs_c,
                    x=D_c,
                    tol=solver_tol,
                    maxiter=solver_maxiter,
                    M=schur_M,
                )

                
                wp.copy(src=Ip, dest=rhs_p)
                
                if self.matrix_free_normal:
                    sparse.bsr_mv(J0wp, D_c, y=scratch_obs, beta=0.0)
                    sparse.bsr_mv(J1wp, scratch_obs, y=rhs_p,
                                  alpha=-1.0, beta=1.0, transpose=True)
                else:
                    sparse.bsr_mv(Wt, D_c, y=rhs_p, alpha=-1.0, beta=1.0)

                D_p = wp.zeros_like(rhs_p)
                linear.cg(
                    A=V,
                    b=rhs_p,
                    x=D_p,
                    tol=solver_tol,
                    maxiter=solver_maxiter,
                    M=linear.preconditioner(V),
                )

                D_c_t = wp.to_torch(D_c).flatten()
                D_p_t = wp.to_torch(D_p).flatten()
                D = torch.cat([D_c_t, D_p_t])
                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                print("Loss:", self.loss, "Last Loss:", self.last, "Reject Count:", self.reject_count, "Damping:", pg['damping'])

                self.strategy.update(
                    pg,
                    last=self.last,
                    loss=self.loss,
                    J=J,
                    Jwp=[J0wp, J1wp],
                    D=[D_c_t, D_p_t],
                    R=R_flat.view(-1, 1),
                )

                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break

        return self.loss