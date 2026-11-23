
from typing import NamedTuple, Optional
import warnings

import pypose as pp
import torch
from torch.func import jacrev

from ..utils.config import jacobian_tuple_enabled
from ..utils.parameter import trim_parameter_jacobian_values
from ..sparse.tensor import BSRTensor
from .function import _INDEXED_TRACE_TAG


class BSRJacobianData(NamedTuple):
    """Dense tensor components of a BSR Jacobian.

    PyTorch's FakeTensor/AOTAutograd stack cannot currently represent a sparse
    BSR graph output. Returning its component tensors keeps Jacobian traversal
    and value computation inside the compiled graph; materialization into a
    sparse Tensor is a zero-copy eager wrapper operation after the call.
    """

    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    values: torch.Tensor
    size: tuple[int, int]


def _is_indexed_trace_arg(arg) -> bool:
    return isinstance(arg, tuple) and len(arg) == 3 and arg[0] == _INDEXED_TRACE_TAG


def _materialize_trace_arg(arg):
    #  produces an ordinary aten.index; it enables Inductor to choose fusion but does not force it.
    if _is_indexed_trace_arg(arg):
        _, tensor, index = arg
        return tensor[index]
    return arg


def _crow_to_row_indices(crow_indices: torch.Tensor) -> torch.Tensor:
    counts = (crow_indices[1:] - crow_indices[:-1]).to(torch.int64)
    row_ids = torch.arange(counts.numel(), device=crow_indices.device, dtype=torch.int64)
    return torch.repeat_interleave(row_ids, counts)


def _row_indices_to_crow(row_indices: torch.Tensor, n_rows: int, dtype: torch.dtype) -> torch.Tensor:
    row_boundaries = torch.arange(
        n_rows + 1,
        device=row_indices.device,
        dtype=row_indices.dtype,
    )
    return torch.searchsorted(row_indices, row_boundaries).to(dtype)


def _slice_upstream_bsr_columns(
    upstream: torch.Tensor,
    col_start: int,
    col_end: int,
    out_cols_blocks: int,
) -> torch.Tensor:
    crow = upstream.crow_indices()
    col = upstream.col_indices()
    values = upstream.values()

    n_rows_blocks = crow.numel() - 1
    row_indices = _crow_to_row_indices(crow)
    mask = (col >= col_start) & (col < col_end)

    row_f = row_indices[mask]
    col_f = (col[mask] - col_start).to(col.dtype)
    val_f = values[mask]

    crow_f = _row_indices_to_crow(row_f, n_rows_blocks, dtype=crow.dtype)
    block_cols = values.shape[-1] if values.ndim > 1 else 1
    new_size = (upstream.shape[0], out_cols_blocks * block_cols)
    return torch.sparse_bsr_tensor(
        crow_indices=crow_f,
        col_indices=col_f,
        values=val_f,
        size=new_size,
        device=upstream.device,
        dtype=upstream.dtype,
    )


def _slice_upstream_tuple_columns(
    indices: Optional[torch.Tensor],
    values: torch.Tensor,
    col_start: int,
    col_end: int,
    out_cols_blocks: int,
) -> torch.Tensor:
    n_rows_blocks = values.shape[0]
    dm = values.shape[-2] if values.ndim > 1 else 1
    dn = values.shape[-1] if values.ndim > 1 else 1

    if indices is None:
        indices = torch.arange(n_rows_blocks, device=values.device, dtype=torch.int32)
    elif indices.device != values.device:
        indices = indices.to(device=values.device)

    mask = (indices >= col_start) & (indices < col_end)
    crow = torch.zeros(n_rows_blocks + 1, device=values.device, dtype=torch.int32)
    crow[1:] = torch.cumsum(mask.to(crow.dtype), dim=0)
    col_f = (indices[mask] - col_start).to(device=values.device, dtype=torch.int32)
    val_f = values[mask]

    return torch.sparse_bsr_tensor(
        crow_indices=crow,
        col_indices=col_f,
        values=val_f,
        size=(n_rows_blocks * dm, out_cols_blocks * dn),
        device=values.device,
        dtype=values.dtype,
    )


def construct_sbt(jac_from_vmap, num, index: Optional[torch.Tensor], type=torch.sparse_bsc):
    if index is None:
        index = torch.arange(num, device=jac_from_vmap.device, dtype=torch.int32)
    n = index.shape[0] # num 2D points
    block_shape = jac_from_vmap.shape[1:]
    idx_dtype = index.dtype

    if type == torch.sparse_bsc:
        i = torch.stack([torch.arange(n, dtype=index.dtype, device=index.device), index])
        dummy_val = torch.arange(n, device=index.device, dtype=torch.int32)
        dummy_coo = torch.sparse_coo_tensor(i, dummy_val, size=(n, num), device=index.device, dtype=torch.int32)
        dummy_csc = dummy_coo.coalesce().to_sparse_csc()
        return torch.sparse_bsc_tensor(ccol_indices=dummy_csc.ccol_indices().to(torch.int32), 
                                    row_indices=dummy_csc.row_indices().to(torch.int32),
                                    values = jac_from_vmap[dummy_csc.values()],
                                    size = (n * block_shape[0], num * block_shape[1]),
                                    device=index.device, dtype=jac_from_vmap.dtype)
    elif type == torch.sparse_bsr:
        return torch.sparse_bsr_tensor(col_indices=index, 
                                    crow_indices=torch.arange(n + 1, device=index.device, dtype=idx_dtype),
                                    values = jac_from_vmap,
                                    size = (n * block_shape[0], num * block_shape[1]))

def _clear_jactrace(output, params):
    seen = set()
    stack = [output, *params]
    while stack:
        tensor = stack.pop()
        if not isinstance(tensor, torch.Tensor) or id(tensor) in seen:
            continue
        seen.add(id(tensor))

        if hasattr(tensor, 'jactrace'):
            delattr(tensor, 'jactrace')

        if not hasattr(tensor, 'optrace'):
            continue

        trace = tensor.optrace
        op = trace[0]
        if op == 'map':
            args = trace[2]
            for arg in args:
                if _is_indexed_trace_arg(arg):
                    stack.append(arg[1])
                elif isinstance(arg, torch.Tensor):
                    stack.append(arg)
        elif op == 'index':
            arg = trace[2]
            if isinstance(arg, torch.Tensor):
                stack.append(arg)
        elif op == 'cat':
            tracked = trace[2]  # ((start, end, arg), ...)
            stack.extend(item[2] for item in tracked if isinstance(item[2], torch.Tensor))


def amend_trace(arg, jac_trace: tuple):
    if hasattr(arg, 'jactrace'):  # convert to sparse_bsr needed for accumulation
        if type(arg.jactrace) is tuple and type(jac_trace) is tuple:
            if arg.jactrace[0] is None and jac_trace[0] is None:
                arg.jactrace = (None, arg.jactrace[1] + jac_trace[1])
                return 
        if type(arg.jactrace) is tuple:
            arg.jactrace = construct_sbt(arg.jactrace[1], arg.shape[0], arg.jactrace[0], type=torch.sparse_bsr)
        if type(jac_trace) is tuple:
            jac_trace = construct_sbt(jac_trace[1], arg.shape[0], jac_trace[0], type=torch.sparse_bsr)
        arg.jactrace = arg.jactrace + jac_trace
    else:
        arg.jactrace = jac_trace

def update_from_trace(bsrt: torch.Tensor, arg, new_col: Optional[torch.Tensor]=None, new_val: Optional[torch.Tensor]=None):
    if new_col is not None:
        jac_trace = torch.sparse_bsr_tensor(
                col_indices=new_col, 
                crow_indices=bsrt.crow_indices(),
                values=bsrt.values(),
                size=(bsrt.shape[0], arg.shape[0] * bsrt.values().shape[-1]),
                device=bsrt.device,
            )
    if new_val is not None:
        jac_trace = torch.sparse_bsr_tensor(
                col_indices=bsrt.col_indices(), 
                crow_indices=bsrt.crow_indices(),
                values=new_val,
                size=(bsrt.shape[0], arg.shape[0] * new_val.shape[-1]),
                device=bsrt.device,
            )
    return jac_trace


def _vmap_in_dims(args):
    """Map Tensor inputs over dim 0 while treating non-Tensors as constants."""
    return tuple(0 if isinstance(arg, torch.Tensor) else None for arg in args)


def _restore_lie_inputs(func, args):
    """Reconstruct LieTensor inputs after functorch strips subclass metadata."""
    ltypes = tuple(
        arg.ltype if isinstance(arg, pp.LieTensor) else None for arg in args
    )
    if not any(ltype is not None for ltype in ltypes):
        return func

    def wrapped(*values):
        values = tuple(
            pp.LieTensor(value, ltype=ltype) if ltype is not None else value
            for value, ltype in zip(values, ltypes)
        )
        return func(*values)

    return wrapped


def _compose_map_trace(jac_block, upstream):
    """Compose a local block Jacobian with an optional downstream trace."""
    if upstream is None:
        rows = torch.arange(
            jac_block.shape[0], device=jac_block.device, dtype=torch.int64
        )
        return (rows, None, jac_block)

    rows, positions, upstream_values = upstream
    if positions is not None:
        jac_block = jac_block[positions]
    return (rows, positions, upstream_values @ jac_block)


def _compose_index_trace(index, upstream, output):
    """Propagate a block trace through an observation-wise index operation."""
    if upstream is None:
        rows = torch.arange(
            output.shape[0], device=output.device, dtype=torch.int64
        )
        if output.ndim == 1:
            values = torch.ones(
                (output.shape[0], 1, 1),
                device=output.device,
                dtype=output.dtype,
            )
        else:
            block_dim = output.shape[-1]
            eye = torch.eye(block_dim, device=output.device, dtype=output.dtype)
            values = eye.unsqueeze(0).repeat(output.shape[0], 1, 1)
        return (rows, index, values)

    rows, positions, values = upstream
    if positions is not None:
        index = index[positions]
    return (rows, index, values)


def _parameter_index(tensor, params):
    """Resolve a traced leaf to its requested parameter using object identity."""
    for index, param in enumerate(params):
        if tensor is param:
            return index
    return None


def _append_functional_trace(tensor, trace, params, contributions):
    """Continue through a traced node or record a parameter contribution."""
    if isinstance(tensor, torch.Tensor) and hasattr(tensor, "optrace"):
        _functional_backward(tensor, trace, params, contributions)
        return

    index = _parameter_index(tensor, params)
    if index is not None:
        contributions[index].append(trace)


def _functional_backward(output, upstream, params, contributions):
    """Compile-safe sparse-Jacobian traversal without Tensor attribute mutation.

    Dynamo unrolls the Python control flow over the immutable ``optrace`` tree.
    Runtime Jacobian state is represented only by tensor tuples, rather than by
    adding and deleting ``jactrace`` attributes on Tensor/FakeTensor objects.
    """
    output_trace = output.optrace
    op = output_trace[0]

    if op == "map":
        func = output_trace[1]
        trace_args = output_trace[2]
        #   This design avoids retaining the intermediate indexed TrackingTensor in the immutable trace tree. Under torch.compile, the
        #   reconstructed cameras[camera_indices] is captured as an ordinary graph operation, so Inductor may fuse its indexed loads with
        #   the derivative computation.
        args = tuple(_materialize_trace_arg(arg) for arg in trace_args)
        argnums = tuple(
            index
            for index, arg in enumerate(args)
            if hasattr(arg, "optrace") or isinstance(arg, torch.nn.Parameter)
        )
        if len(argnums) == 0:
            return

        jac_blocks = torch.vmap(
            jacrev(_restore_lie_inputs(func, args), argnums=argnums),
            in_dims=_vmap_in_dims(args),
        )(*args)

        for jac_block, arg_index in zip(jac_blocks, argnums):
            assert jac_block.ndim == 3, "`func` is not properly vectorized in `torch.vmap`"
            trace = _compose_map_trace(jac_block, upstream)
            trace_arg = trace_args[arg_index]
            if _is_indexed_trace_arg(trace_arg):
                _, parent, index = trace_arg
                trace = _compose_index_trace(index, trace, args[arg_index])
                _append_functional_trace(parent, trace, params, contributions)
            else:
                _append_functional_trace(
                    args[arg_index], trace, params, contributions
                )
        return

    if op == "index":
        index, parent = output_trace[1], output_trace[2]
        trace = _compose_index_trace(index, upstream, output)
        _append_functional_trace(parent, trace, params, contributions)
        return

    if op == "cat":
        dim, tracked = output_trace[1], output_trace[2]
        if dim != 0:
            raise NotImplementedError("Only torch.cat(..., dim=0) is supported")

        if upstream is None:
            rows = torch.arange(
                output.shape[0], device=output.device, dtype=torch.int64
            )
            positions = rows
            if output.ndim == 1:
                values = torch.ones(
                    (output.shape[0], 1, 1),
                    device=output.device,
                    dtype=output.dtype,
                )
            else:
                block_dim = output.shape[-1]
                eye = torch.eye(
                    block_dim, device=output.device, dtype=output.dtype
                )
                values = eye.unsqueeze(0).repeat(output.shape[0], 1, 1)
        else:
            rows, positions, values = upstream
            if positions is None:
                positions = torch.arange(
                    values.shape[0], device=values.device, dtype=torch.int64
                )

        for start, end, arg in tracked:
            mask = (positions >= start) & (positions < end)
            trace = (
                rows[mask],
                positions[mask] - start,
                values[mask],
            )
            _append_functional_trace(arg, trace, params, contributions)
        return

    raise NotImplementedError(f"Unsupported trace operation: {op}")


def jacobian_components(output, params):
    """Return compile-safe BSR component tensors for each requested parameter.

    The result contains one or more components per parameter. Multiple
    components arise when distinct compute-graph branches contribute to the
    same parameter and are combined during eager materialization.
    """
    assert output.optrace[0] in (
        "map",
        "index",
        "cat",
    ), "Unsupported last operation in compute graph"
    contributions = [[] for _ in params]
    _functional_backward(output, None, params, contributions)

    result = []
    for param, traces in zip(params, contributions):
        param_components = []
        for rows, columns, values in traces:
            if columns is None:
                columns = torch.arange(
                    values.shape[0], device=values.device, dtype=torch.int64
                )
            values = trim_parameter_jacobian_values(
                param, values, block_indices=columns
            )
            crow = _row_indices_to_crow(
                rows,
                output.shape[0],
                dtype=columns.dtype,
            )
            param_components.append(
                BSRJacobianData(
                    crow,
                    columns,
                    values,
                    (
                        output.shape[0] * values.shape[-2],
                        param.shape[0] * values.shape[-1],
                    ),
                )
            )
        result.append(tuple(param_components))
    return tuple(result)


def materialize_jacobian_components(components, *, as_tuple: Optional[bool] = None):
    """Create sparse BSR tensors while preserving parameter alignment.

    An empty component group represents a parameter that does not contribute
    to the output and is materialized as ``None``.
    ``as_tuple`` selects tuple-backed :class:`BSRTensor`.
    """
    result = []
    for param_components in components:
        if len(param_components) == 0:
            result.append(None)
            continue
        sparse_components = [
            BSRTensor(*component[:3], size=component.size, is_coalesced=True)
            if as_tuple else torch.sparse_bsr_tensor(
                component.crow_indices,
                component.col_indices,
                component.values,
                size=component.size,
            )
            for component in param_components
        ]
        combined = sparse_components[0]
        for sparse_component in sparse_components[1:]:
            combined = combined + sparse_component
        result.append(combined)
    return result

def backward(output_, is_root=False):
    # For non-root recursion, no incoming trace means no contribution to
    # propagate. This avoids re-initializing identity traces on revisits.
    if (not is_root) and (not hasattr(output_, 'jactrace')):
        return

    output_trace = output_.optrace
    if output_trace[0] == 'map':
        func = output_trace[1]
        args = tuple(_materialize_trace_arg(arg) for arg in output_trace[2])
        argnums = tuple(idx for idx, arg in enumerate(args) if hasattr(arg, 'optrace') or isinstance(arg, torch.nn.Parameter))
        if len(argnums) == 0:
            warnings.warn("No upstream parameters to compute jacobian", stacklevel=2)
            return
        in_dims = _vmap_in_dims(args)
        with pp.retain_ltype():
            jac_blocks = torch.vmap(jacrev(func, argnums=argnums), in_dims=in_dims)(*args)
        for jacidx, argidx in enumerate(argnums):
            jac_block = jac_blocks[jacidx]
            arg = args[argidx]
            assert jac_block.ndim == 3, "`func` is not properly vectorized in `torch.vmap`"
            # TODO: perhaps flatten the jacobian block in the future
            if not hasattr(output_, 'jactrace'):  # check for upstream jacobian
                jac_trace = (None, jac_block)  # leave None for identity indices
            else:
                indices = None
                if type(output_.jactrace) is tuple:
                    indices = output_.jactrace[0]
                    jac_ustrm = output_.jactrace[1]
                elif isinstance(output_.jactrace, torch.Tensor) and output_.jactrace.layout == torch.sparse_bsr:
                    indices = output_.jactrace.col_indices()
                    jac_ustrm = output_.jactrace.values()

                if indices is not None:
                    jac_block = jac_block[indices]
                jac_block = jac_ustrm @ jac_block

                if type(output_.jactrace) is tuple:
                    jac_trace = (indices, jac_block)
                elif isinstance(output_.jactrace, torch.Tensor) and output_.jactrace.layout == torch.sparse_bsr:
                    jac_trace = update_from_trace(output_.jactrace, arg, new_val=jac_block)
            amend_trace(arg, jac_trace)
        # Recurse once per unique upstream tensor after all local contributions
        # have been accumulated.
        seen = set()
        for argidx in argnums:
            arg = args[argidx]
            if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
                arg_id = id(arg)
                if arg_id in seen:
                    continue
                seen.add(arg_id)
                backward(arg, is_root=False)

        # Consume intermediate trace to avoid re-propagating it when this node is
        # reached again from another downstream branch (e.g. two index ops on one cat).
        if hasattr(output_, 'jactrace'):
            delattr(output_, 'jactrace')


    elif output_trace[0] == 'index':
        index = output_trace[1]
        arg = output_trace[2]

        # If the last operation is indexing, there is no downstream map op to
        # populate Jacobian values. In this case, the Jacobian block values are
        # identity matrices placed at the indexed columns.
        if not hasattr(output_, 'jactrace'):
            if not is_root:
                return
            if output_.ndim == 1:
                eye_blocks = torch.ones((output_.shape[0], 1, 1), device=output_.device, dtype=output_.dtype)
            else:
                block_dim = output_.shape[-1]
                eye = torch.eye(block_dim, device=output_.device, dtype=output_.dtype)
                eye_blocks = eye.unsqueeze(0).repeat(output_.shape[0], 1, 1)
            output_.jactrace = (None, eye_blocks)

        if type(output_.jactrace) is tuple:
            if output_.jactrace[0] is not None:
                upstream_index = output_.jactrace[0]
                index = index[upstream_index]
                jac_trace = (index, output_.jactrace[1])
            elif output_.jactrace[0] is None:
                jac_trace = (index, output_.jactrace[1])
        elif type(output_.jactrace) is torch.Tensor and output_.jactrace.layout == torch.sparse_bsr:
            upstream_index = output_.jactrace.col_indices()
            index = index[upstream_index]
            jac_trace = update_from_trace(output_.jactrace, arg, new_col=index)
            
        amend_trace(arg, jac_trace)
        if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
            backward(arg, is_root=False)

        if hasattr(output_, 'jactrace'):
            delattr(output_, 'jactrace')

    elif output_trace[0] == 'cat':
        dim = output_trace[1]
        tracked = output_trace[2]  # ((start, end, arg), ...)
        if dim != 0:
            raise NotImplementedError("Only torch.cat(..., dim=0) is supported")

        if not hasattr(output_, 'jactrace'):
            if not is_root:
                return
            if output_.ndim == 1:
                eye_blocks = torch.ones((output_.shape[0], 1, 1), device=output_.device, dtype=output_.dtype)
            else:
                block_dim = output_.shape[-1]
                eye = torch.eye(block_dim, device=output_.device, dtype=output_.dtype)
                eye_blocks = eye.unsqueeze(0).repeat(output_.shape[0], 1, 1)
            output_.jactrace = (None, eye_blocks)

        upstream = output_.jactrace
        for start, end, arg in tracked:
            n = arg.shape[0]
            if type(upstream) is tuple:
                jac_trace = _slice_upstream_tuple_columns(
                    upstream[0], upstream[1], start, end, out_cols_blocks=n
                )
            elif isinstance(upstream, torch.Tensor) and upstream.layout == torch.sparse_bsr:
                jac_trace = _slice_upstream_bsr_columns(
                    upstream, start, end, out_cols_blocks=n
                )
            else:
                raise TypeError(f"Unsupported upstream jactrace type: {type(upstream)}")

            amend_trace(arg, jac_trace)
            if isinstance(arg, torch.Tensor) and hasattr(arg, 'optrace'):
                backward(arg, is_root=False)

        if hasattr(output_, 'jactrace'):
            delattr(output_, 'jactrace')


def jacobian(output, params):
    # TODO: should we deprecate this in favor of jacobian_components + materialize_jacobian_components?
    if jacobian_tuple_enabled():
        components = jacobian_components(output, params)
        return materialize_jacobian_components(components, as_tuple=True)
    assert output.optrace[0] in ('map', 'index', 'cat'), "Unsupported last operation in compute graph"
    _clear_jactrace(output, params)
    try:
        backward(output, is_root=True)
        res = []
        for param in params:
            if hasattr(param, 'jactrace'):
                if isinstance(param.jactrace, tuple):
                    indices, values = param.jactrace
                    values = trim_parameter_jacobian_values(param, values, block_indices=indices)
                    param.jactrace = (indices, values)
                elif isinstance(param.jactrace, torch.Tensor) and param.jactrace.layout == torch.sparse_bsr:
                    values = trim_parameter_jacobian_values(
                        param,
                        param.jactrace.values(),
                        block_indices=param.jactrace.col_indices(),
                    )
                    if values.shape != param.jactrace.values().shape:
                        param.jactrace = torch.sparse_bsr_tensor(
                            col_indices=param.jactrace.col_indices(),
                            crow_indices=param.jactrace.crow_indices(),
                            values=values,
                            size=(param.jactrace.shape[0], param.shape[0] * values.shape[-1]),
                            device=param.device,
                        )
                if type(param.jactrace) is tuple:
                    param.jactrace = construct_sbt(param.jactrace[1], param.shape[0], param.jactrace[0], type=torch.sparse_bsr)
                res.append(param.jactrace)
        return res
    finally:
        _clear_jactrace(output, params)
