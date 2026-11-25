"""Tensor interfaces for component-backed BSR matrices and their concatenation."""

import torch

from . import tuple_ops


class _SparseTensor(torch.Tensor):
    # Dense storage belongs to the components, not the wrapper. Returning plain
    # tensors from matmul is essential: Tensor's default torch_function would
    # otherwise wrap the result without initializing our component attributes.
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def _new(cls, size, reference):
        return torch.Tensor._make_wrapper_subclass(
            cls, tuple(size), dtype=reference.dtype, device=reference.device,
            requires_grad=False,
        )

    def __repr__(self):
        return f"{type(self).__name__}(size={tuple(self.shape)}, dtype={self.dtype}, device='{self.device}')"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        aten = torch.ops.aten
        if func in (aten.mv.default, aten.mv.out, aten.mm.default, aten.mm.out):
            matrix, x = args
            if isinstance(matrix, _SparseTensor) and not isinstance(x, _SparseTensor):
                return matrix.matvec(x, out=kwargs.get("out"))
        elif func is aten.add.Tensor:
            left, right = args
            if isinstance(left, BSRTensor) and isinstance(right, BSRTensor):
                return left._add(right, kwargs.get("alpha", 1))
        elif func is aten.cat.default:
            return SparseCat(args[0], dim=args[1] if len(args) > 1 else 0)
        elif func in (aten.t.default, aten.mT.default, aten.transpose.int, aten.permute.default):
            matrix = args[0]
            if func is aten.transpose.int:
                if any(dim not in (-2, -1, 0, 1) for dim in args[1:]):
                    raise IndexError("Sparse matrix transpose dimensions must be 0 or 1")
                if args[1] % 2 == args[2] % 2:
                    return matrix.detach()
            if func is aten.permute.default and tuple(args[1]) not in ((1, 0), (-1, -2)):
                if tuple(args[1]) in ((0, 1), (-2, -1)):
                    return matrix.detach()
                raise NotImplementedError("Sparse matrix permutation must swap its two dimensions")
            return matrix._transpose()
        elif func in (aten.detach.default, aten.clone.default, aten._to_copy.default):
            return args[0]._map_components(func, kwargs)
        raise NotImplementedError(f"{func} is not supported for {cls.__name__}")

    def matvec(self, x, out=None):
        if x.ndim not in (1, 2) or x.shape[0] != self.shape[1]:
            raise ValueError("x must be a vector or matrix with compatible leading dimension")
        if x.device != self.device or x.dtype != self.dtype:
            raise ValueError("x must have the matrix's device and dtype")
        shape = (self.shape[0], *x.shape[1:])
        if out is None:
            out = x.new_empty(shape)
        elif out.shape != shape or out.device != self.device or out.dtype != self.dtype:
            raise ValueError("out must have the result's shape, device, and dtype")
        elif not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if torch._C._overlaps(x, out):
            x = x.clone()
        if x.ndim == 2 and x.shape[1] != 1:
            if x.shape[1]:
                out.copy_(torch.stack([self.matvec(column) for column in x.unbind(1)], dim=1))
        else:
            self._mv(x, out, beta=0)
        return out


class BSRTensor(_SparseTensor):
    """BSR matrix backed by dense tensors, including on MPS.

    ``size`` is mandatory so empty rows and unused trailing columns survive.
    The wrapper is intended for fixed solver linearizations, not autograd.
    Components are exposed through the native sparse Tensor accessors. Stored
    transposes are cached and refreshed after ordinary component mutations.
    Transposed component arrays own their storage; update the source matrix's
    components to change the matrix rather than writing to its transpose.
    """

    @staticmethod
    def __new__(cls, crow_indices, col_indices, values, *, size, is_coalesced=False):
        components = (crow_indices, col_indices, values)
        tuple_ops._validate(components, size)
        result = cls._new(cls, size, values)
        result._components = components
        result._source = None
        result._transposed_components = None
        result._source_versions = None
        result._canonical_versions = result._index_versions() if is_coalesced else None
        return result

    @property
    def components(self):
        if self._source is not None:
            source = self._source.components
            versions = tuple(t._version for t in source)
            if self._source_versions != versions:
                cached = self._source._transposed_components
                if cached is None or cached[0] != versions:
                    cached = (versions, tuple_ops.bsr_transposed(source, size=self._source.shape))
                    self._source._transposed_components = cached
                self._components = cached[1]
                self._source_versions = versions
                self._canonical_versions = (
                    self._index_versions() if self._source.is_coalesced() else None
                )
        return self._components

    def crow_indices(self):
        return self.components[0]

    def col_indices(self):
        return self.components[1]

    def values(self):
        return self.components[2]

    def _index_versions(self):
        return tuple(t._version for t in self._components[:2])

    def is_coalesced(self):
        self.components
        return self._canonical_versions == self._index_versions()

    def coalesce(self):
        return self if self.is_coalesced() else self._combine([self])

    @staticmethod
    def _combine(matrices, alpha=1):
        # PyTorch implementation of Warp's duplicate block accumulation semantics:
        # https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L1226-L1273
        # https://github.com/NVIDIA/warp/blob/release-1.19/warp/_src/sparse.py#L1417-L1713
        first = matrices[0]
        parts = [matrix.components for matrix in matrices]
        columns = torch.cat([part[1] for part in parts])
        values = torch.cat([part[2] if i == 0 else part[2] * alpha
                            for i, part in enumerate(parts)])
        if not columns.numel():
            return BSRTensor(parts[0][0].to(columns.dtype), columns, values,
                             size=first.shape, is_coalesced=True)
        rows = torch.cat([
            torch.searchsorted(crow[1:], torch.arange(value.shape[0], device=value.device,
                                                     dtype=crow.dtype), right=True)
            for crow, _, value in parts
        ])
        ncols = first.shape[1] // values.shape[-1]
        keys = rows.long() * ncols + columns.long()
        order = torch.argsort(keys, stable=True)
        keys, values = keys[order], values[order]
        starts = torch.cat([torch.ones(1, dtype=torch.bool, device=keys.device), keys[1:] != keys[:-1]])
        unique_keys = keys[starts]
        groups = starts.long().cumsum(0) - 1
        combined = values.new_zeros((unique_keys.numel(), *values.shape[1:]))
        combined.index_add_(0, groups, values)
        boundaries = torch.arange(first.shape[0] // values.shape[-2] + 1,
                                  device=columns.device, dtype=torch.int64)
        crow = torch.searchsorted(unique_keys // ncols, boundaries).to(columns.dtype)
        col = (unique_keys % ncols).to(columns.dtype)
        return BSRTensor(crow, col, combined, size=first.shape, is_coalesced=True)

    def _add(self, other, alpha):
        if self.shape != other.shape or self.values().shape[1:] != other.values().shape[1:]:
            raise ValueError("BSR addition requires matching matrix and block shapes")
        if self.device != other.device or self.dtype != other.dtype:
            raise ValueError("BSR addition requires matching devices and dtypes")
        return self._combine([self, other], alpha)

    def _transpose(self):
        if self._source is not None:
            return self._source.detach()
        result = self._new(type(self), self.shape[::-1], self.values())
        result._source = self
        result._components = None
        result._transposed_components = None
        result._source_versions = None
        result._canonical_versions = None
        return result

    def _mv(self, x, out, beta):
        components = self.components
        # Preserve the measured MPS dispatch for cached camera transposes.
        scalar = (self._source is not None and self.device.type == "mps"
                  and components[2].shape[-2:] == (9, 2)
                  and components[0].numel() - 1 >= 1024)
        tuple_ops.bsr_mv(components, x, y=out, beta=beta, size=self.shape,
                         tile_size=-1 if scalar else 0)

    def normal_diagonal(self):
        return tuple_ops.bsr_diagonal(self.coalesce().components, size=self.shape)

    def _map_components(self, func, kwargs):
        if func is torch.ops.aten.detach.default and self._source is not None:
            return self._source.detach().mT
        if func is torch.ops.aten.detach.default:
            # Dispatch runs below view tracking. Reenter it for dense detach so
            # aliases share version counters and invalidate transpose caches.
            with torch.overrides.enable_reentrant_dispatch():
                parts = [func(tensor) for tensor in self.components]
            return BSRTensor(*parts, size=self.shape, is_coalesced=self.is_coalesced())
        parts = []
        for index, tensor in enumerate(self.components):
            options = dict(kwargs)
            if func is torch.ops.aten._to_copy.default and index != 2:
                options.pop("dtype", None)
            parts.append(func(tensor, **options))
        return BSRTensor(*parts, size=self.shape, is_coalesced=self.is_coalesced())

    def to_dense(self):
        crow, col, values = self.components
        br, bc = values.shape[-2:]
        rows = torch.searchsorted(crow[1:], torch.arange(col.numel(), device=self.device,
                                                        dtype=crow.dtype), right=True)
        row = rows[:, None, None] * br + torch.arange(br, device=self.device)[None, :, None]
        column = col[:, None, None] * bc + torch.arange(bc, device=self.device)[None, None, :]
        result = values.new_zeros(self.numel())
        result.index_add_(0, (row * self.shape[1] + column).reshape(-1).long(), values.reshape(-1))
        return result.reshape(self.shape)


class SparseCat(_SparseTensor):
    """A tensor concatenation retaining each BSR matrix's own block size."""

    @staticmethod
    def __new__(cls, blocks, dim=1):
        blocks = tuple(blocks)
        if not blocks or any(not isinstance(block, _SparseTensor) for block in blocks):
            raise TypeError("Sparse concatenation requires BSR tensors")
        if dim not in (-2, -1, 0, 1):
            raise IndexError("Sparse concatenation dimension must be 0 or 1")
        dim %= 2
        first = blocks[0]
        if any(block.shape[1 - dim] != first.shape[1 - dim] for block in blocks):
            raise ValueError("Sparse concatenation requires matching non-concatenated dimensions")
        if any(block.device != first.device or block.dtype != first.dtype for block in blocks):
            raise ValueError("Sparse concatenation requires matching devices and dtypes")
        shape = list(first.shape)
        shape[dim] = sum(block.shape[dim] for block in blocks)
        result = cls._new(cls, shape, first)
        result.blocks, result.cat_dim = blocks, dim
        return result

    def _transpose(self):
        return SparseCat([block.mT for block in self.blocks], 1 - self.cat_dim)

    def _mv(self, x, out, beta):
        widths = [block.shape[self.cat_dim] for block in self.blocks]
        if self.cat_dim == 1:
            for block, part in zip(self.blocks, x.split(widths, dim=0)):
                block._mv(part, out, beta)
                beta = 1
        else:
            for block, part in zip(self.blocks, out.split(widths, dim=0)):
                block._mv(x, part, beta)

    def normal_diagonal(self):
        diagonals = [block.normal_diagonal() for block in self.blocks]
        return torch.cat(diagonals) if self.cat_dim == 1 else sum(diagonals)

    def _map_components(self, func, kwargs):
        return SparseCat([func(block, **kwargs) for block in self.blocks], self.cat_dim)

    def to_dense(self):
        return torch.cat([block.to_dense() for block in self.blocks], dim=self.cat_dim)
