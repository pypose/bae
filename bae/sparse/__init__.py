from importlib import import_module

from .py_ops import *
from .tuple_ops import bsr_mv, bsr_diagonal, bsr_transposed
from .tensor import BSRTensor, SparseCat
from .warp_wrappers import *


def __getattr__(name):
    module = {'gebsr2csr': '.conversion', 'CuDirectSparseSolver': '.solve',
              'cusolvesp': '.solve'}.get(name)
    if module is not None:
        return getattr(import_module(module, __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
