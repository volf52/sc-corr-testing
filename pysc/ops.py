from numba import vectorize as nvectorize
import numpy as np
import cupy as cp

UNARY_OPS_SIGNATURE = "boolean(boolean)"
BINARY_OPS_SIGNATURE = "boolean(boolean, boolean)"

# These functions can be used directly on SCStream objects, thanks to the __array__ method
# and are a liiiitle bit faster than the normal np logical methods

@nvectorize(BINARY_OPS_SIGNATURE, nopython=True)
def and_it(x, y):
    return x & y


@nvectorize(BINARY_OPS_SIGNATURE, nopython=True)
def or_it(x, y):
    return x | y


@nvectorize(UNARY_OPS_SIGNATURE, nopython=True)
def not_it(x):
    return ~x


def corr_matrix(x, y):
    a = and_it(x, y)
    b = not_it(or_it(x, y))
    c = and_it(not_it(x), y)
    d = and_it(x, not_it(y))

    return a,b,c,d


corr_matrix_cuda = cp.ElementwiseKernel(
    "bool x, bool y",
    "bool a, bool b, bool c, bool d",
    "a = x & y;b = x & ~y;c = ~x & y;d = ~(x | y)",
    "find_corr_mat",
)
