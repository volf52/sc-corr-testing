from numba import vectorize as nvectorize
import numpy as np
import cupy as cp
from math import sqrt
from pysc.stream import SCStream

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


def _corr_matrix(x, y):
    a = and_it(x, y)
    b = and_it(x, not_it(y))
    c = and_it(not_it(x), y)
    d = not_it(or_it(x, y))

    return a, b, c, d


_corr_matrix_cuda = cp.ElementwiseKernel(
    "bool x, bool y",
    "bool a, bool b, bool c, bool d",
    "a = x & y;b = x & ~y;c = ~x & y;d = ~(x | y)",
    "find_corr_mat",
)


def find_corr_mat(stream1: SCStream, stream2: SCStream, device):
    corr_func = [_corr_matrix_cuda, _corr_matrix][device == "cpu"]
    ax = None if (tmp := stream1.ndim) == 0 else tmp - 1

    a, b, c, d = corr_func(stream1, stream2)
    a, b, c, d = a.sum(axis=ax), b.sum(axis=ax), c.sum(axis=ax), d.sum(axis=ax)

    return a, b, c, d


# Complete ndim implementation


def sc_corr_1d(a, b, c, d, n):
    # valid only for 1D
    numer = a * d - b * c
    apb = a + b
    apc = a + c
    a_b_into_a_c = apb * apc  # Common for both
    if numer > 0:  # ad > bc
        denom = n * min(apb, apc) - a_b_into_a_c
    else:
        denom = a_b_into_a_c - n * max(a - d, 0)

    return numer / denom


def pearson(a, b, c, d):
    numer = a * d - b * c
    denom = sqrt((a + b) * (a + c) * (b + d) * (c + d))
    return numer / denom
