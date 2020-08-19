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
    "a = x & y;b = x & !y;c = !x & y;d = !(x | y)",
    "find_corr_mat",
)


def find_corr_mat(stream1: SCStream, stream2: SCStream, device):
    corr_func = [_corr_matrix_cuda, _corr_matrix][device == "cpu"]
    ax = None if (tmp := stream1.ndim) == 0 else tmp - 1

    a, b, c, d = corr_func(stream1, stream2)
    a, b, c, d = a.sum(axis=ax), b.sum(axis=ax), c.sum(axis=ax), d.sum(axis=ax)

    return a, b, c, d


# TODO: Rewrite separately for np and cp with cp.EWK and numba

def sc_corr(a, b, c, d, n):
    # assumed ndim >= 2 for a,b,c,d

    xp = cp.get_array_module(a)

    numer = (a * d - b * c).astype(np.float32)
    apb = a + b
    apc = a + c
    a_b_into_a_c = apb * apc  # Common for both
    denom1 = n * xp.minimum(apb, apc) - a_b_into_a_c
    denom2 = a_b_into_a_c - n * xp.maximum(a - d, 0)

    denom = xp.where(numer > 0, denom1, denom2)
    numer /= denom

    return numer


def pearson(a, b, c, d):
    xp = cp.get_array_module(a)
    numer = (a * d - b * c).astype(np.float32)
    denom = xp.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    numer /= denom
    return numer
