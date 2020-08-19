import cupy as cp
import numpy as np
from numba import vectorize as nvectorize

from pysc.utils import ARRAY

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


def _corr_matrix(x: ARRAY, y: ARRAY):
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


def find_corr_mat(stream1: ARRAY, stream2: ARRAY, device):
    corr_func = [_corr_matrix_cuda, _corr_matrix][device == "cpu"]
    ax = None if (tmp := stream1.ndim) == 0 else tmp - 1

    a, b, c, d = corr_func(stream1, stream2)

    a = a.sum(axis=ax, dtype=np.uint32)
    b = b.sum(axis=ax, dtype=np.uint32)
    c = c.sum(axis=ax, dtype=np.uint32)
    d = d.sum(axis=ax, dtype=np.uint32)

    return a, b, c, d


def sc_corr(a: ARRAY, b: ARRAY, c: ARRAY, d: ARRAY, n: int, device):
    assert device in ("cpu", "gpu")

    if device == "cpu":
        out = sc_corr_np(a, b, c, d, n)
    else:
        out = sc_corr_cp(a, b, c, d, n)

    return out


# Improves time from 18 us to 1.67 us
@nvectorize("float32(uint32, uint32, uint32, uint32, int32)", nopython=True)
def sc_corr_np(a: ARRAY, b: ARRAY, c: ARRAY, d, n: int):
    out = 1.0 * a * d
    out -= b * c

    if out == 0:
        return out

    apb = a + b
    apc = a + c
    ab_into_ac = apb * apc

    if out > 0:
        denom = n * np.minimum(apb, apc)
        denom -= ab_into_ac
    else:
        denom = ab_into_ac
        denom -= n * np.maximum(a - d, 0)

    out /= denom

    return out


# Improved time from 565 us to 11.2 us
sc_corr_cp = cp.ElementwiseKernel(
    "uint32 a, uint32 b, uint32 c, uint32 d, int32 n",
    "float32 out",
    """
        // Might have to change to unsigned long if required later

        unsigned int apb, apc, ab_into_ac;
        float denom;

        out = 1.0 * a * d;
        out -= b*c;

        if(out != 0){
            apb = a + b;
            apc = a + c;
            ab_into_ac = apb * apc;

            if(out > 0){
                denom = n * min(apb, apc);
                denom -= ab_into_ac;
            }
            else{
                denom = ab_into_ac;
                denom -= n * max(a - d, 0);
            }

            out /= denom;
        }
    """,
    "sc_corr_cp",
    reduce_dims=True,
)


def pearson(a: ARRAY, b: ARRAY, c: ARRAY, d: ARRAY):
    xp = cp.get_array_module(a)

    numer = (a * d - b * c).astype(np.float32)
    denom = xp.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    numer /= denom

    return numer
