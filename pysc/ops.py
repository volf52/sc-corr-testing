from typing import Tuple

import cupy as cp
import numpy as np
from numba import guvectorize, vectorize as nvectorize

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


def find_corr_mat(
    stream1: ARRAY, stream2: ARRAY, device
) -> Tuple[ARRAY, ARRAY, ARRAY, ARRAY]:
    corr_func = [_corr_matrix_cuda, _corr_matrix][device == "cpu"]
    ax = None if stream1.ndim == 0 else stream1.ndim - 1

    a, b, c, d = corr_func(stream1, stream2)

    a = a.sum(axis=ax, dtype=np.uint32)
    b = b.sum(axis=ax, dtype=np.uint32)
    c = c.sum(axis=ax, dtype=np.uint32)
    d = d.sum(axis=ax, dtype=np.uint32)

    return a, b, c, d


def sc_corr(a: ARRAY, b: ARRAY, c: ARRAY, d: ARRAY, n: int, device) -> ARRAY:
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


def pearson_corr(a: ARRAY, b: ARRAY, c: ARRAY, d: ARRAY, device) -> ARRAY:
    assert device in ("cpu", "gpu")

    if device == "cpu":
        out = pearson_np(a, b, c, d)
    else:
        out = pearson_cp(a, b, c, d)

    return out


# Improved: 8.85 us -> 1.05 us
@nvectorize("float32(uint32, uint32, uint32, uint32)", nopython=True)
def pearson_np(a, b, c, d):
    out = 1.0 * a * d
    out -= b * c

    if out == 0:
        return out

    denom = a + b
    denom *= a + c
    denom *= b + d
    denom *= c + d
    denom = np.sqrt(denom)

    out /= denom

    return out


# Improved: 123 us -> 10.5 us
pearson_cp = cp.ElementwiseKernel(
    "uint32 a, uint32 b, uint32 c, uint32 d",
    "float32 out",
    """
        // Might have to change here as well (to unsigned long) if required

        float denom;

        out = 1.0 * a * d;
        out -= b*c;

        if(out != 0){
            // denom = 1.0;
            denom = a + b;
            denom *= a + c;
            denom *= b + d;
            denom *= c + d;
            denom = sqrt(denom);

            out /= denom;
        }
    """,
    "pearson_cp",
    reduce_dims=True,
)

@guvectorize(
    ['void(boolean[:], boolean[:], int32, boolean[:], boolean[:])'],
    '(n),(n),()->(n),(n)', nopython=True
)
def _synchronize(x, y, n, outX, outY):
    s = 1
    for i in range(n):

        if x[i] == y[i]:
            outX[i] = outY[i] = x[i]

        elif s == 0:
            if x[i] and not y[i]:
                outX[i] = 1
                outY[i] = 0
            else:
                outX[i] = outY[i] = 1
                s = 1
        elif s == 1:
            if x[i] and not y[i]:
                outX[i] = outY[i] = 0
                s = 0
            else:
                outX[i] = outY[i] = 0
                s = 2

        elif s == 2:
            if x[i] and not y[i]:
                outX[i] = outY[i] = 1
                s = 1
            else:
                outX[i] = 0
                outY[i] = 1

@guvectorize(
    ['void(boolean[:], boolean[:], int32, boolean[:], boolean[:])'],
    '(n),(n),()->(n),(n)', nopython=True
)
def _desynchronize(x, y, n, outX, outY):
    s = 0
    for i in range(n):
        if x[i] ^ y[i]:
            outX[i] = x[i]
            outY[i] = y[i]

        elif s == 0:
            if x[i] and y[i]:
                outX[i] = 0
                outY[i] = 1
                s = 1
            else:
                outX[i] = outY[i] = 0

        elif s == 1:
            if x[i] and y[i]:
                outX[i] = outY[i] = 1
            else:
                outX[i] = 1
                outY[i] = 0
                s = 2

        elif s == 2:
            if x[i] and y[i]:
                outX[i] = 1
                outY[i] = 0
                s = 3
            else:
                outX[i] = outY[i] = 0

        elif s == 3:
            if x[i] and y[i]:
                outX[i] = outY[i] = 1
            else:
                outX[i] = 0
                outY[1] = 1
                s = 0
