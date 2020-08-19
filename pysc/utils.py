from typing import Union

import cupy as cp
import numpy as np
from numba import guvectorize

ARRAY = Union[cp.ndarray, np.ndarray]


def createProbabilityStream(inp: ARRAY, precision: int, encoding: str, device: str):
    assert encoding in ("upe", "bpe")
    assert device in ("cpu", "gpu", "cuda")
    assert inp.dtype == np.float32

    maxv = 1
    minv = -1 if encoding == "bpe" else 0

    q = (maxv - minv) / precision
    out = inp.copy()
    out.clip(minv, maxv, out=out)
    out /= q
    out.round(out=out)
    out *= q

    if encoding == "bpe":
        out += 1.0
        out /= 2.0

    return out


@guvectorize("void(int32, boolean[:])", "(),(n)", nopython=True)
def npStream(n_ones: ARRAY, out: ARRAY):
    #  Note than the probability stream has already been multiplied with precision and cast to int
    out[:n_ones] = 1
    np.random.shuffle(out)


@guvectorize(
    "void(int32, boolean[:], boolean[:])", "(),(n)->(n)", target="cuda", nopython=True
)
def cpStream(n_ones: ARRAY, _, out: ARRAY):
    out[:n_ones] = 1


def shuffle_along_axis_cp(a, axis):
    idx = cp.random.rand(*a.shape).argsort(axis=axis)
    return cp.take_along_axis(a, idx, axis=axis)
