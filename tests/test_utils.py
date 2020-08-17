import numpy as np
import cupy as cp
import pytest


@pytest.fixture
def base_arrays():
    arrCuda = cp.linspace(-5, 5, 1000, dtype=np.float32).reshape(25, 40)
    arrNorm = cp.asnumpy(arrCuda)

    return arrNorm, arrCuda


def test_quantize(base_arrays):
    assert type(base_arrays[0]) == np.ndarray
    assert type(base_arrays[1]) == cp.ndarray
