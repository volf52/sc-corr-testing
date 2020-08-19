import numpy as np
import cupy as cp
import pytest
from pysc.stream import SCStream


@pytest.fixture
def base_arrays():
    arrCuda = cp.linspace(-5, 5, 1000, dtype=np.float32).reshape(25, 40)
    arrNorm = cp.asnumpy(arrCuda)

    return arrNorm, arrCuda


def test_scstream_init_default(base_arrays):
    stream = SCStream(base_arrays[0], -1.2, 1.0)

    assert isinstance(stream.min_val, int)
    assert isinstance(stream.max_val, int)
    assert isinstance(stream.precision, int)
    assert stream.device == "cpu"
    assert stream.precision == 8
    assert stream.min_val == -1
    assert stream.max_val == 1
    assert isinstance(stream._stream, np.ndarray)


def test_scstream_init_options(base_arrays):
    stream = SCStream(base_arrays[1], precision=16, device="gpu")
    assert stream.device == "gpu"
    assert stream.precision == 16
    assert isinstance(stream._stream, cp.ndarray)
