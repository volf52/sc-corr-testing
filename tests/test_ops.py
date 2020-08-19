import cupy as cp
import numpy as np
import pytest

from pysc.ops import find_corr_mat, sc_corr
from pysc.stream import SCStream


# TODO: Complete the tests. Add for n-dim arrays. Add specific for cupy arrays


@pytest.fixture
def starter_values():
    xvals = [
        "11110000",
        "11110000",
        "11110000",
        "11111100",
        "11111100",
        "11111100",
        "11000000",
    ]
    yvals = [
        "11001100",
        "11110000",
        "00001111",
        "11110000",
        "00001111",
        "11100001",
        "11111100",
    ]
    xvals = [list(map(int, xi)) for xi in xvals]
    yvals = [list(map(int, yi)) for yi in yvals]

    ret = {
        "xvals": np.array(xvals, dtype=np.bool),
        "yvals": np.array(yvals, dtype=np.bool),
        "xvals_cuda": cp.array(xvals, dtype=np.bool),
        "yvals_cuda": cp.array(yvals, dtype=np.bool),
        "expected_scc": [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 1.0],
        "expected_pearson": [0.0, 1.0, -1.0, 0.58, -0.58, 0.0, 0.33],
    }

    ret["expected_scc_cp"] = cp.array(ret["expected_scc"])

    return ret


def test_stream_corr(starter_values):
    xvals, yvals = starter_values["xvals"], starter_values["yvals"]

    sc_one = SCStream(np.zeros(xvals.shape[0], dtype=np.float32))
    sc_two = SCStream(np.zeros(yvals.shape[0], dtype=np.float32))

    sc_one._stream = xvals
    sc_two._stream = yvals

    scc, _ = sc_one.corr_with(sc_two)

    assert np.array_equal(starter_values["expected_scc"], scc)


def test_stream_corr_cuda(starter_values):
    xvals, yvals = starter_values["xvals_cuda"], starter_values["yvals_cuda"]

    sc_one = SCStream(np.zeros(xvals.shape[0], dtype=np.float32), device="gpu")
    sc_two = SCStream(np.zeros(yvals.shape[0], dtype=np.float32), device="gpu")

    sc_one._stream = xvals
    sc_two._stream = yvals

    scc, _ = sc_one.corr_with(sc_two)

    assert np.all(starter_values["expected_scc_cp"] == scc)


def test_sc_corr_test(starter_values):
    xvals = starter_values["xvals"]
    yvals = starter_values["yvals"]

    a, b, c, d = find_corr_mat(xvals, yvals, "cpu")

    sc_corrs = sc_corr(a, b, c, d, xvals.shape[1], "cpu")

    assert np.array_equal(starter_values["expected_scc"], sc_corrs)


def test_combined_corr_test_cuda(starter_values):
    xvals = starter_values["xvals_cuda"]
    yvals = starter_values["yvals_cuda"]

    a, b, c, d = find_corr_mat(xvals, yvals, "gpu")

    sc_corrs = sc_corr(a, b, c, d, xvals.shape[1], "gpu")

    assert np.all(starter_values["expected_scc_cp"] == sc_corrs)
