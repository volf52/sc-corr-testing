import cupy as cp
import numpy as np
import pytest

from pysc.ops import find_corr_mat, pearson_corr, sc_corr
from pysc.stream import SCStream


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

    scc, psc = sc_one.corr_with(sc_two)
    psc.round(2, out=psc)

    assert np.array_equal(starter_values["expected_scc"], scc)
    assert np.allclose(starter_values["expected_pearson"], psc)


def test_stream_corr_cuda(starter_values):
    xvals, yvals = starter_values["xvals_cuda"], starter_values["yvals_cuda"]
    device = "gpu"

    sc_one = SCStream(np.zeros(xvals.shape[0], dtype=np.float32), device=device)
    sc_two = SCStream(np.zeros(yvals.shape[0], dtype=np.float32), device=device)

    sc_one._stream = xvals
    sc_two._stream = yvals

    scc, psc = sc_one.corr_with(sc_two)
    psc.round(2, out=psc)

    assert np.all(starter_values["expected_scc_cp"] == scc)
    assert np.allclose(starter_values["expected_pearson"], psc)


def test_sc_corr(starter_values):
    xvals = starter_values["xvals"]
    yvals = starter_values["yvals"]
    device = "cpu"

    a, b, c, d = find_corr_mat(xvals, yvals, device)

    sc_corrs = sc_corr(a, b, c, d, xvals.shape[1], device)

    assert np.array_equal(starter_values["expected_scc"], sc_corrs)


def test_sc_corr_cuda(starter_values):
    xvals = starter_values["xvals_cuda"]
    yvals = starter_values["yvals_cuda"]
    device = "gpu"

    a, b, c, d = find_corr_mat(xvals, yvals, device)

    sc_corrs = sc_corr(a, b, c, d, xvals.shape[1], device)

    assert np.all(starter_values["expected_scc_cp"] == sc_corrs)


def test_sc_corr(starter_values):
    xvals = starter_values["xvals"]
    yvals = starter_values["yvals"]
    device = "cpu"

    a, b, c, d = find_corr_mat(xvals, yvals, device)

    pearson_corrs = pearson_corr(a, b, c, d, device)
    pearson_corrs.round(2, out=pearson_corrs)

    assert np.allclose(starter_values["expected_pearson"], pearson_corrs)


def test_sc_corr(starter_values):
    xvals = starter_values["xvals_cuda"]
    yvals = starter_values["yvals_cuda"]
    device = "gpu"

    a, b, c, d = find_corr_mat(xvals, yvals, device)

    pearson_corrs = pearson_corr(a, b, c, d, device)
    pearson_corrs.round(2, out=pearson_corrs)

    assert np.allclose(starter_values["expected_pearson"], pearson_corrs)
