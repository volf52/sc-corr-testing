from pysc.ops import find_corr_mat, sc_corr, pearson
import pytest
import numpy as np
import cupy as cp

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
        "xvals": xvals,
        "yvals": yvals,
        "expected_scc": [0, 1, -1, 1, -1, 0, 1],
        "expected_pearson": [0.0, 1.0, -1.0, 0.58, -0.58, 0.0, 0.33],
    }

    return ret


def test_find_corr_mat_one(starter_values):
    assert True


def test_sc_corr_1d(starter_values):
    assert True


def test_combined_test(starter_values):
    xvals = np.array(starter_values["xvals"], dtype=np.bool)
    yvals = np.array(starter_values["yvals"], dtype=np.bool)

    a, b, c, d = find_corr_mat(xvals, yvals, "cpu")

    sc_corrs = sc_corr(a, b, c, d, xvals.shape[1])
    pearson_corrs = pearson(a, b, c, d)

    pearson_corrs.round(2, out=pearson_corrs)

    assert np.array_equal(starter_values["expected_scc"], sc_corrs)
    assert np.allclose(starter_values["expected_pearson"], pearson_corrs)


def test_combined_test_cuda(starter_values):
    xvals = cp.array(starter_values["xvals"], dtype=np.bool)
    yvals = cp.array(starter_values["yvals"], dtype=np.bool)

    a, b, c, d = find_corr_mat(xvals, yvals, "gpu")

    sc_corrs = sc_corr(a, b, c, d, xvals.shape[1])
    expected_sc_corr = cp.array(starter_values['expected_scc'])
    pearson_corrs: cp.ndarray = pearson(a, b, c, d)
    pearson_corrs.round(2, out=pearson_corrs)

    assert np.all(expected_sc_corr == sc_corrs)
    assert np.allclose(starter_values["expected_pearson"], pearson_corrs)
