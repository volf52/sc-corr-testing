from pysc.ops import find_corr_mat, sc_corr_1d, pearson
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
    xvals = [np.array(list(map(int, xi)), dtype=np.bool) for xi in xvals]
    yvals = [np.array(list(map(int, yi)), dtype=np.bool) for yi in yvals]

    return xvals, yvals


def test_find_corr_mat_one(starter_values):
    assert True


def test_sc_corr_1d(starter_values):
    assert True


def test_combined_test(starter_values):
    abcds = [find_corr_mat(xi, yi, "cpu") for xi, yi in zip(*starter_values)]

    expected_sc = [0, 1, -1, 1, -1, 0, 1]
    expected_pearson = [0.0, 1.0, -1.0, 0.58, -0.58, 0.0, 0.33]

    sc_corrs = [int(sc_corr_1d(*r, 8)) for r in abcds]
    pearson_corrs = [round(pearson(*r), 2) for r in abcds]

    assert np.array_equal(expected_sc, sc_corrs)
    assert np.allclose(expected_pearson, pearson_corrs)
