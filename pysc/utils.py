from numba import guvectorize


def _SCStream(x, min_val, max_val, precision, _, out):
    if x < min_val:
        quantX = min_val
    elif x > max_val:
        quantX = max_val
    else:
        q = (max_val - min_val) / precision
        quantX = q * round(x / q)

    prob = (quantX + 1) / 2  # BPE Encoding probability
    n_ones = int(prob * precision)
    out[:n_ones] = 1


Stream = guvectorize(
    ["void(float32, float32, float32, float32, boolean[:], boolean[:])"],
    "(),(),(),(),(n)->(n)",
    nopython=True,
)(_SCStream)

# TODO: Complete this, ofc
StreamCuda = None
