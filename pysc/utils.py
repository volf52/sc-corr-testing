from numba import guvectorize

FTYLIST = ["void(float32, int32, int32, int32, boolean[:], boolean[:])"]
SIGNATURE = "(),(),(),(),(n)->(n)"


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


Stream = guvectorize(FTYLIST, SIGNATURE, nopython=True)(_SCStream)

StreamCuda = guvectorize(FTYLIST, SIGNATURE, nopython=True, target="cuda")(_SCStream)
