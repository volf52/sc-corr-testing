from numba import vectorize as nvectorize


UNARY_OPS_SIGNATURE = 'boolean(boolean)'
BINARY_OPS_SIGNATURE = 'boolean(boolean, boolean)'

# Similar for or, not and xnot

@nvectorize(BINARY_OPS_SIGNATURE, nopython=True)
def and_it(x, y):
    return x and y

@nvectorize(UNARY_OPS_SIGNATURE, nopython=True)
def not_it(x):
    return not x