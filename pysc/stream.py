import numpy as np
import cupy as cp

from typing import Union
from pysc.utils import Stream, StreamCuda

ARRAY = Union[cp.ndarray, np.ndarray]
DEVICE = ["cpu", "gpu"]


def to_device(x: ARRAY, device: DEVICE):
    xp = cp.get_array_module(x)
    if device == "cpu" and xp == cp:
        return cp.asnumpy(x)

    elif device == "gpu" and xp == np:
        return cp.array(x, dtype=x.dtype)

    return x


class SCStream:
    """
    Main class for SC Streams.
    """

    def __init__(
        self, inp, /, min_val=-1.0, max_val=1.0, precision=8, *, device: DEVICE = "cpu"
    ):
        assert device in ("cpu", "gpu")
        self.precision = int(precision)
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.device = device

        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.stream_generator = [Stream, StreamCuda][device == "gpu"]
        self._stream: ARRAY = None
        # self.generate_stream(inp)

    def generate_stream(self, inp):
        inp = to_device(inp, self.device)

        if isinstance(inp, self.xp.ndarray):
            self._stream = self.xp.zeros(inp.shape + (self.precision,), dtype=np.bool)
            self.stream_generator(
                inp,
                self.min_val,
                self.max_val,
                self.precision,
                self._stream,
                self._stream,
            )
