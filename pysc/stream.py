from pprint import pformat
from typing import Tuple

import cupy as cp
import numpy as np

from pysc.ops import find_corr_mat, pearson, sc_corr
from pysc.utils import ARRAY, Stream, StreamCuda


def to_device(x: ARRAY, device):
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

    def __init__(self, inp, /, min_val=-1, max_val=1, precision=8, *, device="cpu"):
        assert device in ("cpu", "gpu")
        self.precision = int(precision)
        self.min_val = int(min_val)
        self.max_val = int(max_val)
        self.__device = device

        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self._stream_generator = [Stream, StreamCuda][device == "gpu"]
        self.__stream: ARRAY = None
        self.__generate_stream(inp)
        self.shuffle_stream()

    def __generate_stream(self, inp):
        if not (isinstance(inp, np.ndarray) or isinstance(inp, cp.ndarray)):
            inp = self.xp.array(inp, dtype=np.float32)[None]
        else:
            inp = to_device(inp, self.__device)

        self.__stream = self.xp.zeros(inp.shape + (self.precision,), dtype=np.bool)
        self._stream_generator(
            inp,
            self.min_val,
            self.max_val,
            self.precision,
            self.__stream,
            self.__stream,
        )

    def shuffle_stream(self):
        last_axis = self.__stream.ndim - 1
        self.xp.random.shuffle(self.__stream.swapaxes(0, last_axis))

    def to_device(self, device):
        assert device in ("cpu", "gpu")
        if device == self.__device:
            return

        self.__device = device

        if device == "cpu":
            self.xp = np

        else:
            self.xp = cp

    def corr_with(self, other: "SCStream") -> Tuple[ARRAY, ARRAY]:
        assert self.shape == other.shape
        assert (
            self.__device == other.__device
        )  # may change to moving to cpu if devices are not the same

        a, b, c, d = find_corr_mat(self.__stream, other.__stream, self.__device)

        scc = sc_corr(a, b, c, d, self.precision)
        pearson_corr = pearson(a, b, c, d)

        return scc, pearson_corr

    @property
    def device(self):
        return self.__device

    @property
    def _stream(self):
        return self.__stream

    @_stream.setter
    def _stream(self, newStream: ARRAY):
        assert newStream.shape == self.__stream.shape
        assert newStream.dtype == self.__stream.dtype
        newStream = to_device(newStream, self.__device)
        self.__stream = newStream

    @property
    def shape(self):
        return self.__stream.shape

    @property
    def ndim(self):
        return self.__stream.ndim

    @property
    def size(self):
        return self.__stream.size

    def __len__(self):
        return len(self.__stream)

    def __iter__(self):
        return iter(self.__stream)

    def __getitem__(self, idx):
        return self.__stream[idx]

    def __repr__(self):
        s = f"Stream :-> '{self.precision}' Device: '{self.device}'"
        s += f" Shape: '{self.__stream.shape}' Total: '{self.__stream.size}'\n"
        s += pformat(self.__stream)

        return s

    def __array__(self):
        """Allows us to use np.array and cp.array directly on SCStream objects."""
        return self.__stream
