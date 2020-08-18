import numpy as np
import cupy as cp

from typing import Union
from pysc.utils import Stream, StreamCuda

ARRAY = Union[cp.ndarray, np.ndarray]


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

    def __init__(
        self, inp, /, min_val=-1, max_val=1, precision=8, *, device = "cpu"
    ):
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
        last_exis = self.__stream.ndim - 1
        self.xp.random.shuffle(self.__stream.swapaxes(0, last_exis))

    def to_device(self, device):
        assert device in ("cpu", "gpu")
        if device == self.__device:
            return

        self.__device = device

        if device == "cpu":
            self.xp = np

        else:
            self.xp = cp


    @property
    def device(self):
        return self.__device

    @property
    def _stream(self):
        return self.__stream

    @property
    def shape(self):
        return self.__stream.shape

    @property
    def ndim(self):
        return self.__stream.ndim

    def __len__(self):
        return len(self.__stream)

    def __iter__(self):
        return iter(self.__stream)

    def __getitem__(self, idx):
        return self.__stream[idx]
