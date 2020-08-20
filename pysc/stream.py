from pprint import pformat
from typing import Tuple, Union

import cupy as cp
import numpy as np

from pysc.ops import find_corr_mat, pearson_corr, sc_corr, _synchronize
from pysc.utils import ARRAY, createProbabilityStream, npStream


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

    def __init__(self, inp, precision=8, *, device="cpu", encoding="bpe"):
        assert device in ("cpu", "gpu")
        assert encoding in ("upe", "bpe")

        self.__precision = int(precision)
        self.min_val = -1 if encoding == "bpe" else 0
        self.max_val = 1
        self.__device = device
        self.__encoding = encoding

        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.__stream: ARRAY = None
        self.__generate_stream(inp)

        cp.get_default_memory_pool().free_all_blocks()

    def __generate_stream(self, inp):
        if not (isinstance(inp, np.ndarray) or isinstance(inp, cp.ndarray)):
            inp = self.xp.array(inp, dtype=np.float32)[None]
        else:
            inp = to_device(inp, self.__device)

        probStream = createProbabilityStream(
            inp, self.__precision, self.__encoding, self.__device
        )
        probStream *= self.__precision

        if self.__device == "cpu":
            probStream = probStream.astype(np.int32)
        else:
            probStream = cp.asnumpy(probStream).astype(np.int32)
            cp.cuda.Stream.null.synchronize()

        self.__stream = np.zeros(inp.shape + (self.__precision,), dtype=np.bool)
        npStream(probStream, self.__stream)

        if self.__device != "cpu":
            self.__stream = cp.array(self.__stream)
            cp.cuda.Stream.null.synchronize()

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

        scc = sc_corr(a, b, c, d, self.__precision, self.__device)
        psc = pearson_corr(a, b, c, d, self.__device)

        return scc, psc

    # To induce positive correlation between 2 SCStreams
    @staticmethod
    def synchronize(x: 'SCStream', y: 'SCStream', inplace=True):
        assert x.precision == y.precision
        if inplace:
            newX = x[:]
            newY = y[:]
        else:
            newX = x[:].copy()
            newY = y[:].copy()

        # Todo : investigate the effect of `depth` argument on the final result
        _synchronize(x[:], y[:], x.precision, newX, newY)

        # Todo : convert ARRAY to SCStream if not inplace
        if not inplace:
            return newX, newY

    # Todo, Implement this (or something similar) ->
    """
    @staticmethod
    def fromArray(stream: ARRAY): -> SCStream:
    """

    def decode(self) -> Union[float, ARRAY]:
        if self.__encoding == 'upe':
            return self.xp.count_nonzero(self.__stream, axis=-1) / self.precision
        else:
            return (2 * self.xp.count_nonzero(self.__stream, axis=-1) - self.precision) / self.precision

    @property
    def _stream(self):
        return self.__stream

    @property
    def encoding(self):
        return self.__encoding

    @property
    def device(self):
        return self.__device

    @property
    def precision(self):
        return self.__precision

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
        s = f"Stream :-> '{self.__precision}' Device: '{self.device}'"
        s += f" Shape: '{self.__stream.shape}' Total: '{self.__stream.size}'\n"
        s += pformat(self.__stream)

        return s

    def __array__(self):
        """Allows us to use np.array and cp.array directly on SCStream objects."""
        return self.__stream
