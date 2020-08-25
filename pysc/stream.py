from pprint import pformat
from typing import Tuple, Union

import cupy as cp
import numpy as np

from pysc.ops import find_corr_mat, pearson_corr, sc_corr, _synchronize, _desynchronize
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

    def __init__(self, precision=8, *, device="cpu", encoding="bpe", printMsg = True):
        assert device in ("cpu", "gpu")
        assert encoding in ("upe", "bpe")

        self.__precision = int(precision)
        self.min_val = -1 if encoding == "bpe" else 0
        self.max_val = 1
        self._device = device
        self._encoding = encoding

        if device == "cpu":
            self.xp = np
        else:
            self.xp = cp

        self.__stream: ARRAY = None

        if printMsg:
            print("Please use the alternative constructors (from_input_array, from_probability_array, from_stream_array) etc")

    @classmethod
    def from_input_array(
        cls, inp: ARRAY, precision=8, *, device="cpu", encoding="bpe"
    ):

        scStream = cls(precision=precision, device=device, encoding=encoding, printMsg=False)
        if not (isinstance(inp, np.ndarray) or isinstance(inp, cp.ndarray)):
            inp = scStream.xp.array(inp, dype=np.float32)[None]
        else:
            inp = to_device(inp, device)

        probStream = createProbabilityStream(inp, precision, encoding, device)

        probStream *= precision

        if scStream._device == "cpu":
            probStream = probStream.astype(np.int32)
        else:
            probStream = cp.asnumpy(probStream).astype(np.int32)
            cp.cuda.Stream.null.synchronize()

        scStream.__generate_stream(probStream)

        return scStream

    @classmethod
    def from_probability_array(cls, probs: ARRAY, precision=8, *, device="cpu", encoding="bpe"):
        scStream = cls(precision=precision, device=device, encoding=encoding, printMsg=False)

        probs = probs.copy()
        probs *= precision

        if scStream._device == "cpu":
            probStream = probs.astype(np.int32)
        else:
            probStream = cp.asnumpy(probs).astype(np.int32)
            cp.cuda.Stream.null.synchronize()

        scStream.__generate_stream(probStream)

        return scStream

    @classmethod
    def from_stream_array(cls, streams: ARRAY, encoding: str, *, device="cpu"):
        # The encoding parameters is non_default here to ensure the user enters the correct encoding scheme
        assert streams.dtype == np.bool
        assert streams.ndim > 1

        scStream = cls(precision=streams.shape[-1], device=device, encoding=encoding, printMsg=False)

        # Copy is made to ensure the changes in the streams array do not cause changes to this object
        streams = to_device(streams.copy(), device)

        scStream.__stream = streams
        cp.get_default_memory_pool().free_all_blocks()

        return scStream



    def __generate_stream(self, probStream: ARRAY):
        self.__stream = np.zeros(probStream.shape + (self.__precision,), dtype=np.bool)
        npStream(probStream, self.__stream)

        if self._device != "cpu":
            self.__stream = cp.array(self.__stream)
            cp.cuda.Stream.null.synchronize()

        cp.get_default_memory_pool().free_all_blocks()

    def to_device(self, device):
        assert device in ("cpu", "gpu")
        if device == self._device:
            return

        self._device = device

        if device == "cpu":
            self.xp = np

        else:
            self.xp = cp

    def corr_with(self, other: "SCStream") -> Tuple[ARRAY, ARRAY]:
        if self.__stream is None or other.__stream is None:
            print("Set the underlying stream array")
            return None, None

        assert self.shape == other.shape
        assert (
            self._device == other._device
        )  # may change to moving to cpu if devices are not the same

        a, b, c, d = find_corr_mat(self.__stream, other.__stream, self._device)

        scc = sc_corr(a, b, c, d, self.__precision, self._device)
        psc = pearson_corr(a, b, c, d, self._device)

        return scc, psc

    # To induce positive correlation between 2 SCStreams
    @staticmethod
    def sync_with(x: 'SCStream', y: 'SCStream', inplace=True):
        assert x.precision == y.precision
        if inplace:
            newX = x[:]
        else:
            newX = x[:].copy()

        _synchronize(x[:], y[:], x.precision, newX)

        if not inplace:
            newX = SCStream.from_stream_array(newX, x._encoding, device=x._device)
            return newX

    @staticmethod
    def desync_with(x: 'SCStream', y: 'SCStream', inplace=True):
        assert x.precision == y.precision
        if inplace:
            newX = x[:]
        else:
            newX = x[:].copy()

        _desynchronize(x[:], y[:], x.precision, newX)

        if not inplace:
            newX = SCStream.from_stream_array(newX, x._encoding, device=x._device)

            return newX


    # Todo, Implement this (or something similar) ->
    """
    @staticmethod
    def fromArray(stream: ARRAY): -> SCStream:
    """

    def decode(self) -> Union[float, ARRAY]:
        if self._encoding == "upe":
            return self.xp.count_nonzero(self.__stream, axis=-1) / self.precision
        else:
            return (
                2 * self.xp.count_nonzero(self.__stream, axis=-1) - self.precision
            ) / self.precision

    @property
    def _stream(self):
        return self.__stream

    @property
    def encoding(self):
        return self._encoding

    @property
    def device(self):
        return self._device

    @property
    def precision(self):
        return self.__precision

    @_stream.setter
    def _stream(self, newStream: ARRAY):
        assert newStream.shape == self.__stream.shape
        assert newStream.dtype == self.__stream.dtype
        newStream = to_device(newStream, self._device)
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
