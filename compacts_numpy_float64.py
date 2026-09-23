from __future__ import annotations

import struct
import zlib
from typing import Iterator

import numpy as np


TOTAL = 9_000_000_000_000
CHUNK = 1_000_000

MAGIC = b"F64C2"

EMPTY = 0
CONSTANT = 1
ARITHMETIC = 2
COMPRESSED = 3
RAW = 4


def compact_float64(a: np.ndarray) -> bytes:
    """
    Losslessly compact a contiguous float64 NumPy array.
    """

    a = np.asarray(a, dtype=np.float64)

    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)

    n = a.size

    if n == 0:
        return (
            MAGIC +
            bytes([EMPTY]) +
            struct.pack("<Q", 0)
        )

    # ------------------------------------------------------------
    # Constant representation.
    #
    # Compare the actual uint64 bit pattern so that:
    #
    #     +0.0 != -0.0
    #
    # and different NaN payloads remain distinguishable.
    # ------------------------------------------------------------

    bits = a.view(np.uint64)

    if np.all(bits == bits[0]):
        return (
            MAGIC +
            bytes([CONSTANT]) +
            struct.pack(
                "<QQ",
                n,
                int(bits[0])
            )
        )

    # ------------------------------------------------------------
    # Arithmetic progression.
    # ------------------------------------------------------------

    if n >= 2 and np.isfinite(a).all():

        start = a[0]
        step = a[1] - a[0]

        expected = (
            start +
            step * np.arange(
                n,
                dtype=np.float64
            )
        )

        if np.array_equal(a, expected):
            return (
                MAGIC +
                bytes([ARITHMETIC]) +
                struct.pack(
                    "<Qdd",
                    n,
                    float(start),
                    float(step)
                )
            )

    # ------------------------------------------------------------
    # Generic lossless compression.
    # ------------------------------------------------------------

    raw = a.tobytes(order="C")

    compressed = zlib.compress(
        raw,
        level=9
    )

    if len(compressed) < len(raw):
        return (
            MAGIC +
            bytes([COMPRESSED]) +
            struct.pack(
                "<QQ",
                n,
                len(compressed)
            ) +
            compressed
        )

    # ------------------------------------------------------------
    # Incompressible data.
    # ------------------------------------------------------------

    return (
        MAGIC +
        bytes([RAW]) +
        struct.pack("<Q", n) +
        raw
    )


def expand_float64(blob: bytes) -> np.ndarray:
    """
    Reconstruct one compacted chunk as a NumPy float64 ndarray.
    """

    if blob[:5] != MAGIC:
        raise ValueError("invalid compact float64 object")

    kind = blob[5]

    # EMPTY
    if kind == EMPTY:
        return np.empty(
            0,
            dtype=np.float64
        )

    # CONSTANT
    if kind == CONSTANT:

        n, bits = struct.unpack_from(
            "<QQ",
            blob,
            6
        )

        result = np.empty(
            n,
            dtype=np.float64
        )

        result.view(np.uint64)[:] = bits

        return result

    # ARITHMETIC
    if kind == ARITHMETIC:

        n, start, step = struct.unpack_from(
            "<Qdd",
            blob,
            6
        )

        return (
            start +
            step *
            np.arange(
                n,
                dtype=np.float64
            )
        )

    # COMPRESSED
    if kind == COMPRESSED:

        n, compressed_size = struct.unpack_from(
            "<QQ",
            blob,
            6
        )

        offset = 22

        compressed = blob[
            offset:
            offset + compressed_size
        ]

        raw = zlib.decompress(compressed)

        if len(raw) != n * 8:
            raise ValueError(
                "corrupt compressed float64 chunk"
            )

        return np.frombuffer(
            raw,
            dtype=np.float64
        ).copy()

    # RAW
    if kind == RAW:

        n = struct.unpack_from(
            "<Q",
            blob,
            6
        )[0]

        offset = 14

        raw = blob[
            offset:
            offset + n * 8
        ]

        if len(raw) != n * 8:
            raise ValueError(
                "corrupt raw float64 chunk"
            )

        return np.frombuffer(
            raw,
            dtype=np.float64
        ).copy()

    raise ValueError(
        f"unknown representation: {kind}"
    )


class CompactFloat64Array:
    """
    Logical NumPy-compatible float64 array.

    Only requested chunks are materialized.
    """

    def __init__(
        self,
        length: int,
        chunk_size: int = 1_000_000,
    ):
        self.length = int(length)
        self.chunk_size = int(chunk_size)

        if self.length < 0:
            raise ValueError("negative length")

        if self.chunk_size <= 0:
            raise ValueError("invalid chunk size")

        self.shape = (self.length,)
        self.dtype = np.dtype(np.float64)
        self.ndim = 1

        self._chunks: dict[int, bytes] = {}

    def store_chunk(
        self,
        offset: int,
        values: np.ndarray,
    ) -> None:

        values = np.asarray(
            values,
            dtype=np.float64
        )

        offset = int(offset)

        if offset < 0:
            raise IndexError(offset)

        if offset + values.size > self.length:
            raise IndexError(
                "chunk exceeds logical array"
            )

        self._chunks[offset] = compact_float64(
            values
        )

    def get_chunk(
        self,
        offset: int,
    ) -> np.ndarray:

        offset = int(offset)

        if offset < 0 or offset >= self.length:
            raise IndexError(offset)

        blob = self._chunks.get(offset)

        if blob is None:
            raise KeyError(
                f"chunk {offset} has not been stored"
            )

        return expand_float64(blob)

    def __getitem__(self, key):

        # Single element.
        if isinstance(key, (int, np.integer)):

            index = int(key)

            if index < 0:
                index += self.length

            if index < 0 or index >= self.length:
                raise IndexError(index)

            chunk_offset = (
                index //
                self.chunk_size
            ) * self.chunk_size

            chunk = self.get_chunk(
                chunk_offset
            )

            return chunk[
                index - chunk_offset
            ]

        # Slice.
        if isinstance(key, slice):

            start, stop, step = key.indices(
                self.length
            )

            if step != 1:
                return np.array(
                    [
                        self[i]
                        for i in range(
                            start,
                            stop,
                            step
                        )
                    ],
                    dtype=np.float64
                )

            if start >= stop:
                return np.empty(
                    0,
                    dtype=np.float64
                )

            pieces = []

            position = start

            while position < stop:

                chunk_offset = (
                    position //
                    self.chunk_size
                ) * self.chunk_size

                chunk = self.get_chunk(
                    chunk_offset
                )

                local_start = (
                    position -
                    chunk_offset
                )

                local_stop = min(
                    chunk.size,
                    stop - chunk_offset
                )

                pieces.append(
                    chunk[
                        local_start:
                        local_stop
                    ]
                )

                position = (
                    chunk_offset +
                    local_stop
                )

            return np.concatenate(pieces)

        raise TypeError(
            "index must be an integer or slice"
        )

    def iter_chunks(self) -> Iterator[np.ndarray]:

        offset = 0

        while offset < self.length:

            yield self.get_chunk(offset)

            offset += self.chunk_size

    def stored_bytes(self) -> int:
        return sum(
            len(blob)
            for blob in self._chunks.values()
        )
