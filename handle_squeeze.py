from __future__ import annotations

import ctypes
import lzma
import pickle
import struct
from dataclasses import dataclass
from typing import Any


# ============================================================
# Ctypes Type Layout Declarations
# ============================================================

class SqueezeHeaderLayout(ctypes.BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("preset", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


# ============================================================
# Squeezed Handle & Pipeline Definitions
# ============================================================

@dataclass(slots=True)
class SqueezedHandle:
    compressed_payload: bytes
    lzma_preset: int
    original_length: int
    width: int
    height: int
    shape: tuple[int, ...]
    dtype: Any

    def to_bytes(self) -> bytes:
        # Pack binary metadata header using ctypes and struct
        header = SqueezeHeaderLayout(magic=0x53515A20, preset=self.lzma_preset, reserved=0)
        header_bytes = ctypes.string_at(ctypes.addressof(header), ctypes.sizeof(header))
        
        meta = {
            "original_length": self.original_length,
            "width": self.width,
            "height": self.height,
            "shape": self.shape,
            "dtype": str(self.dtype),
        }
        meta_pickled = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Combine via structured container format
        body = struct.pack(">I", len(meta_pickled)) + meta_pickled + self.compressed_payload
        return header_bytes + lzma.compress(body, preset=6)

    @classmethod
    def from_bytes(cls, data: bytes) -> SqueezedHandle:
        header_size = ctypes.sizeof(SqueezeHeaderLayout)
        if len(data) < header_size:
            raise ValueError("Truncated squeezed data")
            
        header = SqueezeHeaderLayout.from_buffer_copy(data[:header_size])
        if header.magic != 0x53515A20:
            raise ValueError("Invalid squeezed format magic")
            
        decompressed_body = lzma.decompress(data[header_size:])
        
        meta_len = struct.unpack(">I", decompressed_body[:4])[0]
        meta = pickle.loads(decompressed_body[4:4 + meta_len])
        compressed_payload = decompressed_body[4 + meta_len:]
        
        return cls(
            compressed_payload=compressed_payload,
            lzma_preset=header.preset,
            original_length=meta["original_length"],
            width=meta["width"],
            height=meta["height"],
            shape=tuple(meta["shape"]),
            dtype=meta["dtype"],
        )


def squeeze_handle(handle: Any) -> SqueezedHandle:
    """
    Squeezes a compression handle by wrapping and compressing its existing 
    packed payload bytes directly as an opaque byte stream, avoiding any 
    premature decompression or internal object unpickling.
    """
    source_bytes = handle.packed.payload
    
    preset = 9
    compressed_payload = lzma.compress(source_bytes, preset=preset | lzma.PRESET_EXTREME)
    
    if len(compressed_payload) >= len(source_bytes):
        compressed_payload = source_bytes
        preset = 0

    return SqueezedHandle(
        compressed_payload=compressed_payload,
        lzma_preset=preset,
        original_length=handle.original_length,
        width=handle.width,
        height=handle.height,
        shape=handle.shape,
        dtype=handle.dtype,
    )


def unsqueeze_handle(sq: SqueezedHandle) -> Any:
    """
    Restores a SqueezedHandle back into a container holding the original 
    packed payload bytes without triggering any inner object decoding.
    """
    payload = sq.compressed_payload
    if sq.lzma_preset > 0:
        payload = lzma.decompress(payload)

    class _ProxyPacked:
        def __init__(self, p: bytes):
            self.payload = p
            self.compressed = True
            self.raw_size = len(p)

        @property
        def stored_bytes(self) -> int:
            return len(self.payload)

    class _ProxyHandle:
        def __init__(self, p_bytes, shape, dtype, original_length, width, height):
            self.packed = _ProxyPacked(p_bytes)
            self.shape = shape
            self.dtype = dtype
            self.original_length = original_length
            self.width = width
            self.height = height

        @property
        def stored_bytes(self) -> int:
            return self.packed.stored_bytes

    return _ProxyHandle(
        p_bytes=payload,
        shape=sq.shape,
        dtype=sq.dtype,
        original_length=sq.original_length,
        width=sq.width,
        height=sq.height,
    )
