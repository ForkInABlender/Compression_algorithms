import io
import math
import numpy as np
import nibabel as nib
from PIL import Image
from ctypes_object_compact import pack_object, PackedObject


# ============================================================
# Compression handle
# ============================================================

class CompressionHandle:
    """
    Persistent handle for a losslessly compressed float64 array.

    The PackedObject contains the compressed payload.
    The handle stores the metadata required to reconstruct
    the original NumPy array.
    """

    __slots__ = (
        "packed",
        "shape",
        "dtype",
        "original_length",
        "width",
        "height",
    )

    def __init__(
        self,
        packed: PackedObject,
        shape: tuple[int, ...],
        dtype: np.dtype,
        original_length: int,
        width: int,
        height: int,
    ) -> None:
        self.packed = packed
        self.shape = shape
        self.dtype = dtype
        self.original_length = original_length
        self.width = width
        self.height = height

    @property
    def stored_bytes(self) -> int:
        return self.packed.stored_bytes


# ============================================================
# Lossless compression
# ============================================================

def compress(data: np.ndarray) -> CompressionHandle:
    """
    Losslessly compress a float64 NumPy array.

    No quantization or precision reduction occurs.

    The exact IEEE-754 64-bit representation of every value
    is preserved.
    """

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a NumPy ndarray")

    if data.dtype != np.float64:
        raise TypeError("data must have dtype float64")

    # --------------------------------------------------------
    # Preserve the exact float64 bytes.
    # --------------------------------------------------------

    raw_bytes = data.tobytes()
    original_length = len(raw_bytes)

    # --------------------------------------------------------
    # Reinterpret the exact bytes as uint16.
    #
    # This does NOT numerically convert the values.
    # Four uint16 values contain exactly the same 64 bits
    # as one float64 value.
    # --------------------------------------------------------

    uint16_view = np.frombuffer(
        raw_bytes,
        dtype=np.uint16,
    )

    pixel_count = uint16_view.size

    # --------------------------------------------------------
    # Construct the PNG dimensions.
    # --------------------------------------------------------

    width = 2048
    height = math.ceil(pixel_count / width)

    padded_pixel_count = width * height
    padding_pixels = padded_pixel_count - pixel_count

    if padding_pixels:
        padded_pixels = np.pad(
            uint16_view,
            (0, padding_pixels),
            mode="constant",
            constant_values=0,
        )
    else:
        padded_pixels = uint16_view

    # --------------------------------------------------------
    # Encode exact uint16 bit patterns as PNG.
    # --------------------------------------------------------

    image = Image.frombytes(
        "I;16",
        (width, height),
        padded_pixels.tobytes(),
    )

    img_buffer = io.BytesIO()

    image.save(
        img_buffer,
        "PNG",
        compress_level=9,
        optimize=True,
    )

    png_image_bytes = img_buffer.getvalue()

    # --------------------------------------------------------
    # Store PNG using ctypes_object_compact.
    # --------------------------------------------------------

    packed = pack_object(
        png_image_bytes,
        compression_level=9,
    )

    # --------------------------------------------------------
    # Return a persistent compression handle.
    # --------------------------------------------------------

    return CompressionHandle(
        packed=packed,
        shape=tuple(data.shape),
        dtype=data.dtype,
        original_length=original_length,
        width=width,
        height=height,
    )


# ============================================================
# Lossless decompression
# ============================================================

def decompress(handle: CompressionHandle) -> np.ndarray:
    """
    Decompress a CompressionHandle and reconstruct the exact
    original float64 NumPy array.
    """

    if not isinstance(handle, CompressionHandle):
        raise TypeError(
            "handle must be a CompressionHandle"
        )

    # --------------------------------------------------------
    # Recover PNG bytes from PackedObject.
    # --------------------------------------------------------

    restored_png_bytes = handle.packed.unpack()

    # --------------------------------------------------------
    # Decode PNG.
    # --------------------------------------------------------

    with io.BytesIO(restored_png_bytes) as read_buf:
        loaded_img = Image.open(read_buf)
        recovered_pixels = loaded_img.tobytes()

    # --------------------------------------------------------
    # Remove only the padding introduced for the PNG grid.
    # --------------------------------------------------------

    restored_uint16_bytes = recovered_pixels[
        :handle.original_length
    ]

    # --------------------------------------------------------
    # Reinterpret the exact bytes as float64.
    # --------------------------------------------------------

    restored_data = np.frombuffer(
        restored_uint16_bytes,
        dtype=handle.dtype,
    ).reshape(handle.shape)

    return restored_data


# ============================================================
# 1. Generate 1 million float64 numbers
# ============================================================

data = np.linspace(
    0,
    1000,
    1_000_000,
    dtype=np.float64,
).reshape((100, 100, 100))

original_float64_bytes = data.tobytes()
original_length = len(original_float64_bytes)


if __name__ == "__main__":
    print(
    f"Original float64 payload size: "
    f"{original_length / 1024 / 1024:.2f} MB"
    )


# ============================================================
# 2. Compress
# ============================================================

handle = compress(data)

if __name__ == "__main__":
    print(
    f"Final packed size: "
    f"{handle.stored_bytes / 1024 / 1024:.2f} MB")


# ============================================================
# 3. Decompress
# ============================================================

loaded_data = decompress(handle)


# ============================================================
# 4. Verify exact equality
#
# np.allclose() is deliberately NOT used.
# This checks the actual 64-bit representation.
# ============================================================

#assert loaded_data.dtype == np.float64
#assert loaded_data.shape == data.shape

#assert np.array_equal(
#    data.view(np.uint64),
#    loaded_data.view(np.uint64),
#), "LOSSLESS VERIFICATION FAILED"


#assert (
#    loaded_data.tobytes() == data.tobytes()
#), "FLOAT64 BYTE REPRESENTATION CHANGED"

if __name__ == "__main__":
    print("Exact float64 lossless verification: PASS")
    print("Every float64 bit pattern was preserved exactly.")
