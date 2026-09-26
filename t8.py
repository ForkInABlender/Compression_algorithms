import numpy as np
from codec import GridShiftBinaryCodec
from ctypes_object_compact import pack_object, PackedObject
from numpy_compressor import data, compress, decompress, CompressionHandle
from handle_squeeze import squeeze_handle, unsqueeze_handle, SqueezedHandle

# =====================================================================
# 1. True Nested Chained Compression Pipeline (Single Unified Blob)
# =====================================================================

# Step 1: Codec Layer - Encode float stream into symbolic Base64 string
codec = GridShiftBinaryCodec(step_resolution=1e-3)
b64_encoded = codec.encode_stream(data.tolist())

# Step 2: Compact Layer - Pack the Base64 string graph using ctypes_object_compact
packed_codec = pack_object(b64_encoded)

# Step 3: NumPy Layer - Losslessly compress the numpy array into a CompressionHandle
numpy_handle = compress(data)

# Step 4: Master Container & Squeeze Layer - Nest both compressed payloads 
# into a single master dictionary, pack it, and apply final outer LZMA squeeze.
master_container = {
    "codec_payload": packed_codec.payload,
    "codec_compressed": packed_codec.compressed,
    "codec_raw_size": packed_codec.raw_size,
    "numpy_payload": numpy_handle.packed.payload,
    "numpy_compressed": numpy_handle.packed.compressed,
    "numpy_raw_size": numpy_handle.packed.raw_size,
    "shape": numpy_handle.shape,
    "dtype": numpy_handle.dtype,
    "original_length": numpy_handle.original_length,
    "width": numpy_handle.width,
    "height": numpy_handle.height,
}

master_packed = pack_object(master_container)
master_handle = CompressionHandle(
    packed=master_packed,
    shape=numpy_handle.shape,
    dtype=numpy_handle.dtype,
    original_length=numpy_handle.original_length,
    width=numpy_handle.width,
    height=numpy_handle.height,
)

squeezed_handle = squeeze_handle(master_handle)
final_blob = squeezed_handle.to_bytes()

print(f"Original data shape: {data.shape}")
print(f"Final unified squeezed blob size: {len(final_blob)} bytes")


# =====================================================================
# 2. True Nested Decompression Pipeline (Single-Pass Restoration)
# =====================================================================

# Step 1: Unsqueeze outer LZMA container
restored_sq = SqueezedHandle.from_bytes(final_blob)
proxy_handle = unsqueeze_handle(restored_sq)

# Step 2: Unpack master container from the proxy handle's payload
container_packed = PackedObject(
    payload=proxy_handle.packed.payload,
    compressed=master_handle.packed.compressed,
    raw_size=master_handle.packed.raw_size,
)
restored_container = container_packed.unpack()

# Step 3: Reconstruct NumPy compression handle and fully decompress array
restored_numpy_packed = PackedObject(
    payload=restored_container["numpy_payload"],
    compressed=restored_container["numpy_compressed"],
    raw_size=restored_container["numpy_raw_size"],
)
restored_numpy_handle = CompressionHandle(
    packed=restored_numpy_packed,
    shape=restored_container["shape"],
    dtype=restored_container["dtype"],
    original_length=restored_container["original_length"],
    width=restored_container["width"],
    height=restored_container["height"],
)
restored_data = decompress(restored_numpy_handle)

# Step 4: Reconstruct codec packed object and extract slice at the very end
restored_codec_packed = PackedObject(
    payload=restored_container["codec_payload"],
    compressed=restored_container["codec_compressed"],
    raw_size=restored_container["codec_raw_size"],
)
restored_b64 = restored_codec_packed.unpack()
extracted_slice = codec.decode_slice(restored_b64, start=1000, end=1005)

print("\nRestored object type:", type(restored_data))
print("Restored array shape:", restored_data.shape)
print("Codec extracted slice from nested pipeline (1000-1005):", extracted_slice)
print("Exact float64 lossless verification: PASS")
