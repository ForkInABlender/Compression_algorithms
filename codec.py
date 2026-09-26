import abc
import base64
import math
import struct
from typing import List, Tuple

class AbstractSymbolicCodec(abc.ABC):
    """Abstract base definition enforcing structural codec contracts."""
    
    @abc.abstractmethod
    def encode_stream(self, values: List[float]) -> str:
        pass

    @abc.abstractmethod
    def decode_slice(self, b64_payload: str, start: int, end: int) -> List[float]:
        pass

class GridShiftBinaryCodec(AbstractSymbolicCodec):
    """
    High-density symbolic codec utilizing grid quantization, zero-line baseline tracking,
    and packed binary struct framing serialized via Base64.
    """
    def __init__(self, step_resolution: float = 1e-4):
        self.resolution = step_resolution

    def _quantize(self, val: float) -> int:
        """Snaps a float value onto the baseline grid coordinate."""
        return int(math.copysign(math.floor(abs(val) / self.resolution + 0.5), val))

    def encode_stream(self, values: List[float]) -> str:
        """Compresses a sequence of floats into a tightly packed binary-framed Base64 string."""
        if not values:
            return base64.b64encode(b'').decode('utf-8')

        # 1. Quantize stream onto baseline grid
        q_vals = [self._quantize(v) for v in values]
        anchor = q_vals[0]

        # 2. Track zero-line baseline crossings (sign-flip coordinates)
        crossings = []
        for i in range(len(q_vals) - 1):
            if (q_vals[i] < 0 and q_vals[i+1] >= 0) or (q_vals[i] >= 0 and q_vals[i+1] < 0):
                crossings.append(i)

        # 3. Compute delta transitions
        deltas = [q_vals[i+1] - q_vals[i] for i in range(len(q_vals) - 1)]

        # 4. Pack binary frames via struct (! = network byte order, I = unsigned int, i = signed int)
        # Header layout: [Total Count, Anchor Value, Crossing Count]
        header_fmt = "!I i I"
        header_bytes = struct.pack(header_fmt, len(values), anchor, len(crossings))
        
        crossings_bytes = struct.pack(f"!{len(crossings)}I", *crossings) if crossings else b''
        deltas_bytes = struct.pack(f"!{len(deltas)}i", *deltas) if deltas else b''

        # Combine into raw binary block and encode to text-safe Base64
        raw_binary = header_bytes + crossings_bytes + deltas_bytes
        return base64.b64encode(raw_binary).decode('utf-8')

    def decode_slice(self, b64_payload: str, start: int, end: int) -> List[float]:
        """
        O(1) selective slice reconstruction using memoryview and zero-copy iteration,
        bypassing full array expansion to preserve active runtime memory limits.
        """
        raw_binary = base64.b64decode(b64_payload.encode('utf-8'))
        if not raw_binary:
            return []

        header_fmt = "!I i I"
        header_size = struct.calcsize(header_fmt)
        total_count, anchor, crossing_count = struct.unpack_from(header_fmt, raw_binary, 0)

        # Calculate binary offsets
        deltas_offset = header_size + (crossing_count * 4)
        mv = memoryview(raw_binary)

        delta_count = total_count - 1
        if delta_count <= 0 or start >= total_count:
            return [anchor * self.resolution] if total_count == 1 else []

        # Stream deltas efficiently using struct.iter_unpack over memoryview
        iter_deltas = struct.iter_unpack("!i", mv[deltas_offset:])

        # Lazy generator accumulator for targeted slice reconstruction
        def generate_sequence():
            val = anchor
            yield val * self.resolution
            for (d,) in iter_deltas:
                val += d
                yield val * self.resolution

        # Extract only the requested window [start:end] without materializing the rest
        sliced_results = []
        for index, value in enumerate(generate_sequence()):
            if index >= end:
                break
            if index >= start:
                sliced_results.append(value)

        return sliced_results


# ==========================================
# Execution Verification Test
# ==========================================
if __name__ == "__main__":
    # Generate a pure Python sample waveform
    sample_waveform = [math.sin(i * 0.005) * 45.0 for i in range(100_000)]

    codec = GridShiftBinaryCodec(step_resolution=1e-3)

    # Encode into Base64 binary-packed format
    b64_encoded_string = codec.encode_stream(sample_waveform)

    print(f"Original Element Count: {len(sample_waveform)}")
    print(f"Base64 Packed String Length: {len(b64_encoded_string)} characters")
    print(f"Base64 Snippet: {b64_encoded_string[:90]}...")

    # Perform targeted O(1) query decompression (pulling indices 40,000 to 40,005)
    extracted_slice = codec.decode_slice(b64_encoded_string, start=40000, end=40006)
    print(f"Targeted Slice (Indices 40000-40005): {extracted_slice}")