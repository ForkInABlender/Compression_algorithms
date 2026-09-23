import numpy as np


MASK64 = (1 << 64) - 1
EMPTY = np.uint64(MASK64)


class VirtualFloat64Array:
    """
    Sparse virtual float64 array.

    The logical array may be extremely large, but every element is
    implicitly 0.0 until explicitly assigned.

    Nonzero values are stored in a single contiguous open-addressing
    hash table. No Python dictionary and no chunking are used.
    """

    def __init__(self, size, seed=0, initial_capacity=16):
        size = int(size)

        if size < 0:
            raise ValueError("size must be non-negative")

        self.shape = (size,)
        self.dtype = np.dtype(np.float64)
        self.seed = int(seed) & MASK64

        capacity = 1

        while capacity < max(4, int(initial_capacity)):
            capacity <<= 1

        self._keys = np.full(
            capacity,
            EMPTY,
            dtype=np.uint64,
        )

        self._vals = np.empty(
            capacity,
            dtype=np.float64,
        )

        self._count = 0

    @property
    def size(self):
        return self.shape[0]

    @property
    def ndim(self):
        return 1

    @property
    def nbytes(self):
        # Logical size, not physical allocation.
        return self.size * 8

    @property
    def stored_nbytes(self):
        # Physical memory occupied by the sparse table.
        return self._keys.nbytes + self._vals.nbytes

    @property
    def stored_count(self):
        return self._count

    def _normalize_index(self, index):
        i = int(index)

        if i < 0:
            i += self.size

        if i < 0 or i >= self.size:
            raise IndexError("index out of bounds")

        return i

    @staticmethod
    def _hash(x):
        """
        64-bit SplitMix-style integer hash.
        """
        x = (x + 0x9E3779B97F4A7C15) & MASK64

        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & MASK64

        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & MASK64

        x ^= x >> 31

        return x & MASK64

    def _find_slot(self, key):
        mask = len(self._keys) - 1
        pos = self._hash(key) & mask

        while True:
            existing = int(self._keys[pos])

            if existing == MASK64:
                return pos, False

            if existing == key:
                return pos, True

            pos = (pos + 1) & mask

    def _resize(self, new_capacity):
        old_keys = self._keys
        old_vals = self._vals

        self._keys = np.full(
            new_capacity,
            EMPTY,
            dtype=np.uint64,
        )

        self._vals = np.empty(
            new_capacity,
            dtype=np.float64,
        )

        old_count = self._count
        self._count = 0

        for pos in range(len(old_keys)):
            key = int(old_keys[pos])

            if key != MASK64:
                slot, found = self._find_slot(key)

                self._keys[slot] = np.uint64(key)
                self._vals[slot] = old_vals[pos]

                self._count += 1

        if self._count != old_count:
            raise RuntimeError("hash table resize corruption")

    def _ensure_capacity(self):
        # Keep load factor <= 0.5.
        if (self._count + 1) * 2 >= len(self._keys):
            self._resize(len(self._keys) * 2)

    def _get_stored(self, i):
        slot, found = self._find_slot(i)

        if found:
            return self._vals[slot]

        return np.float64(0.0)

    def _set_stored(self, i, value):
        value = np.float64(value)

        slot, found = self._find_slot(i)

        if value == 0.0:
            if found:
                self._delete_slot(slot)
            return

        if found:
            self._vals[slot] = value
            return

        self._ensure_capacity()

        slot, found = self._find_slot(i)

        if found:
            self._vals[slot] = value
            return

        self._keys[slot] = np.uint64(i)
        self._vals[slot] = value
        self._count += 1

    def _delete_slot(self, slot):
        """
        Delete from an open-addressing table while preserving
        probe chains.
        """
        mask = len(self._keys) - 1

        self._keys[slot] = EMPTY
        self._count -= 1

        pos = (slot + 1) & mask

        while self._keys[pos] != EMPTY:
            key = int(self._keys[pos])
            value = self._vals[pos]

            self._keys[pos] = EMPTY
            self._count -= 1

            new_slot, found = self._find_slot(key)

            if found:
                raise RuntimeError(
                    "hash table deletion corruption"
                )

            self._keys[new_slot] = np.uint64(key)
            self._vals[new_slot] = value
            self._count += 1

            pos = (pos + 1) & mask

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            i = self._normalize_index(index)
            return self._get_stored(i)

        if isinstance(index, slice):
            start, stop, step = index.indices(self.size)

            if step != 1:
                raise ValueError(
                    "only contiguous slices are supported"
                )

            count = max(0, stop - start)

            result = np.zeros(
                count,
                dtype=np.float64,
            )

            for j, i in enumerate(range(start, stop)):
                result[j] = self._get_stored(i)

            return result

        raise TypeError(
            "index must be an integer or slice"
        )

    def __setitem__(self, index, value):
        if isinstance(index, (int, np.integer)):
            i = self._normalize_index(index)
            self._set_stored(i, value)
            return

        if isinstance(index, slice):
            start, stop, step = index.indices(self.size)

            if step != 1:
                raise ValueError(
                    "only contiguous slices are supported"
                )

            count = max(0, stop - start)

            if np.isscalar(value):
                value = np.float64(value)

                for i in range(start, stop):
                    self._set_stored(i, value)

                return

            values = np.asarray(
                value,
                dtype=np.float64,
            )

            if values.ndim != 1:
                raise ValueError(
                    "assigned array must be one-dimensional"
                )

            if values.size != count:
                raise ValueError(
                    f"could not broadcast input array from "
                    f"shape {values.shape} into shape ({count},)"
                )

            for i, v in zip(
                range(start, stop),
                values,
            ):
                self._set_stored(i, v)

            return

        raise TypeError(
            "index must be an integer or slice"
        )
