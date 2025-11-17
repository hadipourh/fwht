"""
Bit-packed Boolean WHT example using pyfwht.

Demonstrates:
- Packing a Boolean truth table into uint64 words
- Computing WHT via fwht.boolean_packed
- Verifying against fwht.from_bool (unpacked)
"""

import numpy as np
import pyfwht as fwht


def pack_bits(truth_table: np.ndarray) -> np.ndarray:
    if truth_table.dtype != np.uint8:
        truth_table = truth_table.astype(np.uint8)
    n = truth_table.size
    n_words = (n + 63) // 64
    packed = np.zeros(n_words, dtype=np.uint64)
    for i in range(n):
        if truth_table[i]:
            packed[i // 64] |= (1 << (i % 64))
    return packed


def main():
    # Small example: [0,1,1,0,1,0,0,1] → 0x96
    truth = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    packed = pack_bits(truth)

    wht_ref = fwht.from_bool(truth, signed=True)
    wht_pk = fwht.boolean_packed(packed, n=truth.size)

    print("Truth:", truth.tolist())
    print("Packed (hex):", [hex(int(x)) for x in packed])
    print("WHT (ref):   ", wht_ref.tolist())
    print("WHT (packed):", wht_pk.tolist())
    assert np.array_equal(wht_ref, wht_pk)
    print("✓ Match for n=8")

    # Larger example
    n = 256
    rng = np.random.default_rng(42)
    truth = rng.integers(0, 2, size=n, dtype=np.uint8)
    packed = pack_bits(truth)

    wht_ref = fwht.from_bool(truth, signed=True)
    wht_pk = fwht.boolean_packed(packed, n=n)
    assert np.array_equal(wht_ref, wht_pk)
    print("✓ Match for n=256")


if __name__ == "__main__":
    main()
