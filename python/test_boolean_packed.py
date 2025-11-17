#!/usr/bin/env python3
"""
Test bit-sliced Boolean WHT Python bindings.
"""

import numpy as np
import pyfwht as fwht

def test_boolean_packed():
    """Test bit-packed Boolean WHT."""
    print("Testing bit-packed Boolean WHT...")
    
    # XOR function: [0,1,1,0,1,0,0,1]
    # Pack into single uint64: bits 1,2,4,7 set → 0x96
    packed = np.array([0x96], dtype=np.uint64)
    wht_packed = fwht.boolean_packed(packed, n=8)
    
    # Compare with unpacked version
    truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    wht_unpacked = fwht.from_bool(truth_table, signed=True)
    
    print(f"  Truth table: {truth_table}")
    print(f"  Packed (0x{packed[0]:02x})")
    print(f"  WHT (packed):   {wht_packed}")
    print(f"  WHT (unpacked): {wht_unpacked}")
    
    if np.array_equal(wht_packed, wht_unpacked):
        print("  ✓ Packed and unpacked results match!\n")
        return True
    else:
        print("  ✗ Mismatch!\n")
        return False

def test_larger_function():
    """Test with n=256."""
    print("Testing larger Boolean function (n=256)...")
    
    # Create random Boolean function
    np.random.seed(42)
    truth_table = np.random.randint(0, 2, 256, dtype=np.uint8)
    
    # Pack into uint64 array
    n_words = (256 + 63) // 64
    packed = np.zeros(n_words, dtype=np.uint64)
    for i in range(256):
        if truth_table[i]:
            word_idx = i // 64
            bit_idx = i % 64
            packed[word_idx] |= (1 << bit_idx)
    
    # Compute both versions
    wht_packed = fwht.boolean_packed(packed, n=256)
    wht_unpacked = fwht.from_bool(truth_table, signed=True)
    
    match = np.array_equal(wht_packed, wht_unpacked)
    
    if match:
        print(f"  ✓ Results match for n=256")
        print(f"  Memory: Packed={packed.nbytes}B, Unpacked={truth_table.nbytes}B")
        print(f"  Savings: {truth_table.nbytes / packed.nbytes:.1f}×\n")
        return True
    else:
        print(f"  ✗ Mismatch!\n")
        return False

def test_backend_selection():
    """Test backend selection."""
    print("Testing backend selection...")
    
    truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    packed = np.array([0x96], dtype=np.uint64)
    
    # Test CPU backend
    wht_cpu = fwht.boolean_packed(packed, n=8, backend=fwht.Backend.CPU)
    
    # Test AUTO backend
    wht_auto = fwht.boolean_packed(packed, n=8, backend=fwht.Backend.AUTO)
    
    wht_ref = fwht.from_bool(truth_table, signed=True)
    
    if np.array_equal(wht_cpu, wht_ref) and np.array_equal(wht_auto, wht_ref):
        print("  ✓ Backend selection works\n")
        return True
    else:
        print("  ✗ Backend mismatch\n")
        return False

def test_memory_efficiency():
    """Demonstrate memory savings."""
    print("Memory efficiency demonstration:")
    print("-" * 50)
    
    for n in [256, 1024, 4096, 16384, 65536]:
        n_words = (n + 63) // 64
        packed_bytes = n_words * 8  # uint64 = 8 bytes
        unpacked_bytes = n  # uint8 = 1 byte per element
        int32_bytes = n * 4  # int32 WHT output
        
        savings = unpacked_bytes / packed_bytes
        total_savings = int32_bytes / packed_bytes
        
        print(f"  n={n:5d}: Packed={packed_bytes:5d}B, "
              f"Unpacked={unpacked_bytes:6d}B, "
              f"Savings={savings:4.1f}×")
    
    print()
    return True

def main():
    print("=" * 60)
    print("Bit-Sliced Boolean WHT - Python Bindings Test")
    print("=" * 60)
    print()
    
    results = []
    results.append(test_boolean_packed())
    results.append(test_larger_function())
    results.append(test_backend_selection())
    results.append(test_memory_efficiency())
    
    print("=" * 60)
    if all(results):
        print("All tests passed! ✓")
    else:
        print(f"Some tests failed ({sum(results)}/{len(results)} passed)")
    print("=" * 60)

if __name__ == "__main__":
    main()
