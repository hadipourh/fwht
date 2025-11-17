#!/usr/bin/env python3
"""Test fp32/fp16 with proper precision-matched reference."""

import numpy as np
import torch
import pyfwht

def test_fp32_precision_matched():
    """Test fp32 against fp32 CPU reference (numpy)."""
    print("Testing fp32 with precision-matched reference...")
    
    n = 1024
    batch_size = 10
    np.random.seed(42)
    
    # Generate test data in fp32
    data_original = np.random.randn(batch_size, n).astype(np.float32)
    
    # Compute CPU reference in fp32 using numpy's Hadamard-equivalent
    # (iterative butterfly on each batch element)
    data_ref = data_original.copy()
    for b in range(batch_size):
        arr = data_ref[b]
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    u = arr[j]
                    v = arr[j + h]
                    arr[j] = u + v
                    arr[j + h] = u - v
            h *= 2
    
    # Compute GPU fp32
    data_gpu = torch.tensor(data_original, dtype=torch.float32, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_gpu)
    result_gpu = data_gpu.cpu().numpy()
    
    # Compare
    error = np.abs(result_gpu - data_ref).max()
    print(f"Max error: {error:.2e}")
    print(f"Expected: ~1e-6 (fp32 machine precision)")
    
    if error < 1e-5:
        print("✓ PASS - fp32 GPU matches fp32 CPU reference!")
        return True
    else:
        print("✗ FAIL - unexpected error")
        print(f"Sample GPU: {result_gpu[0, :8]}")
        print(f"Sample CPU: {data_ref[0, :8]}")
        return False

if __name__ == '__main__':
    test_fp32_precision_matched()
