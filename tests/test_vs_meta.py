#!/usr/bin/env python3
"""
Simple test to verify pyfwht and Meta library produce the same results.
"""

import numpy as np
import torch

# Import libraries
import pyfwht
try:
    import fast_hadamard_transform
    meta_available = True
except ImportError:
    meta_available = False
    print("Meta library not available, skipping comparison")

def test_single_transform(n=4096):
    """Test single transform correctness"""
    print(f"\n{'='*60}")
    print(f"Testing single transform (n={n})")
    print(f"{'='*60}")
    
    # Create random input - use different seed for each size
    np.random.seed(42 + n)  # Different seed per size
    x = np.random.randn(n).astype(np.float64)  # Use float64 for pyfwht
    
    # pyfwht GPU (in-place transform)
    x_pyfwht = x.copy()
    pyfwht.transform(x_pyfwht, backend='gpu')
    result_pyfwht = x_pyfwht
    
    if meta_available:
        # Meta library (requires torch tensor, float16)
        # NOTE: Meta's scale parameter defaults to 1.0 (no normalization)
        # We call it with scale=1.0 to get the raw transform like pyfwht
        x_torch = torch.from_numpy(x.astype(np.float32)).cuda().half()
        result_meta = fast_hadamard_transform.hadamard_transform(x_torch, scale=1.0)
        result_meta_np = result_meta.float().cpu().numpy()
        
        # Compare (both are unnormalized now)
        diff = np.abs(result_pyfwht - result_meta_np)
        max_diff = np.max(diff)
        rel_error = max_diff / (np.abs(result_pyfwht).max() + 1e-10)
        
        print(f"\nResults comparison:")
        print(f"  pyfwht[0:5]:  {result_pyfwht[:5]}")
        print(f"  Meta[0:5]:    {result_meta_np[:5]}")
        print(f"  Max diff:     {max_diff:.6f}")
        print(f"  Relative err: {rel_error:.6e}")
        
        if rel_error < 1e-3:  # float16 has limited precision
            print(f"  ✓ Results match within tolerance!")
        else:
            print(f"  ✗ Results differ significantly!")
            print(f"\nFirst 10 values:")
            for i in range(10):
                print(f"    [{i}] pyfwht: {result_pyfwht[i]:.4f}, Meta: {result_meta_np[i]:.4f}, diff: {diff[i]:.4f}")
    else:
        print(f"\npyfwht result[0:5]: {result_pyfwht[:5]}")
    
    return result_pyfwht

def test_batch_transform(n=2048, batch=10):
    """Test batch transform correctness"""
    print(f"\n{'='*60}")
    print(f"Testing batch transform (n={n}, batch={batch})")
    print(f"{'='*60}")
    
    # Create random input
    np.random.seed(100)  # Different seed for batch test
    x_2d = np.random.randn(batch, n).astype(np.float64)
    
    # pyfwht GPU - transform each row separately
    result_pyfwht = x_2d.copy()
    print(f"  Transforming {batch} arrays of size {n}...")
    for i in range(batch):
        pyfwht.transform(result_pyfwht[i], backend='gpu')
    print(f"  First array after transform: {result_pyfwht[0,:5]}")
    
    if meta_available:
        # Meta library (requires torch tensor, float16)
        x_torch = torch.from_numpy(x_2d.astype(np.float32)).cuda().half()
        result_meta = fast_hadamard_transform.hadamard_transform(x_torch, scale=1.0)
        result_meta_np = result_meta.float().cpu().numpy()
        
        # Compare
        diff = np.abs(result_pyfwht - result_meta_np)
        max_diff = np.max(diff)
        rel_error = max_diff / (np.abs(result_pyfwht).max() + 1e-10)
        
        print(f"\nResults comparison:")
        print(f"  pyfwht[0,0:5]:  {result_pyfwht[0,:5]}")
        print(f"  Meta[0,0:5]:    {result_meta_np[0,:5]}")
        print(f"  Max diff:       {max_diff:.6f}")
        print(f"  Relative err:   {rel_error:.6e}")
        
        if rel_error < 1e-3:
            print(f"  ✓ Results match within tolerance!")
        else:
            print(f"  ✗ Results differ significantly!")
    else:
        print(f"\npyfwht result[0,0:5]: {result_pyfwht[0,:5]}")
    
    return result_pyfwht

if __name__ == '__main__':
    print(f"pyfwht version: {pyfwht.__version__}")
    if meta_available:
        print(f"Meta version: {fast_hadamard_transform.__version__}")
    
    # Test single transforms of different sizes
    for size in [1024, 2048, 4096]:
        test_single_transform(n=size)
    
    # Test batch
    test_batch_transform(n=2048, batch=10)
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"{'='*60}")
