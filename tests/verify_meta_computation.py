#!/usr/bin/env python3
"""
Verify that Meta actually computes the transform correctly.
Check if the constant time is due to lazy evaluation or caching.
"""

import torch
import numpy as np
import time

try:
    import fast_hadamard_transform
    import pyfwht
except ImportError as e:
    print(f"Error: {e}")
    exit(1)

def test_meta_actually_computes():
    """Test if Meta actually computes or returns lazy/cached results"""
    print("="*80)
    print("Testing if Meta actually computes the transform")
    print("="*80)
    
    for n in [1024, 2048, 4096]:
        for batch in [10, 100, 1000]:
            print(f"\nTest: n={n}, batch={batch}")
            
            # Create test data with known pattern
            data_np = np.random.randn(batch, n).astype(np.float32)
            
            # pyfwht reference (float64)
            data_pyfwht = torch.from_numpy(data_np.astype(np.float64)).cuda()
            pyfwht.gpu.batch_transform_dlpack(data_pyfwht)
            torch.cuda.synchronize()
            result_pyfwht = data_pyfwht.cpu().numpy().astype(np.float32)
            
            # Meta library (float16)
            data_meta = torch.from_numpy(data_np).cuda().half()
            
            # Time it with forced materialization
            torch.cuda.synchronize()
            start = time.perf_counter()
            result_meta = fast_hadamard_transform.hadamard_transform(data_meta, scale=1.0)
            torch.cuda.synchronize()
            # Force materialization by accessing data
            _ = result_meta[0, 0].item()
            _ = result_meta[-1, -1].item()
            elapsed = time.perf_counter() - start
            
            result_meta_np = result_meta.float().cpu().numpy()
            
            # Check correctness
            max_diff = np.max(np.abs(result_pyfwht - result_meta_np))
            rel_err = max_diff / np.max(np.abs(result_meta_np))
            
            print(f"  Time: {elapsed*1000:.3f}ms")
            print(f"  Max diff: {max_diff:.3f}")
            print(f"  Rel err: {rel_err:.6f}")
            
            if max_diff > 100:  # Large error = likely not computing
                print(f"  ❌ SUSPICIOUS! Large error suggests Meta might not be computing correctly")
            else:
                print(f"  ✓ Results match (within fp16 precision)")

def test_meta_scaling_with_size():
    """Test if Meta's time actually scales with problem size"""
    print("\n" + "="*80)
    print("Testing if Meta's time scales with problem size")
    print("="*80)
    print()
    
    configs = [
        (1024, 1),
        (2048, 1),
        (4096, 1),
        (8192, 1),
        (1024, 1000),
        (2048, 1000),
        (4096, 1000),
    ]
    
    print(f"{'n':>6} {'Batch':>6} {'Time (ms)':>12} {'GOps/s':>10}")
    print("-"*80)
    
    for n, batch in configs:
        data = torch.randn(batch, n, dtype=torch.float16, device='cuda')
        
        # Warmup
        for _ in range(20):
            _ = fast_hadamard_transform.hadamard_transform(data, scale=1.0)
        torch.cuda.synchronize()
        
        # Timed run
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = fast_hadamard_transform.hadamard_transform(data, scale=1.0)
            torch.cuda.synchronize()
            # Force materialization
            _ = result.sum().item()
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        
        mean_time = np.mean(times)
        
        # Calculate throughput
        ops = batch * n * np.log2(n)
        gops = (ops / 1e9) / (mean_time / 1000)
        
        print(f"{n:>6} {batch:>6} {mean_time:>12.3f} {gops:>10.1f}")

if __name__ == "__main__":
    test_meta_actually_computes()
    test_meta_scaling_with_size()
