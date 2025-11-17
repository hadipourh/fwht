#!/usr/bin/env python3
"""
Batch Transform Benchmark: pyfwht vs Meta's fast-hadamard-transform

Compares performance for batch processing (multiple transforms).
"""

import numpy as np
import torch
import time
import pyfwht

try:
    import fast_hadamard_transform
    meta_available = True
except ImportError:
    meta_available = False
    print("Meta library not available. Install with: pip install fast-hadamard-transform")
    exit(1)

def benchmark_pyfwht_batch(n, batch_size, repeats=100, warmup=10):
    """Benchmark pyfwht batch transforms using native batch API"""
    # Create batch data (2D array)
    data = np.random.randn(batch_size, n).astype(np.float64)
    
    # Warmup
    for _ in range(warmup):
        test_data = data.copy()
        pyfwht.gpu.batch_transform_f64(test_data)
    
    # Timed runs
    times = []
    for _ in range(repeats):
        test_data = data.copy()
        start = time.perf_counter()
        pyfwht.gpu.batch_transform_f64(test_data)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms
    
    return np.mean(times), np.std(times)

def benchmark_meta_batch(n, batch_size, repeats=100, warmup=10):
    """Benchmark Meta batch transforms"""
    # Create batch data (Meta uses float16)
    data = torch.randn(batch_size, n, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        _ = fast_hadamard_transform.hadamard_transform(data, scale=1.0)
    torch.cuda.synchronize()
    
    # Timed runs - MEASURE ACTUAL WORK
    times = []
    for _ in range(repeats):
        # Create fresh data each time to avoid caching
        test_data = torch.randn(batch_size, n, dtype=torch.float16, device='cuda')
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fast_hadamard_transform.hadamard_transform(test_data, scale=1.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # ms
        # Force materialization
        _ = result[0, 0].item()
    
    return np.mean(times), np.std(times)

def main():
    print("="*80)
    print("FWHT Batch Transform Benchmark: pyfwht vs Meta")
    print("="*80)
    print()
    
    # Test different sizes and batch sizes
    test_configs = [
        # (n, batch_size)
        (1024, 1),
        (1024, 10),
        (1024, 100),
        (1024, 1000),
        (2048, 1),
        (2048, 10),
        (2048, 100),
        (2048, 1000),
        (4096, 1),
        (4096, 10),
        (4096, 100),
        (4096, 1000),
    ]
    
    print(f"{'n':>6} {'Batch':>6} {'pyfwht (ms)':>15} {'Meta (ms)':>15} {'Speedup':>10}")
    print("-"*80)
    
    for n, batch in test_configs:
        # Benchmark both
        pyfwht_time, pyfwht_std = benchmark_pyfwht_batch(n, batch, repeats=100, warmup=10)
        meta_time, meta_std = benchmark_meta_batch(n, batch, repeats=100, warmup=10)
        
        speedup = pyfwht_time / meta_time
        
        print(f"{n:>6} {batch:>6} {pyfwht_time:>11.3f}±{pyfwht_std:>5.2f} "
              f"{meta_time:>11.3f}±{meta_std:>5.2f} {speedup:>9.2f}x")
    
    print()
    print("Notes:")
    print("  - Speedup > 1.0: Meta is faster")
    print("  - Speedup < 1.0: pyfwht is faster")
    print("  - pyfwht: float64, native batch API (gpu.batch_transform_f64)")
    print("  - Meta: float16, native batch support")

if __name__ == "__main__":
    main()
