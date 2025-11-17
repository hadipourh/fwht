#!/usr/bin/env python3
"""
Detailed analysis of batch performance to understand where time is spent.
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

def profile_pyfwht_batch(n, batch_size):
    """Profile pyfwht batch with detailed timing"""
    print(f"\n{'='*80}")
    print(f"Profiling pyfwht: n={n}, batch={batch_size}")
    print(f"{'='*80}")
    
    # Enable GPU profiling
    pyfwht.gpu.set_profiling(True)
    
    # Create data
    data = np.random.randn(batch_size, n).astype(np.float64)
    
    # Warmup
    for _ in range(10):
        test_data = data.copy()
        pyfwht.gpu.batch_transform_f64(test_data)
    
    # Timed run with profiling
    test_data = data.copy()
    start = time.perf_counter()
    pyfwht.gpu.batch_transform_f64(test_data)
    total_time = (time.perf_counter() - start) * 1000
    
    # Get metrics
    metrics = pyfwht.gpu.get_last_metrics()
    
    print(f"Total time:    {total_time:.3f} ms")
    if metrics['valid']:
        print(f"H2D time:      {metrics['h2d_ms']:.3f} ms ({metrics['h2d_ms']/total_time*100:.1f}%)")
        print(f"Kernel time:   {metrics['kernel_ms']:.3f} ms ({metrics['kernel_ms']/total_time*100:.1f}%)")
        print(f"D2H time:      {metrics['d2h_ms']:.3f} ms ({metrics['d2h_ms']/total_time*100:.1f}%)")
        print(f"Overhead:      {total_time - metrics['h2d_ms'] - metrics['kernel_ms'] - metrics['d2h_ms']:.3f} ms")
        print(f"Bytes:         {metrics['bytes_transferred']:,}")
        
        # Calculate bandwidth
        total_cuda_time = metrics['h2d_ms'] + metrics['kernel_ms'] + metrics['d2h_ms']
        bandwidth_gbps = (metrics['bytes_transferred'] / 1e9) / (total_cuda_time / 1000)
        print(f"Bandwidth:     {bandwidth_gbps:.1f} GB/s")
        
        # Calculate kernel throughput
        total_ops = batch_size * n * np.log2(n)
        kernel_gops = (total_ops / 1e9) / (metrics['kernel_ms'] / 1000)
        print(f"Kernel GOps/s: {kernel_gops:.1f}")
    
    pyfwht.gpu.set_profiling(False)

def profile_meta_batch(n, batch_size):
    """Profile Meta batch"""
    if not meta_available:
        return
        
    print(f"\n{'='*80}")
    print(f"Profiling Meta: n={n}, batch={batch_size}")
    print(f"{'='*80}")
    
    # Create data
    data = torch.randn(batch_size, n, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = fast_hadamard_transform.hadamard_transform(data, scale=1.0)
    torch.cuda.synchronize()
    
    # Timed run
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fast_hadamard_transform.hadamard_transform(data, scale=1.0)
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time:    {total_time:.3f} ms")
    
    # Calculate throughput
    total_ops = batch_size * n * np.log2(n)
    gops = (total_ops / 1e9) / (total_time / 1000)
    print(f"Kernel GOps/s: {gops:.1f}")
    
    # Data size
    bytes_data = batch_size * n * 2  # float16
    print(f"Data size:     {bytes_data:,} bytes ({bytes_data/1e6:.1f} MB)")

def main():
    print("="*80)
    print("Detailed Batch Performance Analysis")
    print("="*80)
    
    test_configs = [
        (1024, 1),
        (1024, 10),
        (1024, 100),
        (1024, 1000),
        (4096, 1),
        (4096, 10),
        (4096, 100),
        (4096, 1000),
    ]
    
    for n, batch in test_configs:
        profile_pyfwht_batch(n, batch)
        if meta_available:
            profile_meta_batch(n, batch)

if __name__ == "__main__":
    main()
