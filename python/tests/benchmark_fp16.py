#!/usr/bin/env python3
"""
Benchmark fp16/fp32 kernels vs fp64 and Meta's implementation.

This script tests the Meta-inspired kernels to see if we match their performance.
"""

import torch
import numpy as np
import time
import sys

try:
    import pyfwht
except ImportError:
    print("ERROR: pyfwht not installed. Run: pip install -e .")
    sys.exit(1)

try:
    from hadamard_transform import hadamard_transform as meta_hadamard
    HAS_META = True
except ImportError:
    print("WARNING: Meta library not installed. Will only test pyfwht kernels.")
    HAS_META = False


def benchmark_pyfwht_dtype(n, batch_size, dtype, num_trials=100):
    """Benchmark pyfwht with specific dtype."""
    # Create data on GPU
    data = torch.randn(batch_size, n, dtype=dtype, device='cuda')
    
    # Warmup
    for _ in range(10):
        pyfwht.gpu.batch_transform_dlpack(data)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        pyfwht.gpu.batch_transform_dlpack(data)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / num_trials
    ops = batch_size * n * np.log2(n)
    gops_per_sec = ops / avg_time / 1e9
    
    return avg_time * 1000, gops_per_sec  # Return ms and GOps/s


def benchmark_meta(n, batch_size, num_trials=100):
    """Benchmark Meta's library (fp16 only)."""
    if not HAS_META:
        return None, None
    
    # Meta only supports fp16
    data = torch.randn(batch_size, n, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(10):
        meta_hadamard(data)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        meta_hadamard(data)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time = (end - start) / num_trials
    ops = batch_size * n * np.log2(n)
    gops_per_sec = ops / avg_time / 1e9
    
    return avg_time * 1000, gops_per_sec


def verify_correctness(n=1024, batch_size=10):
    """Verify that all kernels produce correct results."""
    print("Verifying correctness...")
    
    # Reference: fp64 on CPU
    data_ref = np.random.randn(batch_size, n)
    
    for i in range(batch_size):
        pyfwht.transform(data_ref[i])  # CPU reference
    
    # Test fp64 GPU
    data_f64 = torch.tensor(data_ref, dtype=torch.float64, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_f64)
    diff_f64 = np.abs(data_f64.cpu().numpy() - data_ref).max()
    
    # Test fp32 GPU
    data_f32 = torch.tensor(data_ref, dtype=torch.float32, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_f32)
    diff_f32 = np.abs(data_f32.cpu().numpy() - data_ref).max()
    
    # Test fp16 GPU
    data_f16 = torch.tensor(data_ref, dtype=torch.float16, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_f16)
    diff_f16 = np.abs(data_f16.cpu().numpy() - data_ref).max()
    
    print(f"  fp64 max error: {diff_f64:.2e} (should be ~1e-15)")
    print(f"  fp32 max error: {diff_f32:.2e} (should be ~1e-6)")
    print(f"  fp16 max error: {diff_f16:.2e} (should be ~1e-3)")
    
    # Verify Meta if available
    if HAS_META:
        data_meta = torch.tensor(data_ref, dtype=torch.float16, device='cuda')
        meta_hadamard(data_meta)
        diff_meta = np.abs(data_meta.cpu().numpy() - data_ref).max()
        print(f"  Meta fp16 max error: {diff_meta:.2e}")
    
    print()
    
    # Check if errors are acceptable
    if diff_f64 > 1e-12:
        print("WARNING: fp64 error too large!")
    if diff_f32 > 1e-5:
        print("WARNING: fp32 error too large!")
    if diff_f16 > 0.1:
        print("WARNING: fp16 error too large!")
    
    return diff_f64 < 1e-12 and diff_f32 < 1e-5 and diff_f16 < 0.1


def main():
    print("=" * 80)
    print("FP16/FP32 Kernel Benchmark - Meta-Inspired Implementation")
    print("=" * 80)
    print()
    
    if not pyfwht.has_gpu():
        print("ERROR: GPU support not available")
        return
    
    # Verify correctness first
    if not verify_correctness():
        print("ERROR: Correctness check failed!")
        return
    
    print("Testing configurations (batch=1000, num_trials=100):")
    print()
    
    configs = [
        (1024, 1000),
        (2048, 1000),
        (4096, 1000),
    ]
    
    print(f"{'Size':<8} {'Type':<10} {'Time (ms)':<12} {'GOps/s':<12} {'vs fp64':<12} {'vs Meta':<12}")
    print("-" * 80)
    
    for n, batch_size in configs:
        # Benchmark fp64 (baseline precision)
        time_f64, gops_f64 = benchmark_pyfwht_dtype(n, batch_size, torch.float64)
        print(f"{n:<8} {'fp64':<10} {time_f64:<12.3f} {gops_f64:<12.1f} {'1.00×':<12} {'-':<12}")
        
        # Benchmark fp32 (balanced)
        time_f32, gops_f32 = benchmark_pyfwht_dtype(n, batch_size, torch.float32)
        speedup_vs_f64 = gops_f32 / gops_f64
        print(f"{n:<8} {'fp32':<10} {time_f32:<12.3f} {gops_f32:<12.1f} {f'{speedup_vs_f64:.2f}×':<12} {'-':<12}")
        
        # Benchmark fp16 (maximum speed)
        time_f16, gops_f16 = benchmark_pyfwht_dtype(n, batch_size, torch.float16)
        speedup_vs_f64 = gops_f16 / gops_f64
        
        # Benchmark Meta
        meta_str = '-'
        if HAS_META:
            time_meta, gops_meta = benchmark_meta(n, batch_size)
            if gops_meta is not None:
                speedup_vs_meta = gops_f16 / gops_meta
                meta_str = f'{speedup_vs_meta:.2f}× ({gops_meta:.1f})'
        
        print(f"{n:<8} {'fp16':<10} {time_f16:<12.3f} {gops_f16:<12.1f} {f'{speedup_vs_f64:.2f}×':<12} {meta_str:<12}")
        print()
    
    print()
    print("Summary:")
    print("  - fp64: High precision (cryptographic applications)")
    print("  - fp32: Balanced speed/precision (2-5× faster)")
    print("  - fp16: Maximum speed (5-15× faster, ML/AI applications)")
    if HAS_META:
        print("  - Meta comparison shows relative performance vs their fp16 implementation")
    print()


if __name__ == '__main__':
    main()
