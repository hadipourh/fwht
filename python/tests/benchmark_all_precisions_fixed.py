#!/usr/bin/env python3
"""
Comprehensive benchmark comparing fp64, fp32, and fp16 kernels.

This compares:
1. pyfwht fp64 (cryptographic precision)
2. pyfwht fp32 (balanced)
3. pyfwht fp16 (maximum speed)
4. Meta fp16 (if available)

Results show speedup and precision tradeoffs.
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
    print("WARNING: Meta library not installed. Comparing pyfwht kernels only.")
    HAS_META = False


def benchmark_kernel(name, transform_fn, data, num_trials=100):
    """Benchmark a specific kernel."""
    # Warmup
    for _ in range(10):
        transform_fn(data)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_trials):
        transform_fn(data)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / num_trials * 1000
    
    # Calculate throughput
    batch_size, n = data.shape
    ops = batch_size * n * np.log2(n)
    gops_per_sec = ops / (avg_time_ms / 1000) / 1e9
    
    return avg_time_ms, gops_per_sec


def verify_correctness(n=1024, batch_size=10, tolerance_fp32=1e-6, tolerance_fp16=0.01):
    """Verify that all kernels produce correct results with precision-matched references."""
    print("=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    print()
    
    # Generate reference data
    np.random.seed(42)
    data_original = np.random.randn(batch_size, n)
    
    # Helper to compute CPU reference in any precision
    def cpu_hadamard_numpy(data, dtype):
        """Compute Hadamard transform using numpy in specified dtype."""
        result = data.astype(dtype)
        for b in range(result.shape[0]):
            arr = result[b]
            h = 1
            while h < arr.shape[0]:
                for i in range(0, arr.shape[0], h * 2):
                    for j in range(i, i + h):
                        u = arr[j]
                        v = arr[j + h]
                        arr[j] = u + v
                        arr[j + h] = u - v
                h *= 2
        return result
    
    print(f"Computing precision-matched CPU references (n={n}, batch={batch_size})...")
    data_ref_f64 = cpu_hadamard_numpy(data_original, np.float64)
    data_ref_f32 = cpu_hadamard_numpy(data_original, np.float32)
    data_ref_f16 = cpu_hadamard_numpy(data_original, np.float16)
    print(f"References computed. Testing GPU kernels...\n")
    
    # Test fp64 GPU
    data_f64 = torch.tensor(data_original, dtype=torch.float64, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_f64)
    result_f64 = data_f64.cpu().numpy()
    error_f64 = np.abs(result_f64 - data_ref_f64).max()
    
    # Test fp32 GPU
    data_f32 = torch.tensor(data_original, dtype=torch.float32, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_f32)
    result_f32 = data_f32.cpu().numpy()
    error_f32 = np.abs(result_f32 - data_ref_f32).max()
    
    # Test fp16 GPU
    data_f16 = torch.tensor(data_original, dtype=torch.float16, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_f16)
    result_f16 = data_f16.cpu().numpy()
    error_f16 = np.abs(result_f16 - data_ref_f16).max()
    
    # Test Meta if available
    error_meta = None
    if HAS_META:
        data_meta = torch.tensor(data_original, dtype=torch.float16, device='cuda')
        meta_hadamard(data_meta)
        result_meta = data_meta.cpu().numpy()
        error_meta = np.abs(result_meta - data_ref_f16).max()
    
    # Print results
    print(f"{'Kernel':<20} {'Max Error':<15} {'Status':<20}")
    print("-" * 55)
    
    status_f64 = "✓ PASS" if error_f64 < 1e-12 else "✗ FAIL"
    print(f"{'pyfwht fp64':<20} {error_f64:<15.2e} {status_f64:<20}")
    
    status_f32 = "✓ PASS" if error_f32 < tolerance_fp32 else "✗ FAIL"
    print(f"{'pyfwht fp32':<20} {error_f32:<15.2e} {status_f32:<20}")
    
    status_f16 = "✓ PASS" if error_f16 < tolerance_fp16 else "✗ FAIL"
    print(f"{'pyfwht fp16':<20} {error_f16:<15.2e} {status_f16:<20}")
    
    if HAS_META:
        status_meta = "✓ PASS" if error_meta < tolerance_fp16 else "✗ FAIL"
        print(f"{'Meta fp16':<20} {error_meta:<15.2e} {status_meta:<20}")
    
    print()
    print("Expected errors (now using precision-matched CPU references!):")
    print(f"  fp64: ~1e-15 (GPU fp64 vs CPU fp64)")
    print(f"  fp32: ~1e-6  (GPU fp32 vs CPU fp32)")
    print(f"  fp16: ~1e-3  (GPU fp16 vs CPU fp16)")
    print()
    
    all_pass = (error_f64 < 1e-12 and error_f32 < tolerance_fp32 and 
                error_f16 < tolerance_fp16)
    if HAS_META:
        all_pass = all_pass and error_meta < tolerance_fp16
    
    return all_pass


def benchmark_all(configs, num_trials=100):
    """Benchmark all kernels across multiple configurations."""
    print("=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80)
    print()
    print(f"Number of trials per configuration: {num_trials}")
    print()
    
    for n, batch_size in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: n={n}, batch_size={batch_size}")
        print(f"{'='*80}\n")
        
        # Create test data for each precision
        data_f64 = torch.randn(batch_size, n, dtype=torch.float64, device='cuda')
        data_f32 = torch.randn(batch_size, n, dtype=torch.float32, device='cuda')
        data_f16 = torch.randn(batch_size, n, dtype=torch.float16, device='cuda')
        
        # Benchmark fp64
        time_f64, gops_f64 = benchmark_kernel(
            "pyfwht fp64",
            pyfwht.gpu.batch_transform_dlpack,
            data_f64.clone(),
            num_trials
        )
        
        # Benchmark fp32
        time_f32, gops_f32 = benchmark_kernel(
            "pyfwht fp32",
            pyfwht.gpu.batch_transform_dlpack,
            data_f32.clone(),
            num_trials
        )
        
        # Benchmark fp16
        time_f16, gops_f16 = benchmark_kernel(
            "pyfwht fp16",
            pyfwht.gpu.batch_transform_dlpack,
            data_f16.clone(),
            num_trials
        )
        
        # Benchmark Meta if available
        if HAS_META:
            data_meta = torch.randn(batch_size, n, dtype=torch.float16, device='cuda')
            time_meta, gops_meta = benchmark_kernel(
                "Meta fp16",
                meta_hadamard,
                data_meta.clone(),
                num_trials
            )
        
        # Print results
        print(f"\n{'Kernel':<20} {'Time (ms)':<15} {'GOps/s':<15} {'Speedup vs fp64':<20}")
        print("-" * 70)
        
        print(f"{'pyfwht fp64':<20} {time_f64:<15.3f} {gops_f64:<15.2f} {'1.00x':<20}")
        print(f"{'pyfwht fp32':<20} {time_f32:<15.3f} {gops_f32:<15.2f} {f'{gops_f32/gops_f64:.2f}x':<20}")
        print(f"{'pyfwht fp16':<20} {time_f16:<15.3f} {gops_f16:<15.2f} {f'{gops_f16/gops_f64:.2f}x':<20}")
        
        if HAS_META:
            print(f"{'Meta fp16':<20} {time_meta:<15.3f} {gops_meta:<15.2f} {f'{gops_meta/gops_f64:.2f}x':<20}")


def main():
    """Main benchmark runner."""
    print("\n" + "="*80)
    print(" pyfwht Multi-Precision GPU Benchmark")
    print("="*80 + "\n")
    
    # First verify correctness
    if not verify_correctness(n=1024, batch_size=10):
        print("\n✗ CORRECTNESS CHECK FAILED!")
        print("Skipping performance benchmark.")
        return
    
    print("\n✓ ALL CORRECTNESS CHECKS PASSED!")
    print("\nProceeding to performance benchmarks...\n")
    
    # Run performance benchmarks
    configs = [
        (1024, 1),
        (1024, 100),
        (2048, 1),
        (2048, 100),
        (4096, 1),
        (4096, 100),
    ]
    
    benchmark_all(configs, num_trials=100)
    
    print("\n" + "="*80)
    print(" Benchmark Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
