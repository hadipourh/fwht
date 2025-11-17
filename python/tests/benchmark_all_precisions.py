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


def verify_correctness(n=1024, batch_size=10, tolerance_fp32=1e-5, tolerance_fp16=0.01):
    """Verify that all kernels produce correct results."""
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
    print("Expected errors:")
        print(f"  fp64: ~1e-15 (vs fp64 CPU reference)")
        print(f"  fp32: ~1e-6  (vs fp32 CPU reference)")
        print(f"  fp16: ~1e-3  (vs fp16 CPU reference)")
        print(f"  All comparisons now use precision-matched CPU references!")
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
        
        results = []
        
        # Benchmark fp64
        print("Benchmarking fp64 (cryptographic precision)...", end=' ', flush=True)
        time_f64, gops_f64 = benchmark_kernel(
            "pyfwht fp64",
            lambda x: pyfwht.gpu.batch_transform_dlpack(x),
            data_f64,
            num_trials
        )
        results.append(("pyfwht fp64", time_f64, gops_f64, 1.0, "-"))
        print(f"{gops_f64:.1f} GOps/s")
        
        # Benchmark fp32
        print("Benchmarking fp32 (balanced)...", end=' ', flush=True)
        time_f32, gops_f32 = benchmark_kernel(
            "pyfwht fp32",
            lambda x: pyfwht.gpu.batch_transform_dlpack(x),
            data_f32,
            num_trials
        )
        speedup_f32 = gops_f32 / gops_f64
        results.append(("pyfwht fp32", time_f32, gops_f32, speedup_f32, "-"))
        print(f"{gops_f32:.1f} GOps/s ({speedup_f32:.2f}× vs fp64)")
        
        # Benchmark fp16
        print("Benchmarking fp16 (maximum speed)...", end=' ', flush=True)
        time_f16, gops_f16 = benchmark_kernel(
            "pyfwht fp16",
            lambda x: pyfwht.gpu.batch_transform_dlpack(x),
            data_f16,
            num_trials
        )
        speedup_f16 = gops_f16 / gops_f64
        results.append(("pyfwht fp16", time_f16, gops_f16, speedup_f16, "-"))
        print(f"{gops_f16:.1f} GOps/s ({speedup_f16:.2f}× vs fp64)")
        
        # Benchmark Meta if available
        if HAS_META:
            print("Benchmarking Meta fp16...", end=' ', flush=True)
            time_meta, gops_meta = benchmark_kernel(
                "Meta fp16",
                meta_hadamard,
                data_f16.clone(),
                num_trials
            )
            speedup_meta = gops_meta / gops_f64
            vs_meta = gops_f16 / gops_meta
            results.append(("Meta fp16", time_meta, gops_meta, speedup_meta, f"{vs_meta:.2f}×"))
            print(f"{gops_meta:.1f} GOps/s ({speedup_meta:.2f}× vs fp64)")
            
            # Update pyfwht fp16 to show comparison with Meta
            results[2] = (results[2][0], results[2][1], results[2][2], 
                         results[2][3], f"{vs_meta:.2f}×")
        
        # Print summary table
        print(f"\n{'Kernel':<15} {'Time (ms)':<12} {'GOps/s':<12} {'vs fp64':<12} {'vs Meta':<12}")
        print("-" * 63)
        for name, time_ms, gops, speedup, vs_meta in results:
            print(f"{name:<15} {time_ms:<12.3f} {gops:<12.1f} {speedup:.2f}×{'':<8} {vs_meta:<12}")


def main():
    print("=" * 80)
    print("PYFWHT PRECISION COMPARISON - FP64 / FP32 / FP16")
    print("=" * 80)
    print()
    
    if not pyfwht.has_gpu():
        print("ERROR: GPU support not available")
        return 1
    
    print(f"Library version: {pyfwht.version()}")
    print(f"GPU available: Yes")
    if HAS_META:
        print(f"Meta library: Available")
    else:
        print(f"Meta library: Not available (install with: pip install hadamard-transform)")
    print()
    
    # Verify correctness first
    if not verify_correctness(n=1024, batch_size=10):
        print("ERROR: Correctness check failed!")
        return 1
    
    print("✓ All kernels produce correct results within expected tolerance")
    print()
    
    # Benchmark different configurations
    configs = [
        (1024, 1000),
        (2048, 1000),
        (4096, 1000),
    ]
    
    benchmark_all(configs, num_trials=100)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Precision tradeoffs:")
    print("  • fp64: Highest precision (~1e-15), slowest, for cryptography")
    print("  • fp32: Medium precision (~1e-6), 2-5× faster, general use")
    print("  • fp16: Lower precision (~1e-3), 5-15× faster, ML/AI workloads")
    print()
    print("Usage in PyTorch:")
    print("  data_fp64 = torch.randn(1000, 4096, dtype=torch.float64, device='cuda')")
    print("  pyfwht.gpu.batch_transform_dlpack(data_fp64)  # Auto-selects fp64 kernel")
    print()
    print("  data_fp16 = torch.randn(1000, 4096, dtype=torch.float16, device='cuda')")
    print("  pyfwht.gpu.batch_transform_dlpack(data_fp16)  # Auto-selects fp16 kernel")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
