#!/usr/bin/env python3
"""
GPU Multi-Precision FWHT Example

Demonstrates how to use fp64, fp32, and fp16 precision modes on GPU
for different speed/accuracy trade-offs.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

import pyfwht


def _fwht_ops(n: int) -> int:
    """Return total floating ops (add/sub) for one FWHT of size n."""
    return 2 * n * int(math.log2(n))


def _measure_fp16_throughput(n: int, batch_sizes: list[int], repeats: int = 20) -> dict[int, float]:
    """Measure fp16 batch throughput (GOps/s) for the requested batch sizes."""
    stats: dict[int, float] = {}
    for batch in batch_sizes:
        data = torch.randn(batch, n, dtype=torch.float16, device="cuda")
        pyfwht.gpu.batch_transform_dlpack(data)  # warmup
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(max(1, repeats)):
            pyfwht.gpu.batch_transform_dlpack(data)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / max(1, repeats)
        if elapsed <= 0:
            continue
        gops = (_fwht_ops(n) * batch) / (elapsed * 1e9)
        stats[batch] = gops
    return stats
def main():
    # Check GPU availability
    if not pyfwht.has_gpu():
        print("GPU not available. This example requires CUDA support.")
        return
    
    print("=" * 70)
    print("GPU Multi-Precision FWHT Demo")
    print("=" * 70)
    print()
    
    # Configuration
    n = 4096
    batch_size = 100
    
    print(f"Configuration: n={n}, batch_size={batch_size}")
    print()
    
    # Generate reference data (same input for all precisions)
    np.random.seed(42)
    data_ref = np.random.randn(batch_size, n)
    
    # Test each precision mode
    precisions = [
        (torch.float64, "fp64", "Cryptographic precision"),
        (torch.float32, "fp32", "Balanced speed/accuracy"),
        (torch.float16, "fp16", "Maximum speed (ML/AI)")
    ]
    
    results = {}
    
    for dtype, name, description in precisions:
        print(f"{name.upper()} - {description}")
        print("-" * 70)
        
        # Create GPU tensor
        data_gpu = torch.tensor(data_ref, dtype=dtype, device='cuda')
        
        # Transform
        pyfwht.gpu.batch_transform_dlpack(data_gpu)
        
        # Move result back to CPU for inspection
        result = data_gpu.cpu().numpy()
        results[name] = result
        
        # Print sample values
        print(f"  Sample output (first 8 values of batch 0):")
        print(f"    {result[0, :8]}")
        print(f"  Result shape: {result.shape}")
        print(f"  Result dtype: {result.dtype}")
        print()
    
    # Compare precision differences
    print("=" * 70)
    print("Precision Comparison")
    print("=" * 70)
    print()
    
    # Convert all to same dtype for comparison
    ref_result = results['fp64']
    
    for name in ['fp32', 'fp16']:
        diff = np.abs(results[name] - ref_result)
        max_error = np.max(diff)
        mean_error = np.mean(diff)
        
        print(f"{name.upper()} vs fp64:")
        print(f"  Max absolute error:  {max_error:.2e}")
        print(f"  Mean absolute error: {mean_error:.2e}")
        print(f"  Relative error:      {max_error / np.max(np.abs(ref_result)):.2e}")
        print()
    
    # Performance guidance
    print("=" * 70)
    print("When to Use Each Precision")
    print("=" * 70)
    print()
    print("fp64 (float64):")
    print("  • Cryptographic analysis requiring exact Walsh coefficients")
    print("  • Scientific computing where precision is critical")
    print("  • Reference/validation computations")
    print()
    print("fp32 (float32):")
    print("  • General-purpose Walsh transforms")
    print("  • 25-30× faster than fp64 with ~1e-6 error")
    print("  • Good balance for most applications")
    print()
    print("fp16 (float16):")
    print("  • Machine learning / neural network applications")
    print("  • 25-36× faster than fp64 with ~1e-3 error")
    print("  • Maximum throughput: up to 738 GOps/s on RTX 4090")
    print("  • Perfect when approximate Walsh spectrum suffices")
    print()
    
    print("Performance Tip: Use larger batches for better GPU utilization!")
    try:
        targets = sorted(set([1, max(1, batch_size)]))
        stats = _measure_fp16_throughput(n, targets)
        if stats:
            for b in sorted(stats):
                print(f"  • batch={b}: ~{stats[b]:.2f} GOps/s (fp16)")
            if len(stats) >= 2:
                smallest = min(stats)
                largest = max(stats)
                ratio = stats[largest] / stats[smallest] if stats[smallest] > 0 else float('inf')
                print(f"    → {ratio:.0f}× throughput jump from batch={smallest} to batch={largest}")
        else:
            print("  • Unable to measure throughput (no statistics recorded)")
    except Exception as exc:  # pragma: no cover - best effort hint
        print(f"  • Skipped dynamic throughput measurement ({exc})")
    print("=" * 70)
    print("=" * 70)

if __name__ == '__main__':
    main()
