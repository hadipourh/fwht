#!/usr/bin/env python3
"""
Test GPU Context performance vs batch API to isolate cudaMalloc/Free overhead.
"""
import pyfwht
import numpy as np
import time

def bench_context(n, repeats=10):
    """Benchmark using GPU Context (pre-allocated memory)"""
    ctx = pyfwht.gpu.Context(max_n=n, batch_size=1)
    x = np.random.randint(-100, 100, size=(n,), dtype=np.int32)
    
    # Warmup
    for _ in range(3):
        work = x.copy()
        ctx.transform_i32(work)
    
    # Measure
    times = []
    for _ in range(repeats):
        work = x.copy()
        t0 = time.perf_counter()
        ctx.transform_i32(work)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    ctx.close()
    return np.mean(times), np.std(times)

def bench_batch_api(n, repeats=10):
    """Benchmark using batch API (cudaMalloc/Free on every call)"""
    x = np.random.randint(-100, 100, size=(1, n), dtype=np.int32)
    
    # Warmup
    for _ in range(3):
        work = x.copy()
        pyfwht.gpu.batch_transform_i32(work)
    
    # Measure
    times = []
    for _ in range(repeats):
        work = x.copy()
        t0 = time.perf_counter()
        pyfwht.gpu.batch_transform_i32(work)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return np.mean(times), np.std(times)

if __name__ == "__main__":
    print("Testing GPU Context vs Batch API performance")
    print("=" * 60)
    
    for n in [512, 1024, 2048, 4096]:
        ctx_mean, ctx_std = bench_context(n)
        batch_mean, batch_std = bench_batch_api(n)
        
        speedup = batch_mean / ctx_mean
        
        print(f"\nn={n}:")
        print(f"  Context:   {ctx_mean:6.3f} ± {ctx_std:5.3f} ms")
        print(f"  Batch API: {batch_mean:6.3f} ± {batch_std:5.3f} ms")
        print(f"  Speedup:   {speedup:.2f}x (Context is faster)")
