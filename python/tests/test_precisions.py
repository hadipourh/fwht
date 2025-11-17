#!/usr/bin/env python3
"""
Quick test to verify fp16/fp32/fp64 kernels are working.
"""

import torch
import pyfwht

print("Testing pyfwht precision support...")
print()

# Test data
n = 4096
batch_size = 10

print(f"Creating test data (n={n}, batch={batch_size})...")
data_f64 = torch.randn(batch_size, n, dtype=torch.float64, device='cuda')
data_f32 = torch.randn(batch_size, n, dtype=torch.float32, device='cuda')
data_f16 = torch.randn(batch_size, n, dtype=torch.float16, device='cuda')

print("Testing fp64 kernel...", end=' ')
pyfwht.gpu.batch_transform_dlpack(data_f64)
print("✓")

print("Testing fp32 kernel...", end=' ')
pyfwht.gpu.batch_transform_dlpack(data_f32)
print("✓")

print("Testing fp16 kernel...", end=' ')
pyfwht.gpu.batch_transform_dlpack(data_f16)
print("✓")

print()
print("All precision levels working! Run benchmark_all_precisions.py for detailed comparison.")
