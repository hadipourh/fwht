#!/usr/bin/env python3
"""
Minimal debug - test if basic transform works at all.
"""

import torch
import numpy as np
import pyfwht

print("Testing n=1024, batch=1")
print()

# Simple pattern
n = 1024
data = np.ones(n, dtype=np.float64)

print("CPU transform:")
print(f"  Input: all ones (first 10: {data[:10]})")
pyfwht.transform(data)
print(f"  Output (first 10): {data[:10]}")
print(f"  Output[0]: {data[0]} (should be {n})")
print()

# GPU fp64
data_gpu = torch.ones(1, n, dtype=torch.float64, device='cuda')
print("GPU fp64 transform:")
print(f"  Input: all ones")
pyfwht.gpu.batch_transform_dlpack(data_gpu)
result = data_gpu.cpu().numpy()[0]
print(f"  Output (first 10): {result[:10]}")
print(f"  Output[0]: {result[0]} (should be {n})")
print()

# Test if old batch API works
print("Testing old batch API (non-DLPack):")
data_old = np.ones((1, n), dtype=np.float64).flatten()
pyfwht.gpu.batch_transform_f64(data_old, n, 1)
print(f"  Output[0]: {data_old[0]} (should be {n})")
