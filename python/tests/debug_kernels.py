#!/usr/bin/env python3
"""
Debug script to see what's actually happening.
"""

import torch
import numpy as np
import pyfwht

print("Debug: Testing small case\n")

# Very small test case
n = 8
batch_size = 2

# Create simple test data
data_cpu = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                     [1, -1, 1, -1, 1, -1, 1, -1]], dtype=np.float64)

print("Input data:")
print(data_cpu)
print()

# CPU reference
data_ref = data_cpu.copy()
for i in range(batch_size):
    pyfwht.transform(data_ref[i])

print("CPU reference result:")
print(data_ref)
print()

# Test fp64 GPU
data_f64 = torch.tensor(data_cpu, dtype=torch.float64, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data_f64)
result_f64 = data_f64.cpu().numpy()

print("GPU fp64 result:")
print(result_f64)
print(f"Error: {np.abs(result_f64 - data_ref).max():.2e}")
print()

# Test fp32 GPU
data_f32 = torch.tensor(data_cpu, dtype=torch.float32, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data_f32)
result_f32 = data_f32.cpu().numpy()

print("GPU fp32 result:")
print(result_f32)
print(f"Error: {np.abs(result_f32 - data_ref).max():.2e}")
print()

# Test fp16 GPU
data_f16 = torch.tensor(data_cpu, dtype=torch.float16, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data_f16)
result_f16 = data_f16.cpu().numpy()

print("GPU fp16 result:")
print(result_f16)
print(f"Error: {np.abs(result_f16 - data_ref).max():.2e}")
