#!/usr/bin/env python3
"""Debug script to understand fp32 errors."""
import torch
import numpy as np
import pyfwht

# Tiny test case
n = 8
batch = 1

print(f"Testing n={n}, batch={batch}")
print("="*60)

# Simple input
data_original = np.array([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float64)
print(f"Input: {data_original}")

# CPU reference (fp64)
data_cpu = data_original.copy()
pyfwht.transform(data_cpu)
print(f"CPU fp64 result: {data_cpu}")

# GPU fp64
data_gpu_f64 = torch.tensor(data_original, dtype=torch.float64, device='cuda').unsqueeze(0)
pyfwht.gpu.batch_transform_dlpack(data_gpu_f64)
result_f64 = data_gpu_f64.cpu().numpy()[0]
print(f"GPU fp64 result: {result_f64}")
print(f"GPU fp64 error:  {np.abs(result_f64 - data_cpu).max():.2e}")

# GPU fp32
data_gpu_f32 = torch.tensor(data_original, dtype=torch.float32, device='cuda').unsqueeze(0)
pyfwht.gpu.batch_transform_dlpack(data_gpu_f32)
result_f32 = data_gpu_f32.cpu().numpy()[0]
print(f"GPU fp32 result: {result_f32}")
print(f"GPU fp32 error:  {np.abs(result_f32 - data_cpu).max():.2e}")

# GPU fp16
data_gpu_f16 = torch.tensor(data_original, dtype=torch.float16, device='cuda').unsqueeze(0)
pyfwht.gpu.batch_transform_dlpack(data_gpu_f16)
result_f16 = data_gpu_f16.cpu().numpy()[0]
print(f"GPU fp16 result: {result_f16}")
print(f"GPU fp16 error:  {np.abs(result_f16 - data_cpu).max():.2e}")

print("\n" + "="*60)
print("Analysis:")
if np.abs(result_f64 - data_cpu).max() < 1e-12:
    print("✓ fp64 correct")
else:
    print("✗ fp64 WRONG")

if np.abs(result_f32 - data_cpu).max() < 1e-5:
    print("✓ fp32 correct")
else:
    print(f"✗ fp32 WRONG - Detailed comparison:")
    for i in range(len(data_cpu)):
        print(f"  [{i}] CPU={data_cpu[i]:10.6f}  GPU={result_f32[i]:10.6f}  diff={result_f32[i]-data_cpu[i]:+10.6f}")

if np.abs(result_f16 - data_cpu).max() < 1e-2:
    print("✓ fp16 correct")
else:
    print(f"✗ fp16 WRONG - Detailed comparison:")
    for i in range(len(data_cpu)):
        print(f"  [{i}] CPU={data_cpu[i]:10.6f}  GPU={result_f16[i]:10.6f}  diff={result_f16[i]-data_cpu[i]:+10.6f}")
