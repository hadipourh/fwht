# Dual-Precision FWHT Strategy

## Overview

We now support **two performance tiers** based on precision requirements:

### 1. **High Precision (Cryptography)** - Existing
- **Type**: `float64` (double precision)
- **Use case**: Cryptanalysis, boolean function analysis, exact computations
- **Precision**: ~15 decimal digits, machine epsilon 2.2e-16
- **Performance**: ~74 GOps/s (batch=1000, n=4096)
- **Status**: ✅ Already implemented and tested

### 2. **High Speed (ML/AI)** - New (Meta-inspired)
- **Type**: `fp16` / `fp32` (half/single precision)
- **Use case**: Machine learning, neural networks, approximate computations
- **Precision**: fp16: ~3 digits, fp32: ~7 digits
- **Target performance**: ~800 GOps/s (matching Meta)
- **Status**: 🚧 Implementation in progress

## Why Dual Precision?

### Cryptographic Requirements (float64)
```python
# Boolean function correlation analysis
data = np.array([...], dtype=np.float64)
pyfwht.transform(data, backend='gpu')
# Need 15 digits precision for:
# - Detecting weak S-boxes
# - Finding linear approximations
# - Differential cryptanalysis
```

### ML/AI Requirements (fp16/fp32)
```python
# Neural network training with Hadamard features
data = torch.randn(1000, 4096, dtype=torch.float16, device='cuda')
pyfwht.gpu.batch_transform_dlpack_fp16(data)  # NEW API
# Speed critical, 3-7 digits sufficient
```

## Implementation Strategy

### Phase 1: Add FP16/FP32 Kernels (Meta-inspired) ✓
```
src/fwht_cuda_fp16.cuh  - New fp16/fp32 kernels
    ├── Small sizes (≤512): All-register + shuffles
    ├── Medium (1K-4K): Hybrid register+shuffle+shared
    └── Large (>4K): Multi-block approach
```

### Phase 2: Integrate Into API
```python
# Low-level C API
fwht_status_t fwht_batch_f16_cuda(half* d_data, size_t n, size_t batch_size);
fwht_status_t fwht_batch_f32_cuda(float* d_data, size_t n, size_t batch_size);

# Python API with DLPack
pyfwht.gpu.batch_transform_dlpack(tensor)  # Auto-detects dtype
```

### Phase 3: Benchmark & Tune
- Target: Match Meta's 800+ GOps/s for fp16
- Keep: Existing 74 GOps/s for float64 (good for precision)

## API Design

### User chooses precision via dtype:

```python
import torch
import pyfwht

# HIGH PRECISION (cryptography)
data = torch.randn(1000, 4096, dtype=torch.float64, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data)
# → Uses existing float64 kernel (74 GOps/s, high precision)

# HIGH SPEED (ML/AI)
data = torch.randn(1000, 4096, dtype=torch.float16, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data)
# → Uses new fp16 kernel (target: 800 GOps/s)

# BALANCED
data = torch.randn(1000, 4096, dtype=torch.float32, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data)
# → Uses fp32 kernel (target: 400 GOps/s, better precision than fp16)
```

## Meta's Optimizations We're Adopting

### 1. **XOR-Based Addressing**
```cuda
// Perfect memory coalescing
int partner_idx = idx ^ stride;
```

### 2. **Optimal Shuffle Order**
```cuda
// Start with smallest strides (better for fp16)
for (int s = 0; s < log_size; ++s) {
    T partner = __shfl_xor_sync(0xFFFFFFFF, data, 1 << s);
    // ...
}
```

### 3. **Size-Specific Launch Configs**
```cuda
// Tuned for each size
{threads_per_block, blocks_per_sm, elements_per_thread}
n=1024: {256, 8, 4}
n=2048: {512, 4, 4}
n=4096: {1024, 2, 4}
```

### 4. **Minimal Shared Memory**
- Only use shared memory when absolutely necessary
- Keep data in registers + shuffles as long as possible

## What We're NOT Copying from Meta

### ❌ Tensor Cores (WMMA)
- Meta uses `mma.sync.aligned.m16n8k16`
- Only beneficial for fp16 matrix ops
- Hadamard transform doesn't naturally map to matrix multiply
- Would complicate code significantly
- **Decision**: Skip for now, focus on simpler optimizations first

### ❌ Async Copy (`cp.async`)
- Requires SM 8.0+ (Ampere)
- Adds complexity
- **Decision**: Add later if needed

## Performance Targets

| Precision | Use Case     | Current | Target  | Strategy              |
|-----------|--------------|---------|---------|----------------------|
| float64   | Crypto       | 74      | 74      | Keep existing (✓)    |
| float32   | Balanced     | N/A     | 400     | New kernel           |
| float16   | ML/AI Speed  | N/A     | 800     | Meta-inspired kernel |

## Migration Path for Users

### No breaking changes!
```python
# Old code still works (float64)
data = np.random.randn(1000, 4096).astype(np.float64)
pyfwht.gpu.batch_transform_f64(data)
# → Uses existing kernel, no change

# New code (fp16 for speed)
data = torch.randn(1000, 4096, dtype=torch.float16, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data)
# → Automatically uses new fp16 kernel
```

## License Compliance

Meta's code: BSD 3-Clause  
Our code: GPL-3.0  
**Compatible** ✓

Attribution added to all files using Meta's techniques.

## Next Steps

1. ✅ Create fp16 kernel header
2. ⏳ Implement fp16/fp32 dispatchers in main CUDA file
3. ⏳ Update Python bindings to support fp16/fp32 via DLPack
4. ⏳ Add comprehensive benchmarks
5. ⏳ Tune launch configurations for RTX 4090
6. ⏳ Document precision/performance tradeoffs

## Summary

**We're not replacing anything** - we're **adding options**:
- Cryptographers: Keep using float64 (high precision)
- ML engineers: Use new fp16 (high speed)
- Everyone else: Choose based on needs

This gives users the best of both worlds! 🎯
