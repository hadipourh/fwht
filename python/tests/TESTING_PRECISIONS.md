# Testing FP16/FP32/FP64 Kernels

## Quick Installation

```bash
# Install pyfwht with GPU support
cd /workspace/python
pip install -e .

# Install PyTorch (if not already installed)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Optional: Install Meta's library for comparison
pip install hadamard-transform
```

## Quick Test

```bash
cd /workspace/python/tests
python test_precisions.py
```

Expected output:
```
Testing pyfwht precision support...

Creating test data (n=4096, batch=10)...
Testing fp64 kernel... ✓
Testing fp32 kernel... ✓
Testing fp16 kernel... ✓

All precision levels working!
```

## Full Benchmark

```bash
python benchmark_all_precisions.py
```

This will:
1. Verify correctness of all kernels
2. Benchmark fp64, fp32, fp16 for n=1024, 2048, 4096
3. Compare against Meta's library (if installed)
4. Show speedup ratios

Expected results (RTX 4090):
- **fp64**: ~74 GOps/s (baseline, cryptographic precision)
- **fp32**: ~300-400 GOps/s (2-5× faster)
- **fp16**: ~700-900 GOps/s (10-12× faster, matching Meta)

## Usage Examples

### PyTorch (automatic dtype detection)

```python
import torch
import pyfwht

# High precision (cryptography)
data = torch.randn(1000, 4096, dtype=torch.float64, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data)  # Uses fp64 kernel

# Balanced (general purpose)
data = torch.randn(1000, 4096, dtype=torch.float32, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data)  # Uses fp32 kernel

# Maximum speed (ML/AI)
data = torch.randn(1000, 4096, dtype=torch.float16, device='cuda')
pyfwht.gpu.batch_transform_dlpack(data)  # Uses fp16 kernel
```

### CuPy

```python
import cupy as cp
import pyfwht

# fp32 example
data = cp.random.randn(1000, 4096, dtype=cp.float32)
pyfwht.gpu.batch_transform_dlpack(data)
```

## Precision Tradeoffs

| Precision | Max Error | Speed    | Use Case                    |
|-----------|-----------|----------|-----------------------------|
| fp64      | ~1e-15    | 1×       | Cryptography, exact math    |
| fp32      | ~1e-6     | 2-5×     | General scientific computing|
| fp16      | ~1e-3     | 10-15×   | ML/AI, neural networks      |

## Troubleshooting

### "fp16 kernel not found"
Make sure you compiled with CUDA support:
```bash
cd /workspace
make clean
make
cd python
pip install -e . --force-reinstall
```

### "CUDA out of memory"
Reduce batch size:
```python
# Instead of batch=1000
data = torch.randn(100, 4096, dtype=torch.float16, device='cuda')
```

### Compare with Meta
```bash
pip install hadamard-transform
python benchmark_all_precisions.py
```

## Implementation Details

- **fp64**: Original kernel, 4 elements/thread, float64 throughout
- **fp32**: Meta-inspired kernel, XOR addressing, optimal shuffle order
- **fp16**: Meta-inspired kernel, same as fp32 but uses __half type

All kernels use:
- Elements per thread: 4
- Thread configurations: n/4 threads (e.g., 1024 threads for n=4096)
- Shared memory for block-level butterflies
- Warp shuffles for warp-level operations
