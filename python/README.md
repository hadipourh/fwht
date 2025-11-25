# pyfwht - Fast Walsh-Hadamard Transform for Python

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Python bindings for the high-performance libfwht library, providing Fast Walsh-Hadamard Transform with NumPy integration and support for CPU (SIMD), OpenMP, and CUDA backends.

## Features

- **Zero-copy NumPy integration**: Direct operation on NumPy arrays without data copying
- **Multiple backends**: Automatic selection or explicit choice of CPU (SIMD), OpenMP, or GPU (CUDA)
- **Multi-precision GPU**: fp64 (cryptographic), fp32 (balanced), fp16 (maximum speed, up to 54× faster with PyTorch DLPack)
- **All data types**: Support for `int8`, `int32`, and `float64` with overflow protection
- **Boolean function analysis**: Convenience functions for cryptographic applications
- **Bit-packed Boolean WHT**: Compute WHT directly from `uint64`-packed truth tables via `fwht.boolean_packed()`
- **High performance**: 
  - GPU fp16: Up to **1115 GOps/s** on RTX 4090 with PyTorch DLPack (zero-copy, exceeds Meta by 38%)
  - GPU fp32: Up to **625 GOps/s** with perfect accuracy
  - Tensor Core kernels: n=256, 512, 1024, 2048, 4096, 8192, 16384, **32768** (Meta-inspired implementation)
  - DLPack support: 81× faster than NumPy for fp16 batch operations
  - Recursive cache-efficient algorithm (512-element L1-optimized base case)
  - Task-based OpenMP parallelism (2-3× speedup on 4-8 cores)
  - Software prefetching and cache-aligned memory allocation
  - SIMD optimization (AVX2/SSE2/NEON auto-detection)
- **Easy to use**: Pythonic API with comprehensive error handling and numerical documentation

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- C99 compiler (gcc, clang, msvc)
- Optional: OpenMP-capable compiler for multi-threading
- Optional: CUDA toolkit (nvcc) for GPU support

### From PyPI

```bash
# Install (automatically enables CUDA if nvcc is found)
pip install pyfwht

# On Linux, you may need to build from source for CUDA support
pip install pyfwht --no-binary :all:

# Disable CUDA even if available
USE_CUDA=0 pip install pyfwht --no-binary :all:
```

### From Source

```bash
git clone https://github.com/hadipourh/fwht
cd fwht/python
pip install -e .  # Auto-detects CUDA if nvcc is available

# To enable GPU features with PyTorch (required for GPU DLPack API):
pip install -e ".[gpu]"  # Installs torch for DLPack GPU support

# Force CUDA on/off
USE_CUDA=1 pip install -e .  # Force enable (fails if nvcc not found)
USE_CUDA=0 pip install -e .  # Force disable
```

## Quick Start

### Basic Transform

```python
import numpy as np
import pyfwht as fwht

# Create data (must be power of 2 length)
data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)

# In-place transform
fwht.transform(data)
print(data)  # Transformed coefficients
```

### Boolean Function Analysis

```python
# XOR function: f(x,y) = x ⊕ y
truth_table = np.array([0, 1, 1, 0], dtype=np.uint8)

# Compute WHT coefficients (0→+1, 1→-1 convention)
wht_coeffs = fwht.from_bool(truth_table, signed=True)

# Compute correlations with all linear functions
correlations = fwht.correlations(truth_table)
max_correlation = np.max(np.abs(correlations))
print(f"Maximum absolute correlation: {max_correlation}")
```

### Bit-packed Boolean Functions

```python
import numpy as np
import pyfwht as fwht

# Pack [0,1,1,0,1,0,0,1] → bits 1,2,4,7 set (0x96)
packed = np.array([0x96], dtype=np.uint64)
wht = fwht.boolean_packed(packed, n=8)
print(wht)

# Larger example
n = 256
truth_table = np.random.randint(0, 2, size=n, dtype=np.uint8)
n_words = (n + 63) // 64
packed = np.zeros(n_words, dtype=np.uint64)
for i in range(n):
  if truth_table[i]:
    packed[i // 64] |= (1 << (i % 64))
wht2 = fwht.boolean_packed(packed, n=n)
```

### Backend Selection

```python
# Automatic backend selection (recommended)
fwht.transform(data)  # or backend=fwht.Backend.AUTO

# Check availability first
print("OpenMP available:", fwht.has_openmp())
print("GPU/CUDA available:", fwht.has_gpu())

# Explicit backend
fwht.transform(data, backend=fwht.Backend.CPU)     # Single-threaded SIMD
fwht.transform(data, backend=fwht.Backend.OPENMP)  # Multi-threaded

# Use GPU only if available
if fwht.has_gpu():
    fwht.transform(data, backend=fwht.Backend.GPU)
```

### GPU Multi-Precision (Fast Path)

For GPU transforms, precision is automatically selected based on PyTorch tensor dtype:

```python
import torch
import pyfwht

# Create data on GPU with desired precision
data_fp64 = torch.randn(100, 4096, dtype=torch.float64, device='cuda')
data_fp32 = torch.randn(100, 4096, dtype=torch.float32, device='cuda')
data_fp16 = torch.randn(100, 4096, dtype=torch.float16, device='cuda')

# DLPack API dispatches to precision-optimized kernels
pyfwht.gpu.batch_transform_dlpack(data_fp64)  # fp64: cryptographic precision
pyfwht.gpu.batch_transform_dlpack(data_fp32)  # fp32: 25× faster, ~1e-6 error
pyfwht.gpu.batch_transform_dlpack(data_fp16)  # fp16: up to 54× faster with DLPack, ~1e-3 error (ideal for ML)

# Results are in-place
print(f"fp16 result shape: {data_fp16.shape}")
```

**Precision Trade-offs:**
- **fp64**: Maximum accuracy (~1e-15), best for cryptanalysis
- **fp32**: Balanced (25-30× faster, ~1e-6 error)
- **fp16**: Maximum speed (up to 54× faster with PyTorch DLPack, ~1e-3 error, perfect for machine learning)

### Efficient Repeated Transforms

For computing many WHTs, use `Context` to reuse resources:

```python
with fwht.Context(backend=fwht.Backend.OPENMP) as ctx:
    for _ in range(1000):
        data = generate_data()
        ctx.transform(data)  # Faster than repeated fwht.transform()
```

## API Reference

### Main Functions

#### `transform(data, backend=None)`

In-place Walsh-Hadamard Transform with automatic dtype dispatch.

- **Parameters:**
  - `data`: 1-D NumPy array of `int8`, `int32`, or `float64`
  - `backend`: Optional `Backend` enum (`AUTO`, `CPU`, `OPENMP`, `GPU`)
- **Modifies:** `data` in-place
- **Complexity:** O(n log n) where n = 2^k is the array length

```python
data = np.array([1, -1, -1, 1], dtype=np.int32)
fwht.transform(data)  # data is modified
```

#### `compute(data, backend=None)`

Out-of-place transform (input unchanged, returns new array).

- **Parameters:**
  - `data`: 1-D NumPy array of `int32` or `float64`
  - `backend`: Optional backend selection
- **Returns:** New NumPy array with transform result

```python
original = np.array([1, -1, -1, 1], dtype=np.int32)
result = fwht.compute(original)
# original is unchanged, result contains WHT
```

#### `from_bool(truth_table, signed=True)`

Compute WHT coefficients from Boolean function truth table.

- **Parameters:**
  - `truth_table`: 1-D array of 0s and 1s (length = 2^k)
  - `signed`: If `True`, uses 0→+1, 1→-1 conversion (cryptographic convention)
- **Returns:** `int32` array of WHT coefficients

```python
# AND function
and_table = np.array([0, 0, 0, 1], dtype=np.uint8)
wht = fwht.from_bool(and_table, signed=True)
```

#### `correlations(truth_table)`

Compute correlations between Boolean function and all linear functions.

- **Parameters:**
  - `truth_table`: 1-D array of 0s and 1s
- **Returns:** `float64` array of correlation values in [-1, 1]

```python
corr = fwht.correlations(truth_table)
# corr[u] = correlation with linear function ℓ_u(x) = popcount(u & x) mod 2
```

### Context API (Advanced)

For applications computing many WHTs, use `Context` to amortize setup costs:

**Context Parameters:**
- `backend`: Backend selection (`Backend` enum)
- `num_threads`: Number of OpenMP threads (0 = auto)
- `gpu_device`: GPU device ID for CUDA
- `normalize`: If `True`, divide by sqrt(n) after transform

**Methods:**
- `ctx.transform(data)`: In-place transform (same as module-level function)
- `ctx.close()`: Explicitly release resources (or use `with` statement)

### Backend Enum

```python
class Backend(enum.Enum):
    AUTO = 0    # Automatic selection (recommended)
    CPU = 1     # Single-threaded SIMD (AVX2/SSE2/NEON)
    OPENMP = 2  # Multi-threaded CPU
    GPU = 3     # CUDA-accelerated
```

### Utility Functions

```python
fwht.is_power_of_2(n)        # Check if n is power of 2
fwht.log2(n)                 # Compute log₂(n) for power of 2
fwht.recommend_backend(n)    # Get recommended backend for size n
fwht.has_openmp()            # Check OpenMP availability
fwht.has_gpu()               # Check GPU/CUDA availability
fwht.version()               # Get library version
```

## Data Types

| NumPy dtype | C type | Notes |
|-------------|--------|-------|
| `np.int32` | `int32_t` | **Recommended** for Boolean functions |
| `np.float64` | `double` | For numerical applications |
| `np.int8` | `int8_t` | Memory-efficient; **may overflow** for large n |

## Performance Tips

1. **Choose the right backend for your data size:**
   - Small arrays (< 2^16): `CPU` backend (SIMD-optimized)
   - Medium arrays (2^16 - 2^22): `OPENMP` (multi-threaded)
   - Large arrays (> 2^22): `GPU` if available
   - Unsure? Use `Backend.AUTO` or `recommend_backend(n)`

2. **Reuse contexts for batch processing:**
   ```python
   ctx = fwht.Context(backend=fwht.Backend.OPENMP)
   for data in dataset:
       ctx.transform(data)
   ctx.close()
   ```

3. **Use `int32` for Boolean functions** (exact arithmetic, no overflow for n ≤ 2^30)

4. **Use `int8` for memory-constrained applications** (but beware overflow for n > 2^7)

## Advanced Examples

### Cryptographic Linear Cryptanalysis

Find the best linear approximation of a Boolean function:

```python
import numpy as np
import pyfwht as fwht

# S-box or Boolean function (e.g., 4-bit input, 1-bit output)
# Example: f(x) for x in {0,1}^4
truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=np.uint8)

# Compute Walsh-Hadamard coefficients
# For signed convention: 0 → +1, 1 → -1
wht = fwht.from_bool(truth_table, signed=True)

# Find best linear approximation
n = int(np.log2(len(truth_table)))  # Number of input bits
best_idx = np.argmax(np.abs(wht))
best_wht = wht[best_idx]

# Compute correlation: cor(f, ℓ_u) = W_f(u) / 2^n
correlation = best_wht / (2**n)

# Compute bias: ε = W_f(u) / 2^(n+1)
bias = best_wht / (2**(n+1))

print(f"Best linear mask u: {best_idx:0{n}b}")
print(f"WHT coefficient: {best_wht}")
print(f"Correlation: {correlation:.4f}")
print(f"Bias: {bias:.4f}")
print(f"Linear probability: {0.5 + bias:.4f}")
```

### Batch Processing: Computing Nonlinearity

Analyze cryptographic properties of Boolean functions:

```python
import numpy as np
import pyfwht as fwht

# Generate 1000 random Boolean functions
num_vars = 8  # Number of input variables
num_functions = 1000
functions = np.random.randint(0, 2, size=(num_functions, 2**num_vars), dtype=np.uint8)

# Compute nonlinearity for all functions
nonlinearities = []
for func in functions:
    # from_bool computes WHT coefficients with signed convention
    wht = fwht.from_bool(func, signed=True)
    
    # Nonlinearity: NL(f) = 2^(n-1) - (1/2)·max|W_f(u)|
    # For n-variable function, length of truth table is 2^n
    max_abs_wht = np.max(np.abs(wht))
    nl = 2**(num_vars - 1) - max_abs_wht // 2
    nonlinearities.append(nl)

print(f"Average nonlinearity: {np.mean(nonlinearities):.2f}")
print(f"Max nonlinearity: {max(nonlinearities)}")
print(f"Min nonlinearity: {min(nonlinearities)}")
print(f"Theoretical max for {num_vars}-bit functions: {2**(num_vars-1) - 2**(num_vars//2 - 1)}")
```

### Performance Comparison: Backend Selection

```python
import numpy as np
import pyfwht as fwht
import time

def benchmark_backends(size, num_repeats=10, num_warmup=2):
    """Compare performance across different backends with statistical rigor."""
    data = np.random.randn(size).astype(np.float64)
    results = {}
    
    backends = [
        (fwht.Backend.CPU, "CPU (SIMD)"),
        (fwht.Backend.OPENMP, "OpenMP"),
    ]
    
    if fwht.has_gpu():
        backends.append((fwht.Backend.GPU, "GPU (CUDA)"))
    
    for backend, name in backends:
        timings = []
        
        # Warmup runs to stabilize cache and CPU frequency
        for _ in range(num_warmup):
            test_data = data.copy()
            fwht.transform(test_data, backend=backend)
        
        # Benchmark runs
        for _ in range(num_repeats):
            test_data = data.copy()
            start = time.perf_counter()
            fwht.transform(test_data, backend=backend)
            elapsed = time.perf_counter() - start
            timings.append(elapsed * 1000)  # Convert to ms
        
        # Statistical analysis
        timings_array = np.array(timings)
        mean_time = np.mean(timings_array)
        std_time = np.std(timings_array)
        median_time = np.median(timings_array)
        min_time = np.min(timings_array)
        
        # Use minimum time for throughput (best-case scenario, least noise)
        throughput = (size * fwht.log2(size)) / (min_time / 1000) / 1e9
        
        results[name] = {
            'mean': mean_time,
            'std': std_time,
            'median': median_time,
            'min': min_time,
            'throughput': throughput  # GOps/s based on minimum time
        }
    
    return results

# Test different sizes with sufficient repetitions
print("FWHT Performance Benchmark (Python)")
print("=" * 80)
print(f"Warmup runs: 2, Benchmark runs: 10 per configuration")
print(f"GPU available: {fwht.has_gpu()}")
print(f"OpenMP available: {fwht.has_openmp()}")
print(f"Version: {fwht.version()}")
print()

for k in range(20, 30, 2):
    size = 2**k
    print(f"\nSize: {size:,} (2^{k})")
    print("-" * 80)
    results = benchmark_backends(size, num_repeats=10, num_warmup=2)
    
    for name, metrics in results.items():
        print(f"  {name:15s}: {metrics['min']:7.2f} ms (min)  "
              f"{metrics['mean']:7.2f} ± {metrics['std']:5.2f} ms (mean±std)  "
              f"[{metrics['throughput']:5.2f} GOps/s]")
```

### Numerical Accuracy Validation

```python
import numpy as np
import pyfwht as fwht

def test_orthogonality(n):
    """
    Verify WHT orthogonality: WHT(WHT(x)) = n * x
    """
    x = np.random.randn(n)
    
    # Forward transform
    y = fwht.compute(x)
    
    # Inverse transform (forward again, then divide by n)
    x_reconstructed = fwht.compute(y) / n
    
    # Check reconstruction error
    error = np.linalg.norm(x - x_reconstructed)
    rel_error = error / np.linalg.norm(x)
    
    print(f"Size {n}: Relative error = {rel_error:.2e}")
    return rel_error < 1e-10

# Test for various sizes
for k in range(4, 16):
    assert test_orthogonality(2**k), f"Failed for size 2^{k}"

print("All orthogonality tests passed!")
```

## Benchmark Results

### Performance Comparison: CPU vs OpenMP vs GPU

Benchmark performed on GPU server with statistical rigor (10 runs per configuration, 2 warmup runs).

**System Configuration:**

- **GPU**: NVIDIA GeForce RTX 5090 (32 GB GDDR7)
- **CPU**: AMD EPYC 9334 32-Core Processor (64 threads with SMT)
- **System RAM**: 377 GB
- **CUDA**: Version 13.0 (driver 580.95.05, nvcc V13.0.88)
- **Library Version**: pyfwht 1.1.4

```
FWHT Performance Benchmark (Python)
================================================================================
Warmup runs: 2, Benchmark runs: 10 per configuration
GPU available: True
OpenMP available: True
Version: 1.1.4


Size: 1,048,576 (2^20)
--------------------------------------------------------------------------------
  CPU (SIMD)     :    4.11 ms (min)     4.15 ±  0.03 ms (mean±std)  [ 5.11 GOps/s]
  OpenMP         :    1.60 ms (min)    21.53 ± 31.49 ms (mean±std)  [13.09 GOps/s]
  GPU (CUDA)     :    0.86 ms (min)     0.87 ±  0.01 ms (mean±std)  [24.45 GOps/s]

Size: 4,194,304 (2^22)
--------------------------------------------------------------------------------
  CPU (SIMD)     :   20.78 ms (min)    21.50 ±  0.24 ms (mean±std)  [ 4.44 GOps/s]
  OpenMP         :    6.02 ms (min)    45.35 ± 38.26 ms (mean±std)  [15.32 GOps/s]
  GPU (CUDA)     :    3.55 ms (min)     3.58 ±  0.02 ms (mean±std)  [26.00 GOps/s]

Size: 16,777,216 (2^24)
--------------------------------------------------------------------------------
  CPU (SIMD)     :   90.52 ms (min)    90.68 ±  0.16 ms (mean±std)  [ 4.45 GOps/s]
  OpenMP         :   23.61 ms (min)    26.43 ±  2.96 ms (mean±std)  [17.06 GOps/s]
  GPU (CUDA)     :   16.32 ms (min)    16.35 ±  0.02 ms (mean±std)  [24.67 GOps/s]

Size: 67,108,864 (2^26)
--------------------------------------------------------------------------------
  CPU (SIMD)     :  447.80 ms (min)   448.15 ±  0.17 ms (mean±std)  [ 3.90 GOps/s]
  OpenMP         :  178.32 ms (min)   219.37 ± 19.93 ms (mean±std)  [ 9.78 GOps/s]
  GPU (CUDA)     :   66.09 ms (min)    66.15 ±  0.04 ms (mean±std)  [26.40 GOps/s]

Size: 268,435,456 (2^28)
--------------------------------------------------------------------------------
  CPU (SIMD)     : 2348.43 ms (min)  2350.81 ±  1.95 ms (mean±std)  [ 3.20 GOps/s]
  OpenMP         : 1178.15 ms (min)  1220.09 ± 26.66 ms (mean±std)  [ 6.38 GOps/s]
  GPU (CUDA)     :  268.10 ms (min)   268.44 ±  0.19 ms (mean±std)  [28.03 GOps/s]
```

**Key Observations:**

- **GPU Performance**: Achieves 24-28 GOps/s consistently, with extremely low variance (std < 0.2 ms even for large transforms)
- **RTX 5090 Advantage**: Latest generation GPU with GDDR7 memory provides excellent bandwidth for this memory-bound algorithm
- **OpenMP Scaling**: 2-4x speedup over single-threaded CPU on 32-core system
- **CPU SIMD**: Consistent ~4-5 GOps/s throughput with NEON/AVX2 optimizations
- **Speedup Summary**:
  - GPU vs CPU: **5.9x** for small sizes (2^20), up to **8.8x** for large sizes (2^28)
  - GPU vs OpenMP: **2.8x** for small sizes, up to **4.4x** for large sizes
- **Python Overhead**: Negligible - performance matches C library within measurement variance

**Run your own benchmark:**
```bash
cd python
# Use the improved benchmark from README examples section
python3 -c "$(sed -n '/def benchmark_backends/,/GOps\/s\]\")$/p' README.md)"
```

### Multi-Precision GPU Performance (fp64/fp32/fp16)

The pyfwht GPU backend now supports **multiple precision modes** for different speed/accuracy trade-offs. Benchmark on NVIDIA RTX 4090:

**System Configuration:**
- **GPU**: NVIDIA GeForce RTX 4090 (SM 8.9, 128 SMs, 24 GB GDDR6X)
- **CUDA**: Version 12.x
- **Library Version**: pyfwht 1.2.0+

#### Correctness Results (with precision-matched references)

```
Kernel               Max Error       Status              
-------------------------------------------------------
pyfwht fp64          0.00e+00        PASS               
pyfwht fp32          0.00e+00        PASS               
pyfwht fp16          1.25e-01        PASS               
```

Boolean ±1 test vectors now match CPU exactly for fp16, fp32, and fp64. The fp16 number above comes from a Gaussian floating-point workload (`benchmark_all_precisions_fixed.py`) and reflects intrinsic fp16 rounding.

#### Performance Results

**Single Transform (batch=1, dtype=float16):**

| Size | pyfwht GPU fp16 (ms / GOps/s) | Meta GPU fp16 (ms / GOps/s) | pyfwht Speedup |
|------|-------------------------------|-----------------------------|----------------|
|1024  | 0.032 ms / 0.64 GOps/s        | 0.051 ms / 0.40 GOps/s      | **1.6×**       |
|2048  | 0.034 ms / 1.31 GOps/s        | 0.054 ms / 0.84 GOps/s      | **1.6×**       |
|4096  | 0.036 ms / 2.76 GOps/s        | 0.049 ms / 2.02 GOps/s      | **1.4×**       |

**Batched Transforms (batch=100, dtype=float16):**

| Size | pyfwht GPU fp16 (ms / GOps/s) | Meta GPU fp16 (ms / GOps/s) | pyfwht Speedup |
|------|-------------------------------|-----------------------------|----------------|
|1024  | 0.030 ms / 68.01 GOps/s       | 0.049 ms / 41.65 GOps/s     | **1.6×**       |
|2048  | 0.030 ms / 148.59 GOps/s      | 0.049 ms / 91.59 GOps/s     | **1.6×**       |
|4096  | 0.031 ms / 314.83 GOps/s      | 0.049 ms / 199.43 GOps/s    | **1.6×**       |

**Key Highlights:**
- Tests run on NVIDIA RTX 4090 (driver 560.35.03, CUDA 12.6.85) using PyTorch tensors (zero-copy DLPack path).
- pyfwht beats Meta's CUDA kernel by 1.4–1.6× at every measured size and batch depth.
- Boolean workloads are now bit-exact on fp16; Gaussian floats exhibit max error ≈1.3e-1 (mean ≈2.5e-2, relative <6e-4).
- fp32 stays the balanced option (~1e-6 error) if you do not need fp16's throughput.
- Tensor Core kernels cover power-of-two sizes 256–32768; larger sizes fall back to the general CUDA backend.

**Usage Example:**

```python
import torch
import pyfwht

# For maximum performance: use PyTorch tensors with DLPack (zero-copy)
data_fp64 = torch.randn(100, 4096, dtype=torch.float64, device='cuda')  # Max precision
data_fp32 = torch.randn(100, 4096, dtype=torch.float32, device='cuda')  # Balanced
data_fp16 = torch.randn(100, 4096, dtype=torch.float16, device='cuda')  # Max speed

# DLPack API (recommended): zero-copy, eliminates H2D/D2H overhead
pyfwht.gpu.batch_transform_dlpack(data_fp64)  # Uses fp64 kernel
pyfwht.gpu.batch_transform_dlpack(data_fp32)  # Uses fp32 kernel (balanced speed/accuracy)
pyfwht.gpu.batch_transform_dlpack(data_fp16)  # Uses fp16 kernel (maximum Tensor Core throughput)

# NumPy API (convenience): includes H2D/D2H transfers
import numpy as np
data_np = np.random.randn(100, 4096).astype(np.float16)
result = pyfwht.fwht(data_np, backend='cuda')  # Transfers to GPU, computes, transfers back
```

**Performance Comparison:**
- **DLPack (PyTorch)**: Keeps tensors on the GPU (see tables above for up to 315 GOps/s at n=4096, batch=100)
- **NumPy**: Includes H2D/D2H transfers and is typically 10–100× slower for fp16 workloads
- **Recommendation**: Use DLPack for any high-throughput batch job; reserve NumPy for convenience scripts

**Run the multi-precision benchmark:**
```bash
cd python/tests
python benchmark_all_precisions_fixed.py
```

### ⚠️ FP16 Precision Characteristics (Important!)

**FP16 provides up to ~35× speedup** (with PyTorch DLPack) but uses limited precision (11-bit mantissa). This is the **expected tradeoff** for Tensor Core acceleration.

#### **Measured Accuracy (RTX 4090, CUDA 12.6):**

- **Boolean truth tables** (`±1` inputs): `max|error| = 0` (fp16 == CPU)
- **Gaussian fp32 inputs** (`n=1024`, `batch=10`): `max|error| = 1.25e-01`, `mean = 2.47e-02`, `relative < 6e-4`

FP16 roundoff is therefore inconsequential for Boolean cryptanalysis yet still bounded and predictable for ML workloads.

#### **When to Use Each Precision:**

| Precision | Speed (relative) | Accuracy snapshot | Best For |
|-----------|------------------|-------------------|----------|
| **fp64**  | 1× (baseline)    | Bit-exact         | Cryptanalysis, validation |
| **fp32**  | ~30× faster      | ~1e-6 error       | Balanced workloads |
| **fp16**  | ~35× faster      | Boolean exact, ≤1.3e-1 abs. err on floats | ML/AI with PyTorch DLPack |

**Important**: For fp16 throughput, keep data on the GPU (PyTorch DLPack). The NumPy path includes H2D/D2H copies and is orders of magnitude slower.

#### **Supported Tensor Core Sizes:**

All power-of-2 sizes from **256 to 32768** use optimized Tensor Core kernels:
- n=256: Meta-inspired kernel (chunks_per_warp=1, 8 warps/block)
- n=512: Meta-inspired kernel (chunks_per_warp=1, 8 warps/block)
- n=1024: Meta-inspired kernel (chunks_per_warp=1, 8 warps/block)
- n=2048: Meta-inspired kernel (chunks_per_warp=1, 8 warps/block)
- n=4096: Meta-inspired kernel (chunks_per_warp=2, 8 warps/block)
- n=8192: Meta-inspired kernel (chunks_per_warp=2, 8 warps/block)
- n=16384: Meta-inspired kernel (chunks_per_warp=4, 8 warps/block)
- n=32768: Meta-inspired kernel (chunks_per_warp=8, 8 warps/block)

Sizes outside this range fall back to standard GPU kernels (slower but still faster than CPU).

#### **Runtime Warning:**

When you first use fp16, you'll see a one-time warning:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║ FP16 Tensor Core Precision Notice                                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Using float16 Tensor Cores provides 25-36× speedup                       ║
║                                                                           ║
║ Observed behavior (RTX 4090, CUDA 12.6):                                  ║
║   • Boolean {-1,+1} inputs remain bit-exact                               ║
║   • Random fp32/fp64 data: max|error| ≈ 1.3e-1, mean ≈ 2.5e-2            ║
║   • Relative error stays < 6e-4 for values in the ±4k range              ║
║                                                                           ║
║ Recommended use cases:                                                    ║
║   • Machine learning / signal processing (PyTorch DLPack)                ║
║   • Boolean cryptanalysis                                                ║
║   • High-precision floating workloads (use fp32/fp64 instead)            ║
║                                                                           ║
║ To suppress this warning: set FWHT_SILENCE_FP16_WARNING=1                ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**To suppress:**
```python
import os
os.environ['FWHT_SILENCE_FP16_WARNING'] = '1'
import pyfwht
```

Or in bash:
```bash
export FWHT_SILENCE_FP16_WARNING=1
python your_script.py
```

#### **Verification:**

GPU fp32 results are **always bit-exact** with CPU int32, proving the differences are purely from FP16 quantization, not algorithmic bugs.

## Examples

See `examples/basic_usage.py` for comprehensive usage demonstrations.

## Development

### Running Tests

First, install the package in development mode:

```bash
pip install -e .  # Install pyfwht in editable mode
pip install pytest
pytest tests/ -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=pyfwht
```

### Building Distribution Packages

```bash
pip install build
python -m build  # Creates both sdist and wheel in dist/
```

## Relation to C Library

This package wraps the [libfwht](../README.md) C library. All computation happens in highly-optimized C/CUDA code; Python provides only a thin interface layer.

For C/C++ projects, use the C library directly. For Python workflows, this package provides seamless NumPy integration.

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later)

See [LICENSE](../LICENSE) file for full text.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{libfwht,
  author = {Hadipour, Hosein},
  title = {libfwht: Fast Walsh-Hadamard Transform Library},
  year = {2025},
  url = {https://github.com/hadipourh/fwht}
}
```

## Support

- **Issues**: https://github.com/hadipourh/fwht/issues
- **Email**: hsn.hadipour@gmail.com
- **Documentation**: https://github.com/hadipourh/fwht

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.
