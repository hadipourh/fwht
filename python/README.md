# pyfwht - Fast Walsh-Hadamard Transform for Python

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Python bindings for the high-performance libfwht library, providing Fast Walsh-Hadamard Transform with NumPy integration and support for CPU (SIMD), OpenMP, and CUDA backends.

## Features

- **Zero-copy NumPy integration**: Direct operation on NumPy arrays without data copying
- **Multiple backends**: Automatic selection or explicit choice of CPU (SIMD), OpenMP, or GPU (CUDA)
- **All data types**: Support for `int8`, `int32`, and `float64` with overflow protection
- **Boolean function analysis**: Convenience functions for cryptographic applications
- **High performance**: 
  - Recursive cache-efficient algorithm (512-element L1-optimized base case)
  - Task-based OpenMP parallelism (2-3× speedup on 4-8 cores)
  - Software prefetching and cache-aligned memory allocation
  - SIMD optimization (AVX2/SSE2/NEON auto-detection)
  - 19% faster CPU, 89% better OpenMP scaling vs v1.0.0
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

## Examples

See `examples/basic_usage.py` for comprehensive usage demonstrations.

## Benchmark Results

### GPU Performance (NVIDIA GPU)

Benchmark performed on GPU server with CUDA backend. Transform sizes match the C library benchmarks (2^24 to 2^28 points).

```
FWHT GPU Benchmark - Python Bindings
================================================================================

GPU available: True
OpenMP available: True
Version: 1.1.1

================================================================================
GPU Performance Benchmark
================================================================================
        Size          Time       Throughput
--------------------------------------------------------------------------------
  16,777,216     12.629 ms     31.88 GOps/s
  33,554,432     27.046 ms     31.02 GOps/s
  67,108,864     53.785 ms     32.44 GOps/s
 134,217,728    107.651 ms     33.66 GOps/s
 268,435,456    215.810 ms     34.83 GOps/s

================================================================================
CPU vs GPU Speedup Comparison
================================================================================
        Size      CPU Time      GPU Time     Speedup
--------------------------------------------------------------------------------
  16,777,216    125.631 ms     12.756 ms       9.85x
  33,554,432    257.413 ms     27.083 ms       9.50x
  67,108,864    527.751 ms     54.160 ms       9.74x
 134,217,728       1.089 s    107.526 ms      10.12x
 268,435,456       2.243 s    216.649 ms      10.35x
```

**Observations:**
- GPU achieves 30+ GOps/s throughput consistently across large problem sizes
- 9-10x speedup over single-threaded AVX2 CPU backend
- Python bindings add negligible overhead compared to the C library

**Run your own benchmark:**
```bash
cd python
python3 gpu_benchmark.py
```

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
