# LibFWHT

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Version](https://img.shields.io/badge/version-1.1.0-green.svg)

High-performance C99 library for computing the Fast Walsh-Hadamard Transform (FWHT), a fundamental tool in cryptanalysis and Boolean function analysis. The library provides multiple backend implementations (vectorized single-threaded CPU, OpenMP, and CUDA) with automatic selection based on problem size, offering optimal performance across different hardware configurations.

<p align="center">
  <img src="examples/butterfly.svg" alt="FWHT Butterfly Diagram" width="200">
</p>

## Key Features

- **Multiple Backends**: Vectorized CPU (AVX2/SSE2/NEON), OpenMP multi-threading, CUDA GPU acceleration
- **Automatic Backend Selection**: Chooses optimal implementation based on problem size and available hardware
- **Memory Efficient**: In-place algorithm with `O(log n)` stack space, cache-aligned allocations
- **High Performance**: Task-based OpenMP parallelism, GPU acceleration for large transforms and batch processing
- **Flexible API**: In-place transforms, out-of-place helpers, batch processing, Boolean function utilities
- **Production Ready**: Comprehensive test suite, numerical stability guarantees, command-line tool included
- **Easy Integration**: C99 standard, minimal dependencies, Python bindings available via PyPI

## Algorithm

The Fast Walsh-Hadamard Transform computes the Walsh spectrum of k-variable Boolean functions using butterfly operations:

- **Time complexity**: `O(n log n) = O(k × 2^k)` for truth tables of size `n = 2^k`
- **Space complexity**: `O(log n)` recursion stack (in-place, no temporary buffers)
- **Divide-and-conquer**: Recursive with cache-efficient base cases (512-element cutoff fits in L1)
- **Multi-backend architecture**:
  - **CPU**: SIMD vectorization (AVX2/SSE2/NEON auto-detected), software prefetching, cache-aligned allocations
  - **OpenMP**: Task-based recursive parallelism for multi-core scaling
  - **GPU**: Persistent buffers, async transfers, shared memory kernels
  - **Auto-tuning**: Runtime backend selection based on problem size and hardware

## Build and Install

- Prerequisites: C99 compiler, `make`; optional OpenMP toolchain; optional CUDA toolkit when GPU support is desired
- Default build (library + regression tests): `make`
- Focused targets: `make lib`, `make test`, `make test-gpu`, `make openmp`, `make NO_CUDA=1`
- Installation (optional): `sudo make install` installs headers and libraries into `/usr/local`

Build outputs are placed in `build/` (executables) and `lib/` (libraries).

## Library Usage

```c
#include <fwht.h>
#include <stdio.h>

int main(void) {
    /* Boolean truth table: 0 → +1, 1 → -1 */
    int32_t data[8] = {1, -1, -1, 1, -1, 1, 1, -1};

    fwht_status_t status = fwht_i32(data, 8);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "%s\n", fwht_error_string(status));
        return 1;
    }

    printf("WHT[0] = %d\n", data[0]);
    return 0;
}
```

Compile with `gcc example.c -lfwht -lm` (or link directly against `libfwht.a` in `lib/`).

### Core API Highlights

- `fwht_i32`: in-place transform for `int32_t` data (default entry point for Boolean spectra)
  - Safe for all n if `|input[i]| ≤ 1`; general rule: `n × max(|input|) < 2^31`
- `fwht_f64`: in-place transform for `double` data when fractional coefficients matter
  - Relative error typically `< log₂(n) × 2.22e-16` (e.g., `< 2e-15` for n=2^20)
- `fwht_i8`: in-place transform for `int8_t` data to minimize memory footprint
  - ⚠️ Only safe for `n ≤ 64` with `|input| = 1` (watch for overflow)
- `fwht_i32_backend`, `fwht_f64_backend`: same transforms with explicit backend selection (`AUTO`, `CPU`, `OPENMP`, `GPU`)
- `fwht_compute_i32`, `fwht_compute_f64`: out-of-place transforms returning cache-aligned memory
  - ⚠️ **Must use `fwht_free()` instead of `free()` to deallocate results**
- `fwht_from_bool`: convert a Boolean truth table to signed Walsh coefficients before transforming
- `fwht_correlations`: normalize Walsh coefficients to per-mask correlation values
- `fwht_has_gpu`, `fwht_has_openmp`, `fwht_backend_name`: query runtime capabilities and selected backend

## Python Package

Python bindings are available via PyPI for seamless NumPy integration:

```bash
# Install from PyPI
pip install pyfwht

# Enable CUDA support (requires CUDA toolkit)
USE_CUDA=1 pip install pyfwht --no-binary :all:
```

### Quick Example

```python
import numpy as np
import pyfwht as fwht

# Boolean function analysis
truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
wht_coeffs = fwht.from_bool(truth_table, signed=True)
print(wht_coeffs)

# Automatic backend selection (CPU/OpenMP/GPU)
data = np.random.randint(-100, 100, size=2**20, dtype=np.int32)
fwht.transform(data)  # In-place, auto-selects best backend
```

**Features:**

- Zero-copy NumPy integration
- Automatic backend selection (CPU SIMD, OpenMP, CUDA)
- Support for `int8`, `int32`, and `float64` data types
- Boolean function utilities for cryptanalysis
- GPU acceleration for large-scale transforms and batch operations

See [`python/README.md`](python/README.md) for complete documentation and API reference.

## Command-Line Interface

A CLI is provided for quick transforms without writing C code.

### Build

```
make cli
```

The executable is written to `build/fwht_cli`.

### Usage

```
./build/fwht_cli [--input file | --values list] [options]
```

Key options:

- `--input <path>`: read whitespace/comma separated tokens from a file
- `--values <list>`: inline comma/space separated integers (e.g. `--values 0,1,1,0`)
- `--input-format bool|signed`: interpret tokens as 0/1 (default) or signed integers
- `--backend auto|cpu|openmp|gpu`: choose backend (default `auto`)
- `--normalize`: print coefficients divided by `sqrt(n)`
- `--precision <digits>`: decimal places for normalized output (default 6)
- `--no-index`: omit the index column (useful for piping downstream)
- `--quiet`: suppress header metadata

Examples:

```
./build/fwht_cli --values 0,1,1,0,1,0,0,1 --normalize
./build/fwht_cli --input data/walsh.txt --backend gpu
```

Exit status `0` indicates success; non-zero signals parse or transform errors.

## Testing and Tooling

- `make test`: CPU regression suite (mathematical properties, edge cases, API coverage)
- `make test-gpu`: CUDA regression and consistency tests (skips cleanly if CUDA is unavailable)
- `make bench`: build benchmarking utility at `build/fwht_bench`

Always run these targets from the `libfwht` root so generated artefacts remain in the build tree.

## Benchmark Reference

Measurements gathered with `./build/fwht_bench` using `--repeats=10` (GPU runs include `--warmup=1`).

### Reproduction Commands

```bash
# Build library with OpenMP and benchmark (run from the libfwht root)
make openmp bench

# CPU benchmarks (single-threaded and multi-threaded)
./build/fwht_bench \
    --backend=cpu \
    --sizes=16777216,33554432,67108864,134217728,268435456,1073741824 \
    --repeats=10

./build/fwht_bench \
    --backend=openmp \
    --sizes=16777216,33554432,67108864,134217728,268435456,1073741824 \
    --repeats=10

# GPU benchmarks (rebuild with CUDA support first)
make clean && make bench
./build/fwht_bench \
    --backend=gpu \
    --sizes=16777216,33554432,67108864,134217728,268435456,1073741824 \
    --repeats=10 \
    --warmup=1
```

### CPU Performance

#### Apple M4 (macOS 15.7.1)

**System Configuration:**
- CPU: Apple M4 (10 cores, ARM NEON)
- Memory: 24 GB unified

| Mode                     |    Size | Mean (ms) | StdDev (ms) |
| :----------------------- | ------: | --------: | ----------: |
| cpu (single-threaded)    |  2^24   |      27.4 |         1.0 |
|                          |  2^25   |      57.6 |         0.8 |
|                          |  2^26   |     123.4 |         2.8 |
|                          |  2^27   |     262.7 |        16.3 |
|                          |  2^28   |     547.3 |        17.7 |
|                          |  2^30   |   2,417.8 |       147.4 |
| openmp (multi-threaded)  |  2^24   |      15.3 |         0.5 |
|                          |  2^25   |      27.8 |         2.0 |
|                          |  2^26   |      56.8 |         4.5 |
|                          |  2^27   |     115.3 |         4.6 |
|                          |  2^28   |     248.6 |         8.3 |
|                          |  2^30   |   1,119.7 |        35.9 |

#### AMD EPYC 9254 (Linux)

**System Configuration:**
- CPU: AMD EPYC 9254 24-Core Processor (48 threads, x86_64 AVX2)
- Memory: 377 GB

| Mode                     |    Size | Mean (ms) | StdDev (ms) |
| :----------------------- | ------: | --------: | ----------: |
| cpu (single-threaded)    |  2^24   |      56.1 |         0.0 |
|                          |  2^25   |     116.2 |         0.1 |
|                          |  2^26   |     240.9 |         0.1 |
|                          |  2^27   |     589.0 |         0.1 |
|                          |  2^28   |   1,393.2 |         0.3 |
|                          |  2^30   |   7,286.1 |         1.1 |
| openmp (multi-threaded)  |  2^24   |      29.5 |         5.8 |
|                          |  2^25   |      40.8 |         7.3 |
|                          |  2^26   |      68.6 |         9.4 |
|                          |  2^27   |     221.6 |         8.6 |
|                          |  2^28   |     514.9 |         7.0 |
|                          |  2^30   |   2,235.8 |       140.0 |

### GPU Performance (NVIDIA A30, Linux)

**System Configuration:**
- GPU: NVIDIA A30 (CUDA 13.0 runtime, driver 580.95.05, nvcc 12.6.68)
- Host CPU: Dual AMD EPYC 9254 (48 hardware threads)
- System RAM: 377 GB, GPU RAM: 24 GB HBM2

|    Size | Mean (ms) | StdDev (ms) |
| ------: | --------: | ----------: |
|  2^24   |      10.7 |         0.0 |
|  2^25   |      22.8 |         1.0 |
|  2^26   |      47.3 |         0.1 |
|  2^27   |      86.9 |         3.5 |
|  2^28   |     171.9 |         0.1 |
|  2^30   |     714.6 |         5.6 |

## Performance Insights

**Memory-Bandwidth Bound Algorithm:**
- FWHT performance depends on memory subsystem bandwidth, not FLOPS
- Each element is accessed log₂(n) times with irregular stride patterns (low arithmetic intensity)
- CPU optimizations focus on cache efficiency and SIMD vectorization

**Backend Selection Guidelines:**
- **CPU single-threaded**: Small transforms (n < 1M) or when latency matters
- **OpenMP multi-threaded**: Medium to large transforms on multi-core systems (near-linear scaling observed)
- **GPU**: Large single transforms (n ≥ 64M) or batch operations (10+ transforms)
  - HBM-based datacenter GPUs (A30, A100, H100) provide consistent low-variance performance
  - Consumer GPUs with GDDR6X may show higher timing variance
- **Auto mode**: Let the library choose based on problem size and available hardware

**Numerical Stability:**
- `fwht_i32`: Safe for all n if `|input[i]| ≤ 1`; general rule: `n × max(|input|) < 2^31`
- `fwht_f64`: Relative error typically `< log₂(n) × 2.22e-16`
- `fwht_i8`: Only safe for `n ≤ 64` with `|input| = 1` (overflow risk)

## Repository Layout

```
libfwht/
├── include/
│   └── fwht.h              Public C header
├── src/
│   ├── fwht_core.c         Core CPU implementation with SIMD
│   ├── fwht_backend.c      Backend dispatcher
│   ├── fwht_cuda.cu        CUDA GPU implementation
│   └── fwht_internal.h     Internal definitions
├── python/
│   ├── pyfwht/             Python package with NumPy integration
│   ├── tests/              Python test suite
│   ├── examples/           Python usage examples
│   ├── setup.py            Build configuration
│   └── README.md           Python package documentation
├── examples/               Minimal C usage samples
├── tests/                  CPU and GPU regression programs
├── tools/
│   └── fwht_cli.c          Command-line interface
└── Makefile                Build orchestration
```

## Support and Licensing

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

- License: GNU General Public License v3.0 (see `LICENSE`)
- Maintainer: Hosein Hadipour <hsn.hadipour@gmail.com>
- Please report issues or propose patches via this repository
