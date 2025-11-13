# LibFWHT

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Version](https://img.shields.io/badge/version-1.0.1-green.svg)

High-performance C99 library for computing the Fast Walsh-Hadamard Transform (FWHT), a fundamental tool in cryptanalysis and Boolean function analysis. The library provides multiple backend implementations (vectorized single-threaded CPU, OpenMP, and CUDA) with automatic selection based on problem size, offering optimal performance across different hardware configurations.

<p align="center">
  <img src="examples/butterfly.svg" alt="FWHT Butterfly Diagram" width="200">
</p>

## Key Features

- **Multiple Backends**: Vectorized CPU (AVX2/SSE2/NEON), OpenMP multi-threading, CUDA GPU acceleration
- **Automatic Backend Selection**: Chooses optimal implementation based on problem size and available hardware
- **Memory Efficient**: In-place algorithm with `O(log n)` stack space, cache-aligned allocations
- **High Performance**: Task-based OpenMP parallelism (2-3× speedup), GPU acceleration (2.7-3.5× on datacenter GPUs)
- **Flexible API**: In-place transforms, out-of-place helpers, batch processing, Boolean function utilities
- **Production Ready**: Comprehensive test suite, numerical stability guarantees, command-line tool included
- **Easy Integration**: C99 standard, minimal dependencies, Python bindings available via PyPI

## Algorithm

- Computes Walsh-Hadamard Transform for k-variable Boolean functions using butterfly operations
- For a truth table of size `n = 2^k`, runs in `O(n log n) = O(k × 2^k)` time
- **Space complexity**: `O(log n)` for recursion stack (in-place algorithm, no temporary buffers needed)
- **Recursive divide-and-conquer** with cache-efficient base cases (512-element cutoff fits in L1 cache)
- **SIMD acceleration**: Auto-detects and uses AVX2, SSE2, or NEON instructions
- **Task-based OpenMP**: Recursive parallelism for excellent multi-core scaling
- **GPU optimization**: Persistent buffers, async transfers, shared memory kernels
- **Auto-tuning**: Dynamically selects best backend and configuration based on hardware

## Performance Characteristics

- **Memory-bandwidth bound**: Performance depends primarily on memory subsystem, not raw compute
- **CPU optimizations**: Software prefetching, cache-line aligned allocations, SIMD vectorization
- **GPU considerations**: HBM-based datacenter GPUs (A30, A100, H100) preferred for consistent performance
- **Batch processing**: GPU batch mode achieves 10-100× throughput vs sequential CPU for multiple transforms
- **Numerical stability**: Documented overflow bounds and precision guarantees for all data types

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
- GPU achieves 30+ GOps/s with 9-10× speedup over CPU

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

# CPU timings on the Apple M4 host
./build/fwht_bench \
    --backend=cpu \
    --sizes=16777216,33554432,67108864,134217728,268435456,1073741824 \
    --repeats=10

# GPU timings on the NVIDIA A30 host  
# (rebuild with CUDA support first)
make clean && make bench
./build/fwht_bench \
    --backend=gpu \
    --sizes=16777216,33554432,67108864,134217728,268435456,1073741824 \
    --repeats=10 \
    --warmup=1
```

Each command samples the same transform sizes reported in the tables below. Adjust `--backend` and the size list as needed for other hardware.

**Apple M4 desktop (macOS 15.7.1)**
CPU: Apple M4 (10 physical / 10 logical cores)
Memory: 24 GiB unified

| Mode                     | Size (points) | Mean (ms) | StdDev (ms) |
| :----------------------- | ------------: | --------: | ----------: |
| cpu (single-threaded)    |    16,777,216 |      28.8 |         0.6 |
|                          |    33,554,432 |      59.9 |         0.2 |
|                          |    67,108,864 |     127.6 |         2.8 |
|                          |   134,217,728 |     262.2 |         2.3 |
|                          |   268,435,456 |     548.8 |        15.5 |
|                          | 1,073,741,824 |   2,499.8 |       205.4 |
| openmp (multi-threaded)  |    16,777,216 |      16.0 |         1.4 |
|                          |    33,554,432 |      27.9 |         1.1 |
|                          |    67,108,864 |      57.8 |         6.4 |
|                          |   134,217,728 |     119.3 |         6.5 |
|                          |   268,435,456 |     256.7 |        43.7 |
|                          | 1,073,741,824 |   1,186.2 |        98.6 |
| auto (runtime selection) |    16,777,216 |      18.1 |         4.9 |
|                          |    33,554,432 |      31.6 |         6.7 |
|                          |    67,108,864 |      61.4 |         6.7 |
|                          |   134,217,728 |     124.4 |         7.2 |
|                          |   268,435,456 |     273.8 |        11.0 |
|                          | 1,073,741,824 |   1,253.4 |        97.5 |

**NVIDIA A30 server (Linux 5.14.0-570.49.1.el9_6.x86_64)**
GPU: NVIDIA A30 (CUDA 13.0 runtime, driver 580.95.05, nvcc 12.6.68)
Host CPU: Dual AMD EPYC 9254 (48 hardware threads)
System RAM: 377 GiB, GPU RAM: 24 GiB HBM2

| Size (points) | Mean (ms) | StdDev (ms) | Speedup vs CPU |
| ------------: | --------: | ----------: | -------------: |
|    16,777,216 |      10.7 |         0.0 |          2.7× |
|    33,554,432 |      22.8 |         1.0 |          2.6× |
|    67,108,864 |      47.3 |         0.1 |          2.7× |
|   134,217,728 |      86.9 |         3.5 |          3.0× |
|   268,435,456 |     171.9 |         0.1 |          3.2× |
| 1,073,741,824 |     714.6 |         5.6 |          3.5× |

## Performance Insights

- **FWHT is extremely memory-bandwidth bound**: Performance depends on memory subsystem, not raw TFLOPS
  - Each element accessed log₂(n) times with irregular stride patterns (low arithmetic intensity)
- **GPU architecture matters**: A30 (HBM2) achieves 2.7-3.5× speedup with ±0.1ms variance; RTX 4090 (GDDR6X) shows 1.8-2.4× with ±3-22ms variance
- **OpenMP scales well**: 2.7× speedup on 10 cores via task-based recursive parallelism
- **Best practices**:
  - Single transforms < 64M: Use OpenMP CPU (lower latency)
  - Single transforms ≥ 64M: GPU benefits from reduced PCIe overhead ratio
  - Batch operations (10+ transforms): GPU strongly preferred
  - GPU selection: Prefer HBM-based datacenter GPUs for consistent performance

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
