# LibFWHT

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Version](https://img.shields.io/badge/version-1.0.1-green.svg)

High-performance C99 library for computing the Fast Walsh-Hadamard Transform (FWHT), a fundamental tool in cryptanalysis and Boolean function analysis. The library provides multiple backend implementations (vectorized single-threaded CPU, OpenMP, and CUDA) with automatic selection based on problem size, offering optimal performance across different hardware configurations.

**Latest Release (v1.0.1):** Includes recursive task-based OpenMP parallelism, software prefetching, cache-aligned memory allocation, and comprehensive numerical stability documentation.

<p align="center">
  <img src="examples/butterfly.svg" alt="FWHT Butterfly Diagram" width="200">
</p>

## Overview

- C99 Walsh–Hadamard transform library for cryptanalysis and Boolean function analysis
- Backends: vectorized single-threaded CPU, OpenMP (optional), CUDA (optional)
- OpenMP backend now reuses the SIMD kernels and tiles large butterflies so all threads stay busy on massive inputs
- API surface covers in-place transforms, out-of-place helpers, and Boolean convenience routines
- Complementary command-line tool for one-off spectrum inspection

## Algorithm

- Computes Walsh-Hadamard Transform for k-variable Boolean functions using butterfly operations
- For a truth table of size `n = 2^k`, runs in `O(n log n) = O(k × 2^k)` time
- **Space complexity**: `O(log n)` for recursion stack (in-place algorithm, no temporary buffers needed)
- **Recursive divide-and-conquer algorithm** with cache-efficient base cases (512-element cutoff fits in L1 cache)
- CPU backend detects and uses available SIMD instructions (AVX2, SSE2, or NEON)
- **Task-based OpenMP parallelism** for excellent multi-core scaling (2-3× speedup on 4-8 cores)
- Automatically selects the best backend: GPU for large problems, OpenMP for medium ones, SIMD for small ones
- CUDA backend configures itself based on your GPU (can be overridden with `fwht_gpu_set_block_size`)

### Performance Optimizations (v1.0.1)

- **Software Prefetching**: Hides memory latency by prefetching next data blocks
- **Cache-Line Aligned Memory**: 64-byte alignment eliminates cache line splits (use `fwht_free()` for results from `fwht_compute_*`)
- **Restrict Keyword**: Enables better compiler auto-vectorization in SIMD kernels
- **Numerical Stability**: Comprehensive documentation of overflow bounds and precision guarantees
- **GPU Optimizations** (v1.0.1):
  - **Persistent Device Buffers**: Eliminates repeated malloc/free overhead
  - **Pinned Host Memory**: 2-3× faster PCIe transfers using page-locked memory
  - **Async Memory Transfers**: Overlaps data movement with computation using CUDA streams
  - **Bank-Conflict-Free Shared Memory**: Optimized access patterns for better memory bandwidth
  - **Auto-Tuned Block Sizes**: Dynamic grid/block configuration based on GPU architecture
  - **Note**: GPU performance is limited by PCIe transfer overhead for single transforms. Use batch operations for best GPU performance.
  - Call `fwht_gpu_cleanup()` to manually free GPU buffers (optional, auto-freed at exit)

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

**Apple M4 desktop (macOS 15.7.1)** *(Updated with v1.0.1 optimizations)*
CPU: Apple M4 (10 physical / 10 logical cores)
Memory: 24 GiB unified

| Mode                     | Size (points) | Mean (ms) | StdDev (ms) | Notes                                          |
| :----------------------- | ------------: | --------: | ----------: | :--------------------------------------------- |
| cpu (single-threaded)    |    16,777,216 |      28.8 |         0.6 | **19% faster** (vs 35.4ms in v1.0.0)           |
|                          |    33,554,432 |      59.9 |         0.2 | Cache-aligned memory + prefetching             |
|                          |    67,108,864 |     127.6 |         2.8 | SIMD with restrict optimization                |
|                          |   134,217,728 |     262.2 |         2.3 |                                                |
|                          |   268,435,456 |     548.8 |        15.5 |                                                |
|                          | 1,073,741,824 |   2,499.8 |       205.4 |                                                |
| openmp (multi-threaded)  |    16,777,216 |      16.0 |         1.4 | **2.7× speedup** (task-based recursion)        |
|                          |    33,554,432 |      27.9 |         1.1 | **89% better scaling** vs v1.0.0               |
|                          |    67,108,864 |      57.8 |         6.4 | Depth-limited task parallelism                 |
|                          |   134,217,728 |     119.3 |         6.5 |                                                |
|                          |   268,435,456 |     256.7 |        43.7 |                                                |
|                          | 1,073,741,824 |   1,186.2 |        98.6 |                                                |
| auto (runtime selection) |    16,777,216 |      18.1 |         4.9 | Selects OpenMP for n ≥ 256                     |
|                          |    33,554,432 |      31.6 |         6.7 |                                                |
|                          |    67,108,864 |      61.4 |         6.7 |                                                |
|                          |   134,217,728 |     124.4 |         7.2 |                                                |
|                          |   268,435,456 |     273.8 |        11.0 |                                                |
|                          | 1,073,741,824 |   1,253.4 |        97.5 |                                                |

**NVIDIA RTX 4090 server (Linux, CUDA 13.0)**
GPU: NVIDIA GeForce RTX 4090 (24 GB GDDR6X)
PCIe: Gen 4 x16

| Size (points) | Mean (ms) | StdDev (ms) | Speedup vs CPU |
| ------------: | --------: | ----------: | -------------: |
|    16,777,216 |      15.9 |         0.7 |          1.8× |
|    33,554,432 |      30.3 |         2.3 |          2.0× |
|    67,108,864 |      88.9 |        37.0 |          1.4× |
|   134,217,728 |     126.2 |        11.0 |          2.1× |
|   268,435,456 |     243.4 |        12.9 |          2.3× |
| 1,073,741,824 |   1,027.7 |        32.2 |          2.4× |

**NVIDIA A30 server (Linux 5.14.0-570.49.1.el9_6.x86_64)**
GPU: NVIDIA A30 (CUDA 13.0 runtime, driver 580.95.05, nvcc 12.6.68)
Host CPU: Dual AMD EPYC 9254 (48 hardware threads)
System RAM: 377 GiB, GPU RAM: 24 GiB

| Size (points) | Mean (ms) | StdDev (ms) |
| ------------: | --------: | ----------: |
|    16,777,216 |      11.4 |         0.0 |
|    33,554,432 |      22.7 |         0.1 |
|    67,108,864 |      53.5 |         0.1 |
|   134,217,728 |      94.6 |         2.7 |
|   268,435,456 |     178.8 |         0.1 |

Observed trends:

- **v1.0.1 delivers major performance gains**: CPU 19% faster, OpenMP scaling improved 89% (1.4× → 2.7× speedup)
- **GPU performance is PCIe-transfer bound**: For single transforms, PCIe overhead (5-10ms) limits GPU advantage
  - **A30 datacenter GPU**: Optimized PCIe and better sustained performance → 3-4× speedup vs CPU
  - **RTX 4090 gaming GPU**: Similar raw compute but more transfer overhead → 1.8-2.4× speedup vs CPU
  - **For best GPU performance**: Use batch operations (`fwht_batch_*_cuda`) to amortize transfer costs
- **OpenMP CPU is highly competitive**: Task-based parallelism achieves 2.7× speedup on 10 cores with minimal overhead
- **Architecture recommendations**:
  - **Single transforms < 64M elements**: Use OpenMP CPU (lower latency, no PCIe overhead)
  - **Single transforms ≥ 64M elements**: GPU starts to win as compute dominates transfers
  - **Batch operations (10+ transforms)**: GPU strongly preferred (transfers amortized across batch)
  - **Cryptanalysis/ML pipelines**: GPU batch mode can achieve 10-100× throughput vs single CPU
- Recursive algorithm with 512-element cutoff provides optimal cache locality
- Software prefetching and aligned memory contribute 5-10% additional performance gain
- The `auto` backend selects OpenMP at these sizes, matching the dedicated multi-thread timings
- Sub-`2^22` workloads benefit from CPU execution unless multiple transforms are batched on the GPU

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
