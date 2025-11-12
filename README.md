# LibFWHT

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

High-performance C99 library for computing the Fast Walsh-Hadamard Transform (FWHT), a fundamental tool in cryptanalysis and Boolean function analysis. The library provides multiple backend implementations (vectorized single-threaded CPU, OpenMP, and CUDA) with automatic selection based on problem size, offering optimal performance across different hardware configurations.

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

- Fast Walsh-Hadamard Transform with iterative butterfly operations running in-place at `O(n log n)` complexity
- Total memory usage is `O(n)` for the input array with `O(1)` auxiliary storage (no temporary buffers), where `n = 2^k` is the truth table size for a k-variable Boolean function
- CPU backend auto-detects SIMD support (AVX2, SSE2, or NEON) and falls back to scalar code when unavailable
- Automatic backend selection prefers GPU for large instances and OpenMP for medium-sized workloads
- CUDA backend auto-tunes its grid/block configuration from the active device (override with `fwht_gpu_set_block_size` when needed)

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
- `fwht_f64`: in-place transform for `double` data when fractional coefficients matter
- `fwht_i8`: in-place transform for `int8_t` data to minimize memory footprint (watch for overflow)
- `fwht_i32_backend`, `fwht_f64_backend`: same transforms with explicit backend selection (`AUTO`, `CPU`, `OPENMP`, `GPU`)
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

```
# build the benchmark harness (run from the libfwht root)
make bench

# CPU timings on the Apple M4 host
make openmp
./build/fwht_bench \
    --backend=cpu \
    --sizes=16777216,33554432,67108864,134217728,268435456 \
    --repeats=10

# GPU timings on the NVIDIA A30 host
./build/fwht_bench \
    --backend=gpu \
    --sizes=16777216,33554432,67108864,134217728,268435456 \
    --repeats=10 \
    --warmup=1
```

Each command samples the same transform sizes reported in the tables below. Adjust `--backend` and the size list as needed for other hardware.

**Apple M4 desktop (macOS 15.7.1)**
CPU: Apple M4 (10 physical / 10 logical cores)
Memory: 24 GiB unified

| Mode | Size (points) | Mean (ms) | StdDev (ms) |
| :--- | ------------: | --------: | ----------: |
| cpu (single-threaded) |    16,777,216 |    37.935 |       1.685 |
|  |    33,554,432 |    89.638 |       8.207 |
|  |    67,108,864 |   172.907 |       4.434 |
|  |   134,217,728 |   353.781 |       2.846 |
|  |   268,435,456 |   733.108 |      10.635 |
| openmp (multi-threaded) |    16,777,216 |    27.602 |       1.109 |
|  |    33,554,432 |    61.920 |       0.695 |
|  |    67,108,864 |   133.844 |       1.170 |
|  |   134,217,728 |   282.285 |       7.859 |
|  |   268,435,456 |   586.918 |       6.049 |
| auto (runtime selection) |    16,777,216 |    27.461 |       0.468 |
|  |    33,554,432 |    62.273 |       0.937 |
|  |    67,108,864 |   133.579 |       1.280 |
|  |   134,217,728 |   279.426 |       1.640 |
|  |   268,435,456 |   584.172 |       2.441 |

**NVIDIA A30 server (Linux 5.14.0-570.49.1.el9_6.x86_64)**
GPU: NVIDIA A30 (CUDA 13.0 runtime, driver 580.95.05, nvcc 12.6.68)
Host CPU: Dual AMD EPYC 9254 (48 hardware threads)
System RAM: 377 GiB, GPU RAM: 24 GiB

| Size (points) | Mean (ms) | StdDev (ms) |
| ------------: | --------: | ----------: |
|    16,777,216 |    11.440 |       0.032 |
|    33,554,432 |    22.668 |       0.079 |
|    67,108,864 |    53.521 |       0.062 |
|   134,217,728 |    94.560 |       2.663 |
|   268,435,456 |   178.840 |       0.082 |

Observed trends:

- GPU overtakes the CPU decisively beyond `2^24` points, delivering ~3–4× speedup even with PCIe transfers.
- Building with `make openmp` engages the SIMD-aware tiling pass, roughly halves CPU runtime on 10-core Apple silicon for 32M–256M point transforms relative to the single-thread baseline, and keeps scaling through the final stages.
- The `auto` backend selects OpenMP at these sizes, matching the dedicated multi-thread timings.
- Sub-`2^22` workloads benefit from CPU execution unless multiple transforms are batched on the GPU.
- Adjust `fwht_gpu_set_block_size` and reuse device buffers to minimise launch overhead for long-running jobs.

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
