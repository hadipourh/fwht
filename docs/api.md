# libfwht API Guide

This document explains how to use libfwht at three levels: the C/CUDA library, the command-line interface, and the Python bindings. Use it as a reference when integrating the Walsh-Hadamard transform into your own tooling.

---

## 1. Getting Started

| Step | Command | Notes |
|------|---------|-------|
| Build (CPU only) | `make` | Produces `lib/libfwht.{a,so}` and runs the regression suite |
| Build with CUDA | `make clean && make` | Auto-enables `USE_CUDA=1` when `nvcc` is in `PATH` |
| Build the CLI | `make cli` | Creates `build/fwht_cli` linked against the shared library |
| Build examples | `make examples` | Generates sample programs under `examples/` |

To link libfwht into your own C program, include the header and link against `libfwht`:

```c
#include "fwht.h"
// gcc app.c -Iinclude -Llib -lfwht -lm -o app -Wl,-rpath,$(pwd)/lib
```

Fundamentals:
- All transform sizes must be powers of two (`fwht_is_power_of_2` helps with validation).
- Transforms run in-place; allocate `n` elements before calling any routine.
- The API is thread-safe as long as every thread works on its own buffers.

---

## 2. Core Integer Transforms

Convert Boolean truth tables into ±1 integers and call the int32 interface:

```c
int32_t data[8] = { 1, -1, -1, 1, -1, 1, 1, -1 };
fwht_status_t st = fwht_i32(data, 8);
if (st != FWHT_SUCCESS) {
    fprintf(stderr, "FWHT failed: %s\n", fwht_error_string(st));
}
```

Common variants:
- `fwht_i32_safe(data, n)` – identical signature with overflow detection.
- `fwht_f64(data, n)` – double-precision transform when you need fractional outputs immediately.
- `fwht_i8(data, n)` – compact path for very small transforms.
- `fwht_i32_backend(data, n, backend)` / `fwht_f64_backend(...)` – pin a specific backend instead of relying on auto-selection.

Boolean helpers save boilerplate:

```c
uint8_t truth[256] = { /* 0/1 values */ };
int32_t walsh[256];
fwht_from_bool(truth, walsh, 256, /*signed_rep=*/true);

double corr[256];
fwht_correlations(truth, corr, 256);  // normalized to [-1, 1]
```

---

## 3. Batch and Bit-Packed APIs

### SIMD batch transforms

Batch APIs eliminate per-transform overhead when you have many signals of the same length:

```c
int32_t *blocks[32];
for (size_t i = 0; i < 32; ++i) {
    blocks[i] = malloc(256 * sizeof(int32_t));
    // populate blocks[i]
}
fwht_status_t st = fwht_i32_batch(blocks, 256, 32);
```

`fwht_f64_batch` mirrors the same interface for double precision. Expect 3–5× speedups for `n ≤ 256`.

### Bit-packed Boolean transforms

Popcount-based routines avoid expanding Boolean data to 32-bit integers:

```c
uint64_t packed[4] = { /* 256 bits */ };
int32_t walsh[256];
fwht_boolean_packed(packed, walsh, 256);

fwht_boolean_packed_backend(packed, walsh, 256, FWHT_BACKEND_OPENMP);
```

`fwht_boolean_batch` processes arrays of packed truth tables (ideal for S-box analysis).

---

## 4. GPU Acceleration

Compiling with CUDA enables the GPU backend automatically. At runtime:

```c
if (!fwht_has_gpu()) {
    fprintf(stderr, "No CUDA device detected.\n");
    return;
}

fwht_i32_backend(data, 1024, FWHT_BACKEND_GPU);
```

GPU features in `fwht.h`:
- **Batch kernels**: `fwht_batch_i32_cuda`, `fwht_batch_f64_cuda`, `fwht_batch_f32_cuda` operate on host arrays (copies handled internally).
- **Device-pointer APIs**: `fwht_batch_*_cuda_device` skip host copies when you already hold device memory.
- **FP16 Tensor Core path**: `fwht_batch_f16_cuda_device` accelerates fp16 workloads.
- **Persistent contexts**: `fwht_gpu_context_create(max_n, max_batch)` plus `fwht_gpu_context_compute_*` amortize allocations across repeated calls.
- **Profiling and tuning**: enable metrics via `fwht_gpu_set_profiling(true)` and read them with `fwht_gpu_get_last_metrics`. Adjust kernel behavior with `fwht_gpu_set_block_size`, `fwht_gpu_set_multi_shuffle`, and inspect device info (`fwht_gpu_get_device_name`, `fwht_gpu_get_compute_capability`, etc.).
- **Memory helpers**: `fwht_gpu_host_alloc`, `fwht_gpu_device_alloc`, and the copy utilities simplify pinned-memory transfers.

All GPU-only APIs return `FWHT_ERROR_BACKEND_UNAVAILABLE` when CUDA is not present, so the same code can run on CPU-only deployments.

---

## 5. Command-Line Interface

The CLI lets you test data or integrate WHT computations into shell pipelines:

```bash
./build/fwht_cli --values 0,1,1,0,1,0,0,1 --backend cpu --normalize
```

Key options (see `./build/fwht_cli --help` for the full list):
- `--input <file>` / `--values <list>` – read whitespace or comma-separated tokens.
- `--dtype i32|f64` plus `--input-format bool|signed|float` – control parsing.
- `--batch-size <n>` – split the stream into `n` equal transforms and use the batch APIs automatically.
- `--backend auto|cpu|cpu-safe|openmp|gpu` – mirror the C backend selector.
- `--gpu-profile`, `--gpu-block-size <pow2>` – expose CUDA profiling/tuning knobs when the library is built with CUDA.
- `--normalize`, `--precision <digits>`, `--no-index`, `--quiet` – adjust output formatting for downstream tools.

---

## 6. Python Bindings

`pyfwht` mirrors the C API with NumPy and DLPack support:

```python
import numpy as np
import pyfwht as fwht

truth = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
spectrum = fwht.from_bool(truth, signed=True)

vec = np.random.randint(-50, 50, size=1024, dtype=np.int32)
fwht.transform(vec, backend="auto")  # in-place
```

Highlights:
- Zero-copy NumPy interoperability when arrays are aligned.
- Backend auto-selection identical to the C implementation.
- GPU support for fp64/fp32/fp16, including Tensor Core acceleration via DLPack tensors (PyTorch, JAX, etc.).
- Example scripts in `python/examples/` cover Boolean analysis, GPU batching, and precision benchmarking.

Install locally with `pip install -e python/` or build a wheel via `python -m build` inside the `python` directory.

---

## 7. Troubleshooting & Best Practices

- **Power-of-two enforcement** – use `fwht_is_power_of_2(n)` and `fwht_log2(n)` before running transforms.
- **Overflow concerns** – switch to `fwht_i32_safe` or `fwht_f64` when `n * max(|data|)` approaches `2^31`.
- **Backend availability** – query `fwht_has_openmp()` and `fwht_has_gpu()` before forcing those backends.
- **Batch layout** – each pointer passed to batch APIs must reference `n` contiguous elements; mismatches trigger `FWHT_ERROR_INVALID_SIZE`.
- **GPU contexts** – destroy contexts with `fwht_gpu_context_destroy` to free device memory.
- **CLI parsing** – float tokens require `--dtype f64 --input-format float`; otherwise the CLI intentionally rejects non-integer data.

---

## 8. Further Reading

- `examples/example_basic.c` – core API walkthrough.
- `examples/example_boolean_packed.c` – bit-packed Boolean workflow.
- `examples/example_batch.c` – overflow-safe mode, SIMD batches, and Boolean batches.
- `examples/example_gpu_multi_precision.cu` – GPU profiling, device-pointer APIs, and multi-precision transforms.
- `docs/wht.md` – background on Walsh-Hadamard theory.
- `README.md` – repository overview, build instructions, and benchmarks.

Use this guide as your map: start with the quick snippets above, then explore the example programs for complete reference implementations.
