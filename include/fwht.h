/*
 * Fast Walsh-Hadamard Transform (FWHT) Library
 *
 * High-performance implementation of the Walsh-Hadamard Transform
 * for cryptanalysis and Boolean function analysis.
 *
 * Copyright (C) 2025 Hosein Hadipour
 *
 * Author: Hosein Hadipour <hsn.hadipour@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * Version: 1.1.4
 */

#ifndef FWHT_H
#define FWHT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ============================================================================
 * VERSION INFORMATION
 * ============================================================================ */

#define FWHT_VERSION_MAJOR 1
#define FWHT_VERSION_MINOR 1
#define FWHT_VERSION_PATCH 4
#define FWHT_VERSION "1.1.4"

/* ============================================================================
 * BACKEND SELECTION
 * ============================================================================ */

typedef enum {
    FWHT_BACKEND_AUTO = 0,    /* Automatic selection based on size */
    FWHT_BACKEND_CPU,         /* Single-threaded reference implementation */
    FWHT_BACKEND_CPU_SAFE,    /* CPU with runtime overflow detection */
    FWHT_BACKEND_OPENMP,      /* Multi-threaded CPU (OpenMP) */
    FWHT_BACKEND_GPU          /* GPU-accelerated (CUDA) */
} fwht_backend_t;

/* Query available backends at runtime */
bool fwht_has_openmp(void);
bool fwht_has_gpu(void);
const char* fwht_backend_name(fwht_backend_t backend);

/* ============================================================================
 * ERROR HANDLING
 * ============================================================================ */

typedef enum {
    FWHT_SUCCESS = 0,
    FWHT_ERROR_INVALID_SIZE,        /* Size not a power of 2 */
    FWHT_ERROR_NULL_POINTER,        /* Null pointer argument */
    FWHT_ERROR_BACKEND_UNAVAILABLE, /* Requested backend not available */
    FWHT_ERROR_OUT_OF_MEMORY,       /* Memory allocation failed */
    FWHT_ERROR_INVALID_ARGUMENT,    /* Other invalid argument */
    FWHT_ERROR_CUDA,                /* CUDA runtime error */
    FWHT_ERROR_OVERFLOW             /* Integer overflow detected */
} fwht_status_t;

const char* fwht_error_string(fwht_status_t status);

/* ============================================================================
 * CORE API - SIMPLE INTERFACE
 * 
 * These are the primary functions most users need.
 * All transforms are in-place and use the standard butterfly algorithm.
 * ============================================================================ */

/*
 * In-place Walsh-Hadamard Transform for 32-bit signed integers.
 * 
 * Parameters:
 *   data - Array of n signed 32-bit integers (modified in-place)
 *   n    - Size of array (must be power of 2)
 * 
 * Returns: FWHT_SUCCESS or error code
 * 
 * Mathematical Definition:
 *   WHT[u] = Σ_{x=0}^{n-1} data[x] * (-1)^{popcount(u & x)}
 * 
 * Typical Usage:
 *   For Boolean function f: {0,1}^k → {0,1} where n = 2^k
 *   Convert: data[x] = (f(x) == 0) ? +1 : -1
 *   After transform: data[u] = WHT coefficient for mask u
 *   Correlation: Cor(f, u) = data[u] / n
 * 
 * Complexity: O(n log n)
 * Thread-safe: Yes (different arrays can be processed concurrently)
 */
fwht_status_t fwht_i32(int32_t* data, size_t n);

/*
 * In-place WHT for 32-bit integers with runtime overflow detection.
 * 
 * This variant uses compiler builtins (__builtin_add_overflow,
 * __builtin_sub_overflow) to detect integer overflow during computation.
 * Returns FWHT_ERROR_OVERFLOW if any overflow is detected.
 * 
 * Performance: ~5-10% slower than fwht_i32() due to overflow checks.
 * 
 * Use when:
 *   - Input magnitudes are large or unknown
 *   - Safety is more important than performance
 *   - Validating that n * max(|input|) < 2^31
 * 
 * Returns:
 *   FWHT_SUCCESS - Transform completed without overflow
 *   FWHT_ERROR_OVERFLOW - Integer overflow detected, data may be corrupted
 */
fwht_status_t fwht_i32_safe(int32_t* data, size_t n);

/*
 * In-place Walsh-Hadamard Transform for double precision floats.
 * 
 * Same as fwht_i32 but for floating-point data.
 * Useful when normalization or fractional values are needed.
 */
fwht_status_t fwht_f64(double* data, size_t n);

/*
 * In-place WHT for 8-bit signed integers (memory-efficient).
 * WARNING: May overflow for large n. Use only when n * max(|data|) < 128
 */
fwht_status_t fwht_i8(int8_t* data, size_t n);

/* ============================================================================
 * VECTORIZED BATCH API - HIGH PERFORMANCE FOR SMALL TRANSFORMS
 * 
 * Process multiple independent transforms simultaneously using SIMD.
 * Ideal for cryptanalysis: compute WHT for thousands of S-boxes in parallel.
 * 
 * Performance gain: 3-5× faster than sequential processing for n ≤ 256
 * 
 * Requirements:
 *   - All transforms must have the same size n
 *   - n must be a power of 2
 *   - batch_size can be any positive integer
 * 
 * Memory layout:
 *   data_array[0] points to first transform (n elements)
 *   data_array[1] points to second transform (n elements)
 *   ...
 *   data_array[batch_size-1] points to last transform
 * 
 * Example - S-box linear cryptanalysis:
 *   int32_t* sboxes[256];  // 256 different S-boxes
 *   for (int i = 0; i < 256; i++) {
 *       sboxes[i] = malloc(256 * sizeof(int32_t));
 *       // Fill with truth table...
 *   }
 *   fwht_i32_batch(sboxes, 256, 256);  // 3-5× faster than loop
 * 
 * ============================================================================ */

/*
 * Vectorized batch WHT for int32 arrays.
 * Processes batch_size transforms of size n in parallel using SIMD.
 * 
 * Parameters:
 *   data_array - Array of pointers to transforms
 *   n          - Size of each transform (must be power of 2)
 *   batch_size - Number of transforms to process
 * 
 * Returns:
 *   FWHT_SUCCESS on success
 *   FWHT_ERROR_INVALID_SIZE if n is not power of 2
 *   FWHT_ERROR_NULL_POINTER if data_array is NULL
 * 
 * Performance:
 *   - n ≤ 256:  3-5× faster than sequential (uses AVX2/NEON)
 *   - n > 256:  1.2-1.5× faster (memory-bound, less SIMD benefit)
 */
fwht_status_t fwht_i32_batch(int32_t** data_array, size_t n, size_t batch_size);

/*
 * Vectorized batch WHT for float64 arrays.
 * Same as fwht_i32_batch but for double precision.
 */
fwht_status_t fwht_f64_batch(double** data_array, size_t n, size_t batch_size);

/* ============================================================================
 * CORE API - BACKEND CONTROL
 * 
 * Explicit backend selection for performance tuning.
 * ============================================================================ */

fwht_status_t fwht_i32_backend(int32_t* data, size_t n, fwht_backend_t backend);
fwht_status_t fwht_f64_backend(double* data, size_t n, fwht_backend_t backend);

/* ============================================================================
 * GPU/CUDA BATCH PROCESSING
 * 
 * Process multiple WHTs in parallel on GPU.
 * Only available when compiled with CUDA support.
 * ============================================================================ */

#ifdef USE_CUDA
/*
 * Configure CUDA execution parameters (optional).
 * Provide a power-of-two block size in [1, 1024] to override auto-tuning.
 * Pass 0 to revert to automatic selection based on the active GPU.
 */
fwht_status_t fwht_gpu_set_block_size(unsigned int block_size);
unsigned int  fwht_gpu_get_block_size(void);

/*
 * Lightweight profiling support for the CUDA backend.
 * Enable to collect host-to-device, kernel, and device-to-host timings.
 */
typedef struct fwht_gpu_metrics {
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    size_t n;
    size_t batch_size;
    size_t bytes_transferred;
    int    samples;
    bool   valid;
} fwht_gpu_metrics_t;

fwht_status_t fwht_gpu_set_profiling(bool enable);
bool fwht_gpu_profiling_enabled(void);
fwht_gpu_metrics_t fwht_gpu_get_last_metrics(void);

/*
 * GPU device information queries.
 * These functions auto-initialize device state if needed.
 * Useful for understanding GPU architecture and debugging performance.
 */
unsigned int fwht_gpu_get_smem_banks(void);         /* Returns 16 or 32 */
unsigned int fwht_gpu_get_compute_capability(void);  /* e.g., 75=Turing, 80=Ampere, 89=Ada, 90=Hopper */
const char* fwht_gpu_get_device_name(void);         /* e.g., "NVIDIA A100-SXM4-40GB" */
unsigned int fwht_gpu_get_sm_count(void);           /* Number of streaming multiprocessors */

/* Control small-N multi-element warp-shuffle optimization (32 < N ≤ 512). */
fwht_status_t fwht_gpu_set_multi_shuffle(bool enable);
bool          fwht_gpu_multi_shuffle_enabled(void);

/*
 * Batch processing of multiple WHTs on GPU.
 *
 * Parameters:
 *   data       - Flat array containing batch_size WHTs of size n each
 *   n          - Size of each WHT (must be power of 2)
 *   batch_size - Number of WHTs to process
 *
 * Layout: data[0..n-1] = first WHT, data[n..2n-1] = second WHT, etc.
 */
fwht_status_t fwht_batch_i32_cuda(int32_t* data, size_t n, size_t batch_size);
fwht_status_t fwht_batch_f64_cuda(double* data, size_t n, size_t batch_size);

/* Device-pointer APIs: operate on GPU-resident buffers (no H2D/D2H copies). */
fwht_status_t fwht_batch_i32_cuda_device(int32_t* d_data, size_t n, size_t batch_size);
fwht_status_t fwht_batch_f64_cuda_device(double* d_data, size_t n, size_t batch_size);

/* ============================================================================
 * PERSISTENT GPU CONTEXT API
 * 
 * For applications that compute many WHTs repeatedly, creating a persistent
 * context pre-allocates GPU memory and eliminates repeated cudaMalloc/cudaFree
 * overhead. This can provide 5-10x speedup for cryptanalysis workloads.
 * 
 * Usage:
 *   fwht_gpu_context_t* ctx = fwht_gpu_context_create(max_n, max_batch_size);
 *   for (many iterations) {
 *       fwht_gpu_context_compute_i32(ctx, data, n, batch_size);
 *   }
 *   fwht_gpu_context_destroy(ctx);
 * ============================================================================ */

typedef struct fwht_gpu_context fwht_gpu_context_t;

/*
 * Create a persistent GPU context with pre-allocated device memory.
 * 
 * Parameters:
 *   max_n          - Maximum transform size (must be power of 2)
 *   max_batch_size - Maximum batch size
 * 
 * Returns: Context pointer, or NULL on error
 * 
 * The context pre-allocates max_n * max_batch_size elements on the GPU.
 * Subsequent transforms with n <= max_n and batch <= max_batch_size
 * will reuse this allocation without cudaMalloc/cudaFree overhead.
 */
fwht_gpu_context_t* fwht_gpu_context_create(size_t max_n, size_t max_batch_size);

/*
 * Destroy GPU context and free all allocated resources.
 */
void fwht_gpu_context_destroy(fwht_gpu_context_t* ctx);

/*
 * Compute WHT using persistent context (int32).
 * Must have: n <= ctx->max_n && batch_size <= ctx->max_batch_size
 */
fwht_status_t fwht_gpu_context_compute_i32(fwht_gpu_context_t* ctx, 
                                            int32_t* data, size_t n, size_t batch_size);

/*
 * Compute WHT using persistent context (double).
 * Must have: n <= ctx->max_n && batch_size <= ctx->max_batch_size
 */
fwht_status_t fwht_gpu_context_compute_f64(fwht_gpu_context_t* ctx,
                                            double* data, size_t n, size_t batch_size);

/*
 * Host memory utilities for faster transfers.
 * Allocate/free page-locked (pinned) host memory when available.
 * Safe to call even if CUDA isn’t initialized; allocation will fail with
 * FWHT_ERROR_BACKEND_UNAVAILABLE in that case.
 */
fwht_status_t fwht_gpu_host_alloc(void** ptr, size_t bytes);
void          fwht_gpu_host_free(void* ptr);

/* Optional low-level device memory helpers (CUDA only). */
fwht_status_t fwht_gpu_device_alloc(void** d_ptr, size_t bytes);
void          fwht_gpu_device_free(void* d_ptr);
fwht_status_t fwht_gpu_memcpy_h2d(void* d_dst, const void* h_src, size_t bytes);
fwht_status_t fwht_gpu_memcpy_d2h(void* h_dst, const void* d_src, size_t bytes);

/* ============================================================================
 * GPU LOAD/STORE CALLBACKS (Advanced)
 * 
 * For advanced users who want to fuse custom preprocessing or postprocessing
 * with the FWHT on the GPU. This eliminates redundant memory transfers and
 * separate kernel launches.
 * 
 * Callbacks are device function pointers that execute on the GPU. They can:
 * - Preprocess data before transform (e.g., apply mask, convert format)
 * - Postprocess results after transform (e.g., normalize, extract features)
 * 
 * Expected performance gain: 10-20% reduction in total kernel time by
 * eliminating redundant global memory access and extra kernel launches.
 * 
 * Usage:
 *   // Define device function
 *   __device__ int32_t my_preprocess(int32_t val, size_t idx, void* params) {
 *       return val ^ mask;  // XOR with mask
 *   }
 *   
 *   // Get function pointer
 *   __device__ fwht_load_fn_i32 d_preprocess = my_preprocess;
 *   fwht_load_fn_i32 h_preprocess;
 *   cudaMemcpyFromSymbol(&h_preprocess, d_preprocess, sizeof(void*));
 *   
 *   // Use with context
 *   fwht_gpu_context_set_callbacks_i32(ctx, h_preprocess, NULL, NULL);
 *   fwht_gpu_context_compute_i32(ctx, data, n, 1);
 * 
 * IMPORTANT: This is an advanced API. Callbacks must be __device__ functions.
 * Incorrect usage can cause GPU crashes. Use NULL to disable callbacks.
 * ============================================================================ */

#if defined(__CUDACC__) || defined(USE_CUDA)
/* Device function pointer types for load/store callbacks */
typedef int32_t (*fwht_load_fn_i32)(int32_t value, size_t index, void* user_params);
typedef void (*fwht_store_fn_i32)(int32_t* dest, int32_t value, size_t index, void* user_params);
typedef double (*fwht_load_fn_f64)(double value, size_t index, void* user_params);
typedef void (*fwht_store_fn_f64)(double* dest, double value, size_t index, void* user_params);

/*
 * Set load/store callbacks for int32 transforms.
 * 
 * Parameters:
 *   ctx         - GPU context
 *   load_fn     - Device function called when loading data (NULL = no preprocessing)
 *   store_fn    - Device function called when storing results (NULL = no postprocessing)
 *   user_params - User-defined parameter pointer passed to callbacks
 * 
 * Returns: FWHT_SUCCESS or error code
 */
fwht_status_t fwht_gpu_context_set_callbacks_i32(fwht_gpu_context_t* ctx,
                                                   fwht_load_fn_i32 load_fn,
                                                   fwht_store_fn_i32 store_fn,
                                                   void* user_params);

/*
 * Set load/store callbacks for double transforms.
 */
fwht_status_t fwht_gpu_context_set_callbacks_f64(fwht_gpu_context_t* ctx,
                                                   fwht_load_fn_f64 load_fn,
                                                   fwht_store_fn_f64 store_fn,
                                                   void* user_params);
#endif /* __CUDACC__ || USE_CUDA */

#endif

/* ============================================================================
 * ADVANCED API - OUT-OF-PLACE TRANSFORMS
 * 
 * Allocates output array and returns pointer.
 * User must free() the result.
 * ============================================================================ */

/*
 * Compute WHT and return new array (input unchanged).
 * Returns: Pointer to newly allocated array, or NULL on error
 * User responsibility: Call free() on the result
 */
int32_t* fwht_compute_i32(const int32_t* input, size_t n);
double*  fwht_compute_f64(const double* input, size_t n);

/* With backend control */
int32_t* fwht_compute_i32_backend(const int32_t* input, size_t n, fwht_backend_t backend);
double*  fwht_compute_f64_backend(const double* input, size_t n, fwht_backend_t backend);

/* ============================================================================
 * ADVANCED API - CONTEXT FOR REPEATED CALLS
 * 
 * For applications that compute many WHTs, creating a context amortizes
 * setup costs (thread pools, GPU memory allocation, etc.)
 * ============================================================================ */

typedef struct fwht_context fwht_context_t;

typedef struct {
    fwht_backend_t backend;
    int num_threads;        /* For OpenMP (0 = auto-detect) */
    int gpu_device;         /* GPU device ID (default: 0) */
    bool normalize;         /* Divide by sqrt(n) after transform */
} fwht_config_t;

/* Default configuration */
fwht_config_t fwht_default_config(void);

/* Create/destroy context */
fwht_context_t* fwht_create_context(const fwht_config_t* config);
void            fwht_destroy_context(fwht_context_t* ctx);

/* Compute using context (more efficient for repeated calls) */
fwht_status_t fwht_transform_i32(fwht_context_t* ctx, int32_t* data, size_t n);
fwht_status_t fwht_transform_f64(fwht_context_t* ctx, double* data, size_t n);

/* ============================================================================
 * ADVANCED API - BATCH PROCESSING
 * 
 * Compute multiple WHTs in parallel (optimal for GPU).
 * All arrays must have the same size.
 * ============================================================================ */

/*
 * Batch transform: compute WHT for multiple arrays in parallel.
 * 
 * Parameters:
 *   ctx        - Context (use NULL for default)
 *   data_array - Array of pointers to data arrays
 *   n          - Size of each array (must be same for all)
 *   batch_size - Number of arrays to process
 * 
 * This is significantly faster than calling fwht_i32 in a loop,
 * especially on GPU where batch operations amortize transfer costs.
 */
fwht_status_t fwht_batch_i32(fwht_context_t* ctx, int32_t** data_array, 
                             size_t n, int batch_size);
fwht_status_t fwht_batch_f64(fwht_context_t* ctx, double** data_array,
                             size_t n, int batch_size);

/* ============================================================================
 * CONVENIENCE API - BOOLEAN FUNCTIONS
 * 
 * Direct operations on Boolean functions represented as bit arrays.
 * ============================================================================ */

/*
 * Compute WHT of Boolean function.
 * 
 * Parameters:
 *   bool_func - Boolean function as array of 0/1 values
 *   wht_out   - Output array for WHT coefficients (size n)
 *   n         - Size (must be power of 2)
 *   signed_rep - If true: converts 0→+1, 1→-1 before transform
 *                If false: uses values as-is
 * 
 * This is a convenience wrapper that handles the conversion:
 *   signed_rep=true:  wht_out[u] = Σ (-1)^{bool_func[x] ⊕ popcount(u&x)}
 *   signed_rep=false: wht_out[u] = Σ bool_func[x] * (-1)^{popcount(u&x)}
 */
fwht_status_t fwht_from_bool(const uint8_t* bool_func, int32_t* wht_out, 
                             size_t n, bool signed_rep);

/*
 * Compute correlations between Boolean function and all linear functions.
 * 
 * Parameters:
 *   bool_func - Boolean function as array of 0/1 values
 *   corr_out  - Output array for correlations (size n)
 *   n         - Size (must be power of 2)
 * 
 * Output: corr_out[u] = Cor(f, ℓ_u) where ℓ_u(x) = popcount(u & x) mod 2
 *         Values in range [-1.0, +1.0]
 */
fwht_status_t fwht_correlations(const uint8_t* bool_func, double* corr_out, size_t n);

/* ============================================================================
 * BIT-SLICED BOOLEAN WHT (HIGH PERFORMANCE FOR CRYPTOGRAPHY)
 * 
 * Optimized WHT for bit-packed Boolean functions using popcount instructions.
 * This is 32-64× faster than unpacked representation for cryptanalysis.
 * 
 * Key advantages:
 * - Memory efficient: 32× less memory (1 bit vs 32 bits per element)
 * - Cache friendly: Entire truth table fits in L1 cache
 * - SIMD optimized: Uses __builtin_popcount (CPU) or __popc (GPU)
 * - Perfect for crypto: Designed for ±1 Boolean functions
 * 
 * Use cases:
 * - S-box analysis (compute WHT for thousands of Boolean components)
 * - Nonlinearity computation (find max |Walsh coefficient|)
 * - Correlation immunity testing
 * - Bent function verification
 * ============================================================================ */

/*
 * Compute WHT from bit-packed Boolean function (CPU-optimized).
 * 
 * Input format: Bit-packed truth table where bit i of word j represents
 *               bool_func[j*64 + i] for uint64_t, or bool_func[j*32 + i] for uint32_t.
 * 
 * Parameters:
 *   packed_bits - Bit-packed Boolean function (n/64 uint64_t elements)
 *   wht_out     - Output WHT spectrum (n int32_t elements)
 *   n           - Transform size (must be power of 2, n ≤ 65536)
 * 
 * Performance: 32-64× faster than fwht_from_bool() for n ≥ 256
 * 
 * Example (n=8):
 *   Truth table: [0,1,1,0,1,0,0,1]
 *   Packed: 0b10010110 = 0x96 (one uint64_t with bits 1,2,4,7 set)
 *   
 *   uint64_t packed = 0x96;
 *   int32_t wht[8];
 *   fwht_boolean_packed(&packed, wht, 8);
 */
fwht_status_t fwht_boolean_packed(const uint64_t* packed_bits, int32_t* wht_out, size_t n);

/*
 * Compute WHT from bit-packed Boolean function with backend selection.
 * Allows explicit CPU vs GPU backend choice for bit-sliced transforms.
 */
fwht_status_t fwht_boolean_packed_backend(const uint64_t* packed_bits, int32_t* wht_out, 
                                           size_t n, fwht_backend_t backend);

/*
 * Batch bit-sliced WHT for vectorial Boolean functions (S-box cryptanalysis).
 * 
 * Process multiple bit-packed Boolean functions in parallel.
 * Ideal for analyzing all component functions of an S-box simultaneously.
 * 
 * Parameters:
 *   packed_batch - Array of pointers to bit-packed functions
 *   wht_batch    - Array of pointers to output WHT spectra
 *   n            - Transform size (same for all functions)
 *   batch_size   - Number of Boolean functions to process
 * 
 * Performance: 50-100× faster than sequential unpacked transforms
 * 
 * Example (analyze 8-bit S-box with 8 component functions):
 *   uint64_t* sbox_bits[8];  // 8 component functions
 *   int32_t* sbox_wht[8];    // 8 WHT spectra
 *   for (int i = 0; i < 8; i++) {
 *       sbox_bits[i] = malloc((256/64) * sizeof(uint64_t));
 *       sbox_wht[i] = malloc(256 * sizeof(int32_t));
 *       // Pack S-box component function i into sbox_bits[i]
 *   }
 *   fwht_boolean_batch(sbox_bits, sbox_wht, 256, 8);
 */
fwht_status_t fwht_boolean_batch(const uint64_t** packed_batch, int32_t** wht_batch,
                                  size_t n, size_t batch_size);

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/* Check if n is a power of 2 */
bool fwht_is_power_of_2(size_t n);

/* Compute log2(n) for power of 2 (returns -1 if not power of 2) */
int fwht_log2(size_t n);

/* Get recommended backend for given size */
fwht_backend_t fwht_recommend_backend(size_t n);

/* Get version string */
const char* fwht_version(void);

/* ============================================================================
 * C11 GENERIC INTERFACE (OPTIONAL)
 * 
 * Type-safe polymorphic interface using C11 _Generic.
 * Only available when compiling with C11 or later.
 * ============================================================================ */

#if __STDC_VERSION__ >= 201112L

#define fwht(data, n) _Generic((data), \
    int32_t*: fwht_i32, \
    int8_t*:  fwht_i8, \
    double*:  fwht_f64  \
)(data, n)

#define fwht_compute(input, n) _Generic((input), \
    const int32_t*: fwht_compute_i32, \
    int32_t*:       fwht_compute_i32, \
    const double*:  fwht_compute_f64, \
    double*:        fwht_compute_f64  \
)(input, n)

#endif /* C11 */

#ifdef __cplusplus
}
#endif

#endif /* FWHT_H */
