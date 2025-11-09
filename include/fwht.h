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
 * Version: 1.0.0
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
#define FWHT_VERSION_MINOR 0
#define FWHT_VERSION_PATCH 0
#define FWHT_VERSION "1.0.0"

/* ============================================================================
 * BACKEND SELECTION
 * ============================================================================ */

typedef enum {
    FWHT_BACKEND_AUTO = 0,    /* Automatic selection based on size */
    FWHT_BACKEND_CPU,         /* Single-threaded reference implementation */
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
    FWHT_ERROR_CUDA                 /* CUDA runtime error */
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
