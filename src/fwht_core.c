/*
 * Fast Walsh-Hadamard Transform - Core CPU Implementation
 *
 * Reference implementation using the butterfly algorithm.
 * This is the "ground truth" - correctness is paramount.
 * All other backends must match this exactly.
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
 */

#include "fwht.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define FWHT_HAVE_AVX2 1
#define FWHT_HAVE_SSE2 1
#elif defined(__SSE2__)
#include <emmintrin.h>
#define FWHT_HAVE_SSE2 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FWHT_HAVE_NEON 1
#endif

#ifdef _OPENMP
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <omp.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif

/* Tile size used when slicing OpenMP work inside large butterflies. */
#ifdef _OPENMP
#define FWHT_OMP_TILE_I32 256u
#define FWHT_OMP_TILE_F64 128u
#endif

static void fwht_print_simd_banner(void) {
#if defined(FWHT_HAVE_AVX2)
    fprintf(stderr, "[libfwht] CPU backend: AVX2 vector path active\n");
#elif defined(FWHT_HAVE_SSE2) && !defined(FWHT_HAVE_NEON)
    fprintf(stderr, "[libfwht] CPU backend: SSE2 vector path active\n");
#elif defined(FWHT_HAVE_NEON)
    fprintf(stderr, "[libfwht] CPU backend: NEON vector path active\n");
#else
    fprintf(stderr, "[libfwht] CPU backend: scalar path active\n");
#endif
}

static void fwht_report_simd_mode(void) {
#if defined(_WIN32)
    static LONG reported = 0;
    if (InterlockedCompareExchange(&reported, 1, 0) != 0) {
        return;
    }
    fwht_print_simd_banner();
#elif defined(__GNUC__) || defined(__clang__)
    static int reported = 0;
    if (__sync_lock_test_and_set(&reported, 1)) {
        return;
    }
    fwht_print_simd_banner();
#elif defined(__unix__) || defined(__APPLE__)
    static pthread_once_t once_control = PTHREAD_ONCE_INIT;
    pthread_once(&once_control, fwht_print_simd_banner);
#else
    static int reported = 0;
    if (reported) {
        return;
    }
    reported = 1;
    fwht_print_simd_banner();
#endif
}

static inline void fwht_process_range_i32(int32_t* even, int32_t* odd, size_t count) {
    if (count == 0) {
        return;
    }

    size_t j = 0;

#if defined(FWHT_HAVE_AVX2)
    if (count >= 8) {
        size_t avx_end = count & (size_t)~7;
        for (; j < avx_end; j += 8) {
            __m256i a = _mm256_loadu_si256((const __m256i*)(even + j));
            __m256i b = _mm256_loadu_si256((const __m256i*)(odd + j));
            __m256i sum = _mm256_add_epi32(a, b);
            __m256i diff = _mm256_sub_epi32(a, b);
            _mm256_storeu_si256((__m256i*)(even + j), sum);
            _mm256_storeu_si256((__m256i*)(odd + j), diff);
        }
    }
#endif

#if defined(FWHT_HAVE_SSE2) && !defined(FWHT_HAVE_NEON)
    if (count >= 4) {
        size_t sse_end = count & (size_t)~3;
        for (; j < sse_end; j += 4) {
            __m128i a = _mm_loadu_si128((const __m128i*)(even + j));
            __m128i b = _mm_loadu_si128((const __m128i*)(odd + j));
            __m128i sum = _mm_add_epi32(a, b);
            __m128i diff = _mm_sub_epi32(a, b);
            _mm_storeu_si128((__m128i*)(even + j), sum);
            _mm_storeu_si128((__m128i*)(odd + j), diff);
        }
    }
#endif

#if defined(FWHT_HAVE_NEON)
    if (count >= 4) {
        size_t neon_end = count & (size_t)~3;
        for (; j < neon_end; j += 4) {
            int32x4_t a = vld1q_s32(even + j);
            int32x4_t b = vld1q_s32(odd + j);
            int32x4_t sum = vaddq_s32(a, b);
            int32x4_t diff = vsubq_s32(a, b);
            vst1q_s32(even + j, sum);
            vst1q_s32(odd + j, diff);
        }
    }
#endif

    for (; j < count; ++j) {
        int32_t a = even[j];
        int32_t b = odd[j];
        even[j] = a + b;
        odd[j]  = a - b;
    }
}

static inline void fwht_process_block_i32(int32_t* data, size_t base, size_t h) {
    fwht_process_range_i32(data + base, data + base + h, h);
}

static inline void fwht_process_range_f64(double* even, double* odd, size_t count) {
    if (count == 0) {
        return;
    }

    for (size_t j = 0; j < count; ++j) {
        double a = even[j];
        double b = odd[j];
        even[j] = a + b;
        odd[j]  = a - b;
    }
}

static inline void fwht_process_block_f64(double* data, size_t base, size_t h) {
    fwht_process_range_f64(data + base, data + base + h, h);
}

#ifdef _OPENMP
static inline void fwht_process_segment_i32(int32_t* data,
                                            size_t base,
                                            size_t h,
                                            size_t offset,
                                            size_t length) {
    size_t start = base + offset;
    size_t upper = base + h;
    if (start >= upper) {
        return;
    }

    size_t clipped = length;
    if (start + clipped > upper) {
        clipped = upper - start;
    }

    fwht_process_range_i32(data + start, data + start + h, clipped);
}

static inline void fwht_process_segment_f64(double* data,
                                            size_t base,
                                            size_t h,
                                            size_t offset,
                                            size_t length) {
    size_t start = base + offset;
    size_t upper = base + h;
    if (start >= upper) {
        return;
    }

    size_t clipped = length;
    if (start + clipped > upper) {
        clipped = upper - start;
    }

    fwht_process_range_f64(data + start, data + start + h, clipped);
}
#endif /* _OPENMP */

/* =========================================================================
 * VALIDATION HELPERS
 * ========================================================================== */

static bool is_power_of_2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static fwht_status_t validate_input(const void* data, size_t n) {
    if (data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (!is_power_of_2(n)) return FWHT_ERROR_INVALID_SIZE;
    if (n == 0) return FWHT_ERROR_INVALID_SIZE;
    return FWHT_SUCCESS;
}

/* =========================================================================
 * CORE BUTTERFLY ALGORITHM - INT32
 * 
 * This is the reference implementation. Correctness verified against:
 * 1. Mathematical definition of WHT
 * 2. sboxU library (during development)
 * 3. Self-consistency (WHT(WHT(f)) = n*f property)
 * 
 * Algorithm: Cooley-Tukey butterfly (in-place, decimation-in-frequency)
 * Complexity: O(n log n)
 * Memory: O(1) auxiliary space (in-place)
 * ============================================================================ */

static void fwht_butterfly_i32(int32_t* data, size_t n) {
    fwht_report_simd_mode();
    /*
     * Butterfly algorithm with optional SIMD acceleration.
     * For each stage h = 1, 2, 4, ..., n/2:
     *   Process pairs (data[i], data[i+h]) for all applicable i
     *   Compute: sum = a + b, diff = a - b
     *   Store: data[i] = sum, data[i+h] = diff
     */
    for (size_t h = 1; h < n; h <<= 1) {
        size_t stride = h << 1;
        for (size_t i = 0; i < n; i += stride) {
            fwht_process_block_i32(data, i, h);
        }
    }
}

/* ============================================================================
 * CORE BUTTERFLY ALGORITHM - DOUBLE
 * 
 * Same algorithm, double precision for numerical applications.
 * ============================================================================ */

static void fwht_butterfly_f64(double* data, size_t n) {
    fwht_report_simd_mode();
    for (size_t h = 1; h < n; h <<= 1) {
        for (size_t i = 0; i < n; i += (h << 1)) {
            fwht_process_block_f64(data, i, h);
        }
    }
}

#ifdef _OPENMP
/* ==========================================================================
 * OPENMP PARALLEL VARIANTS
 * ========================================================================== */

static void fwht_butterfly_i32_openmp(int32_t* data, size_t n) {
    if (n < 2) {
        return;
    }

#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        for (size_t h = 1; h < n; h <<= 1) {
            size_t stride = h << 1;
            size_t blocks = n / stride;
            if (blocks >= (size_t)threads) {
#pragma omp for schedule(static)
                for (ptrdiff_t block = 0; block < (ptrdiff_t)blocks; ++block) {
                    size_t base = (size_t)block * stride;
                    fwht_process_block_i32(data, base, h);
                }
            } else {
                size_t tile = FWHT_OMP_TILE_I32;
                if (tile > h) {
                    tile = h;
                }
                size_t segments_per_block = (h + tile - 1) / tile;
                size_t total_segments = blocks * segments_per_block;
#pragma omp for schedule(static)
                for (ptrdiff_t seg = 0; seg < (ptrdiff_t)total_segments; ++seg) {
                    size_t block = (size_t)seg / segments_per_block;
                    size_t seg_in_block = (size_t)seg % segments_per_block;
                    size_t offset = seg_in_block * tile;
                    size_t len = tile;
                    if (offset + len > h) {
                        len = h - offset;
                    }
                    size_t base = block * stride;
                    fwht_process_segment_i32(data, base, h, offset, len);
                }
            }
        }
    }
}

static void fwht_butterfly_f64_openmp(double* data, size_t n) {
    if (n < 2) {
        return;
    }

#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        for (size_t h = 1; h < n; h <<= 1) {
            size_t stride = h << 1;
            size_t blocks = n / stride;
            if (blocks >= (size_t)threads) {
#pragma omp for schedule(static)
                for (ptrdiff_t block = 0; block < (ptrdiff_t)blocks; ++block) {
                    size_t base = (size_t)block * stride;
                    fwht_process_block_f64(data, base, h);
                }
            } else {
                size_t tile = FWHT_OMP_TILE_F64;
                if (tile > h) {
                    tile = h;
                }
                size_t segments_per_block = (h + tile - 1) / tile;
                size_t total_segments = blocks * segments_per_block;
#pragma omp for schedule(static)
                for (ptrdiff_t seg = 0; seg < (ptrdiff_t)total_segments; ++seg) {
                    size_t block = (size_t)seg / segments_per_block;
                    size_t seg_in_block = (size_t)seg % segments_per_block;
                    size_t offset = seg_in_block * tile;
                    size_t len = tile;
                    if (offset + len > h) {
                        len = h - offset;
                    }
                    size_t base = block * stride;
                    fwht_process_segment_f64(data, base, h, offset, len);
                }
            }
        }
    }
}
#endif /* _OPENMP */

/* ============================================================================
 * CORE BUTTERFLY ALGORITHM - INT8
 * 
 * Memory-efficient version for small values.
 * WARNING: Overflow possible if n * max(|data[i]|) >= 128
 * ============================================================================ */

static void fwht_butterfly_i8(int8_t* data, size_t n) {
    fwht_report_simd_mode();
    for (size_t h = 1; h < n; h <<= 1) {
        for (size_t i = 0; i < n; i += (h << 1)) {
            for (size_t j = i; j < i + h; ++j) {
                int8_t a = data[j];
                int8_t b = data[j + h];
                data[j]     = a + b;
                data[j + h] = a - b;
            }
        }
    }
}

/* ============================================================================
 * PUBLIC API - BASIC IN-PLACE TRANSFORMS
 * ============================================================================ */

/* CPU-only versions for internal use and fallback */
fwht_status_t fwht_i32_cpu(int32_t* data, size_t n) {
    fwht_status_t status = validate_input(data, n);
    if (status != FWHT_SUCCESS) return status;
    
    fwht_butterfly_i32(data, n);
    return FWHT_SUCCESS;
}

fwht_status_t fwht_f64_cpu(double* data, size_t n) {
    fwht_status_t status = validate_input(data, n);
    if (status != FWHT_SUCCESS) return status;

    fwht_butterfly_f64(data, n);
    return FWHT_SUCCESS;
}

/* Default API routes to AUTO backend */
fwht_status_t fwht_i32(int32_t* data, size_t n) {
    return fwht_i32_backend(data, n, FWHT_BACKEND_AUTO);
}

fwht_status_t fwht_f64(double* data, size_t n) {
    return fwht_f64_backend(data, n, FWHT_BACKEND_AUTO);
}

fwht_status_t fwht_i8(int8_t* data, size_t n) {
    fwht_status_t status = validate_input(data, n);
    if (status != FWHT_SUCCESS) return status;
    
    fwht_butterfly_i8(data, n);
    return FWHT_SUCCESS;
}

/* ============================================================================
 * PUBLIC API - BACKEND CONTROL
 * ============================================================================ */

/* CUDA function declarations (when available) */
#ifdef USE_CUDA
extern fwht_status_t fwht_i32_cuda(int32_t* data, size_t n);
extern fwht_status_t fwht_f64_cuda(double* data, size_t n);
#endif

fwht_status_t fwht_i32_backend(int32_t* data, size_t n, fwht_backend_t backend) {
    fwht_status_t status = validate_input(data, n);
    if (status != FWHT_SUCCESS) return status;
    
    /* Select backend */
    if (backend == FWHT_BACKEND_AUTO) {
        backend = fwht_recommend_backend(n);
    }
    
    /* Execute on selected backend */
    switch (backend) {
        case FWHT_BACKEND_CPU:
            fwht_report_simd_mode();
            fwht_butterfly_i32(data, n);
            return FWHT_SUCCESS;
            
#ifdef USE_CUDA
        case FWHT_BACKEND_GPU:
            return fwht_i32_cuda(data, n);
#endif
            
        case FWHT_BACKEND_OPENMP:
#ifdef _OPENMP
            fwht_report_simd_mode();
            fwht_butterfly_i32_openmp(data, n);
            return FWHT_SUCCESS;
#else
            return FWHT_ERROR_BACKEND_UNAVAILABLE;
#endif
            
        default:
            return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }
}

fwht_status_t fwht_f64_backend(double* data, size_t n, fwht_backend_t backend) {
    fwht_status_t status = validate_input(data, n);
    if (status != FWHT_SUCCESS) return status;
    
    /* Select backend */
    if (backend == FWHT_BACKEND_AUTO) {
        backend = fwht_recommend_backend(n);
    }
    
    /* Execute on selected backend */
    switch (backend) {
        case FWHT_BACKEND_CPU:
            fwht_report_simd_mode();
            fwht_butterfly_f64(data, n);
            return FWHT_SUCCESS;
            
#ifdef USE_CUDA
        case FWHT_BACKEND_GPU:
            return fwht_f64_cuda(data, n);
#endif
            
        case FWHT_BACKEND_OPENMP:
#ifdef _OPENMP
            fwht_report_simd_mode();
            fwht_butterfly_f64_openmp(data, n);
            return FWHT_SUCCESS;
#else
            return FWHT_ERROR_BACKEND_UNAVAILABLE;
#endif
            
        default:
            return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }
}

/* ============================================================================
 * PUBLIC API - OUT-OF-PLACE TRANSFORMS
 * ============================================================================ */

int32_t* fwht_compute_i32(const int32_t* input, size_t n) {
    if (validate_input(input, n) != FWHT_SUCCESS) return NULL;
    
    int32_t* output = (int32_t*)malloc(n * sizeof(int32_t));
    if (output == NULL) return NULL;
    
    memcpy(output, input, n * sizeof(int32_t));
    fwht_butterfly_i32(output, n);
    
    return output;
}

double* fwht_compute_f64(const double* input, size_t n) {
    if (validate_input(input, n) != FWHT_SUCCESS) return NULL;
    
    double* output = (double*)malloc(n * sizeof(double));
    if (output == NULL) return NULL;
    
    memcpy(output, input, n * sizeof(double));
    fwht_butterfly_f64(output, n);
    
    return output;
}

int32_t* fwht_compute_i32_backend(const int32_t* input, size_t n, fwht_backend_t backend) {
    (void)backend;
    return fwht_compute_i32(input, n);
}

double* fwht_compute_f64_backend(const double* input, size_t n, fwht_backend_t backend) {
    (void)backend;
    return fwht_compute_f64(input, n);
}

/* ============================================================================
 * PUBLIC API - BOOLEAN FUNCTION CONVENIENCE
 * ============================================================================ */

fwht_status_t fwht_from_bool(const uint8_t* bool_func, int32_t* wht_out, 
                             size_t n, bool signed_rep) {
    fwht_status_t status = validate_input(bool_func, n);
    if (status != FWHT_SUCCESS) return status;
    if (wht_out == NULL) return FWHT_ERROR_NULL_POINTER;
    
    /* Convert boolean function to signed representation */
    if (signed_rep) {
        /* Cryptographic convention: 0 → +1, 1 → -1 */
        for (size_t i = 0; i < n; ++i) {
            wht_out[i] = (bool_func[i] == 0) ? 1 : -1;
        }
    } else {
        /* Use values as-is */
        for (size_t i = 0; i < n; ++i) {
            wht_out[i] = (int32_t)bool_func[i];
        }
    }
    
    /* Compute WHT */
    fwht_butterfly_i32(wht_out, n);
    
    return FWHT_SUCCESS;
}

fwht_status_t fwht_correlations(const uint8_t* bool_func, double* corr_out, size_t n) {
    fwht_status_t status = validate_input(bool_func, n);
    if (status != FWHT_SUCCESS) return status;
    if (corr_out == NULL) return FWHT_ERROR_NULL_POINTER;
    
    /* Convert to signed and compute WHT */
    int32_t* wht = (int32_t*)malloc(n * sizeof(int32_t));
    if (wht == NULL) return FWHT_ERROR_OUT_OF_MEMORY;
    
    status = fwht_from_bool(bool_func, wht, n, true);
    if (status != FWHT_SUCCESS) {
        free(wht);
        return status;
    }
    
    /* Convert WHT to correlations: Cor(f, u) = WHT[u] / n */
    double n_inv = 1.0 / (double)n;
    for (size_t i = 0; i < n; ++i) {
        corr_out[i] = (double)wht[i] * n_inv;
    }
    
    free(wht);
    return FWHT_SUCCESS;
}

/* ============================================================================
 * CONTEXT API - STUB IMPLEMENTATION
 * 
 * Full implementation will be in fwht_context.c
 * For now, provide minimal stubs.
 * ============================================================================ */

struct fwht_context {
    fwht_config_t config;
};

fwht_config_t fwht_default_config(void) {
    fwht_config_t config;
    config.backend = FWHT_BACKEND_AUTO;
    config.num_threads = 0;  /* Auto-detect */
    config.gpu_device = 0;
    config.normalize = false;
    return config;
}

fwht_context_t* fwht_create_context(const fwht_config_t* config) {
    fwht_context_t* ctx = (fwht_context_t*)malloc(sizeof(fwht_context_t));
    if (ctx == NULL) return NULL;
    
    if (config != NULL) {
        ctx->config = *config;
    } else {
        ctx->config = fwht_default_config();
    }
    
    return ctx;
}

void fwht_destroy_context(fwht_context_t* ctx) {
    if (ctx != NULL) {
        free(ctx);
    }
}

fwht_status_t fwht_transform_i32(fwht_context_t* ctx, int32_t* data, size_t n) {
    if (ctx == NULL) {
        return fwht_i32(data, n);
    }
    return fwht_i32_backend(data, n, ctx->config.backend);
}

fwht_status_t fwht_transform_f64(fwht_context_t* ctx, double* data, size_t n) {
    if (ctx == NULL) {
        return fwht_f64(data, n);
    }
    return fwht_f64_backend(data, n, ctx->config.backend);
}

fwht_status_t fwht_batch_i32(fwht_context_t* ctx, int32_t** data_array, 
                             size_t n, int batch_size) {
    if (data_array == NULL) return FWHT_ERROR_NULL_POINTER;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    
    /* Simple sequential implementation for now */
    /* GPU version will parallelize this */
    fwht_backend_t backend = (ctx != NULL) ? ctx->config.backend : FWHT_BACKEND_AUTO;
    
    for (int i = 0; i < batch_size; ++i) {
        fwht_status_t status = fwht_i32_backend(data_array[i], n, backend);
        if (status != FWHT_SUCCESS) return status;
    }
    
    return FWHT_SUCCESS;
}

fwht_status_t fwht_batch_f64(fwht_context_t* ctx, double** data_array,
                             size_t n, int batch_size) {
    if (data_array == NULL) return FWHT_ERROR_NULL_POINTER;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    
    fwht_backend_t backend = (ctx != NULL) ? ctx->config.backend : FWHT_BACKEND_AUTO;
    
    for (int i = 0; i < batch_size; ++i) {
        fwht_status_t status = fwht_f64_backend(data_array[i], n, backend);
        if (status != FWHT_SUCCESS) return status;
    }
    
    return FWHT_SUCCESS;
}
