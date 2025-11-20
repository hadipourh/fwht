/*
 * Fast Walsh-Hadamard Transform - Vectorized Batch Implementation
 *
 * Process multiple small transforms simultaneously using SIMD instructions.
 * Ideal for cryptanalysis: analyze thousands of S-boxes in parallel.
 *
 * Copyright (C) 2025 Hosein Hadipour
 *
 * Author: Hosein Hadipour <hsn.hadipour@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "../include/fwht.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Portable aligned allocation for batch operations */
static void* fwht_batch_aligned_alloc(size_t alignment, size_t size) {
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    return aligned_alloc(alignment, size);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return NULL;
#else
    /* Fallback: just use malloc (alignment not guaranteed but safe to free) */
    (void)alignment;  /* Suppress unused parameter warning */
    return malloc(size);
#endif
}

/* Restrict keyword for optimization */
#if defined(__GNUC__) || defined(__clang__)
#define FWHT_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define FWHT_RESTRICT __restrict
#else
#define FWHT_RESTRICT
#endif
#include <stdint.h>

#if defined(__AVX2__)
#include <immintrin.h>
#define FWHT_BATCH_HAVE_AVX2 1
#define FWHT_SIMD_WIDTH_I32 8
#define FWHT_SIMD_WIDTH_F64 4
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FWHT_BATCH_HAVE_NEON 1
#define FWHT_SIMD_WIDTH_I32 4
#define FWHT_SIMD_WIDTH_F64 2
#else
/* Scalar fallback */
#define FWHT_SIMD_WIDTH_I32 1
#define FWHT_SIMD_WIDTH_F64 1
#endif

/* ============================================================================
 * AVX2 IMPLEMENTATION (x86-64)
 * ============================================================================ */

#if defined(FWHT_BATCH_HAVE_AVX2)

/*
 * Process 8 independent size-n transforms simultaneously using AVX2.
 * Each __m256i register holds one element from 8 different transforms.
 * 
 * Memory layout (transposed):
 *   lane[0] = [data0[0], data1[0], data2[0], ..., data7[0]]
 *   lane[1] = [data0[1], data1[1], data2[1], ..., data7[1]]
 *   ...
 * 
 * This allows vectorized butterfly operations across 8 transforms at once.
 */
static void fwht_i32_batch_avx2_kernel(int32_t** FWHT_RESTRICT data_array, 
                                        size_t n, size_t batch_offset) {
    const size_t simd_width = 8;
    
    /* Allocate transposed buffer for SIMD processing */
    __m256i* lanes = (__m256i*)fwht_batch_aligned_alloc(32, n * sizeof(__m256i));
    if (!lanes) return;  /* Fallback to scalar if allocation fails */
    
    /* Transpose: gather elements from 8 arrays into SIMD lanes */
    for (size_t i = 0; i < n; i++) {
        int32_t tmp[8];
        for (size_t j = 0; j < simd_width; j++) {
            tmp[j] = data_array[batch_offset + j][i];
        }
        lanes[i] = _mm256_loadu_si256((__m256i*)tmp);
    }
    
    /* Butterfly passes - vectorized across 8 transforms */
    for (size_t h = 1; h < n; h <<= 1) {
        size_t stride = h << 1;
        
        for (size_t i = 0; i < n; i += stride) {
            for (size_t j = 0; j < h; j++) {
                size_t even_idx = i + j;
                size_t odd_idx = even_idx + h;
                
                __m256i a = lanes[even_idx];
                __m256i b = lanes[odd_idx];
                
                /* Vectorized butterfly: 8 transforms in parallel */
                lanes[even_idx] = _mm256_add_epi32(a, b);
                lanes[odd_idx] = _mm256_sub_epi32(a, b);
            }
        }
    }
    
    /* Transpose back: scatter SIMD lanes to 8 arrays */
    for (size_t i = 0; i < n; i++) {
        int32_t tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, lanes[i]);
        for (size_t j = 0; j < simd_width; j++) {
            data_array[batch_offset + j][i] = tmp[j];
        }
    }
    
    free(lanes);
}

static void fwht_f64_batch_avx2_kernel(double** FWHT_RESTRICT data_array,
                                        size_t n, size_t batch_offset) {
    const size_t simd_width = 4;  /* AVX2: 4 doubles per register */
    
    __m256d* lanes = (__m256d*)fwht_batch_aligned_alloc(32, n * sizeof(__m256d));
    if (!lanes) return;
    
    /* Transpose */
    for (size_t i = 0; i < n; i++) {
        double tmp[4];
        for (size_t j = 0; j < simd_width; j++) {
            tmp[j] = data_array[batch_offset + j][i];
        }
        lanes[i] = _mm256_loadu_pd(tmp);
    }
    
    /* Butterfly passes */
    for (size_t h = 1; h < n; h <<= 1) {
        size_t stride = h << 1;
        
        for (size_t i = 0; i < n; i += stride) {
            for (size_t j = 0; j < h; j++) {
                size_t even_idx = i + j;
                size_t odd_idx = even_idx + h;
                
                __m256d a = lanes[even_idx];
                __m256d b = lanes[odd_idx];
                
                lanes[even_idx] = _mm256_add_pd(a, b);
                lanes[odd_idx] = _mm256_sub_pd(a, b);
            }
        }
    }
    
    /* Transpose back */
    for (size_t i = 0; i < n; i++) {
        double tmp[4];
        _mm256_storeu_pd(tmp, lanes[i]);
        for (size_t j = 0; j < simd_width; j++) {
            data_array[batch_offset + j][i] = tmp[j];
        }
    }
    
    free(lanes);
}

#endif /* FWHT_BATCH_HAVE_AVX2 */

/* ============================================================================
 * NEON IMPLEMENTATION (ARM)
 * ============================================================================ */

#if defined(FWHT_BATCH_HAVE_NEON)

static void fwht_i32_batch_neon_kernel(int32_t** FWHT_RESTRICT data_array,
                                        size_t n, size_t batch_offset) {
    const size_t simd_width = 4;  /* NEON: 4 int32s per register */
    
    int32x4_t* lanes = (int32x4_t*)fwht_batch_aligned_alloc(16, n * sizeof(int32x4_t));
    if (!lanes) return;
    
    /* Transpose */
    for (size_t i = 0; i < n; i++) {
        int32_t tmp[4];
        for (size_t j = 0; j < simd_width; j++) {
            tmp[j] = data_array[batch_offset + j][i];
        }
        lanes[i] = vld1q_s32(tmp);
    }
    
    /* Butterfly passes */
    for (size_t h = 1; h < n; h <<= 1) {
        size_t stride = h << 1;
        
        for (size_t i = 0; i < n; i += stride) {
            for (size_t j = 0; j < h; j++) {
                size_t even_idx = i + j;
                size_t odd_idx = even_idx + h;
                
                int32x4_t a = lanes[even_idx];
                int32x4_t b = lanes[odd_idx];
                
                lanes[even_idx] = vaddq_s32(a, b);
                lanes[odd_idx] = vsubq_s32(a, b);
            }
        }
    }
    
    /* Transpose back */
    for (size_t i = 0; i < n; i++) {
        int32_t tmp[4];
        vst1q_s32(tmp, lanes[i]);
        for (size_t j = 0; j < simd_width; j++) {
            data_array[batch_offset + j][i] = tmp[j];
        }
    }
    
    free(lanes);
}

static void fwht_f64_batch_neon_kernel(double** FWHT_RESTRICT data_array,
                                        size_t n, size_t batch_offset) {
    const size_t simd_width = 2;  /* NEON: 2 doubles per register */
    
    float64x2_t* lanes = (float64x2_t*)fwht_batch_aligned_alloc(16, n * sizeof(float64x2_t));
    if (!lanes) return;
    
    /* Transpose */
    for (size_t i = 0; i < n; i++) {
        double tmp[2];
        for (size_t j = 0; j < simd_width; j++) {
            tmp[j] = data_array[batch_offset + j][i];
        }
        lanes[i] = vld1q_f64(tmp);
    }
    
    /* Butterfly passes */
    for (size_t h = 1; h < n; h <<= 1) {
        size_t stride = h << 1;
        
        for (size_t i = 0; i < n; i += stride) {
            for (size_t j = 0; j < h; j++) {
                size_t even_idx = i + j;
                size_t odd_idx = even_idx + h;
                
                float64x2_t a = lanes[even_idx];
                float64x2_t b = lanes[odd_idx];
                
                lanes[even_idx] = vaddq_f64(a, b);
                lanes[odd_idx] = vsubq_f64(a, b);
            }
        }
    }
    
    /* Transpose back */
    for (size_t i = 0; i < n; i++) {
        double tmp[2];
        vst1q_f64(tmp, lanes[i]);
        for (size_t j = 0; j < simd_width; j++) {
            data_array[batch_offset + j][i] = tmp[j];
        }
    }
    
    free(lanes);
}

#endif /* FWHT_BATCH_HAVE_NEON */

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

fwht_status_t fwht_i32_batch(int32_t** data_array, size_t n, size_t batch_size) {
    /* Validate inputs */
    if (data_array == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    
    /* For large n or small batch, use context-based batch API which handles GPU properly */
    /* Note: n >= 256 routes to context API to allow GPU for n=256 if recommended */
    if (n >= 256 || batch_size < FWHT_SIMD_WIDTH_I32) {
        /* Use context API which properly handles GPU batches */
        return fwht_batch_i32(NULL, data_array, n, (int)batch_size);
    }
    
    /* Process in SIMD-width chunks */
#if defined(FWHT_BATCH_HAVE_AVX2)
    const size_t simd_width = 8;
    size_t num_simd_chunks = batch_size / simd_width;
    
    for (size_t i = 0; i < num_simd_chunks; i++) {
        fwht_i32_batch_avx2_kernel(data_array, n, i * simd_width);
    }
    
    /* Process remainder serially */
    for (size_t i = num_simd_chunks * simd_width; i < batch_size; i++) {
        fwht_status_t status = fwht_i32(data_array[i], n);
        if (status != FWHT_SUCCESS) return status;
    }
    
#elif defined(FWHT_BATCH_HAVE_NEON)
    const size_t simd_width = 4;
    size_t num_simd_chunks = batch_size / simd_width;
    
    for (size_t i = 0; i < num_simd_chunks; i++) {
        fwht_i32_batch_neon_kernel(data_array, n, i * simd_width);
    }
    
    /* Process remainder serially */
    for (size_t i = num_simd_chunks * simd_width; i < batch_size; i++) {
        fwht_status_t status = fwht_i32(data_array[i], n);
        if (status != FWHT_SUCCESS) return status;
    }
    
#else
    /* Scalar fallback */
    for (size_t i = 0; i < batch_size; i++) {
        fwht_status_t status = fwht_i32(data_array[i], n);
        if (status != FWHT_SUCCESS) return status;
    }
#endif
    
    return FWHT_SUCCESS;
}

fwht_status_t fwht_f64_batch(double** data_array, size_t n, size_t batch_size) {
    /* Validate inputs */
    if (data_array == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    
    /* For large n or small batch, use context-based batch API which handles GPU properly */
    /* Note: n >= 256 routes to context API to allow GPU for n=256 if recommended */
    if (n >= 256 || batch_size < FWHT_SIMD_WIDTH_F64) {
        /* Use context API which properly handles GPU batches */
        return fwht_batch_f64(NULL, data_array, n, (int)batch_size);
    }
    
    /* Process in SIMD-width chunks */
#if defined(FWHT_BATCH_HAVE_AVX2)
    const size_t simd_width = 4;
    size_t num_simd_chunks = batch_size / simd_width;
    
    for (size_t i = 0; i < num_simd_chunks; i++) {
        fwht_f64_batch_avx2_kernel(data_array, n, i * simd_width);
    }
    
    for (size_t i = num_simd_chunks * simd_width; i < batch_size; i++) {
        fwht_status_t status = fwht_f64(data_array[i], n);
        if (status != FWHT_SUCCESS) return status;
    }
    
#elif defined(FWHT_BATCH_HAVE_NEON)
    const size_t simd_width = 2;
    size_t num_simd_chunks = batch_size / simd_width;
    
    for (size_t i = 0; i < num_simd_chunks; i++) {
        fwht_f64_batch_neon_kernel(data_array, n, i * simd_width);
    }
    
    for (size_t i = num_simd_chunks * simd_width; i < batch_size; i++) {
        fwht_status_t status = fwht_f64(data_array[i], n);
        if (status != FWHT_SUCCESS) return status;
    }
    
#else
    for (size_t i = 0; i < batch_size; i++) {
        fwht_status_t status = fwht_f64(data_array[i], n);
        if (status != FWHT_SUCCESS) return status;
    }
#endif
    
    return FWHT_SUCCESS;
}
