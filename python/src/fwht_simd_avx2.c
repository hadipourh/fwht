#include <stddef.h>
#include <stdint.h>

#if defined(__x86_64__) && defined(__AVX2__)

#include <immintrin.h>
#include "fwht_internal.h"

/*
 * AVX2 implementation of in-place FWHT for int32_t.
 *
 * Optimized with aggressive unrolling and software pipelining.
 * FFHT uses hand-written assembly - we get as close as possible with intrinsics.
 */

void fwht_butterfly_i32_avx2(int32_t* data, size_t n) {
    if (n <= 1) return;

    for (size_t len = 1; len < n; len <<= 1) {
        size_t step = len << 1;
        
        for (size_t i = 0; i < n; i += step) {
            size_t j = 0;
            
            /* Aggressive 4-way unroll with software pipelining */
            for (; j + 32 <= len; j += 32) {
                int32_t* a0 = data + i + j;
                int32_t* b0 = a0 + len;
                int32_t* a1 = a0 + 8;
                int32_t* b1 = b0 + 8;
                int32_t* a2 = a0 + 16;
                int32_t* b2 = b0 + 16;
                int32_t* a3 = a0 + 24;
                int32_t* b3 = b0 + 24;
                
                /* Load all 8 vectors */
                __m256i va0 = _mm256_loadu_si256((const __m256i*)a0);
                __m256i vb0 = _mm256_loadu_si256((const __m256i*)b0);
                __m256i va1 = _mm256_loadu_si256((const __m256i*)a1);
                __m256i vb1 = _mm256_loadu_si256((const __m256i*)b1);
                __m256i va2 = _mm256_loadu_si256((const __m256i*)a2);
                __m256i vb2 = _mm256_loadu_si256((const __m256i*)b2);
                __m256i va3 = _mm256_loadu_si256((const __m256i*)a3);
                __m256i vb3 = _mm256_loadu_si256((const __m256i*)b3);
                
                /* Compute all butterflies */
                __m256i sum0  = _mm256_add_epi32(va0, vb0);
                __m256i diff0 = _mm256_sub_epi32(va0, vb0);
                __m256i sum1  = _mm256_add_epi32(va1, vb1);
                __m256i diff1 = _mm256_sub_epi32(va1, vb1);
                __m256i sum2  = _mm256_add_epi32(va2, vb2);
                __m256i diff2 = _mm256_sub_epi32(va2, vb2);
                __m256i sum3  = _mm256_add_epi32(va3, vb3);
                __m256i diff3 = _mm256_sub_epi32(va3, vb3);
                
                /* Store all results */
                _mm256_storeu_si256((__m256i*)a0, sum0);
                _mm256_storeu_si256((__m256i*)b0, diff0);
                _mm256_storeu_si256((__m256i*)a1, sum1);
                _mm256_storeu_si256((__m256i*)b1, diff1);
                _mm256_storeu_si256((__m256i*)a2, sum2);
                _mm256_storeu_si256((__m256i*)b2, diff2);
                _mm256_storeu_si256((__m256i*)a3, sum3);
                _mm256_storeu_si256((__m256i*)b3, diff3);
            }
            
            /* 8-element cleanup */
            for (; j + 8 <= len; j += 8) {
                int32_t* a_ptr = data + i + j;
                int32_t* b_ptr = a_ptr + len;
                
                __m256i a = _mm256_loadu_si256((const __m256i*)a_ptr);
                __m256i b = _mm256_loadu_si256((const __m256i*)b_ptr);
                
                __m256i sum  = _mm256_add_epi32(a, b);
                __m256i diff = _mm256_sub_epi32(a, b);
                
                _mm256_storeu_si256((__m256i*)a_ptr, sum);
                _mm256_storeu_si256((__m256i*)b_ptr, diff);
            }
            
            /* Scalar tail */
            for (; j < len; ++j) {
                int32_t a = data[i + j];
                int32_t b = data[i + j + len];
                data[i + j] = a + b;
                data[i + j + len] = a - b;
            }
        }
    }
}

#endif /* defined(__x86_64__) && defined(__AVX2__) */
