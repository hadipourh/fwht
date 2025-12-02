#include <stddef.h>
#include <stdint.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>
#include "fwht_internal.h"

/* Inline scalar fallback for small sizes where NEON overhead isn't worth it */
static void fwht_butterfly_i32_scalar(int32_t* data, size_t n) {
    if (n <= 1) return;
    for (size_t len = 1; len < n; len <<= 1) {
        size_t step = len << 1;
        for (size_t i = 0; i < n; i += step) {
            for (size_t j = 0; j < len; ++j) {
                int32_t a = data[i + j];
                int32_t b = data[i + j + len];
                data[i + j]         = a + b;
                data[i + j + len]   = a - b;
            }
        }
    }
}

void fwht_butterfly_i32_neon(int32_t* data, size_t n) {
    if (n <= 1) return;

    /* For small sizes, use the scalar fallback directly */
    if (n < 4096) {
        fwht_butterfly_i32_scalar(data, n);
        return;
    }

    for (size_t len = 1; len < n; len <<= 1) {
        size_t step = len << 1; /* 2 * len */

        for (size_t i = 0; i < n; i += step) {
            size_t j = 0;

            /* Main NEON core: process 8 elements (2 vectors) per iteration */
            for (; j + 8 <= len; j += 8) {
                int32_t* a0 = data + i + j;
                int32_t* b0 = a0 + len;
                int32_t* a1 = a0 + 4;
                int32_t* b1 = b0 + 4;

                int32x4_t va0 = vld1q_s32(a0);
                int32x4_t vb0 = vld1q_s32(b0);
                int32x4_t va1 = vld1q_s32(a1);
                int32x4_t vb1 = vld1q_s32(b1);

                int32x4_t sum0  = vaddq_s32(va0, vb0);
                int32x4_t diff0 = vsubq_s32(va0, vb0);
                int32x4_t sum1  = vaddq_s32(va1, vb1);
                int32x4_t diff1 = vsubq_s32(va1, vb1);

                vst1q_s32(a0, sum0);
                vst1q_s32(b0, diff0);
                vst1q_s32(a1, sum1);
                vst1q_s32(b1, diff1);
            }

            /* 4-wide NEON tail */
            for (; j + 4 <= len; j += 4) {
                int32_t* a = data + i + j;
                int32_t* b = a + len;

                int32x4_t va = vld1q_s32(a);
                int32x4_t vb = vld1q_s32(b);

                int32x4_t sum  = vaddq_s32(va, vb);
                int32x4_t diff = vsubq_s32(va, vb);

                vst1q_s32(a, sum);
                vst1q_s32(b, diff);
            }

            /* Scalar tail for this block */
            for (; j < len; ++j) {
                int32_t a = data[i + j];
                int32_t b = data[i + j + len];
                data[i + j]         = a + b;
                data[i + j + len]   = a - b;
            }
        }
    }
}

#endif /* defined(__ARM_NEON) || defined(__ARM_NEON__) */
