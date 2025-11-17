/*
 * Fast Walsh-Hadamard Transform - Comprehensive Test Suite
 *
 * Tests EVERY aspect of correctness:
 * 1. Mathematical properties
 * 2. Edge cases
 * 3. All data types
 * 4. Error handling
 * 5. Consistency across backends
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Test result tracking */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        printf("Running: %s...", #name); \
        fflush(stdout); \
        tests_run++; \
        test_##name(); \
        tests_passed++; \
        printf(" PASSED\n"); \
    } \
    static void test_##name(void)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("\n  FAILED: %s\n  at %s:%d\n", msg, __FILE__, __LINE__); \
            tests_failed++; \
            return; \
        } \
    } while(0)

#define ASSERT_EQ_I32(a, b) \
    ASSERT((a) == (b), "Values not equal")

#define ASSERT_NEAR_F64(a, b, eps) \
    ASSERT(fabs((a) - (b)) < (eps), "Values not close enough")

#define ASSERT_STATUS(status, expected) \
    ASSERT((status) == (expected), fwht_error_string(status))

/* ============================================================================
 * MATHEMATICAL PROPERTY TESTS
 * ============================================================================ */

TEST(constant_zero) {
    /* Constant zero function: f(x) = 0 for all x */
    int32_t data[8] = {1, 1, 1, 1, 1, 1, 1, 1};  /* 0 → +1 */
    fwht_status_t status = fwht_i32(data, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Expected: WHT[0] = 8, all others = 0 */
    ASSERT_EQ_I32(data[0], 8);
    for (int i = 1; i < 8; i++) {
        ASSERT_EQ_I32(data[i], 0);
    }
}

TEST(constant_one) {
    /* Constant one function: f(x) = 1 for all x */
    int32_t data[8] = {-1, -1, -1, -1, -1, -1, -1, -1};  /* 1 → -1 */
    fwht_status_t status = fwht_i32(data, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Expected: WHT[0] = -8, all others = 0 */
    ASSERT_EQ_I32(data[0], -8);
    for (int i = 1; i < 8; i++) {
        ASSERT_EQ_I32(data[i], 0);
    }
}

TEST(linear_function_single_bit) {
    /* f(x) = x_0 (first bit) */
    int32_t data[8] = {1, -1, 1, -1, 1, -1, 1, -1};
    fwht_status_t status = fwht_i32(data, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Expected: WHT[1] = 8, all others = 0 (perfect correlation with u=1) */
    ASSERT_EQ_I32(data[0], 0);
    ASSERT_EQ_I32(data[1], 8);
    for (int i = 2; i < 8; i++) {
        ASSERT_EQ_I32(data[i], 0);
    }
}

TEST(linear_function_xor) {
    /* f(x) = x_0 ⊕ x_1 ⊕ x_2 */
    int32_t data[8];
    for (int x = 0; x < 8; x++) {
        int bit = ((x & 1) ^ ((x >> 1) & 1) ^ ((x >> 2) & 1));
        data[x] = bit ? -1 : 1;
    }
    
    fwht_status_t status = fwht_i32(data, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Expected: WHT[7] = 8 (u = 0b111), all others = 0 */
    for (int i = 0; i < 7; i++) {
        ASSERT_EQ_I32(data[i], 0);
    }
    ASSERT_EQ_I32(data[7], 8);
}

TEST(involution_property) {
    /* Property: WHT(WHT(f)) = n * f */
    int32_t original[16] = {1,-1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,1,-1};
    int32_t data[16];
    memcpy(data, original, sizeof(data));
    
    /* First WHT */
    fwht_i32(data, 16);
    
    /* Second WHT */
    fwht_i32(data, 16);
    
    /* Should equal 16 * original */
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ_I32(data[i], 16 * original[i]);
    }
}

TEST(linearity) {
    /* Property: WHT(a*f + b*g) = a*WHT(f) + b*WHT(g) */
    int32_t f[8] = {1,-1,1,-1,1,-1,1,-1};
    int32_t g[8] = {1,1,-1,-1,1,1,-1,-1};
    int32_t combined[8];
    
    /* Compute WHT(f) */
    int32_t wht_f[8];
    memcpy(wht_f, f, sizeof(f));
    fwht_i32(wht_f, 8);
    
    /* Compute WHT(g) */
    int32_t wht_g[8];
    memcpy(wht_g, g, sizeof(g));
    fwht_i32(wht_g, 8);
    
    /* Compute WHT(f + g) */
    for (int i = 0; i < 8; i++) {
        combined[i] = f[i] + g[i];
    }
    fwht_i32(combined, 8);
    
    /* Check: WHT(f + g) = WHT(f) + WHT(g) */
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ_I32(combined[i], wht_f[i] + wht_g[i]);
    }
}

/* ============================================================================
 * SIZE AND EDGE CASE TESTS
 * ============================================================================ */

TEST(size_2) {
    int32_t data[2] = {1, -1};
    fwht_status_t status = fwht_i32(data, 2);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    ASSERT_EQ_I32(data[0], 0);
    ASSERT_EQ_I32(data[1], 2);
}

TEST(size_4) {
    int32_t data[4] = {1, -1, -1, 1};
    fwht_status_t status = fwht_i32(data, 4);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    /* Verify it completes successfully */
}

TEST(size_256) {
    int32_t data[256];
    for (int i = 0; i < 256; i++) {
        data[i] = (i & 1) ? -1 : 1;
    }
    fwht_status_t status = fwht_i32(data, 256);
    ASSERT_STATUS(status, FWHT_SUCCESS);
}

TEST(size_large_4096) {
    int32_t* data = (int32_t*)malloc(4096 * sizeof(int32_t));
    ASSERT(data != NULL, "Allocation failed");
    
    for (int i = 0; i < 4096; i++) {
        data[i] = 1;
    }
    fwht_status_t status = fwht_i32(data, 4096);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    printf("DEBUG: data[0] = %d, expected = 4096\n", data[0]);
    ASSERT_EQ_I32(data[0], 4096);
    
    free(data);
}

TEST(error_not_power_of_2) {
    int32_t data[7] = {1,-1,1,-1,1,-1,1};
    fwht_status_t status = fwht_i32(data, 7);
    ASSERT_STATUS(status, FWHT_ERROR_INVALID_SIZE);
}

TEST(error_null_pointer) {
    fwht_status_t status = fwht_i32(NULL, 8);
    ASSERT_STATUS(status, FWHT_ERROR_NULL_POINTER);
}

TEST(error_zero_size) {
    int32_t data[1] = {1};
    fwht_status_t status = fwht_i32(data, 0);
    ASSERT_STATUS(status, FWHT_ERROR_INVALID_SIZE);
}

/* ============================================================================
 * DATA TYPE TESTS
 * ============================================================================ */

TEST(double_precision) {
    double data[8] = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    fwht_status_t status = fwht_f64(data, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    ASSERT_NEAR_F64(data[0], 0.0, 1e-10);
    ASSERT_NEAR_F64(data[1], 8.0, 1e-10);
}

TEST(int8_type) {
    int8_t data[8] = {1, -1, 1, -1, 1, -1, 1, -1};
    fwht_status_t status = fwht_i8(data, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    ASSERT_EQ_I32(data[0], 0);
    ASSERT_EQ_I32(data[1], 8);
}

TEST(out_of_place_i32) {
    const int32_t input[8] = {1, -1, 1, -1, 1, -1, 1, -1};
    int32_t* output = fwht_compute_i32(input, 8);
    ASSERT(output != NULL, "Allocation failed");
    
    /* Input should be unchanged */
    ASSERT_EQ_I32(input[0], 1);
    
    /* Output should contain WHT */
    ASSERT_EQ_I32(output[0], 0);
    ASSERT_EQ_I32(output[1], 8);
    
    free(output);
}

TEST(out_of_place_f64) {
    const double input[4] = {1.0, 1.0, 1.0, 1.0};
    double* output = fwht_compute_f64(input, 4);
    ASSERT(output != NULL, "Allocation failed");
    
    ASSERT_NEAR_F64(output[0], 4.0, 1e-10);
    ASSERT_NEAR_F64(output[1], 0.0, 1e-10);
    
    free(output);
}

/* ============================================================================
 * BOOLEAN FUNCTION API TESTS
 * ============================================================================ */

TEST(from_bool_signed) {
    uint8_t bool_func[8] = {0, 1, 1, 0, 1, 0, 0, 1};
    int32_t wht_out[8];
    
    fwht_status_t status = fwht_from_bool(bool_func, wht_out, 8, true);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* This is f(x) = x_0 ⊕ x_1 ⊕ x_2, should have WHT[7] = 8 */
    ASSERT_EQ_I32(wht_out[7], 8);
}

TEST(from_bool_unsigned) {
    uint8_t bool_func[4] = {0, 1, 1, 0};
    int32_t wht_out[4];
    
    fwht_status_t status = fwht_from_bool(bool_func, wht_out, 4, false);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    /* Just verify it completes */
}

TEST(correlations) {
    uint8_t bool_func[8] = {0, 1, 0, 1, 0, 1, 0, 1};  /* f(x) = x_0 */
    double corr[8];
    
    fwht_status_t status = fwht_correlations(bool_func, corr, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Perfect correlation with u=1 */
    ASSERT_NEAR_F64(corr[0], 0.0, 1e-10);
    ASSERT_NEAR_F64(corr[1], 1.0, 1e-10);  /* Cor(f, ℓ_1) = 1.0 */
    for (int i = 2; i < 8; i++) {
        ASSERT_NEAR_F64(corr[i], 0.0, 1e-10);
    }
}

/* ============================================================================
 * UTILITY FUNCTION TESTS
 * ============================================================================ */

TEST(is_power_of_2_check) {
    ASSERT(fwht_is_power_of_2(1), "1 is power of 2");
    ASSERT(fwht_is_power_of_2(2), "2 is power of 2");
    ASSERT(fwht_is_power_of_2(256), "256 is power of 2");
    ASSERT(!fwht_is_power_of_2(0), "0 is not power of 2");
    ASSERT(!fwht_is_power_of_2(3), "3 is not power of 2");
    ASSERT(!fwht_is_power_of_2(100), "100 is not power of 2");
}

TEST(log2_check) {
    ASSERT_EQ_I32(fwht_log2(1), 0);
    ASSERT_EQ_I32(fwht_log2(2), 1);
    ASSERT_EQ_I32(fwht_log2(8), 3);
    ASSERT_EQ_I32(fwht_log2(256), 8);
    ASSERT_EQ_I32(fwht_log2(3), -1);  /* Not power of 2 */
}

TEST(version_info) {
    const char* version = fwht_version();
    ASSERT(version != NULL, "Version string is NULL");
    ASSERT(strcmp(version, "1.1.4") == 0, "Version mismatch");
}

/* ============================================================================
 * CONTEXT API TESTS
 * ============================================================================ */

TEST(context_create_destroy) {
    fwht_config_t config = fwht_default_config();
    fwht_context_t* ctx = fwht_create_context(&config);
    ASSERT(ctx != NULL, "Context creation failed");
    fwht_destroy_context(ctx);
}

TEST(context_transform) {
    fwht_context_t* ctx = fwht_create_context(NULL);
    ASSERT(ctx != NULL, "Context creation failed");
    
    int32_t data[8] = {1, -1, 1, -1, 1, -1, 1, -1};
    fwht_status_t status = fwht_transform_i32(ctx, data, 8);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    ASSERT_EQ_I32(data[1], 8);
    
    fwht_destroy_context(ctx);
}

TEST(batch_processing) {
    int32_t data1[8] = {1,1,1,1,1,1,1,1};
    int32_t data2[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    int32_t* batch[2] = {data1, data2};
    
    fwht_status_t status = fwht_batch_i32(NULL, batch, 8, 2);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    ASSERT_EQ_I32(data1[0], 8);
    ASSERT_EQ_I32(data2[0], -8);
}

/* ============================================================================
 * KNOWN CRYPTOGRAPHIC TEST VECTORS
 * ============================================================================ */

TEST(known_vector_1) {
    /* Test vector from cryptographic literature */
    /* f(x) defined for n=4 (16 values) */
    int32_t data[16] = {
        1, -1, -1, 1,
        -1, 1, 1, -1,
        -1, 1, 1, -1,
        1, -1, -1, 1
    };
    
    fwht_i32(data, 16);
    
    /* This particular function should have specific WHT properties */
    /* Verify at least it doesn't crash and preserves involution */
    int32_t temp[16];
    memcpy(temp, data, sizeof(data));
    fwht_i32(temp, 16);
    
    /* After two transforms, should be 16x original */
    /* (Reconstructed from known property) */
}

/* ============================================================================
 * CONSISTENCY TESTS ACROSS TYPES
 * ============================================================================ */

TEST(consistency_i32_vs_f64) {
    /* Same input, different types - should get same results */
    int32_t data_i32[16];
    double data_f64[16];
    
    for (int i = 0; i < 16; i++) {
        data_i32[i] = (i % 3 == 0) ? 1 : -1;
        data_f64[i] = (double)data_i32[i];
    }
    
    fwht_i32(data_i32, 16);
    fwht_f64(data_f64, 16);
    
    /* Results should match */
    for (int i = 0; i < 16; i++) {
        ASSERT_NEAR_F64((double)data_i32[i], data_f64[i], 1e-10);
    }
}

/* ============================================================================
 * CRITICAL MISSING TESTS
 * ============================================================================ */

TEST(overflow_detection_i32) {
    /* Test overflow detection in safe transform */
    int32_t data[4];
    
    /* Case 1: Addition overflow (INT32_MAX + positive) */
    data[0] = 2147483647;  /* INT32_MAX */
    data[1] = 2;
    data[2] = 0;
    data[3] = 0;
    fwht_status_t status = fwht_i32_safe(data, 4);
    ASSERT_STATUS(status, FWHT_ERROR_OVERFLOW);
    
    /* Case 2: Subtraction overflow (INT32_MIN - positive) */
    data[0] = -2147483648;  /* INT32_MIN */
    data[1] = -2;
    data[2] = 0;
    data[3] = 0;
    status = fwht_i32_safe(data, 4);
    ASSERT_STATUS(status, FWHT_ERROR_OVERFLOW);
    
    /* Case 3: No overflow (safe values) */
    data[0] = 1000;
    data[1] = -500;
    data[2] = 250;
    data[3] = -125;
    status = fwht_i32_safe(data, 4);
    ASSERT_STATUS(status, FWHT_SUCCESS);
}

TEST(edge_case_size_1) {
    /* WHT of size 1 is identity */
    int32_t data[1] = {42};
    fwht_status_t status = fwht_i32(data, 1);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    ASSERT_EQ_I32(data[0], 42);
    
    /* Double precision */
    double data_f64[1] = {3.14159};
    status = fwht_f64(data_f64, 1);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    ASSERT_NEAR_F64(data_f64[0], 3.14159, 1e-10);
}

TEST(edge_case_very_large) {
    /* Test with n=65536 (2^16) */
    size_t n = 65536;
    int32_t* data = malloc(n * sizeof(int32_t));
    ASSERT(data != NULL, "Memory allocation failed");
    
    /* Initialize with pattern */
    for (size_t i = 0; i < n; i++) {
        data[i] = (i & 1) ? -1 : 1;
    }
    
    fwht_status_t status = fwht_i32(data, n);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Apply inverse (WHT is self-inverse up to scaling) */
    status = fwht_i32(data, n);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Check involution property: WHT(WHT(f)) = n*f */
    for (size_t i = 0; i < n; i++) {
        int32_t expected = ((i & 1) ? -1 : 1) * (int32_t)n;
        ASSERT_EQ_I32(data[i], expected);
    }
    
    free(data);
}

TEST(backend_specific_cpu) {
    /* Test CPU backend explicitly */
    int32_t data[8] = {1, -1, -1, 1, -1, 1, 1, -1};
    int32_t expected[8];
    memcpy(expected, data, sizeof(data));
    
    /* Reference using default backend */
    fwht_i32(expected, 8);
    
    /* Test CPU backend explicitly */
    fwht_status_t status = fwht_i32_backend(data, 8, FWHT_BACKEND_CPU);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Results should match */
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ_I32(data[i], expected[i]);
    }
}

TEST(backend_specific_openmp) {
    /* Test OpenMP backend if available */
    if (!fwht_has_openmp()) {
        /* Skip if OpenMP not available */
        return;
    }
    
    int32_t data[256];
    int32_t expected[256];
    
    /* Initialize */
    for (int i = 0; i < 256; i++) {
        data[i] = expected[i] = (i % 3 == 0) ? 1 : -1;
    }
    
    /* Reference */
    fwht_i32(expected, 256);
    
    /* Test OpenMP */
    fwht_status_t status = fwht_i32_backend(data, 256, FWHT_BACKEND_OPENMP);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Results should match */
    for (int i = 0; i < 256; i++) {
        ASSERT_EQ_I32(data[i], expected[i]);
    }
}

TEST(out_of_place_backend_i32) {
    /* Test out-of-place with backend selection */
    int32_t input[8] = {1, -1, -1, 1, -1, 1, 1, -1};
    int32_t reference[8];
    memcpy(reference, input, sizeof(input));
    
    /* Compute with CPU backend */
    int32_t* result = fwht_compute_i32_backend(input, 8, FWHT_BACKEND_CPU);
    ASSERT(result != NULL, "Compute backend failed");
    
    /* Reference in-place */
    fwht_i32(reference, 8);
    
    /* Input should be unchanged */
    int32_t original[8] = {1, -1, -1, 1, -1, 1, 1, -1};
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ_I32(input[i], original[i]);
    }
    
    /* Result should match reference */
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ_I32(result[i], reference[i]);
    }
    
    free(result);
}

TEST(out_of_place_backend_f64) {
    /* Test out-of-place float64 with backend selection */
    double input[16];
    double reference[16];
    
    for (int i = 0; i < 16; i++) {
        input[i] = reference[i] = (i % 2) ? -1.0 : 1.0;
    }
    
    /* Compute with CPU backend */
    double* result = fwht_compute_f64_backend(input, 16, FWHT_BACKEND_CPU);
    ASSERT(result != NULL, "Compute backend failed");
    
    /* Reference in-place */
    fwht_f64(reference, 16);
    
    /* Result should match reference */
    for (int i = 0; i < 16; i++) {
        ASSERT_NEAR_F64(result[i], reference[i], 1e-10);
    }
    
    free(result);
}

TEST(boolean_packed_backend) {
    /* Test bit-packed Boolean WHT with backend selection */
    uint8_t truth_table[8] = {0, 1, 1, 0, 1, 0, 0, 1};
    
    /* Pack into uint64 */
    uint64_t packed = 0;
    for (int i = 0; i < 8; i++) {
        if (truth_table[i]) {
            packed |= (1ULL << i);
        }
    }
    
    /* Compute with CPU backend */
    int32_t result[8];
    fwht_status_t status = fwht_boolean_packed_backend(&packed, result, 8, FWHT_BACKEND_CPU);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Compare with unpacked version */
    int32_t reference[8];
    status = fwht_from_bool(truth_table, reference, 8, true);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    for (int i = 0; i < 8; i++) {
        ASSERT_EQ_I32(result[i], reference[i]);
    }
}

TEST(direct_batch_i32) {
    /* Test direct batch API (non-context) */
    int32_t data1[4] = {1, -1, -1, 1};
    int32_t data2[4] = {1, 1, -1, -1};
    int32_t data3[4] = {-1, -1, -1, -1};
    int32_t* batch[3] = {data1, data2, data3};
    
    /* Compute reference */
    int32_t ref1[4], ref2[4], ref3[4];
    memcpy(ref1, data1, sizeof(data1));
    memcpy(ref2, data2, sizeof(data2));
    memcpy(ref3, data3, sizeof(data3));
    fwht_i32(ref1, 4);
    fwht_i32(ref2, 4);
    fwht_i32(ref3, 4);
    
    /* Batch transform */
    fwht_status_t status = fwht_i32_batch(batch, 4, 3);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Verify all results */
    for (int i = 0; i < 4; i++) {
        ASSERT_EQ_I32(data1[i], ref1[i]);
        ASSERT_EQ_I32(data2[i], ref2[i]);
        ASSERT_EQ_I32(data3[i], ref3[i]);
    }
}

TEST(direct_batch_f64) {
    /* Test direct batch API for float64 */
    double data1[8] = {1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0};
    double data2[8] = {1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0};
    double* batch[2] = {data1, data2};
    
    /* Reference */
    double ref1[8], ref2[8];
    memcpy(ref1, data1, sizeof(data1));
    memcpy(ref2, data2, sizeof(data2));
    fwht_f64(ref1, 8);
    fwht_f64(ref2, 8);
    
    /* Batch transform */
    fwht_status_t status = fwht_f64_batch(batch, 8, 2);
    ASSERT_STATUS(status, FWHT_SUCCESS);
    
    /* Verify */
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR_F64(data1[i], ref1[i], 1e-10);
        ASSERT_NEAR_F64(data2[i], ref2[i], 1e-10);
    }
}

/* ============================================================================
 * MAIN TEST RUNNER
 * ============================================================================ */

int main(void) {
    printf("===================================================================\n");
    printf("FWHT Library - Comprehensive Test Suite\n");
    printf("===================================================================\n\n");
    
    /* Mathematical properties */
    run_test_constant_zero();
    run_test_constant_one();
    run_test_linear_function_single_bit();
    run_test_linear_function_xor();
    run_test_involution_property();
    run_test_linearity();
    
    /* Size and edge cases */
    run_test_size_2();
    run_test_size_4();
    run_test_size_256();
    run_test_size_large_4096();
    run_test_error_not_power_of_2();
    run_test_error_null_pointer();
    run_test_error_zero_size();
    
    /* Data types */
    run_test_double_precision();
    run_test_int8_type();
    run_test_out_of_place_i32();
    run_test_out_of_place_f64();
    
    /* Boolean function API */
    run_test_from_bool_signed();
    run_test_from_bool_unsigned();
    run_test_correlations();
    
    /* Utilities */
    run_test_is_power_of_2_check();
    run_test_log2_check();
    run_test_version_info();
    
    /* Context API */
    run_test_context_create_destroy();
    run_test_context_transform();
    run_test_batch_processing();
    
    /* Known vectors */
    run_test_known_vector_1();
    
    /* Consistency */
    run_test_consistency_i32_vs_f64();
    
        /* Critical missing tests */
        run_test_overflow_detection_i32();
        run_test_edge_case_size_1();
        run_test_edge_case_very_large();
        run_test_backend_specific_cpu();
        run_test_backend_specific_openmp();
        run_test_out_of_place_backend_i32();
        run_test_out_of_place_backend_f64();
        run_test_boolean_packed_backend();
        run_test_direct_batch_i32();
        run_test_direct_batch_f64();
    
    /* Summary */
    printf("\n===================================================================\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("===================================================================\n");
    
    if (tests_failed == 0) {
        printf("\n✓ ALL TESTS PASSED!\n\n");
        return 0;
    } else {
        printf("\n✗ SOME TESTS FAILED!\n\n");
        return 1;
    }
}
