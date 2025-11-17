/*
 * Test suite for bit-sliced Boolean WHT
 * 
 * Tests the optimized bit-packed WHT implementation for cryptography.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../include/fwht.h"

#define TEST_PASS "\033[32m✓\033[0m"
#define TEST_FAIL "\033[31m✗\033[0m"

static int test_count = 0;
static int pass_count = 0;

static void test_result(const char* name, int passed) {
    test_count++;
    if (passed) {
        pass_count++;
        printf("  %s %s\n", TEST_PASS, name);
    } else {
        printf("  %s %s\n", TEST_FAIL, name);
    }
}

/*
 * Test 1: Simple 4-bit Boolean function
 * Truth table: [0,1,1,0] (XOR function)
 * Expected WHT: [0, 0, 0, 4] (maximum correlation with x1 XOR x0)
 */
static int test_simple_xor(void) {
    uint8_t truth_table[] = {0, 1, 1, 0};
    size_t n = 4;
    
    /* Test unpacked version first */
    int32_t wht_unpacked[4];
    fwht_status_t status = fwht_from_bool(truth_table, wht_unpacked, n, true);
    if (status != FWHT_SUCCESS) {
        printf("    Unpacked WHT failed: %s\n", fwht_error_string(status));
        return 0;
    }
    
    /* Pack the truth table into bits */
    uint64_t packed = 0;
    for (size_t i = 0; i < n; i++) {
        if (truth_table[i]) {
            packed |= (1ULL << i);
        }
    }
    
    /* Test bit-sliced version */
    int32_t wht_packed[4];
    status = fwht_boolean_packed(&packed, wht_packed, n);
    if (status != FWHT_SUCCESS) {
        printf("    Packed WHT failed: %s\n", fwht_error_string(status));
        return 0;
    }
    
    /* Compare results */
    for (size_t i = 0; i < n; i++) {
        if (wht_unpacked[i] != wht_packed[i]) {
            printf("    Mismatch at position %zu: unpacked=%d, packed=%d\n",
                   i, wht_unpacked[i], wht_packed[i]);
            return 0;
        }
    }
    
    /* Verify expected XOR pattern: maximum at position 3 (binary 11) */
    if (wht_packed[3] != 4 && wht_packed[3] != -4) {
        printf("    Expected |WHT[3]| = 4, got %d\n", wht_packed[3]);
        return 0;
    }
    
    return 1;
}

/*
 * Test 2: All-zeros function (constant 0)
 * Expected: WHT[0] = n, all others = 0
 */
static int test_constant_zero(void) {
    size_t n = 16;
    uint8_t truth_table[16] = {0};  /* All zeros */
    
    /* Unpacked */
    int32_t wht_unpacked[16];
    fwht_status_t status = fwht_from_bool(truth_table, wht_unpacked, n, true);
    if (status != FWHT_SUCCESS) return 0;
    
    /* Packed */
    uint64_t packed = 0;  /* All bits zero */
    int32_t wht_packed[16];
    status = fwht_boolean_packed(&packed, wht_packed, n);
    if (status != FWHT_SUCCESS) return 0;
    
    /* Compare */
    for (size_t i = 0; i < n; i++) {
        if (wht_unpacked[i] != wht_packed[i]) {
            printf("    Mismatch at %zu: %d vs %d\n", i, wht_unpacked[i], wht_packed[i]);
            return 0;
        }
    }
    
    /* Verify constant function property */
    if (abs(wht_packed[0]) != (int32_t)n) {
        printf("    Expected |WHT[0]| = %zu, got %d\n", n, wht_packed[0]);
        return 0;
    }
    
    for (size_t i = 1; i < n; i++) {
        if (wht_packed[i] != 0) {
            printf("    Expected WHT[%zu] = 0, got %d\n", i, wht_packed[i]);
            return 0;
        }
    }
    
    return 1;
}

/*
 * Test 3: Balanced function
 * Equal number of 0s and 1s → WHT[0] = 0
 */
static int test_balanced(void) {
    size_t n = 8;
    uint8_t truth_table[] = {0, 1, 1, 0, 1, 0, 0, 1};  /* 4 zeros, 4 ones */
    
    /* Unpacked */
    int32_t wht_unpacked[8];
    fwht_status_t status = fwht_from_bool(truth_table, wht_unpacked, n, true);
    if (status != FWHT_SUCCESS) return 0;
    
    /* Packed */
    uint64_t packed = 0;
    for (size_t i = 0; i < n; i++) {
        if (truth_table[i]) {
            packed |= (1ULL << i);
        }
    }
    
    int32_t wht_packed[8];
    status = fwht_boolean_packed(&packed, wht_packed, n);
    if (status != FWHT_SUCCESS) return 0;
    
    /* Compare */
    for (size_t i = 0; i < n; i++) {
        if (wht_unpacked[i] != wht_packed[i]) {
            printf("    Mismatch at %zu: %d vs %d\n", i, wht_unpacked[i], wht_packed[i]);
            return 0;
        }
    }
    
    /* Verify balanced property: WHT[0] = 0 */
    if (wht_packed[0] != 0) {
        printf("    Balanced function should have WHT[0] = 0, got %d\n", wht_packed[0]);
        return 0;
    }
    
    return 1;
}

/*
 * Test 4: Larger function (n=256, typical for 8-bit S-box)
 */
static int test_large_function(void) {
    size_t n = 256;
    
    /* Create a pseudo-random Boolean function */
    uint8_t truth_table[256];
    for (size_t i = 0; i < n; i++) {
        /* Simple pseudo-random pattern */
        truth_table[i] = ((i * 7 + 13) % 2);
    }
    
    /* Unpacked */
    int32_t* wht_unpacked = (int32_t*)malloc(n * sizeof(int32_t));
    if (!wht_unpacked) return 0;
    
    fwht_status_t status = fwht_from_bool(truth_table, wht_unpacked, n, true);
    if (status != FWHT_SUCCESS) {
        free(wht_unpacked);
        return 0;
    }
    
    /* Packed */
    size_t n_words = (n + 63) / 64;
    uint64_t* packed = (uint64_t*)calloc(n_words, sizeof(uint64_t));
    if (!packed) {
        free(wht_unpacked);
        return 0;
    }
    
    for (size_t i = 0; i < n; i++) {
        if (truth_table[i]) {
            packed[i / 64] |= (1ULL << (i % 64));
        }
    }
    
    int32_t* wht_packed = (int32_t*)malloc(n * sizeof(int32_t));
    if (!wht_packed) {
        free(wht_unpacked);
        free(packed);
        return 0;
    }
    
    status = fwht_boolean_packed(packed, wht_packed, n);
    if (status != FWHT_SUCCESS) {
        free(wht_unpacked);
        free(packed);
        free(wht_packed);
        return 0;
    }
    
    /* Compare all coefficients */
    int all_match = 1;
    for (size_t i = 0; i < n; i++) {
        if (wht_unpacked[i] != wht_packed[i]) {
            printf("    Mismatch at %zu: %d vs %d\n", i, wht_unpacked[i], wht_packed[i]);
            all_match = 0;
            break;
        }
    }
    
    free(wht_unpacked);
    free(packed);
    free(wht_packed);
    
    return all_match;
}

/*
 * Test 5: Batch processing
 */
static int test_batch(void) {
    size_t n = 16;
    size_t batch_size = 4;
    
    /* Create batch of Boolean functions */
    uint64_t* packed_batch[4];
    int32_t* wht_batch[4];
    int32_t* wht_reference[4];
    
    for (size_t i = 0; i < batch_size; i++) {
        packed_batch[i] = (uint64_t*)calloc(1, sizeof(uint64_t));
        wht_batch[i] = (int32_t*)malloc(n * sizeof(int32_t));
        wht_reference[i] = (int32_t*)malloc(n * sizeof(int32_t));
        
        if (!packed_batch[i] || !wht_batch[i] || !wht_reference[i]) {
            /* Cleanup and fail */
            for (size_t j = 0; j <= i; j++) {
                free(packed_batch[j]);
                free(wht_batch[j]);
                free(wht_reference[j]);
            }
            return 0;
        }
        
        /* Create unique pattern for each function */
        packed_batch[i][0] = (uint64_t)(0xAAAA << i);
        
        /* Compute reference */
        fwht_boolean_packed(packed_batch[i], wht_reference[i], n);
    }
    
    /* Batch computation */
    fwht_status_t status = fwht_boolean_batch(
        (const uint64_t**)packed_batch, wht_batch, n, batch_size);
    
    if (status != FWHT_SUCCESS) {
        for (size_t i = 0; i < batch_size; i++) {
            free(packed_batch[i]);
            free(wht_batch[i]);
            free(wht_reference[i]);
        }
        return 0;
    }
    
    /* Verify all results match */
    int all_match = 1;
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < n; j++) {
            if (wht_batch[i][j] != wht_reference[i][j]) {
                printf("    Batch %zu mismatch at %zu: %d vs %d\n",
                       i, j, wht_batch[i][j], wht_reference[i][j]);
                all_match = 0;
                break;
            }
        }
    }
    
    /* Cleanup */
    for (size_t i = 0; i < batch_size; i++) {
        free(packed_batch[i]);
        free(wht_batch[i]);
        free(wht_reference[i]);
    }
    
    return all_match;
}

int main(void) {
    printf("\n");
    printf("==============================================\n");
    printf("  Bit-Sliced Boolean WHT Test Suite\n");
    printf("==============================================\n\n");
    
    printf("Running tests:\n");
    test_result("Simple XOR function (n=4)", test_simple_xor());
    test_result("Constant zero function (n=16)", test_constant_zero());
    test_result("Balanced function (n=8)", test_balanced());
    test_result("Large function (n=256)", test_large_function());
    test_result("Batch processing (4 functions)", test_batch());
    
    printf("\n");
    printf("==============================================\n");
    printf("  Results: %d/%d tests passed\n", pass_count, test_count);
    printf("==============================================\n\n");
    
    return (pass_count == test_count) ? 0 : 1;
}
