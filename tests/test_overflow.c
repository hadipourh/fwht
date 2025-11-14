/*
 * Test overflow detection in FWHT
 */

#include "../include/fwht.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TEST_PASS "\033[32mPASSED\033[0m"
#define TEST_FAIL "\033[31mFAILED\033[0m"

/* Test that safe mode detects overflow */
static int test_overflow_detection(void) {
    const size_t n = 256;
    int32_t* data = (int32_t*)malloc(n * sizeof(int32_t));
    
    /* Create data that will overflow: use large values */
    for (size_t i = 0; i < n; i++) {
        data[i] = 100000000;  /* Large enough that n * max will overflow int32 */
    }
    
    fwht_status_t status = fwht_i32_safe(data, n);
    
    int passed = (status == FWHT_ERROR_OVERFLOW);
    printf("  Overflow detection: %s\n", passed ? TEST_PASS : TEST_FAIL);
    
    if (!passed) {
        printf("    Expected FWHT_ERROR_OVERFLOW, got: %s\n", fwht_error_string(status));
    }
    
    free(data);
    return passed;
}

/* Test that safe mode passes for valid data */
static int test_safe_mode_valid(void) {
    const size_t n = 256;
    int32_t* data = (int32_t*)malloc(n * sizeof(int32_t));
    
    /* Create data that won't overflow */
    for (size_t i = 0; i < n; i++) {
        data[i] = (i % 2 == 0) ? 1 : -1;
    }
    
    fwht_status_t status = fwht_i32_safe(data, n);
    
    int passed = (status == FWHT_SUCCESS);
    printf("  Safe mode valid data: %s\n", passed ? TEST_PASS : TEST_FAIL);
    
    if (!passed) {
        printf("    Expected FWHT_SUCCESS, got: %s\n", fwht_error_string(status));
    }
    
    free(data);
    return passed;
}

/* Test that safe mode produces same results as unsafe mode for valid data */
static int test_safe_correctness(void) {
    const size_t n = 128;
    int32_t* data1 = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* data2 = (int32_t*)malloc(n * sizeof(int32_t));
    
    /* Initialize with same data */
    for (size_t i = 0; i < n; i++) {
        data1[i] = data2[i] = (i % 2 == 0) ? 1 : -1;
    }
    
    /* Transform with safe and unsafe modes */
    fwht_status_t status1 = fwht_i32_safe(data1, n);
    fwht_status_t status2 = fwht_i32(data2, n);
    
    /* Check both succeeded */
    if (status1 != FWHT_SUCCESS || status2 != FWHT_SUCCESS) {
        printf("  Safe mode correctness: %s\n", TEST_FAIL);
        printf("    Safe status: %s, Unsafe status: %s\n", 
               fwht_error_string(status1), fwht_error_string(status2));
        free(data1);
        free(data2);
        return 0;
    }
    
    /* Compare results */
    int passed = 1;
    for (size_t i = 0; i < n; i++) {
        if (data1[i] != data2[i]) {
            passed = 0;
            break;
        }
    }
    
    printf("  Safe mode correctness: %s\n", passed ? TEST_PASS : TEST_FAIL);
    
    free(data1);
    free(data2);
    return passed;
}

/* Test backend name reporting */
static int test_backend_names(void) {
    const char* name = fwht_backend_name(FWHT_BACKEND_CPU_SAFE);
    int passed = (strcmp(name, "cpu_safe") == 0);
    
    printf("  Backend name: %s\n", passed ? TEST_PASS : TEST_FAIL);
    
    if (!passed) {
        printf("    Expected 'cpu_safe', got: '%s'\n", name);
    }
    
    return passed;
}

int main(void) {
    printf("=============================================================\n");
    printf("FWHT Overflow Detection Test Suite\n");
    printf("=============================================================\n\n");
    
    int total = 0;
    int passed = 0;
    
    total++; if (test_overflow_detection()) passed++;
    total++; if (test_safe_mode_valid()) passed++;
    total++; if (test_safe_correctness()) passed++;
    total++; if (test_backend_names()) passed++;
    
    printf("\n=============================================================\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", total);
    printf("  Passed: %d\n", passed);
    printf("  Failed: %d\n", total - passed);
    printf("=============================================================\n\n");
    
    if (passed == total) {
        printf("✓ ALL TESTS PASSED!\n");
        return 0;
    } else {
        printf("✗ SOME TESTS FAILED!\n");
        return 1;
    }
}
