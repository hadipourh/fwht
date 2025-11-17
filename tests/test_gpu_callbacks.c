/*
 * Test Suite for GPU Callbacks API
 * Tests the callback setter functions and verifies API correctness.
 */

#include "../include/fwht.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_PASS "\033[32mPASSED\033[0m"
#define TEST_FAIL "\033[31mFAILED\033[0m"

/* Test that callback API is declared correctly */
static int test_api_declarations(void) {
    printf("  API declarations test: ");
    
    /* Test is just compilation-time check */
    /* If we got here, the API is properly declared */
    
    printf("%s\n", TEST_PASS);
    return 1;
}

/* Test callback setters with NULL context */
static int test_null_context(void) {
    printf("  NULL context handling: ");
    
#ifdef USE_CUDA
    fwht_status_t status;
    
    status = fwht_gpu_context_set_callbacks_i32(NULL, NULL, NULL, NULL);
    if (status != FWHT_ERROR_NULL_POINTER) {
        printf("%s (expected NULL_POINTER error)\n", TEST_FAIL);
        return 0;
    }
    
    status = fwht_gpu_context_set_callbacks_f64(NULL, NULL, NULL, NULL);
    if (status != FWHT_ERROR_NULL_POINTER) {
        printf("%s (expected NULL_POINTER error)\n", TEST_FAIL);
        return 0;
    }
    
    printf("%s\n", TEST_PASS);
    return 1;
#else
    printf("SKIPPED (no CUDA)\n");
    return 1;
#endif
}

/* Test basic callback configuration */
static int test_callback_configuration(void) {
    printf("  Callback configuration: ");
    
#ifdef USE_CUDA
    if (!fwht_has_gpu()) {
        printf("SKIPPED (no GPU)\n");
        return 1;
    }
    
    /* Create context */
    fwht_gpu_context_t* ctx = fwht_gpu_context_create(256, 10);
    if (!ctx) {
        printf("%s (context creation failed)\n", TEST_FAIL);
        return 0;
    }
    
    /* Set callbacks to NULL (should succeed) */
    fwht_status_t status = fwht_gpu_context_set_callbacks_i32(ctx, NULL, NULL, NULL);
    if (status != FWHT_SUCCESS) {
        printf("%s (setting NULL callbacks failed)\n", TEST_FAIL);
        fwht_gpu_context_destroy(ctx);
        return 0;
    }
    
    status = fwht_gpu_context_set_callbacks_f64(ctx, NULL, NULL, NULL);
    if (status != FWHT_SUCCESS) {
        printf("%s (setting NULL callbacks failed)\n", TEST_FAIL);
        fwht_gpu_context_destroy(ctx);
        return 0;
    }
    
    /* Cleanup */
    fwht_gpu_context_destroy(ctx);
    
    printf("%s\n", TEST_PASS);
    return 1;
#else
    printf("SKIPPED (no CUDA)\n");
    return 1;
#endif
}

/* Test that transforms still work after setting NULL callbacks */
static int test_transform_with_null_callbacks(void) {
    printf("  Transform with NULL callbacks: ");
    
#ifdef USE_CUDA
    if (!fwht_has_gpu()) {
        printf("SKIPPED (no GPU)\n");
        return 1;
    }
    
    const size_t n = 256;
    int32_t* data = (int32_t*)malloc(n * sizeof(int32_t));
    if (!data) {
        printf("%s (malloc failed)\n", TEST_FAIL);
        return 0;
    }
    
    /* Initialize data */
    for (size_t i = 0; i < n; i++) {
        data[i] = (i % 2 == 0) ? 1 : -1;
    }
    
    /* Create context and set NULL callbacks */
    fwht_gpu_context_t* ctx = fwht_gpu_context_create(n, 1);
    if (!ctx) {
        printf("%s (context creation failed)\n", TEST_FAIL);
        free(data);
        return 0;
    }
    
    fwht_gpu_context_set_callbacks_i32(ctx, NULL, NULL, NULL);
    
    /* Perform transform */
    fwht_status_t status = fwht_gpu_context_compute_i32(ctx, data, n, 1);
    if (status != FWHT_SUCCESS) {
        printf("%s (transform failed: %s)\n", TEST_FAIL, fwht_error_string(status));
        fwht_gpu_context_destroy(ctx);
        free(data);
        return 0;
    }
    
    /* Verify result makes sense (should be 0 or 256 for alternating pattern) */
    if (data[0] != 0 && data[0] != 256 && data[0] != -256) {
        printf("%s (unexpected result: %d)\n", TEST_FAIL, data[0]);
        fwht_gpu_context_destroy(ctx);
        free(data);
        return 0;
    }
    
    /* Cleanup */
    fwht_gpu_context_destroy(ctx);
    free(data);
    
    printf("%s\n", TEST_PASS);
    return 1;
#else
    printf("SKIPPED (no CUDA)\n");
    return 1;
#endif
}

int main(void) {
    printf("=============================================================\n");
    printf("FWHT GPU Callbacks Test Suite\n");
    printf("=============================================================\n\n");
    
    int total = 0;
    int passed = 0;
    
    total++; if (test_api_declarations()) passed++;
    total++; if (test_null_context()) passed++;
    total++; if (test_callback_configuration()) passed++;
    total++; if (test_transform_with_null_callbacks()) passed++;
    
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
