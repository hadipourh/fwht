/*
 * Test Suite for Vectorized Batch FWHT
 */

#include "../include/fwht.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define TEST_PASS "\033[32mPASSED\033[0m"
#define TEST_FAIL "\033[31mFAILED\033[0m"

/* Get wall-clock time in microseconds */
static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

/* Test batch correctness against sequential */
static int test_batch_correctness_i32(void) {
    const size_t n = 256;
    const size_t batch_size = 16;
    
    int32_t** data_batch = malloc(batch_size * sizeof(int32_t*));
    int32_t** data_seq = malloc(batch_size * sizeof(int32_t*));
    
    /* Initialize with random data */
    srand(12345);
    for (size_t i = 0; i < batch_size; i++) {
        data_batch[i] = malloc(n * sizeof(int32_t));
        data_seq[i] = malloc(n * sizeof(int32_t));
        
        for (size_t j = 0; j < n; j++) {
            int32_t val = (rand() % 3) - 1;  /* -1, 0, or 1 */
            data_batch[i][j] = val;
            data_seq[i][j] = val;
        }
    }
    
    /* Batch transform */
    fwht_status_t status = fwht_i32_batch(data_batch, n, batch_size);
    if (status != FWHT_SUCCESS) {
        printf("  Batch i32 correctness: %s (batch transform failed)\n", TEST_FAIL);
        return 0;
    }
    
    /* Sequential transform */
    for (size_t i = 0; i < batch_size; i++) {
        fwht_i32(data_seq[i], n);
    }
    
    /* Compare results */
    int passed = 1;
    for (size_t i = 0; i < batch_size && passed; i++) {
        for (size_t j = 0; j < n; j++) {
            if (data_batch[i][j] != data_seq[i][j]) {
                passed = 0;
                printf("  Mismatch at batch[%zu][%zu]: %d vs %d\n", 
                       i, j, data_batch[i][j], data_seq[i][j]);
                break;
            }
        }
    }
    
    printf("  Batch i32 correctness (n=%zu, batch=%zu): %s\n", 
           n, batch_size, passed ? TEST_PASS : TEST_FAIL);
    
    /* Cleanup */
    for (size_t i = 0; i < batch_size; i++) {
        free(data_batch[i]);
        free(data_seq[i]);
    }
    free(data_batch);
    free(data_seq);
    
    return passed;
}

/* Test batch performance vs sequential */
static int test_batch_performance_i32(void) {
    const size_t n = 256;
    const size_t batch_size = 128;
    
    int32_t** data = malloc(batch_size * sizeof(int32_t*));
    
    srand(54321);
    for (size_t i = 0; i < batch_size; i++) {
        data[i] = malloc(n * sizeof(int32_t));
        for (size_t j = 0; j < n; j++) {
            data[i][j] = (rand() % 3) - 1;
        }
    }
    
    /* Warmup runs */
    fwht_i32_batch(data, n, batch_size);
    fwht_i32_batch(data, n, batch_size);
    
    /* Reset and benchmark batch (multiple runs for accuracy) */
    srand(54321);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < n; j++) {
            data[i][j] = (rand() % 3) - 1;
        }
    }
    
    double start = get_time_us();
    for (int r = 0; r < 10; r++) {
        fwht_i32_batch(data, n, batch_size);
    }
    double end = get_time_us();
    double batch_time = (end - start) / 10.0 / 1000.0;  /* Average, convert to ms */
    
    /* Reset data and benchmark sequential */
    srand(54321);
    for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < n; j++) {
            data[i][j] = (rand() % 3) - 1;
        }
    }
    
    start = get_time_us();
    for (int r = 0; r < 10; r++) {
        for (size_t i = 0; i < batch_size; i++) {
            fwht_i32(data[i], n);
        }
    }
    end = get_time_us();
    double seq_time = (end - start) / 10.0 / 1000.0;  /* Average, convert to ms */
    
    double speedup = seq_time / batch_time;
    
    printf("  Batch i32 performance (n=%zu, batch=%zu, 10 runs avg):\n", n, batch_size);
    printf("    Sequential: %.3f ms\n", seq_time);
    printf("    Batch:      %.3f ms\n", batch_time);
    printf("    Speedup:    %.2fx\n", speedup);
    
    /* Cleanup */
    for (size_t i = 0; i < batch_size; i++) {
        free(data[i]);
    }
    free(data);
    
    /* Consider it passing if batch is at least as fast as sequential */
    int passed = (batch_time <= seq_time * 1.1);  /* Allow 10% margin */
    printf("    Status: %s\n", passed ? TEST_PASS : TEST_FAIL);
    
    return passed;
}

/* Test various sizes */
static int test_batch_various_sizes(void) {
    size_t sizes[] = {2, 4, 8, 16, 32, 64, 128, 256};
    int all_passed = 1;
    
    printf("  Testing various sizes:\n");
    
    for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        size_t n = sizes[s];
        size_t batch_size = 8;
        
        int32_t** data_batch = malloc(batch_size * sizeof(int32_t*));
        int32_t** data_seq = malloc(batch_size * sizeof(int32_t*));
        
        for (size_t i = 0; i < batch_size; i++) {
            data_batch[i] = malloc(n * sizeof(int32_t));
            data_seq[i] = malloc(n * sizeof(int32_t));
            
            for (size_t j = 0; j < n; j++) {
                int32_t val = (j % 2 == 0) ? 1 : -1;
                data_batch[i][j] = val;
                data_seq[i][j] = val;
            }
        }
        
        fwht_i32_batch(data_batch, n, batch_size);
        for (size_t i = 0; i < batch_size; i++) {
            fwht_i32(data_seq[i], n);
        }
        
        int passed = 1;
        for (size_t i = 0; i < batch_size && passed; i++) {
            for (size_t j = 0; j < n; j++) {
                if (data_batch[i][j] != data_seq[i][j]) {
                    passed = 0;
                    break;
                }
            }
        }
        
        printf("    n=%3zu: %s\n", n, passed ? TEST_PASS : TEST_FAIL);
        
        if (!passed) all_passed = 0;
        
        for (size_t i = 0; i < batch_size; i++) {
            free(data_batch[i]);
            free(data_seq[i]);
        }
        free(data_batch);
        free(data_seq);
    }
    
    return all_passed;
}

int main(void) {
    printf("=============================================================\n");
    printf("FWHT Vectorized Batch Test Suite\n");
    printf("=============================================================\n\n");
    
    int total = 0;
    int passed = 0;
    
    total++; if (test_batch_correctness_i32()) passed++;
    total++; if (test_batch_performance_i32()) passed++;
    total++; if (test_batch_various_sizes()) passed++;
    
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
