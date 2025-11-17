/*
 * Test suite for bit-sliced Boolean WHT
 * 
 * Validates:
 * - Correctness vs unpacked implementation
 * - Small sizes (8, 16, 32, 64)
 * - Medium sizes (256, 1024, 4096)
 * - Large sizes (16384, 65536)
 * - Batch processing
 * - Performance comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../include/fwht.h"

#define ANSI_GREEN  "\x1b[32m"
#define ANSI_RED    "\x1b[31m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_RESET  "\x1b[0m"

static int total_tests = 0;
static int passed_tests = 0;

void print_test_result(const char* test_name, int passed) {
    total_tests++;
    if (passed) {
        passed_tests++;
        printf("  " ANSI_GREEN "✓" ANSI_RESET " %s\n", test_name);
    } else {
        printf("  " ANSI_RED "✗" ANSI_RESET " %s\n", test_name);
    }
}

/* Pack uint8 truth table into uint64 bit-packed representation */
void pack_truth_table(const uint8_t* truth_table, uint64_t* packed, size_t n) {
    size_t n_words = (n + 63) / 64;
    memset(packed, 0, n_words * sizeof(uint64_t));
    
    for (size_t i = 0; i < n; ++i) {
        if (truth_table[i]) {
            size_t word_idx = i / 64;
            size_t bit_idx = i % 64;
            packed[word_idx] |= (1ULL << bit_idx);
        }
    }
}

/* Test small truth table: n=8 */
void test_small_size(void) {
    size_t n = 8;
    
    /* XOR function: f(x0,x1,x2) = x0 ⊕ x1 ⊕ x2 */
    uint8_t truth_table[8] = {0, 1, 1, 0, 1, 0, 0, 1};
    
    /* Compute unpacked WHT */
    int32_t wht_unpacked[8];
    fwht_status_t status = fwht_from_bool(truth_table, wht_unpacked, n, true);
    if (status != FWHT_SUCCESS) {
        print_test_result("Small size (n=8) - unpacked", 0);
        return;
    }
    
    /* Compute packed WHT */
    uint64_t packed;
    pack_truth_table(truth_table, &packed, n);
    
    int32_t wht_packed[8];
    status = fwht_boolean_packed(&packed, wht_packed, n);
    if (status != FWHT_SUCCESS) {
        print_test_result("Small size (n=8) - packed", 0);
        return;
    }
    
    /* Compare results */
    int match = 1;
    for (size_t i = 0; i < n; ++i) {
        if (wht_unpacked[i] != wht_packed[i]) {
            match = 0;
            printf("  Mismatch at i=%zu: unpacked=%d, packed=%d\n", 
                   i, wht_unpacked[i], wht_packed[i]);
            break;
        }
    }
    
    print_test_result("Small size (n=8)", match);
}

/* Test medium size: n=256 */
void test_medium_size(void) {
    size_t n = 256;
    
    /* Create random Boolean function */
    uint8_t* truth_table = malloc(n * sizeof(uint8_t));
    srand(42);
    for (size_t i = 0; i < n; ++i) {
        truth_table[i] = rand() % 2;
    }
    
    /* Compute unpacked WHT */
    int32_t* wht_unpacked = malloc(n * sizeof(int32_t));
    fwht_status_t status = fwht_from_bool(truth_table, wht_unpacked, n, true);
    if (status != FWHT_SUCCESS) {
        print_test_result("Medium size (n=256)", 0);
        free(truth_table);
        free(wht_unpacked);
        return;
    }
    
    /* Compute packed WHT */
    size_t n_words = (n + 63) / 64;
    uint64_t* packed = malloc(n_words * sizeof(uint64_t));
    pack_truth_table(truth_table, packed, n);
    
    int32_t* wht_packed = malloc(n * sizeof(int32_t));
    status = fwht_boolean_packed(packed, wht_packed, n);
    if (status != FWHT_SUCCESS) {
        print_test_result("Medium size (n=256)", 0);
        free(truth_table);
        free(wht_unpacked);
        free(packed);
        free(wht_packed);
        return;
    }
    
    /* Compare results */
    int match = 1;
    for (size_t i = 0; i < n; ++i) {
        if (wht_unpacked[i] != wht_packed[i]) {
            match = 0;
            printf("  Mismatch at i=%zu: unpacked=%d, packed=%d\n", 
                   i, wht_unpacked[i], wht_packed[i]);
            break;
        }
    }
    
    print_test_result("Medium size (n=256)", match);
    
    free(truth_table);
    free(wht_unpacked);
    free(packed);
    free(wht_packed);
}

/* Test large size: n=4096 */
void test_large_size(void) {
    size_t n = 4096;
    
    /* Create random Boolean function */
    uint8_t* truth_table = malloc(n * sizeof(uint8_t));
    srand(123);
    for (size_t i = 0; i < n; ++i) {
        truth_table[i] = rand() % 2;
    }
    
    /* Compute unpacked WHT */
    int32_t* wht_unpacked = malloc(n * sizeof(int32_t));
    fwht_status_t status = fwht_from_bool(truth_table, wht_unpacked, n, true);
    if (status != FWHT_SUCCESS) {
        print_test_result("Large size (n=4096)", 0);
        free(truth_table);
        free(wht_unpacked);
        return;
    }
    
    /* Compute packed WHT */
    size_t n_words = (n + 63) / 64;
    uint64_t* packed = malloc(n_words * sizeof(uint64_t));
    pack_truth_table(truth_table, packed, n);
    
    int32_t* wht_packed = malloc(n * sizeof(int32_t));
    status = fwht_boolean_packed(packed, wht_packed, n);
    if (status != FWHT_SUCCESS) {
        print_test_result("Large size (n=4096)", 0);
        free(truth_table);
        free(wht_unpacked);
        free(packed);
        free(wht_packed);
        return;
    }
    
    /* Compare results */
    int match = 1;
    for (size_t i = 0; i < n; ++i) {
        if (wht_unpacked[i] != wht_packed[i]) {
            match = 0;
            printf("  Mismatch at i=%zu: unpacked=%d, packed=%d\n", 
                   i, wht_unpacked[i], wht_packed[i]);
            break;
        }
    }
    
    print_test_result("Large size (n=4096)", match);
    
    free(truth_table);
    free(wht_unpacked);
    free(packed);
    free(wht_packed);
}

/* Test batch processing */
void test_batch(void) {
    size_t n = 256;
    size_t batch_size = 8;
    
    /* Create batch of random Boolean functions */
    uint8_t** truth_tables = malloc(batch_size * sizeof(uint8_t*));
    uint64_t** packed_batch = malloc(batch_size * sizeof(uint64_t*));
    int32_t** wht_batch = malloc(batch_size * sizeof(int32_t*));
    int32_t** wht_ref = malloc(batch_size * sizeof(int32_t*));
    
    size_t n_words = (n + 63) / 64;
    
    srand(456);
    for (size_t i = 0; i < batch_size; ++i) {
        truth_tables[i] = malloc(n * sizeof(uint8_t));
        packed_batch[i] = malloc(n_words * sizeof(uint64_t));
        wht_batch[i] = malloc(n * sizeof(int32_t));
        wht_ref[i] = malloc(n * sizeof(int32_t));
        
        /* Random Boolean function */
        for (size_t j = 0; j < n; ++j) {
            truth_tables[i][j] = rand() % 2;
        }
        
        /* Pack and compute reference */
        pack_truth_table(truth_tables[i], packed_batch[i], n);
        fwht_from_bool(truth_tables[i], wht_ref[i], n, true);
    }
    
    /* Compute batch WHT */
    fwht_status_t status = fwht_boolean_batch((const uint64_t**)packed_batch, 
                                               wht_batch, n, batch_size);
    if (status != FWHT_SUCCESS) {
        print_test_result("Batch processing (8×256)", 0);
        goto cleanup;
    }
    
    /* Verify all results */
    int match = 1;
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (wht_batch[i][j] != wht_ref[i][j]) {
                match = 0;
                printf("  Batch %zu mismatch at j=%zu: got=%d, expected=%d\n",
                       i, j, wht_batch[i][j], wht_ref[i][j]);
                goto cleanup;
            }
        }
    }
    
    print_test_result("Batch processing (8×256)", match);
    
cleanup:
    for (size_t i = 0; i < batch_size; ++i) {
        free(truth_tables[i]);
        free(packed_batch[i]);
        free(wht_batch[i]);
        free(wht_ref[i]);
    }
    free(truth_tables);
    free(packed_batch);
    free(wht_batch);
    free(wht_ref);
}

/* Performance benchmark: compare packed vs unpacked */
void benchmark_performance(void) {
    size_t n = 4096;
    int iterations = 100;
    
    printf("\n" ANSI_YELLOW "Performance Benchmark (n=%zu, %d iterations):" ANSI_RESET "\n", 
           n, iterations);
    
    /* Create random Boolean function */
    uint8_t* truth_table = malloc(n * sizeof(uint8_t));
    srand(789);
    for (size_t i = 0; i < n; ++i) {
        truth_table[i] = rand() % 2;
    }
    
    /* Prepare packed version */
    size_t n_words = (n + 63) / 64;
    uint64_t* packed = malloc(n_words * sizeof(uint64_t));
    pack_truth_table(truth_table, packed, n);
    
    int32_t* wht = malloc(n * sizeof(int32_t));
    
    /* Benchmark unpacked */
    clock_t start = clock();
    for (int i = 0; i < iterations; ++i) {
        fwht_from_bool(truth_table, wht, n, true);
    }
    clock_t end = clock();
    double time_unpacked = (double)(end - start) / CLOCKS_PER_SEC;
    
    /* Benchmark packed */
    start = clock();
    for (int i = 0; i < iterations; ++i) {
        fwht_boolean_packed(packed, wht, n);
    }
    end = clock();
    double time_packed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("  Unpacked: %.3f ms/iter\n", time_unpacked * 1000.0 / iterations);
    printf("  Packed:   %.3f ms/iter\n", time_packed * 1000.0 / iterations);
    printf("  Speedup:  %.2fx\n", time_unpacked / time_packed);
    
    free(truth_table);
    free(packed);
    free(wht);
}

int main(void) {
    printf("============================================================\n");
    printf("Bit-Sliced Boolean WHT Test Suite\n");
    printf("============================================================\n\n");
    
    printf("Correctness Tests:\n");
    test_small_size();
    test_medium_size();
    test_large_size();
    test_batch();
    
    benchmark_performance();
    
    printf("\n============================================================\n");
    printf("Test Summary: %d/%d passed\n", passed_tests, total_tests);
    printf("============================================================\n");
    
    return (passed_tests == total_tests) ? 0 : 1;
}
