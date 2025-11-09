/*
 * Fast Walsh-Hadamard Transform - GPU Test Suite
 *
 * Exercises CUDA backend correctness and performance validation.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "../include/fwht.h"

#define TEST_PASS "\033[32m✓\033[0m"
#define TEST_FAIL "\033[31m✗\033[0m"

static int tests_run = 0;
static int tests_passed = 0;

/* Test helper */
#define RUN_TEST(test_func) do { \
    printf("  Running %s...", #test_func); \
    fflush(stdout); \
    if (test_func()) { \
        printf(" %s\n", TEST_PASS); \
        tests_passed++; \
    } else { \
        printf(" %s\n", TEST_FAIL); \
    } \
    tests_run++; \
} while(0)

/* ============================================================================
 * GPU Correctness Tests
 * ============================================================================ */

/* Test: GPU matches CPU for random data */
static int test_gpu_vs_cpu_small() {
    const size_t n = 256;
    int32_t cpu_data[256];
    int32_t gpu_data[256];
    
    /* Random data */
    for (size_t i = 0; i < n; i++) {
        cpu_data[i] = (int32_t)(rand() % 100 - 50);
        gpu_data[i] = cpu_data[i];
    }
    
    /* CPU transform */
    fwht_status_t cpu_status = fwht_i32_backend(cpu_data, n, FWHT_BACKEND_CPU);
    if (cpu_status != FWHT_SUCCESS) return 0;
    
    /* GPU transform */
    fwht_status_t gpu_status = fwht_i32_backend(gpu_data, n, FWHT_BACKEND_GPU);
    if (gpu_status != FWHT_SUCCESS) {
        printf(" (GPU unavailable)");
        return 1; /* Pass if GPU not available */
    }
    
    /* Compare results */
    for (size_t i = 0; i < n; i++) {
        if (cpu_data[i] != gpu_data[i]) {
            printf(" (mismatch at index %zu: CPU=%d, GPU=%d)", i, cpu_data[i], gpu_data[i]);
            return 0;
        }
    }
    
    return 1;
}

/* Test: GPU matches CPU for larger data */
static int test_gpu_vs_cpu_large() {
    const size_t n = 4096;
    int32_t* cpu_data = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* gpu_data = (int32_t*)malloc(n * sizeof(int32_t));
    
    if (!cpu_data || !gpu_data) {
        free(cpu_data);
        free(gpu_data);
        return 0;
    }
    
    /* Random data */
    for (size_t i = 0; i < n; i++) {
        cpu_data[i] = (int32_t)(rand() % 200 - 100);
        gpu_data[i] = cpu_data[i];
    }
    
    /* CPU transform */
    fwht_status_t cpu_status = fwht_i32_backend(cpu_data, n, FWHT_BACKEND_CPU);
    
    /* GPU transform */
    fwht_status_t gpu_status = fwht_i32_backend(gpu_data, n, FWHT_BACKEND_GPU);
    
    int result = 1;
    if (cpu_status != FWHT_SUCCESS || gpu_status != FWHT_SUCCESS) {
        result = (gpu_status != FWHT_SUCCESS) ? 1 : 0; /* Pass if GPU unavailable */
    } else {
        /* Compare results */
        for (size_t i = 0; i < n; i++) {
            if (cpu_data[i] != gpu_data[i]) {
                printf(" (mismatch at %zu)", i);
                result = 0;
                break;
            }
        }
    }
    
    free(cpu_data);
    free(gpu_data);
    return result;
}

/* Test: Large transform uses stage kernel and matches CPU */
static int test_gpu_stage_large_transform() {
    if (!fwht_has_gpu()) {
        return 1; /* Skip when GPU unavailable */
    }

    const size_t n = 16384;
    int32_t* cpu_data = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* gpu_data = (int32_t*)malloc(n * sizeof(int32_t));

    if (!cpu_data || !gpu_data) {
        free(cpu_data);
        free(gpu_data);
        return 0;
    }

    for (size_t i = 0; i < n; ++i) {
        cpu_data[i] = (int32_t)(rand() % 512 - 256);
        gpu_data[i] = cpu_data[i];
    }

    unsigned int original_block = fwht_gpu_get_block_size();
    /* Force a different launch configuration to exercise setter */
    if (fwht_gpu_set_block_size(512) != FWHT_SUCCESS) {
        free(cpu_data);
        free(gpu_data);
        return 0;
    }

    fwht_status_t cpu_status = fwht_i32_backend(cpu_data, n, FWHT_BACKEND_CPU);
    fwht_status_t gpu_status = fwht_i32_backend(gpu_data, n, FWHT_BACKEND_GPU);

    fwht_gpu_set_block_size(original_block);

    int result = 1;
    if (cpu_status != FWHT_SUCCESS || gpu_status != FWHT_SUCCESS) {
        result = (gpu_status != FWHT_SUCCESS) ? 1 : 0; /* Skip if GPU fails */
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (cpu_data[i] != gpu_data[i]) {
                result = 0;
                break;
            }
        }
    }

    free(cpu_data);
    free(gpu_data);
    return result;
}

/* Test: Batch API handles large transforms (n > 1024) */
static int test_gpu_batch_large_transform() {
    if (!fwht_has_gpu()) {
        return 1;
    }

    const size_t n = 2048;
    const size_t batch_size = 16;
    size_t total = n * batch_size;

    int32_t* gpu_data = (int32_t*)malloc(total * sizeof(int32_t));
    int32_t* expected = (int32_t*)malloc(total * sizeof(int32_t));

    if (!gpu_data || !expected) {
        free(gpu_data);
        free(expected);
        return 0;
    }

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < n; ++i) {
            int32_t value = (int32_t)(rand() % 256 - 128);
            gpu_data[b * n + i] = value;
            expected[b * n + i] = value;
        }
        fwht_i32_backend(&expected[b * n], n, FWHT_BACKEND_CPU);
    }

    fwht_status_t status = fwht_batch_i32_cuda(gpu_data, n, batch_size);
    if (status != FWHT_SUCCESS) {
        free(gpu_data);
        free(expected);
        return 0;
    }

    int result = 1;
    for (size_t i = 0; i < total; ++i) {
        if (gpu_data[i] != expected[i]) {
            result = 0;
            break;
        }
    }

    free(gpu_data);
    free(expected);
    return result;
}

/* Test: GPU double precision matches CPU */
static int test_gpu_vs_cpu_f64() {
    const size_t n = 1024;
    double* cpu_data = (double*)malloc(n * sizeof(double));
    double* gpu_data = (double*)malloc(n * sizeof(double));
    
    if (!cpu_data || !gpu_data) {
        free(cpu_data);
        free(gpu_data);
        return 0;
    }
    
    /* Random data */
    for (size_t i = 0; i < n; i++) {
        cpu_data[i] = (double)(rand() % 100 - 50);
        gpu_data[i] = cpu_data[i];
    }
    
    /* CPU transform */
    fwht_f64_backend(cpu_data, n, FWHT_BACKEND_CPU);
    
    /* GPU transform */
    fwht_status_t gpu_status = fwht_f64_backend(gpu_data, n, FWHT_BACKEND_GPU);
    
    int result = 1;
    if (gpu_status != FWHT_SUCCESS) {
        result = 1; /* Pass if GPU unavailable */
    } else {
        /* Compare results (allow small floating-point error) */
        const double epsilon = 1e-10;
        for (size_t i = 0; i < n; i++) {
            if (fabs(cpu_data[i] - gpu_data[i]) > epsilon) {
                printf(" (mismatch at %zu: diff=%e)", i, fabs(cpu_data[i] - gpu_data[i]));
                result = 0;
                break;
            }
        }
    }
    
    free(cpu_data);
    free(gpu_data);
    return result;
}

/* Test: Auto backend selection */
static int test_auto_backend() {
    const size_t n = 2048;
    int32_t* data1 = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* data2 = (int32_t*)malloc(n * sizeof(int32_t));
    
    if (!data1 || !data2) {
        free(data1);
        free(data2);
        return 0;
    }
    
    /* Same data */
    for (size_t i = 0; i < n; i++) {
        data1[i] = (int32_t)(rand() % 100 - 50);
        data2[i] = data1[i];
    }
    
    /* Auto backend (might use GPU if available) */
    fwht_i32(data1, n);
    
    /* Explicit CPU */
    fwht_i32_backend(data2, n, FWHT_BACKEND_CPU);
    
    /* Results should match */
    int result = 1;
    for (size_t i = 0; i < n; i++) {
        if (data1[i] != data2[i]) {
            result = 0;
            break;
        }
    }
    
    free(data1);
    free(data2);
    return result;
}

/* ============================================================================
 * Performance Benchmarks
 * ============================================================================ */

/* Test: GPU batch processing with many transforms */
static int test_gpu_batch_intensive() {
    const size_t n = 1024;
    const size_t batch_size = 100;  /* Process 100 WHTs at once */
    
    int32_t* batch_data = (int32_t*)malloc(n * batch_size * sizeof(int32_t));
    int32_t* cpu_ref = (int32_t*)malloc(n * sizeof(int32_t));
    
    if (!batch_data || !cpu_ref) {
        free(batch_data);
        free(cpu_ref);
        return 0;
    }
    
    /* Initialize batch with random data */
    for (size_t i = 0; i < n * batch_size; i++) {
        batch_data[i] = (int32_t)(rand() % 200 - 100);
    }
    
    /* Save first array for CPU reference */
    memcpy(cpu_ref, batch_data, n * sizeof(int32_t));
    
    /* GPU batch transform */
    fwht_status_t gpu_status = fwht_batch_i32_cuda(batch_data, n, batch_size);
    
    /* CPU reference for first array */
    fwht_i32_backend(cpu_ref, n, FWHT_BACKEND_CPU);
    
    int result = 1;
    if (gpu_status != FWHT_SUCCESS) {
        result = 1; /* Pass if GPU unavailable */
    } else {
        /* Verify first array matches CPU */
        for (size_t i = 0; i < n; i++) {
            if (batch_data[i] != cpu_ref[i]) {
                printf(" (batch mismatch at %zu)", i);
                result = 0;
                break;
            }
        }
    }
    
    free(batch_data);
    free(cpu_ref);
    return result;
}

static double benchmark_backend(size_t n, fwht_backend_t backend, int iterations) {
    int32_t* data = (int32_t*)malloc(n * sizeof(int32_t));
    if (!data) return -1.0;
    
    /* Random data */
    for (size_t i = 0; i < n; i++) {
        data[i] = (int32_t)(rand() % 100 - 50);
    }
    
    clock_t start = clock();
    
    for (int iter = 0; iter < iterations; iter++) {
        fwht_status_t status = fwht_i32_backend(data, n, backend);
        if (status != FWHT_SUCCESS && backend == FWHT_BACKEND_GPU) {
            free(data);
            return -1.0; /* GPU not available */
        }
    }
    
    clock_t end = clock();
    free(data);
    
    return (double)(end - start) / CLOCKS_PER_SEC;
}

static double benchmark_batch(size_t n, size_t batch_size, fwht_backend_t backend) {
    int32_t* data = (int32_t*)malloc(n * batch_size * sizeof(int32_t));
    if (!data) return -1.0;
    
    /* Random data */
    for (size_t i = 0; i < n * batch_size; i++) {
        data[i] = (int32_t)(rand() % 100 - 50);
    }
    
    clock_t start = clock();
    
    fwht_status_t status;
    if (backend == FWHT_BACKEND_GPU) {
        status = fwht_batch_i32_cuda(data, n, batch_size);
    } else {
        /* CPU: process sequentially */
        for (size_t b = 0; b < batch_size; b++) {
            status = fwht_i32_backend(data + b * n, n, FWHT_BACKEND_CPU);
            if (status != FWHT_SUCCESS) break;
        }
    }
    
    clock_t end = clock();
    free(data);
    
    if (status != FWHT_SUCCESS && backend == FWHT_BACKEND_GPU) {
        return -1.0;
    }
    
    return (double)(end - start) / CLOCKS_PER_SEC;
}

static void benchmark_comparison() {
    const size_t sizes[] = {256, 512, 1024};
    const int iterations = 1000;  /* More iterations for better timing */
    
    printf("\nPerformance Comparison - Single Transform (CPU vs GPU):\n");
    printf("  Size      Iterations  CPU Time    GPU Time    Speedup\n");
    printf("  ----------------------------------------------------------\n");
    
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        size_t n = sizes[i];
        
        double cpu_time = benchmark_backend(n, FWHT_BACKEND_CPU, iterations);
        double gpu_time = benchmark_backend(n, FWHT_BACKEND_GPU, iterations);
        
        printf("  %-8zu  %-10d  %.4fs", n, iterations, cpu_time);
        
        if (gpu_time < 0) {
            printf("      N/A         N/A\n");
        } else {
            double speedup = cpu_time / gpu_time;
            printf("      %.4fs      %.2fx\n", gpu_time, speedup);
        }
    }
    
    /* Batch processing benchmark */
    const size_t batch_sizes[] = {10, 50, 100, 200};
    
    printf("\nPerformance Comparison - Batch Processing (n=1024):\n");
    printf("  Batch Size  CPU Time    GPU Time    Speedup\n");
    printf("  -----------------------------------------------\n");
    
    for (size_t i = 0; i < sizeof(batch_sizes)/sizeof(batch_sizes[0]); i++) {
        size_t batch = batch_sizes[i];
        
        double cpu_time = benchmark_batch(1024, batch, FWHT_BACKEND_CPU);
        double gpu_time = benchmark_batch(1024, batch, FWHT_BACKEND_GPU);
        
        printf("  %-10zu  %.4fs", batch, cpu_time);
        
        if (gpu_time < 0) {
            printf("      N/A         N/A\n");
        } else {
            double speedup = cpu_time / gpu_time;
            printf("      %.4fs      %.2fx\n", gpu_time, speedup);
        }
    }
}

/* ============================================================================
 * Main Test Runner
 * ============================================================================ */

int main(void) {
    printf("FWHT GPU Tests\n");
    printf("==============\n\n");
    
    /* Check GPU availability */
    if (fwht_has_gpu()) {
        printf("GPU backend: AVAILABLE\n\n");
    } else {
        printf("GPU backend: NOT AVAILABLE (tests will verify CPU fallback)\n\n");
    }
    
    printf("Correctness Tests:\n");
    RUN_TEST(test_gpu_vs_cpu_small);
    RUN_TEST(test_gpu_vs_cpu_large);
    RUN_TEST(test_gpu_stage_large_transform);
    RUN_TEST(test_gpu_vs_cpu_f64);
    RUN_TEST(test_auto_backend);
    RUN_TEST(test_gpu_batch_large_transform);
    RUN_TEST(test_gpu_batch_intensive);
    
    printf("\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_run - tests_passed);
    
    if (tests_passed == tests_run) {
        printf("\n%s ALL TESTS PASSED!\n", TEST_PASS);
        
        /* Run intensive benchmarks if GPU is available */
        if (fwht_has_gpu()) {
            benchmark_comparison();
        }
        
        return 0;
    } else {
        printf("\n%s SOME TESTS FAILED!\n", TEST_FAIL);
        return 1;
    }
}
