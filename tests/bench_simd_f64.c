/*
 * Benchmark SIMD acceleration for float64 transforms
 * Compares performance across different sizes
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../include/fwht.h"

#define ITERATIONS 1000

void benchmark_size(size_t n) {
    double* data = malloc(n * sizeof(double));
    if (!data) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }
    
    // Initialize with random data
    srand(42);
    for (size_t i = 0; i < n; ++i) {
        data[i] = (double)(rand() % 1000) - 500.0;
    }
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        fwht_f64(data, n);
    }
    
    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; ++i) {
        fwht_f64(data, n);
    }
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double per_iter_ms = (elapsed * 1000.0) / ITERATIONS;
    double throughput = (n * ITERATIONS) / (elapsed * 1e6); // Million elements/sec
    
    printf("  n=%6zu: %8.3f µs/iter  %8.2f M elem/s\n", 
           n, per_iter_ms * 1000.0, throughput);
    
    free(data);
}

int main(void) {
    printf("============================================================\n");
    printf("Float64 SIMD Performance Benchmark\n");
    printf("============================================================\n");
    printf("SIMD Status: ");
    
    // Trigger SIMD banner
    double test[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    fwht_f64(test, 8);
    
    printf("\n");
    printf("Iterations per size: %d\n", ITERATIONS);
    printf("------------------------------------------------------------\n");
    
    size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    size_t n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (size_t i = 0; i < n_sizes; ++i) {
        benchmark_size(sizes[i]);
    }
    
    printf("============================================================\n");
    
    return 0;
}
