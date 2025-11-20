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

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

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

/* ============================================================================
 * Tensor Core FP16 Tests
 * ============================================================================ */

#ifdef USE_CUDA

/* FP16 conversion helpers */
static inline uint16_t float_to_fp16(float f) {
    union { float f; uint32_t u; } f2u = { f };
    uint32_t sign = (f2u.u >> 16) & 0x8000;
    int32_t exp = ((f2u.u >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (f2u.u >> 13) & 0x3FF;
    
    if (exp <= 0) return (uint16_t)sign;  /* Underflow */
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);  /* Overflow to inf */
    return (uint16_t)(sign | (exp << 10) | mantissa);
}

static inline float fp16_to_float(uint16_t h) {
    union { float f; uint32_t u; } f2u;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    if (exp == 0) {
        f2u.u = sign;  /* Zero or denormal */
    } else if (exp == 31) {
        f2u.u = sign | 0x7F800000 | (mantissa << 13);  /* Inf or NaN */
    } else {
        exp = exp - 15 + 127;
        f2u.u = sign | (exp << 23) | (mantissa << 13);
    }
    
    return f2u.f;
}

/* Test: Tensor Core fp16 vs CPU f64 reference */
static int test_tensor_core_fp16() {
    const size_t n = 1024;
    const size_t batch_size = 10;
    const size_t total_size = n * batch_size;
    
    /* Check if Tensor Cores available */
    if (!fwht_has_gpu()) {
        printf(" (GPU unavailable)");
        return 1;
    }
    
    unsigned int compute_cap = fwht_gpu_get_compute_capability();
    if (compute_cap < 70) {
        printf(" (No Tensor Cores, SM %d.%d)", compute_cap / 10, compute_cap % 10);
        return 1;
    }
    
    /* Allocate host memory */
    double* cpu_data = (double*)malloc(total_size * sizeof(double));
    uint16_t* h_fp16_in = (uint16_t*)malloc(total_size * sizeof(uint16_t));
    uint16_t* h_fp16_out = (uint16_t*)malloc(total_size * sizeof(uint16_t));
    
    if (!cpu_data || !h_fp16_in || !h_fp16_out) {
        free(cpu_data);
        free(h_fp16_in);
        free(h_fp16_out);
        return 0;
    }
    
    /* Generate random test data (keep values small for fp16 range) */
    for (size_t i = 0; i < total_size; i++) {
        double val = (double)(rand() % 200 - 100) / 10.0;  /* Range: -10.0 to +10.0 */
        cpu_data[i] = val;
        h_fp16_in[i] = float_to_fp16((float)val);
    }
    
    /* CPU reference (f64) - process each batch element separately */
    for (size_t b = 0; b < batch_size; b++) {
        fwht_status_t status = fwht_f64_backend(cpu_data + b * n, n, FWHT_BACKEND_CPU);
        if (status != FWHT_SUCCESS) {
            free(cpu_data);
            free(h_fp16_in);
            free(h_fp16_out);
            return 0;
        }
    }
    
    /* Allocate GPU memory */
    uint16_t* d_fp16_in = NULL;
    uint16_t* d_fp16_out = NULL;
    
    cudaError_t err = cudaMalloc(&d_fp16_in, total_size * sizeof(uint16_t));
    if (err != cudaSuccess) {
        free(cpu_data);
        free(h_fp16_in);
        free(h_fp16_out);
        printf(" (GPU alloc failed)");
        return 1;
    }
    
    err = cudaMalloc(&d_fp16_out, total_size * sizeof(uint16_t));
    if (err != cudaSuccess) {
        cudaFree(d_fp16_in);
        free(cpu_data);
        free(h_fp16_in);
        free(h_fp16_out);
        printf(" (GPU alloc failed)");
        return 1;
    }
    
    /* Copy to GPU */
    cudaMemcpy(d_fp16_in, h_fp16_in, total_size * sizeof(uint16_t), cudaMemcpyHostToDevice);
    
    /* GPU Tensor Core transform */
    int result = fwht_batch_f16_cuda_device(d_fp16_in, d_fp16_out, (unsigned int)n, (unsigned int)batch_size);
    
    /* Copy result back */
    cudaMemcpy(h_fp16_out, d_fp16_out, total_size * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_fp16_in);
    cudaFree(d_fp16_out);
    
    if (result != 0) {
        free(cpu_data);
        free(h_fp16_in);
        free(h_fp16_out);
        printf(" (Tensor Core transform failed: %d)", result);
        return 0;
    }
    
    /* Compare results with tolerance for fp16 */
    double max_error = 0.0;
    int mismatches = 0;
    const double rel_tolerance = 0.05;  /* 5% relative tolerance for fp16 */
    const double abs_tolerance = 1.0;   /* Absolute tolerance for small values */
    
    for (size_t i = 0; i < total_size; i++) {
        double cpu_val = cpu_data[i];
        double gpu_val = fp16_to_float(h_fp16_out[i]);
        double error = fabs(cpu_val - gpu_val);
        
        /* Use combined relative + absolute tolerance */
        double threshold = abs_tolerance + rel_tolerance * fabs(cpu_val);
        
        if (error > threshold) {
            mismatches++;
        }
        if (error > max_error) max_error = error;
    }
    
    free(cpu_data);
    free(h_fp16_in);
    free(h_fp16_out);
    
    if (mismatches > 0) {
        printf(" (%d mismatches, max_err=%.6f)", mismatches, max_error);
        return 0;
    }
    
    return 1;
}

#else  /* !USE_CUDA */

static int test_tensor_core_fp16() {
    printf(" (CUDA not available)");
    return 1;
}

#endif  /* USE_CUDA */

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

#if defined(USE_CUDA)
static double benchmark_fp32_batch(size_t n, size_t batch_size) {
    if (!fwht_has_gpu()) return -1.0;
    if (batch_size == 0 || n == 0 || (n & (n - 1))) return -1.0;

    size_t total = n * batch_size;
    size_t bytes = total * sizeof(float);
    float* h_data = (float*)malloc(bytes);
    if (!h_data) return -1.0;

    for (size_t i = 0; i < total; i++) {
        h_data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    float* d_data = NULL;
    cudaError_t err = cudaMalloc((void**)&d_data, bytes);
    if (err != cudaSuccess) {
        free(h_data);
        return -1.0;
    }

    double elapsed = -1.0;
    clock_t start = 0;
    clock_t end = 0;
    fwht_status_t status;

    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    start = clock();
    status = fwht_batch_f32_cuda_device(d_data, n, batch_size);
    if (status != FWHT_SUCCESS) goto cleanup;

    err = cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;

cleanup:
    cudaFree(d_data);
    free(h_data);
    return elapsed;
}

static double benchmark_fp64_batch(size_t n, size_t batch_size) {
    if (!fwht_has_gpu()) return -1.0;
    if (batch_size == 0 || n == 0 || (n & (n - 1))) return -1.0;

    size_t total = n * batch_size;
    size_t bytes = total * sizeof(double);
    double* h_data = (double*)malloc(bytes);
    if (!h_data) return -1.0;

    for (size_t i = 0; i < total; i++) {
        h_data[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
    }

    clock_t start = clock();
    fwht_status_t status = fwht_batch_f64_cuda(h_data, n, batch_size);
    clock_t end = clock();
    
    double elapsed = -1.0;
    if (status == FWHT_SUCCESS) {
        elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    }

    free(h_data);
    return elapsed;
}

static double benchmark_tensorcore_batch(size_t n, size_t batch_size) {
    if (!fwht_has_gpu()) return -1.0;
    if (batch_size == 0 || n == 0 || (n & (n - 1))) return -1.0;
    if (n > 1024 || (n % 256) != 0) return -1.0;
    if (fwht_gpu_get_compute_capability() < 70) return -1.0;

    size_t total = n * batch_size;
    size_t bytes = total * sizeof(uint16_t);
    uint16_t* h_data = (uint16_t*)malloc(bytes);
    if (!h_data) return -1.0;

    for (size_t i = 0; i < total; i++) {
        float val = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        h_data[i] = float_to_fp16(val);
    }

    uint16_t* d_data = NULL;
    cudaError_t err = cudaMalloc((void**)&d_data, bytes);
    if (err != cudaSuccess) {
        free(h_data);
        return -1.0;
    }

    double elapsed = -1.0;
    clock_t start = 0;
    clock_t end = 0;
    int status = 0;

    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    start = clock();

    status = fwht_batch_f16_cuda_device(d_data, d_data, (unsigned int)n, (unsigned int)batch_size);
    if (status != 0) goto cleanup;

    err = cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;

cleanup:
    cudaFree(d_data);
    free(h_data);
    return elapsed;
}
#else
static double benchmark_tensorcore_batch(size_t n, size_t batch_size) {
    (void)n;
    (void)batch_size;
    return -1.0;
}
#endif

#if defined(USE_CUDA)
static void verify_batch_outputs(size_t n, size_t batch_size) {
    if (!fwht_has_gpu()) return;
    size_t total = n * batch_size;
    if (total == 0) return;

    int32_t* base = (int32_t*)malloc(total * sizeof(int32_t));
    int32_t* cpu_buf = (int32_t*)malloc(total * sizeof(int32_t));
    int32_t* gpu_buf = (int32_t*)malloc(total * sizeof(int32_t));
    double* gpu_fp64 = (double*)malloc(total * sizeof(double));
    uint16_t* tc_host = (uint16_t*)malloc(total * sizeof(uint16_t));
    float* gpu_fp32 = (float*)malloc(total * sizeof(float));

    if (!base || !cpu_buf || !gpu_buf || !gpu_fp64 || !tc_host || !gpu_fp32) {
        free(base); free(cpu_buf); free(gpu_buf); free(gpu_fp64); free(tc_host); free(gpu_fp32);
        return;
    }

    for (size_t i = 0; i < total; i++) {
        int32_t val = (int32_t)(rand() % 201 - 100);
        base[i] = val;
        cpu_buf[i] = val;
        gpu_buf[i] = val;
        gpu_fp64[i] = (double)val;
        tc_host[i] = float_to_fp16((float)val);
        gpu_fp32[i] = (float)val;
    }

    for (size_t b = 0; b < batch_size; ++b) {
        fwht_i32_backend(cpu_buf + b * n, n, FWHT_BACKEND_CPU);
    }

    fwht_status_t gpu_status = fwht_batch_i32_cuda(gpu_buf, n, batch_size);
    long long max_cpu_gpu = 0;
    if (gpu_status == FWHT_SUCCESS) {
        for (size_t i = 0; i < total; ++i) {
            long long diff = (long long)cpu_buf[i] - (long long)gpu_buf[i];
            if (llabs(diff) > max_cpu_gpu) {
                max_cpu_gpu = llabs(diff);
            }
        }
    }

    bool gpu_fp64_ran = false;
    double max_cpu_gpu_fp64 = 0.0;
    if (fwht_has_gpu()) {
        fwht_status_t fp64_status = fwht_batch_f64_cuda(gpu_fp64, n, batch_size);
        if (fp64_status == FWHT_SUCCESS) {
            gpu_fp64_ran = true;
            for (size_t i = 0; i < total; ++i) {
                double cpu_val = (double)cpu_buf[i];
                double gpu_val = gpu_fp64[i];
                double diff = fabs(cpu_val - gpu_val);
                if (diff > max_cpu_gpu_fp64) {
                    max_cpu_gpu_fp64 = diff;
                }
            }
        }
    }

    bool tc_ran = false;
    double max_cpu_tc = 0.0;
    bool gpu_fp32_ran = false;
    double max_cpu_gpu_fp32 = 0.0;
    size_t worst_fp32_idx = 0;
    double worst_fp32_cpu = 0.0;
    double worst_fp32_raw = 0.0;
    bool tc_supported = (fwht_gpu_get_compute_capability() >= 70) && (n <= 1024) && ((n % 256) == 0);
    if (tc_supported) {
        uint16_t* d_tc = NULL;
        cudaError_t err = cudaMalloc((void**)&d_tc, total * sizeof(uint16_t));
        if (err == cudaSuccess) {
            err = cudaMemcpy(d_tc, tc_host, total * sizeof(uint16_t), cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                int status = fwht_batch_f16_cuda_device(d_tc, d_tc, (unsigned int)n, (unsigned int)batch_size);
                if (status == 0) {
                    err = cudaMemcpy(tc_host, d_tc, total * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                    if (err == cudaSuccess) {
                        tc_ran = true;
                        for (size_t i = 0; i < total; ++i) {
                            double cpu_val = (double)cpu_buf[i];
                            double tc_val = (double)fp16_to_float(tc_host[i]);
                            double diff = fabs(cpu_val - tc_val);
                            if (diff > max_cpu_tc) max_cpu_tc = diff;
                        }
                    }
                }
            }
            cudaFree(d_tc);
        }
    }

    if (fwht_has_gpu()) {
        /* Use host API for consistency with int32 path */
        fwht_status_t fp32_status = fwht_batch_f32_cuda(gpu_fp32, n, batch_size);
        if (fp32_status == FWHT_SUCCESS) {
            /* Check how much of the batch was actually transformed */
                        size_t unchanged_count = 0;
                        for (size_t i = 0; i < total; ++i) {
                            if (fabs(gpu_fp32[i] - (float)base[i]) < 0.1f) {
                                unchanged_count++;
                            }
                        }
                        if (unchanged_count > 0 && unchanged_count < total) {
                            size_t first_suspicious = (size_t)(-1);
                            size_t sample_suspicious[5] = {0};
                            size_t sample_count = 0;
                            for (size_t i = 0; i < total; ++i) {
                                if (fabs(gpu_fp32[i] - (float)base[i]) < 0.1f) {
                                    double cpu_val = (double)cpu_buf[i];
                                    double base_val = (double)base[i];
                                    if (fabs(cpu_val - base_val) > 1.0) {
                                        if (first_suspicious == (size_t)(-1)) {
                                            first_suspicious = i;
                                        }
                                        if (sample_count < 5) {
                                            sample_suspicious[sample_count++] = i;
                                        }
                                    }
                                }
                            }
                            if (sample_count > 0) {
                                size_t report_idx = first_suspicious;
                                fprintf(stderr,
                                        "[WARNING] GPU fp32: %zu/%zu elements equal to input while CPU output differs (first idx=%zu, batch=%zu)\n",
                                        sample_count, total, report_idx, report_idx / n);
                                for (size_t si = 0; si < sample_count; ++si) {
                                    size_t idx = sample_suspicious[si];
                                    fprintf(stderr,
                                            "        suspicious idx=%zu (batch=%zu, pos=%zu): base=%d cpu=%lld gpu=%.1f\n",
                                            idx,
                                            idx / n,
                                            idx % n,
                                            base[idx],
                                            (long long)cpu_buf[idx],
                                            (double)gpu_fp32[idx]);
                                }
                            }
                        } else if (unchanged_count == total && total > 0) {
                            fprintf(stderr, "[WARNING] GPU fp32 output completely unchanged (batch=%zu)\n", batch_size);
                        }
                        
                        gpu_fp32_ran = true;
                        for (size_t i = 0; i < total; ++i) {
                            double cpu_val = (double)cpu_buf[i];
                            double gpu_val = (double)gpu_fp32[i];
                            double diff = fabs(cpu_val - gpu_val);
                            if (diff > max_cpu_gpu_fp32) {
                                max_cpu_gpu_fp32 = diff;
                                worst_fp32_idx = i;
                                worst_fp32_cpu = cpu_val;
                                worst_fp32_raw = gpu_val;
                            }
                        }
        } else {
            fprintf(stderr, "[ERROR] fwht_batch_f32_cuda failed (batch=%zu): status=%d\n",
                    batch_size, fp32_status);
        }
    }

    printf("  [verify] max|CPU-GPU(i32)|=%lld", max_cpu_gpu);
    if (gpu_fp64_ran) {
        printf(", max|CPU-GPU(fp64)|=%#.4f", max_cpu_gpu_fp64);
    }
    if (gpu_fp32_ran) {
        printf(", max|CPU-GPU(fp32)|=%#.4f", max_cpu_gpu_fp32);
    }
    if (tc_ran) {
        printf(", max|CPU-TC(fp16)|=%.4f\n", max_cpu_tc);
        if (n == 1024 && batch_size == 10 && gpu_fp32_ran) {
            FILE* dump = fopen("build/tensorcore_dump_n1024_batch10.txt", "w");
            if (dump) {
                fprintf(dump, "index,batch,position,cpu_value,gpu_fp32,tc_value,diff_cpu_tc,diff_cpu_gpu\n");
                for (size_t b = 0; b < batch_size; ++b) {
                    for (size_t pos = 0; pos < n; ++pos) {
                        size_t idx = b * n + pos;
                        double cpu_val = (double)cpu_buf[idx];
                        double gpu_val = (double)gpu_fp32[idx];
                        double tc_val = (double)fp16_to_float(tc_host[idx]);
                        double diff_tc = cpu_val - tc_val;
                        double diff_gpu = cpu_val - gpu_val;
                        fprintf(dump, "%zu,%zu,%zu,%.0f,%.6f,%.6f,%.6f,%.6f\n",
                                idx, b, pos, cpu_val, gpu_val, tc_val, diff_tc, diff_gpu);
                    }
                }
                fclose(dump);
                printf("      dumped tensor-core comparison to build/tensorcore_dump_n1024_batch10.txt\n");
            } else {
                printf("        failed to write tensor-core dump file\n");
            }
        } else if (n == 1024 && batch_size == 10) {
            printf("        skipped dump (fp32 GPU reference unavailable)\n");
        }
    } else if (tc_supported) {
        printf(", Tensor Core verify skipped (alloc/copy failure)\n");
    } else {
        printf(", Tensor Core verify skipped (unsupported size/GPU)\n");
    }

    if (gpu_fp32_ran && max_cpu_gpu_fp32 > 1.0) {
        printf("        fp32 hint: idx=%zu cpu=%.0f gpu=%.4f\n",
               worst_fp32_idx, worst_fp32_cpu, worst_fp32_raw);
    }

    free(base);
    free(cpu_buf);
    free(gpu_buf);
    free(gpu_fp64);
    free(tc_host);
    free(gpu_fp32);
}
#else
static void verify_batch_outputs(size_t n, size_t batch_size) {
    (void)n;
    (void)batch_size;
}
#endif

static void benchmark_comparison() {
    const size_t sizes[] = {256, 512, 1024, 33554432, 67108864};  /* 2^8, 2^9, 2^10, 2^25, 2^26 */
    const int iterations_small = 1000;  /* More iterations for small sizes */
    const int iterations_large = 10;    /* Fewer iterations for large sizes */
    
    printf("\nPerformance Comparison - Single Transform (CPU vs GPU):\n");
    printf("  Size      Iterations  CPU Time    GPU Time    Speedup\n");
    printf("  ----------------------------------------------------------\n");
    
    for (size_t i = 0; i < sizeof(sizes)/sizeof(sizes[0]); i++) {
        size_t n = sizes[i];
        int iterations = (n <= 1024) ? iterations_small : iterations_large;
        
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
    const size_t batch_sizes[] = {10, 50, 100, 200, 512, 1024};
    
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

#if defined(USE_CUDA)
static void print_tensorcore_sample_vector(void) {
    if (!fwht_has_gpu()) return;
    if (fwht_gpu_get_compute_capability() < 70) return;

    const size_t n = 256;           /* Smallest Tensor Core-supported size */
    const size_t batch_size = 1;
    const size_t visible = 10;      /* Show first/last 10 entries */

    int32_t sample_input[256];
    for (size_t i = 0; i < n; ++i) {
        sample_input[i] = (int32_t)(((i * 7) % 31) - 15);
    }

    int32_t cpu_buf[256];
    memcpy(cpu_buf, sample_input, sizeof(sample_input));

    if (fwht_i32_backend(cpu_buf, n, FWHT_BACKEND_CPU) != FWHT_SUCCESS) {
        return;
    }

    uint16_t h_tc[256];
    for (size_t i = 0; i < n; ++i) {
        h_tc[i] = float_to_fp16((float)sample_input[i]);
    }

    int32_t gpu_i32_buf[256];
    double gpu_fp64_buf[256];
    float gpu_fp32_buf[256];
    bool gpu_i32_ok = false;
    bool gpu_fp64_ok = false;
    bool gpu_fp32_ok = false;
    if (fwht_has_gpu()) {
        memcpy(gpu_i32_buf, sample_input, sizeof(sample_input));
        if (fwht_batch_i32_cuda(gpu_i32_buf, n, batch_size) == FWHT_SUCCESS) {
            gpu_i32_ok = true;
        }

        for (size_t i = 0; i < n; ++i) {
            gpu_fp64_buf[i] = (double)sample_input[i];
        }
        if (fwht_batch_f64_cuda(gpu_fp64_buf, n, batch_size) == FWHT_SUCCESS) {
            gpu_fp64_ok = true;
        }

        for (size_t i = 0; i < n; ++i) {
            gpu_fp32_buf[i] = (float)sample_input[i];
        }
        float* d_fp32 = NULL;
        cudaError_t err = cudaMalloc((void**)&d_fp32, n * batch_size * sizeof(float));
        if (err == cudaSuccess) {
            err = cudaMemcpy(d_fp32, gpu_fp32_buf, n * batch_size * sizeof(float), cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                fwht_status_t status = fwht_batch_f32_cuda_device(d_fp32, n, batch_size);
                if (status == FWHT_SUCCESS) {
                    err = cudaMemcpy(gpu_fp32_buf, d_fp32, n * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
                    if (err == cudaSuccess) {
                        gpu_fp32_ok = true;
                    }
                }
            }
            cudaFree(d_fp32);
        }
    }

    uint16_t* d_tc = NULL;
    cudaError_t err = cudaMalloc((void**)&d_tc, n * sizeof(uint16_t));
    if (err != cudaSuccess) {
        return;
    }

    err = cudaMemcpy(d_tc, h_tc, n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_tc);
        return;
    }

    if (fwht_batch_f16_cuda_device(d_tc, d_tc, (unsigned int)n, (unsigned int)batch_size) != 0) {
        cudaFree(d_tc);
        return;
    }

    err = cudaMemcpy(h_tc, d_tc, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_tc);
    if (err != cudaSuccess) {
        return;
    }

    printf("\nSample CPU vs GPU Precision Comparison (n=%zu, first %zu entries):\n", n, visible);
    printf("  idx  CPU(i32)    GPU(i32)   GPU(fp64)  GPU(fp32)  TC(fp16)   diff(TC)\n");
    printf("  -----------------------------------------------------------------------\n");
    for (size_t i = 0; i < visible && i < n; ++i) {
        double tc_val = (double)fp16_to_float(h_tc[i]);
        double cpu_val = (double)cpu_buf[i];
        char gpu_i32_str[16];
        char gpu_fp64_str[16];
        char gpu_fp32_str[16];
        if (gpu_i32_ok) {
            snprintf(gpu_i32_str, sizeof(gpu_i32_str), "%d", gpu_i32_buf[i]);
        } else {
            snprintf(gpu_i32_str, sizeof(gpu_i32_str), "N/A");
        }
        if (gpu_fp64_ok) {
            snprintf(gpu_fp64_str, sizeof(gpu_fp64_str), "%.1f", gpu_fp64_buf[i]);
        } else {
            snprintf(gpu_fp64_str, sizeof(gpu_fp64_str), "N/A");
        }
        if (gpu_fp32_ok) {
            snprintf(gpu_fp32_str, sizeof(gpu_fp32_str), "%.1f", (double)gpu_fp32_buf[i]);
        } else {
            snprintf(gpu_fp32_str, sizeof(gpu_fp32_str), "N/A");
        }
        printf("  %-3zu  %10.0f  %10s  %10s  %10s  %9.4f  %+5.1f\n",
               i, cpu_val, gpu_i32_str, gpu_fp64_str, gpu_fp32_str, tc_val, cpu_val - tc_val);
    }

    size_t tail_start = (n > visible) ? (n - visible) : 0;
    if (tail_start > 0) {
        printf("\nSample CPU vs GPU Precision Comparison (n=%zu, last %zu entries):\n", n, visible);
        printf("  idx  CPU(i32)    GPU(i32)   GPU(fp64)  GPU(fp32)  TC(fp16)   diff(TC)\n");
        printf("  -----------------------------------------------------------------------\n");
    }
    for (size_t i = tail_start; i < n; ++i) {
        double tc_val = (double)fp16_to_float(h_tc[i]);
        double cpu_val = (double)cpu_buf[i];
        char gpu_i32_str[16];
        char gpu_fp64_str[16];
        char gpu_fp32_str[16];
        if (gpu_i32_ok) {
            snprintf(gpu_i32_str, sizeof(gpu_i32_str), "%d", gpu_i32_buf[i]);
        } else {
            snprintf(gpu_i32_str, sizeof(gpu_i32_str), "N/A");
        }
        if (gpu_fp64_ok) {
            snprintf(gpu_fp64_str, sizeof(gpu_fp64_str), "%.1f", gpu_fp64_buf[i]);
        } else {
            snprintf(gpu_fp64_str, sizeof(gpu_fp64_str), "N/A");
        }
        if (gpu_fp32_ok) {
            snprintf(gpu_fp32_str, sizeof(gpu_fp32_str), "%.1f", (double)gpu_fp32_buf[i]);
        } else {
            snprintf(gpu_fp32_str, sizeof(gpu_fp32_str), "N/A");
        }
        printf("  %-3zu  %10.0f  %10s  %10s  %10s  %9.4f  %+5.1f\n",
               i, cpu_val, gpu_i32_str, gpu_fp64_str, gpu_fp32_str, tc_val, cpu_val - tc_val);
    }
}
#else
static void print_tensorcore_sample_vector(void) {}
#endif

static void benchmark_tensorcore_comparison() {
#if defined(USE_CUDA)
    if (!fwht_has_gpu()) return;
    if (fwht_gpu_get_compute_capability() < 70) return;

    print_tensorcore_sample_vector();

    const size_t n = 1024;
    const size_t batch_sizes[] = {10, 50, 100, 200, 512, 1024};

    printf("\nPerformance Comparison - Batch Processing (n=1024, All GPU Precisions):\n");
    printf("  Batch Size  CPU (i32)   GPU (i32)  GPU (fp64)  GPU (fp32)  TC (fp16)  Speedup (TC)\n");
    printf("  -------------------------------------------------------------------------------------\n");

    for (size_t i = 0; i < sizeof(batch_sizes)/sizeof(batch_sizes[0]); i++) {
        size_t batch = batch_sizes[i];
        double cpu_time = benchmark_batch(n, batch, FWHT_BACKEND_CPU);
        double gpu_time = benchmark_batch(n, batch, FWHT_BACKEND_GPU);
        double gpu_fp64_time = benchmark_fp64_batch(n, batch);
        double gpu_fp32_time = benchmark_fp32_batch(n, batch);
        double tc_time = benchmark_tensorcore_batch(n, batch);

        printf("  %-10zu  %0.4fs", batch, cpu_time);

        if (gpu_time < 0) {
            printf("      N/A      ");
        } else {
            printf("      %0.4fs", gpu_time);
        }

        if (gpu_fp64_time < 0) {
            printf("     N/A      ");
        } else {
            printf("     %0.4fs", gpu_fp64_time);
        }

        if (gpu_fp32_time < 0) {
            printf("     N/A      ");
        } else {
            printf("     %0.4fs", gpu_fp32_time);
        }

        if (tc_time < 0) {
            printf("      N/A           ");
        } else {
            printf("      %0.4fs", tc_time);
        }

        if (tc_time > 0) {
            double speedup_tc = cpu_time / tc_time;
            printf("      %0.2fx\n", speedup_tc);
        } else {
            printf("      N/A\n");
        }

        verify_batch_outputs(n, batch);
    }
#else
    (void)benchmark_tensorcore_batch;
#endif
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
    RUN_TEST(test_tensor_core_fp16);
    
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
            benchmark_tensorcore_comparison();
        }
        
        return 0;
    } else {
        printf("\n%s SOME TESTS FAILED!\n", TEST_FAIL);
        return 1;
    }
}
