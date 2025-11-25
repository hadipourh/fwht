/*
 * Fast Walsh-Hadamard Transform - GPU Multi-Precision Example
 *
 * Demonstrates fwht_batch_*_cuda host APIs, Tensor Core fp16 device API,
 * and lightweight profiling metrics. Requires libfwht built with CUDA.
 *
 * Build (from repo root):
 *   nvcc -O3 -DUSE_CUDA -Iinclude -Llib \
 *        examples/example_gpu_multi_precision.cu -lfwht -o examples/example_gpu_multi_precision
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../include/fwht.h"

#ifdef USE_CUDA

static void fill_alt32(int32_t* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = (i & 1) ? -1 : 1;
    }
}

static void fill_alt_double(double* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = (i & 1) ? -1.0 : 1.0;
    }
}

static void fill_alt_float(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = (i & 1) ? -1.0f : 1.0f;
    }
}

static void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

int main(void) {
    if (!fwht_has_gpu()) {
        printf("No CUDA device detected; skipping multi-precision demo.\n");
        return 0;
    }

    const size_t n = 4096;
    const size_t batch = 8;
    const size_t total = n * batch;

    printf("============================================================\n");
    printf("GPU Multi-Precision Batch Example\n");
    printf("============================================================\n\n");

    printf("Device: %s (SM %u)\n", fwht_gpu_get_device_name(), fwht_gpu_get_compute_capability());
    printf("Batch size = %zu, n = %zu\n\n", batch, n);

    int32_t* host_i32 = (int32_t*)malloc(total * sizeof(int32_t));
    double* host_f64 = (double*)malloc(total * sizeof(double));
    float* host_f32 = (float*)malloc(total * sizeof(float));
    if (!host_i32 || !host_f64 || !host_f32) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    fill_alt32(host_i32, total);
    fill_alt_double(host_f64, total);
    fill_alt_float(host_f32, total);

    fwht_status_t status = fwht_batch_i32_cuda(host_i32, n, batch);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "fwht_batch_i32_cuda failed: %s\n", fwht_error_string(status));
        return 1;
    }
    printf("✓ fwht_batch_i32_cuda complete (WHT[0]=%d)\n", host_i32[0]);

    status = fwht_batch_f64_cuda(host_f64, n, batch);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "fwht_batch_f64_cuda failed: %s\n", fwht_error_string(status));
        return 1;
    }
    printf("✓ fwht_batch_f64_cuda complete (WHT[0]=%.1f)\n", host_f64[0]);

    fwht_gpu_set_profiling(true);
    status = fwht_batch_f32_cuda(host_f32, n, batch);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "fwht_batch_f32_cuda failed: %s\n", fwht_error_string(status));
        return 1;
    }
    fwht_gpu_metrics_t metrics = fwht_gpu_get_last_metrics();
    if (metrics.valid) {
        printf("✓ fwht_batch_f32_cuda complete (kernel %.3f ms, H2D %.3f ms, D2H %.3f ms)\n",
               metrics.kernel_ms, metrics.h2d_ms, metrics.d2h_ms);
    }

    /* Tensor Core / device-pointer path */
    const size_t fp16_bytes = total * sizeof(uint16_t);
    uint16_t* host_fp16 = (uint16_t*)malloc(fp16_bytes);
    if (!host_fp16) {
        fprintf(stderr, "Host fp16 allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < total; ++i) {
        host_fp16[i] = (i & 1) ? 0xBC00 : 0x3C00;  /* ±1 in IEEE fp16 */
    }

    uint16_t* d_in = NULL;
    uint16_t* d_out = NULL;
    check_cuda(cudaMalloc((void**)&d_in, fp16_bytes), "cudaMalloc d_in");
    check_cuda(cudaMalloc((void**)&d_out, fp16_bytes), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_in, host_fp16, fp16_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    int fp16_status = fwht_batch_f16_cuda_device(d_in, d_out, (unsigned int)n, (unsigned int)batch);
    if (fp16_status != 0) {
        fprintf(stderr, "fwht_batch_f16_cuda_device failed with code %d\n", fp16_status);
    } else {
        check_cuda(cudaMemcpy(host_fp16, d_out, fp16_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        printf("✓ fwht_batch_f16_cuda_device complete (first word = 0x%04x)\n", host_fp16[0]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(host_fp16);
    free(host_i32);
    free(host_f64);
    free(host_f32);

    printf("\nAll GPU batch examples finished successfully.\n");
    return 0;
}

#else
int main(void) {
    printf("Rebuild libfwht with USE_CUDA=1 to run this example.\n");
    return 0;
}
#endif
