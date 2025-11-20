#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/fwht.h"

int main() {
    size_t n = 16;
    size_t batch = 2;
    size_t total = n * batch;
    
    int32_t* cpu_buf = (int32_t*)malloc(total * sizeof(int32_t));
    float* gpu_fp32 = (float*)malloc(total * sizeof(float));
    
    // Initialize with simple values
    for (size_t i = 0; i < total; i++) {
        int32_t val = (int32_t)(i % 10);
        cpu_buf[i] = val;
        gpu_fp32[i] = (float)val;
    }
    
    printf("Input data (first batch):\n");
    for (size_t i = 0; i < n; i++) {
        printf("  [%zu] cpu=%d fp32=%.1f\n", i, cpu_buf[i], gpu_fp32[i]);
    }
    
    // CPU transform
    for (size_t b = 0; b < batch; ++b) {
        fwht_i32_backend(cpu_buf + b * n, n, FWHT_BACKEND_CPU);
    }
    
    // GPU transform
    float* d_fp32 = NULL;
    cudaMalloc((void**)&d_fp32, total * sizeof(float));
    cudaMemcpy(d_fp32, gpu_fp32, total * sizeof(float), cudaMemcpyHostToDevice);
    fwht_batch_f32_cuda_device(d_fp32, n, batch);
    cudaMemcpy(gpu_fp32, d_fp32, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_fp32);
    
    printf("\nOutput data (first batch):\n");
    for (size_t i = 0; i < n; i++) {
        printf("  [%zu] cpu=%d fp32=%.1f diff=%.1f\n", 
               i, cpu_buf[i], gpu_fp32[i], fabs((double)cpu_buf[i] - (double)gpu_fp32[i]));
    }
    
    // Check all
    double max_diff = 0.0;
    for (size_t i = 0; i < total; i++) {
        double diff = fabs((double)cpu_buf[i] - (double)gpu_fp32[i]);
        if (diff > max_diff) {
            max_diff = diff;
            printf("\nWorst diff at idx=%zu: cpu=%d gpu=%.1f diff=%.1f\n",
                   i, cpu_buf[i], gpu_fp32[i], diff);
        }
    }
    printf("\nMax difference: %.1f\n", max_diff);
    
    free(cpu_buf);
    free(gpu_fp32);
    return 0;
}
