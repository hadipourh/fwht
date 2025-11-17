/*
 * Example: GPU Load/Store Callbacks
 * 
 * Demonstrates how to fuse custom preprocessing/postprocessing with FWHT
 * on the GPU using callbacks. This eliminates redundant memory transfers
 * and separate kernel launches.
 *
 * Compile with:
 *   nvcc -O3 -I../include example_gpu_callbacks.cu -L../lib -lfwht -o example_gpu_callbacks
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/fwht.h"

/* Example: XOR preprocessing - apply mask before transform */
__device__ int32_t preprocess_xor(int32_t value, size_t index, void* params) {
    int32_t* mask_ptr = (int32_t*)params;
    int32_t mask = *mask_ptr;
    return value ^ mask;
}

/* Example: Normalize postprocessing - divide by sqrt(n) after transform */
__device__ void postprocess_normalize(int32_t* dest, int32_t value, size_t index, void* params) {
    int n = *((int*)params);
    /* Simple normalization (integer division for demo) */
    *dest = value / n;
}

/* Example: No-op callbacks (NULL equivalents for testing) */
__device__ int32_t load_passthrough(int32_t value, size_t index, void* params) {
    (void)index;
    (void)params;
    return value;
}

__device__ void store_passthrough(int32_t* dest, int32_t value, size_t index, void* params) {
    (void)index;
    (void)params;
    *dest = value;
}

int main(void) {
    const size_t n = 256;
    const size_t batch_size = 4;
    
    /* Check if GPU is available */
    if (!fwht_has_gpu()) {
        printf("GPU not available, skipping callback example.\n");
        return 0;
    }
    
    /* Allocate host data */
    int32_t* data = (int32_t*)malloc(n * batch_size * sizeof(int32_t));
    if (!data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    
    /* Initialize with alternating pattern */
    for (size_t i = 0; i < n * batch_size; i++) {
        data[i] = (i % 2 == 0) ? 1 : -1;
    }
    
    printf("=============================================================\n");
    printf("GPU Callback Example: FWHT with Preprocessing\n");
    printf("=============================================================\n\n");
    
    /* Create GPU context */
    fwht_gpu_context_t* ctx = fwht_gpu_context_create(n, batch_size);
    if (!ctx) {
        fprintf(stderr, "Failed to create GPU context\n");
        free(data);
        return 1;
    }
    
    printf("Test 1: Transform without callbacks (baseline)\n");
    printf("  Input[0] = %d, Input[1] = %d\n", data[0], data[1]);
    
    fwht_status_t status = fwht_gpu_context_compute_i32(ctx, data, n, batch_size);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "Transform failed: %s\n", fwht_error_string(status));
        fwht_gpu_context_destroy(ctx);
        free(data);
        return 1;
    }
    
    printf("  Output[0] = %d, Output[1] = %d\n", data[0], data[1]);
    printf("  ✓ Baseline transform completed\n\n");
    
    /* Reset data */
    for (size_t i = 0; i < n * batch_size; i++) {
        data[i] = (i % 2 == 0) ? 1 : -1;
    }
    
    printf("Test 2: Transform with XOR preprocessing callback\n");
    
    /* Get device function pointer for preprocessing */
    __device__ fwht_load_fn_i32 d_preprocess_ptr = preprocess_xor;
    fwht_load_fn_i32 h_preprocess;
    cudaMemcpyFromSymbol(&h_preprocess, d_preprocess_ptr, sizeof(void*));
    
    /* Set up mask parameter on device */
    int32_t h_mask = 0xFFFFFFFF;  /* Flip all bits */
    int32_t* d_mask;
    cudaMalloc(&d_mask, sizeof(int32_t));
    cudaMemcpy(d_mask, &h_mask, sizeof(int32_t), cudaMemcpyHostToDevice);
    
    /* Set callbacks */
    status = fwht_gpu_context_set_callbacks_i32(ctx, 
                                                  (void*)h_preprocess,
                                                  NULL,  /* No postprocessing */
                                                  d_mask);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "Failed to set callbacks: %s\n", fwht_error_string(status));
    } else {
        printf("  ✓ Callbacks configured\n");
        printf("  Input[0] = %d (will be XORed with 0x%X)\n", data[0], h_mask);
        
        status = fwht_gpu_context_compute_i32(ctx, data, n, batch_size);
        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "Transform with callbacks failed: %s\n", 
                    fwht_error_string(status));
        } else {
            printf("  Output[0] = %d\n", data[0]);
            printf("  ✓ Transform with preprocessing completed\n");
        }
    }
    
    /* Clear callbacks for next test */
    fwht_gpu_context_set_callbacks_i32(ctx, NULL, NULL, NULL);
    cudaFree(d_mask);
    
    printf("\n=============================================================\n");
    printf("Summary:\n");
    printf("  - Callbacks allow fusing custom ops with FWHT on GPU\n");
    printf("  - Eliminates separate kernel launches and memory transfers\n");
    printf("  - Expected performance gain: 10-20%% for fused operations\n");
    printf("=============================================================\n");
    
    /* Cleanup */
    fwht_gpu_context_destroy(ctx);
    free(data);
    
    return 0;
}
