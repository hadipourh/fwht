/*
 * Fast Walsh-Hadamard Transform - CUDA GPU Implementation
 *
 * Simple, correct implementation that mirrors the CPU butterfly algorithm.
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

#include "../include/fwht.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>
#include <limits>

struct fwht_cuda_device_state {
    cudaDeviceProp props;
    bool initialized;
};

/* Maximum threads per block (CUDA architectural limit) */
#define MAX_THREADS_PER_BLOCK 1024
/* Maximum grid.y size that is widely supported */
#define CUDA_BATCH_LIMIT 65535u

static fwht_cuda_device_state g_cuda_device_state = {{}, false};

/* Global configuration: stage kernel block size (0 => auto) */
static unsigned int g_fwht_block_override = 0;

static fwht_status_t fwht_cuda_report(cudaError_t err, const char* file, int line) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
    return FWHT_ERROR_CUDA;
}

static fwht_status_t fwht_cuda_ensure_device_state(void) {
    if (g_cuda_device_state.initialized) {
        return FWHT_SUCCESS;
    }

    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaErrorNoDevice) {
        return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }

    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }

    g_cuda_device_state.props = props;
    g_cuda_device_state.initialized = true;
    return FWHT_SUCCESS;
}

static unsigned int fwht_cuda_warp_size(void) {
    if (g_cuda_device_state.initialized && g_cuda_device_state.props.warpSize > 0) {
        return static_cast<unsigned int>(g_cuda_device_state.props.warpSize);
    }
    return 32u;
}

static unsigned int fwht_cuda_max_threads_per_block(void) {
    if (g_cuda_device_state.initialized && g_cuda_device_state.props.maxThreadsPerBlock > 0) {
        return static_cast<unsigned int>(g_cuda_device_state.props.maxThreadsPerBlock);
    }
    return MAX_THREADS_PER_BLOCK;
}

static unsigned int fwht_cuda_sm_count(void) {
    if (g_cuda_device_state.initialized && g_cuda_device_state.props.multiProcessorCount > 0) {
        return static_cast<unsigned int>(g_cuda_device_state.props.multiProcessorCount);
    }
    return 1u;
}

static unsigned int fwht_cuda_max_grid_x(void) {
    if (g_cuda_device_state.initialized && g_cuda_device_state.props.maxGridSize[0] > 0) {
        return static_cast<unsigned int>(g_cuda_device_state.props.maxGridSize[0]);
    }
    return 2147483647u; /* CUDA runtime guarantees at least this on modern devices */
}

/* CUDA error checking */
#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        return fwht_cuda_report(err__, __FILE__, __LINE__); \
    } \
} while(0)

/**
 * CUDA kernel for Walsh-Hadamard Transform (int32_t)
 * 
 * Simple butterfly algorithm in shared memory.
 * Each block processes one WHT independently.
 */
__global__ void fwht_kernel_i32(int32_t* __restrict__ data, int n) {
    extern __shared__ int32_t shared_i32[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * n;
    
    /* Load data into shared memory */
    if (tid < n) {
        shared_i32[tid] = data[block_offset + tid];
    }
    __syncthreads();
    
    /* Butterfly stages */
    for (int h = 1; h < n; h *= 2) {
        int mask = (h << 1) - 1;
        int i = tid & ~mask;  /* Round down to multiple of 2*h */
        int j = i + (tid & (h - 1));
        
        if (tid < n && (tid & h) == 0) {
            int a = shared_i32[j];
            int b = shared_i32[j + h];
            shared_i32[j] = a + b;
            shared_i32[j + h] = a - b;
        }
        __syncthreads();
    }
    
    /* Write back to global memory */
    if (tid < n) {
        data[block_offset + tid] = shared_i32[tid];
    }
}

/**
 * CUDA kernel for Walsh-Hadamard Transform (double)
 */
__global__ void fwht_kernel_f64(double* __restrict__ data, int n) {
    extern __shared__ double shared_f64[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * n;
    
    /* Load data into shared memory */
    if (tid < n) {
        shared_f64[tid] = data[block_offset + tid];
    }
    __syncthreads();
    
    /* Butterfly stages */
    for (int h = 1; h < n; h *= 2) {
        int mask = (h << 1) - 1;
        int i = tid & ~mask;
        int j = i + (tid & (h - 1));
        
        if (tid < n && (tid & h) == 0) {
            double a = shared_f64[j];
            double b = shared_f64[j + h];
            shared_f64[j] = a + b;
            shared_f64[j + h] = a - b;
        }
        __syncthreads();
    }
    
    /* Write back to global memory */
    if (tid < n) {
        data[block_offset + tid] = shared_f64[tid];
    }
}

/* ============================================================================
 * Stage kernels and helpers for large transforms
 * ============================================================================ */

template <typename T>
__global__ void fwht_stage_kernel(T* __restrict__ data,
                                  size_t n,
                                  size_t h,
                                  size_t pairs_per_transform,
                                  size_t batch_size) {
    size_t transform_idx = blockIdx.y;
    if (transform_idx >= batch_size) {
        return;
    }

    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (stride == 0) {
        stride = blockDim.x;
    }

    unsigned long long h_ll = static_cast<unsigned long long>(h);

    for (; pair_idx < pairs_per_transform; pair_idx += stride) {
        unsigned long long pair_ll = static_cast<unsigned long long>(pair_idx);
        unsigned long long block = pair_ll / h_ll;
        unsigned long long offset = pair_ll - block * h_ll;
        unsigned long long base = static_cast<unsigned long long>(transform_idx) * n
                                + block * (h_ll << 1) + offset;

        T a = data[base];
        T b = data[base + h];
        data[base]     = a + b;
        data[base + h] = a - b;
    }
}

static unsigned int fwht_effective_block_size(size_t pairs_per_transform) {
    if (pairs_per_transform == 0) {
        return 1u;
    }

    unsigned int override = g_fwht_block_override;
    unsigned int max_unsigned = std::numeric_limits<unsigned int>::max();
    size_t capped_pairs = std::min(pairs_per_transform, static_cast<size_t>(max_unsigned));
    unsigned int limit = std::min<unsigned int>(fwht_cuda_max_threads_per_block(),
        static_cast<unsigned int>(capped_pairs));

    if (limit == 0) {
        return 1u;
    }

    unsigned int block_size = override;
    if (block_size == 0) {
        unsigned int warp = fwht_cuda_warp_size();
        unsigned int candidate = 1;
        while ((candidate << 1) <= limit) {
            candidate <<= 1;
        }
        block_size = candidate;
        if (block_size < warp && limit >= warp) {
            block_size = warp;
        }
    }

    if (block_size > limit) {
        block_size = limit;
    }

    unsigned int warp = fwht_cuda_warp_size();
    if (block_size > warp) {
        block_size = (block_size / warp) * warp;
    }
    if (block_size == 0) {
        block_size = std::min<unsigned int>(warp, limit);
        if (block_size == 0) {
            block_size = 1u;
        }
    }

    return block_size;
}

template <typename T>
static fwht_status_t fwht_launch_small(T* d_data, size_t n, size_t batch_size);

template <>
fwht_status_t fwht_launch_small<int32_t>(int32_t* d_data, size_t n, size_t batch_size) {
    unsigned int max_threads = fwht_cuda_max_threads_per_block();
    if (n > max_threads) {
        return FWHT_ERROR_INVALID_SIZE;
    }
    size_t processed = 0;
    size_t shared_bytes = n * sizeof(int32_t);
    unsigned int threads = static_cast<unsigned int>(n);

    while (processed < batch_size) {
        unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                               ? CUDA_BATCH_LIMIT
                               : static_cast<unsigned int>(batch_size - processed);
        fwht_kernel_i32<<<current, threads, shared_bytes>>>(d_data + processed * n, static_cast<int>(n));
        CUDA_CHECK(cudaGetLastError());
        processed += current;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return FWHT_SUCCESS;
}

template <>
fwht_status_t fwht_launch_small<double>(double* d_data, size_t n, size_t batch_size) {
    unsigned int max_threads = fwht_cuda_max_threads_per_block();
    if (n > max_threads) {
        return FWHT_ERROR_INVALID_SIZE;
    }
    size_t processed = 0;
    size_t shared_bytes = n * sizeof(double);
    unsigned int threads = static_cast<unsigned int>(n);

    while (processed < batch_size) {
        unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                               ? CUDA_BATCH_LIMIT
                               : static_cast<unsigned int>(batch_size - processed);
        fwht_kernel_f64<<<current, threads, shared_bytes>>>(d_data + processed * n, static_cast<int>(n));
        CUDA_CHECK(cudaGetLastError());
        processed += current;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return FWHT_SUCCESS;
}

template <typename T>
static fwht_status_t fwht_launch_large(T* d_data, size_t n, size_t batch_size) {
    size_t pairs_per_transform = n >> 1;
    unsigned int threads = fwht_effective_block_size(pairs_per_transform);
    unsigned long long work_items = pairs_per_transform;
    unsigned int blocks_x = (threads == 0)
        ? 1u
        : static_cast<unsigned int>((work_items + threads - 1) / threads);

    if (blocks_x == 0) {
        blocks_x = 1;
    }

    unsigned int min_blocks = fwht_cuda_sm_count() * 4u;
    unsigned int max_grid_x = fwht_cuda_max_grid_x();

    if (min_blocks == 0) {
        min_blocks = 1;
    }

    unsigned int max_useful_blocks = static_cast<unsigned int>(std::min<size_t>(
        std::max<size_t>(1, pairs_per_transform),
        static_cast<size_t>(max_grid_x > 0 ? max_grid_x : std::numeric_limits<unsigned int>::max())));

    if (max_useful_blocks > 0 && min_blocks > max_useful_blocks) {
        min_blocks = max_useful_blocks;
    }

    unsigned int target_blocks = std::max(blocks_x, min_blocks);
    if (max_useful_blocks > 0 && target_blocks > max_useful_blocks) {
        target_blocks = max_useful_blocks;
    }

    if (max_grid_x > 0 && target_blocks > max_grid_x) {
        target_blocks = max_grid_x;
    }

    if (target_blocks == 0) {
        target_blocks = 1;
    }

    blocks_x = target_blocks;

    for (size_t h = 1; h < n; h <<= 1) {
        size_t processed = 0;
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            dim3 grid(blocks_x, current);
            fwht_stage_kernel<T><<<grid, threads>>>(d_data + processed * n,
                                                    n,
                                                    h,
                                                    pairs_per_transform,
                                                    current);
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return FWHT_SUCCESS;
}

template <typename T>
static fwht_status_t fwht_execute_cuda(T* data, size_t n, size_t batch_size) {
    if (batch_size == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    fwht_status_t init_status = fwht_cuda_ensure_device_state();
    if (init_status != FWHT_SUCCESS) {
        return init_status;
    }

    if (n > 0 && batch_size > (SIZE_MAX / n)) {
        return FWHT_ERROR_INVALID_SIZE;
    }

    size_t element_count = n * batch_size;
    if (element_count > SIZE_MAX / sizeof(T)) {
        return FWHT_ERROR_INVALID_SIZE;
    }
    size_t bytes = element_count * sizeof(T);

    T* d_data = NULL;
    fwht_status_t status = FWHT_SUCCESS;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_data), bytes);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }

    err = cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        status = fwht_cuda_report(err, __FILE__, __LINE__);
        goto cleanup;
    }

    if (n <= fwht_cuda_max_threads_per_block()) {
        status = fwht_launch_small<T>(d_data, n, batch_size);
    } else {
        status = fwht_launch_large<T>(d_data, n, batch_size);
    }

    if (status != FWHT_SUCCESS) {
        goto cleanup;
    }

    err = cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        status = fwht_cuda_report(err, __FILE__, __LINE__);
        goto cleanup;
    }

cleanup:
    if (d_data != NULL) {
        cudaError_t free_err = cudaFree(d_data);
        if (free_err != cudaSuccess && status == FWHT_SUCCESS) {
            status = fwht_cuda_report(free_err, __FILE__, __LINE__);
        }
    }

    return status;
}

/* ============================================================================
 * Host Functions (C linkage for interoperability)
 * ============================================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/* GPU launch configuration */
fwht_status_t fwht_gpu_set_block_size(unsigned int block_size) {
    if (block_size == 0) {
        g_fwht_block_override = 0;
        return FWHT_SUCCESS;
    }

    if (block_size > MAX_THREADS_PER_BLOCK) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    if ((block_size & (block_size - 1)) != 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    g_fwht_block_override = block_size;
    return FWHT_SUCCESS;
}

unsigned int fwht_gpu_get_block_size(void) {
    return g_fwht_block_override;
}

fwht_status_t fwht_i32_cuda(int32_t* data, size_t n) {
    if (data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    return fwht_execute_cuda<int32_t>(data, n, 1);
}

fwht_status_t fwht_f64_cuda(double* data, size_t n) {
    if (data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    return fwht_execute_cuda<double>(data, n, 1);
}

fwht_status_t fwht_batch_i32_cuda(int32_t* data, size_t n, size_t batch_size) {
    if (data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda<int32_t>(data, n, batch_size);
}

fwht_status_t fwht_batch_f64_cuda(double* data, size_t n, size_t batch_size) {
    if (data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda<double>(data, n, batch_size);
}

/* Context-based API (stub - not implemented yet) */
struct fwht_context {
    void* device_buffer;
    size_t buffer_size;
    size_t max_n;
    fwht_backend_t backend;
};

fwht_context_t* fwht_create_context_cuda(size_t max_n, fwht_backend_t backend) {
    (void)max_n;
    (void)backend;
    return NULL;  /* Not implemented */
}

void fwht_destroy_context_cuda(fwht_context_t* ctx) {
    (void)ctx;
}

fwht_status_t fwht_transform_i32_cuda(fwht_context_t* ctx, int32_t* data, size_t n) {
    (void)ctx;
    (void)data;
    (void)n;
    return FWHT_ERROR_BACKEND_UNAVAILABLE;
}

fwht_status_t fwht_transform_f64_cuda(fwht_context_t* ctx, double* data, size_t n) {
    (void)ctx;
    (void)data;
    (void)n;
    return FWHT_ERROR_BACKEND_UNAVAILABLE;
}

#ifdef __cplusplus
}
#endif
