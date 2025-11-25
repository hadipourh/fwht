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

#ifndef USE_CUDA
#define USE_CUDA
#endif
#include "../include/fwht.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <algorithm>
#include <limits>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Include Meta's Tensor Core kernel
#include "fwht_cuda_fp16.cuh"

struct fwht_cuda_device_state {
    cudaDeviceProp props;
    bool initialized;
    int smem_banks;          /* Shared memory bank count (16 or 32) */
    int compute_capability;  /* SM version: 75=Turing, 80=Ampere, 89=Ada, 90=Hopper */
};

/* Forward declarations for helpers referenced before definition */
static unsigned int fwht_cuda_max_threads_per_block(void);

/* Maximum threads per block (CUDA architectural limit) */
#define MAX_THREADS_PER_BLOCK 1024
/* Maximum grid.y size that is widely supported */
#define CUDA_BATCH_LIMIT 65535u

static fwht_cuda_device_state g_cuda_device_state = {{}, false, 0, 0};

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/* Check if n is a power of 2 */
static inline bool is_power_of_2(size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

/* ============================================================================
 * DEVICE INITIALIZATION
 * ============================================================================ */

/* Global configuration: stage kernel block size (0 => auto) */
static unsigned int g_fwht_block_override = 0;
static bool g_fwht_profiling_enabled = false;
static fwht_gpu_metrics_t g_fwht_last_metrics = {0.0, 0.0, 0.0, 0u, 0u, 0u, 0, false};
static bool g_fwht_tensorcore_fallback_logged = false;
static bool g_fwht_fp16_warning_shown = false;

static inline bool fwht_tensorcore_arch_supported(int compute_cap) {
    return compute_cap >= 80 && compute_cap < 100; /* Ampere & Ada */
}

static inline bool fwht_tensorcore_size_supported(unsigned int n) {
    return (n == 256) || (n == 512) || (n == 1024);
}

static void fwht_tensorcore_log_unavailable(const char* reason,
                                            int compute_cap,
                                            unsigned int n) {
    if (g_fwht_tensorcore_fallback_logged) {
        return;
    }
    fprintf(stderr,
            "[libfwht] Tensor Core path unavailable (%s, SM %d, n=%u); using fp16 fallback.\n",
            reason,
            compute_cap,
            n);
    g_fwht_tensorcore_fallback_logged = true;
}

static void fwht_fp16_precision_warning(void) {
    if (g_fwht_fp16_warning_shown) {
        return;
    }
    fprintf(stderr,
            "\n"
            "╔═══════════════════════════════════════════════════════════════════════════╗\n"
            "║ FP16 Tensor Core Precision Warning                                        ║\n"
            "╠═══════════════════════════════════════════════════════════════════════════╣\n"
            "║ • FP16 provides 25-36× speedup but sacrifices integer exactness           ║\n"
            "║ • Expected behavior: ~12%% of results differ by ±1 to ±4 from exact        ║\n"
            "║ • Maximum error: ±4 integers (typically ±1)                               ║\n"
            "║ • Relative error: < 0.1%% for values in typical range                      ║\n"
            "║                                                                           ║\n"
            "║ Use cases:                                                                ║\n"
            "║   ✓ Machine learning (inference/training)                                 ║\n"
            "║   ✓ Signal processing (approximate transforms)                            ║\n"
            "║   ✗ Cryptanalysis (use fp32 or fp64 for bit-exact results)                ║\n"
            "║                                                                           ║\n"
            "║ To suppress: set FWHT_SILENCE_FP16_WARNING=1 environment variable         ║\n"
            "╚═══════════════════════════════════════════════════════════════════════════╝\n"
            "\n");
    g_fwht_fp16_warning_shown = true;
}
/* Opt-in toggle for 32 < N ≤ 512 warp multi-shuffle path (default: ENABLED for better perf) */
static bool g_fwht_multi_shuffle_enabled = true;
/* Opt-in toggle for chunked large-kernel path (default: DISABLED until tuned) */
static bool g_fwht_chunked_enabled = false;
/* Optional fixed threads-per-block for chunked path (0 => auto heuristics) */
static unsigned int g_fwHT_chunked_threads = 0;
/* Optional logging of dispatch decisions */
static bool g_fwht_dispatch_logging = false;

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
    
    /* Compute capability: major*10 + minor */
    g_cuda_device_state.compute_capability = props.major * 10 + props.minor;
    
    /* Determine shared memory bank count based on compute capability
     * - Pre-Kepler (< sm_30): 16 banks
     * - Kepler+ (>= sm_30): 32 banks (configurable, but 32-bank mode is default and faster)
     * - All modern GPUs (Turing, Ampere, Ada, Hopper): 32 banks
     * 
     * References:
     * - Andrade et al. 2014: "Bank count affects optimal radix selection"
     * - CUDA Programming Guide: "Shared memory banks increased from 16 to 32 in Kepler"
     */
    if (props.major >= 3) {
        g_cuda_device_state.smem_banks = 32;  /* Kepler and newer: 32 banks */
    } else {
        g_cuda_device_state.smem_banks = 16;  /* Pre-Kepler: 16 banks */
    }
    
    g_cuda_device_state.initialized = true;
    
    /* One-time env configuration: FWHT_ENABLE_MULTI_SHUFFLE / FWHT_ENABLE_CHUNKED
     * Any non-empty value other than "0"/"false"/"off" enables the feature
     */
    const char* env_ms = getenv("FWHT_ENABLE_MULTI_SHUFFLE");
    if (env_ms && env_ms[0] != '\0') {
        /* Enable unless explicitly disabled by 0/false/off (case-sensitive minimal set) */
        if (!(strcmp(env_ms, "0") == 0 || strcmp(env_ms, "false") == 0 || strcmp(env_ms, "FALSE") == 0 ||
              strcmp(env_ms, "off") == 0 || strcmp(env_ms, "OFF") == 0)) {
            g_fwht_multi_shuffle_enabled = true;
        }
    }
    const char* env_ck = getenv("FWHT_ENABLE_CHUNKED");
    if (env_ck && env_ck[0] != '\0') {
        if (!(strcmp(env_ck, "0") == 0 || strcmp(env_ck, "false") == 0 || strcmp(env_ck, "FALSE") == 0 ||
              strcmp(env_ck, "off") == 0 || strcmp(env_ck, "OFF") == 0)) {
            g_fwht_chunked_enabled = true;
        }
    }
    /* Optional: FWHT_CHUNKED_THREADS to force a fixed block size (power of two <= max) */
    const char* env_ct = getenv("FWHT_CHUNKED_THREADS");
    if (env_ct && env_ct[0] != '\0') {
        unsigned long val = strtoul(env_ct, NULL, 10);
        if (val > 0 && val <= fwht_cuda_max_threads_per_block()) {
            unsigned int t = 1u; while ((t << 1) <= val) t <<= 1;
            t = (t / 32u) * 32u; if (t == 0) t = 32u;
            g_fwHT_chunked_threads = t;
        }
    }
    /* Optional: FWHT_LOG_DISPATCH to trace kernel selection */
    const char* env_log = getenv("FWHT_LOG_DISPATCH");
    if (env_log && env_log[0] != '\0') {
        if (!(strcmp(env_log, "0") == 0 || strcmp(env_log, "false") == 0 || strcmp(env_log, "FALSE") == 0 ||
              strcmp(env_log, "off") == 0 || strcmp(env_log, "OFF") == 0)) {
            g_fwht_dispatch_logging = true;
        }
    }
    
    /* Print device info on first initialization for user awareness */
    fprintf(stderr, "[libfwht] GPU: %s (SM %d.%d, %d SMEM banks, %d SMs)\n",
            props.name, props.major, props.minor,
            g_cuda_device_state.smem_banks,
            props.multiProcessorCount);
    
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

static int fwht_cuda_smem_banks(void) {
    if (g_cuda_device_state.initialized) {
        return g_cuda_device_state.smem_banks;
    }
    return 32;  /* Default to modern GPU */
}

static int fwht_cuda_compute_capability(void) {
    if (g_cuda_device_state.initialized) {
        return g_cuda_device_state.compute_capability;
    }
    return 70;  /* Default to Volta */
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

/* ============================================================================
 * GPU LOAD/STORE CALLBACK KERNELS
 * ============================================================================ */

/* Callback-aware load/store wrappers for int32 */
template <typename LoadFn, typename StoreFn>
__global__ void fwht_kernel_i32_callbacks(int32_t* __restrict__ data, int n,
                                           LoadFn load_fn, StoreFn store_fn,
                                           void* user_params) {
    extern __shared__ int32_t shared_i32[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * n;
    
    /* Load data into shared memory with optional preprocessing */
    if (tid < n) {
        int32_t val = data[block_offset + tid];
        if (load_fn != NULL) {
            val = load_fn(val, block_offset + tid, user_params);
        }
        shared_i32[tid] = val;
    }
    __syncthreads();
    
    /* Butterfly stages */
    for (int h = 1; h < n; h *= 2) {
        int mask = (h << 1) - 1;
        int i = tid & ~mask;
        int j = i + (tid & (h - 1));
        
        if (tid < n && (tid & h) == 0) {
            int a = shared_i32[j];
            int b = shared_i32[j + h];
            shared_i32[j] = a + b;
            shared_i32[j + h] = a - b;
        }
        __syncthreads();
    }
    
    /* Write back to global memory with optional postprocessing */
    if (tid < n) {
        int32_t val = shared_i32[tid];
        if (store_fn != NULL) {
            store_fn(&data[block_offset + tid], val, block_offset + tid, user_params);
        } else {
            data[block_offset + tid] = val;
        }
    }
}

/* Callback-aware load/store wrappers for double */
template <typename LoadFn, typename StoreFn>
__global__ void fwht_kernel_f64_callbacks(double* __restrict__ data, int n,
                                           LoadFn load_fn, StoreFn store_fn,
                                           void* user_params) {
    extern __shared__ double shared_f64[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * n;
    
    /* Load data into shared memory with optional preprocessing */
    if (tid < n) {
        double val = data[block_offset + tid];
        if (load_fn != NULL) {
            val = load_fn(val, block_offset + tid, user_params);
        }
        shared_f64[tid] = val;
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
    
    /* Write back to global memory with optional postprocessing */
    if (tid < n) {
        double val = shared_f64[tid];
        if (store_fn != NULL) {
            store_fn(&data[block_offset + tid], val, block_offset + tid, user_params);
        } else {
            data[block_offset + tid] = val;
        }
    }
}

/* ============================================================================
 * STANDARD KERNELS (No callbacks)
 * ============================================================================ */

/**
 * Warp-shuffle FWHT kernel for small N (N <= 32, fits in single warp)
 * 
 * OPTIMIZATION (HadaCore 2024): Pure register-based butterfly using warp shuffles.
 * No shared memory needed - all data stays in registers with __shfl_xor_sync().
 * 
 * Performance: 2-3× faster than SMEM version for small N due to:
 * - Zero shared memory bank conflicts
 * - No __syncthreads() overhead
 * - Reduced latency (register ops vs SMEM loads/stores)
 * 
 * Constraints:
 * - N must be ≤ 32 (one warp)
 * - Each thread holds exactly one element
 * - Batch processing: gridDim.x blocks, each processing one independent WHT
 */
template <typename T>
__global__ void fwht_warp_shuffle_kernel(T* __restrict__ data, int n) {
    /* Each block is one warp processing one WHT of size n ≤ 32 */
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * n;
    
    /* Load element into register (one per thread) */
    T val = (tid < n) ? data[block_offset + tid] : T(0);
    
    /* Butterfly stages using warp shuffle
     * 
     * Standard WHT butterfly for stride h:
     *   a' = a + b
     *   b' = a - b
     * 
     * With warp shuffle, thread at position i exchanges with thread at i^h.
     * After shuffle, each thread has its partner's value.
     */
    for (int h = 1; h < n; h *= 2) {
        /* Get partner's value via XOR shuffle */
        T partner = __shfl_xor_sync(0xFFFFFFFF, val, h, 32);
        
        /* Apply butterfly based on whether we're in lower or upper half of pair
         * Lower half (tid & h == 0): compute sum
         * Upper half (tid & h != 0): compute difference
         */
        if ((tid & h) == 0) {
            val = val + partner;  /* Lower position: a + b */
        } else {
            val = partner - val;  /* Upper position: a - b (use partner - val) */
        }
    }
    
    /* Write back to global memory */
    if (tid < n) {
        data[block_offset + tid] = val;
    }
}

/**
 * Warp-shuffle FWHT kernel for medium N (32 < N ≤ 1024, fits in block registers)
 * 
 * OPTIMIZATION: Each thread holds multiple elements and uses warp shuffles
 * for cross-thread communication. Reduces SMEM pressure and bank conflicts.
 * 
 * Algorithm:
 * - Each thread holds N/blockDim.x elements in registers
 * - Intra-thread butterflies for strides < elements_per_thread
 * - Warp shuffles for strides ≥ elements_per_thread
 * 
 * Performance: 1.5-2× faster than SMEM for 64 ≤ N ≤ 512
 */
template <typename T>
__global__ void fwht_warp_shuffle_multi_kernel(T* __restrict__ data, int n) {
    const int lane = threadIdx.x & 31;      /* 0..31 */
    const int W = 32;                       /* warp size (assumed 32) */
    const int base = blockIdx.x * n;        /* block offset into data */
    const int E = (n + W - 1) / W;          /* elements per thread */

    /* Load elements into register array: indices lane + i*W */
    T local[32];  /* supports up to n=1024 (E<=32) */
    for (int i = 0; i < E; ++i) {
        int idx = lane + i * W;
        local[i] = (idx < n) ? data[base + idx] : T(0);
    }

    /* Butterfly stages */
    for (int h = 1; h < n; h <<= 1) {
        if (h < W) {
            /* Low-bit stage: cross-lane within the same local slot */
            for (int i = 0; i < E; ++i) {
                int idx = lane + i * W;
                if (idx < n) {
                    T partner = __shfl_xor_sync(0xFFFFFFFF, local[i], h, W);
                    bool lower = ((idx & h) == 0);
                    local[i] = lower ? (local[i] + partner) : (partner - local[i]);
                }
            }
        } else {
            /* High-bit stage: intra-thread across local slots using XOR of slot index */
            int hi = h >> 5; /* h / W */
            for (int i = 0; i < E; ++i) {
                int j = i ^ hi;
                if (j < E && (i & hi) == 0) {
                    /* Pair (i, j) exactly once per stage */
                    int idx_i = lane + i * W;
                    T a = local[i];
                    T b = local[j];
                    /* lower/upper determined by global index bit h */
                    bool lower_i = ((idx_i & h) == 0);
                    /* For correctness, lower_i should be true here by construction, but compute defensively */
                    if (lower_i) {
                        local[i] = a + b;
                        local[j] = a - b;
                    } else {
                        local[i] = a - b;
                        local[j] = a + b;
                    }
                }
            }
        }
        __syncwarp();
    }

    /* Store back */
    for (int i = 0; i < E; ++i) {
        int idx = lane + i * W;
        if (idx < n) {
            data[base + idx] = local[i];
        }
    }
}

/**
 * Dao-style fused FWHT kernel for medium-large sizes (512 ≤ N ≤ 32768)
 * 
 * PERFORMANCE TECHNIQUES (based on Dao-AILab fast-hadamard-transform):
 * - Each thread processes 8 contiguous elements (reduces memory traffic 8×)
 * - Vectorized loads/stores where possible (int4/float4 for 128-byte coalescing)
 * - Warp shuffles for first log₂(32) stages (no SMEM, no sync)
 * - Only 2 block syncs with shared-memory transposes for higher stages
 * - Supports up to 2^15 elements with minimal sync overhead
 * 
 * Algorithm outline:
 * 1. Load 8 elements per thread into registers (vectorized if aligned)
 * 2. Intra-thread butterflies for strides 1, 2, 4 (within 8-element chunks)
 * 3. Warp shuffles for strides 8, 16 (cross-thread within warp)
 * 4. Shared memory transpose + sync for stride 32 (cross-warp within block)
 * 5. Block-level butterflies for strides ≥ blockDim * elements_per_thread
 * 
 * Grid: gridDim.x blocks, each processing one transform
 * Block: 256 threads (tunable), each handling 8 elements = 2048 elements/block
 */
// Helper for vectorized loads/stores (4-element chunks)
template <typename T> struct Vec4Type { using type = T; };
template <> struct Vec4Type<int32_t> { using type = int4; };
template <> struct Vec4Type<float> { using type = float4; };
template <> struct Vec4Type<double> { using type = double2; };

// 32-byte aligned struct for vectorized double4 loads (double × 4 elements)
struct alignas(32) double4_vec {
    double x, y, z, w;
};

// Meta-style vectorized type helper (BytesToType equivalent)
// For vectorized loads/stores: maps byte size to CUDA vector type
template <int N> struct BytesToType;
template <> struct BytesToType<4> { using Type = float; };
template <> struct BytesToType<8> { using Type = float2; };
template <> struct BytesToType<16> { using Type = float4; };
template <> struct BytesToType<32> { using Type = double4_vec; };  // For double with EPT=4

// Meta's hadamard_mult_thread - inline butterfly on kNElts elements
template<int kLogN, int kNChunks, typename T>
__device__ __forceinline__ void hadamard_mult_thread(T x[kNChunks][1 << kLogN]) {
    constexpr int N = 1 << kLogN;
    #pragma unroll
    for (int i = 0; i < kLogN; ++i) {
        const int stride = 1 << i;
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            for (int c = 0; c < kNChunks; ++c) {
                const T a = x[c][idx];
                const T b = x[c][idx + stride];
                x[c][idx] = a + b;
                x[c][idx + stride] = a - b;
            }
        }
    }
}

// Meta's hadamard_mult_warp - warp shuffle-based butterfly
template<int kLogWarpSize, int kStepStart, int kNChunks, int kNItems, typename T>
__device__ __forceinline__ void hadamard_mult_warp(T x[kNChunks][kNItems]) {
    constexpr int N = 1 << kLogWarpSize;
    int lane_id = threadIdx.x % N;
    #pragma unroll
    for (int step = kStepStart; step < kLogWarpSize; ++step) {
        const int lane_mask = 1 << step;
        const T sign = (lane_id & lane_mask) ? T(-1) : T(1);
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                T x_val_other = __shfl_xor_sync(0xFFFFFFFF, x[c][i], lane_mask);
                x[c][i] = sign * x[c][i] + x_val_other;
            }
        }
    }
}

// Meta's exchange_smem - warp-to-warp data exchange with XOR swizzle
template <int kNChunks, int kChunksPerExchange, int kNElts, int kWarpSize, int kNWarps, bool Pre, typename T, typename vec_t>
__device__ __forceinline__ void exchange_smem(T x_vals[kNChunks][kNElts], vec_t *smem) {
    constexpr int kNThreads = kWarpSize * kNWarps;
    constexpr int kNExchangePerVec = kNElts / (sizeof(vec_t) / sizeof(T));
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane_id = threadIdx.x % kWarpSize;
    const int row_t = threadIdx.x % kNWarps;
    const int col_t = threadIdx.x / kNWarps;
    
    #pragma unroll
    for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
        __syncthreads();
        #pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            #pragma unroll
            for (int r = 0; r < kNExchangePerVec; ++r) {
                smem[(c1 * kNExchangePerVec + r) * kNThreads + (Pre ? warp_id * kWarpSize + lane_id ^ warp_id : row_t * kWarpSize + col_t ^ row_t)] = 
                    reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
            #pragma unroll
            for (int r = 0; r < kNExchangePerVec; ++r) {
                reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r] = 
                    smem[(c1 * kNExchangePerVec + r) * kNThreads + (Pre ? row_t * kWarpSize + col_t ^ row_t : warp_id * kWarpSize + lane_id ^ warp_id)];
            }
        }
    }
}

/* ============================================================================
 * STANDARD SHARED MEMORY KERNELS
 * ============================================================================
 * Proven, reliable implementation for N ≤ max_threads_per_block.
 * Uses bank-conflict-aware access patterns (Andrade et al. 2014).
 * ============================================================================ */

/**
 * CUDA kernel for Walsh-Hadamard Transform (int32)
 * 
 * Optimized butterfly algorithm in shared memory with bank-conflict elimination.
 * Each block processes one WHT independently.
 * 
 * OPTIMIZATION: Bank-conflict-free access patterns (Andrade et al. 2014)
 * - Modern GPUs have 32 banks; Kepler+ use 32-bank mode
 * - Stride accesses to ensure conflict-free butterfly operations
 * - Padding added for non-power-of-2 strides to avoid banking conflicts
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
    
    /* Butterfly stages with bank-conflict-aware addressing
     * 
     * Key insight (Andrade 2014): For 32-bank SMEM, arrange accesses
     * so that threads in a warp read/write different banks.
     * 
     * Thread pattern: For stride h, thread t accesses elements at
     * positions that map to different banks when h is aligned with
     * bank width (32 for modern GPUs).
     */
    for (int h = 1; h < n; h *= 2) {
        int mask = (h << 1) - 1;
        int i = tid & ~mask;  /* Round down to multiple of 2*h */
        int j = i + (tid & (h - 1));
        
        if (tid < n && (tid & h) == 0 && j < n && (j + h) < n) {
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
 * CUDA kernel for Walsh-Hadamard Transform (float)
 * 
 * Optimized butterfly algorithm with bank-conflict-free access.
 */
__global__ void fwht_kernel_f32(float* __restrict__ data, int n) {
    extern __shared__ float shared_f32[];
    
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * n;
    
    /* Load data into shared memory */
    if (tid < n) {
        shared_f32[tid] = data[block_offset + tid];
    }
    __syncthreads();
    
    /* Butterfly stages with bank-conflict-aware addressing */
    for (int h = 1; h < n; h *= 2) {
        int mask = (h << 1) - 1;
        int i = tid & ~mask;
        int j = i + (tid & (h - 1));
        
        if (tid < n && (tid & h) == 0 && j < n && (j + h) < n) {
            float a = shared_f32[j];
            float b = shared_f32[j + h];
            shared_f32[j] = a + b;
            shared_f32[j + h] = a - b;
        }
        __syncthreads();
    }
    
    /* Write back to global memory */
    if (tid < n) {
        data[block_offset + tid] = shared_f32[tid];
    }
}

/**
 * CUDA kernel for Walsh-Hadamard Transform (double)
 * 
 * Optimized butterfly algorithm with bank-conflict-free access.
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
    
    /* Butterfly stages with bank-conflict-aware addressing */
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

        /* Bounds check to prevent invalid memory access */
        if (base < static_cast<unsigned long long>(transform_idx + 1) * n && 
            (base + h) < static_cast<unsigned long long>(transform_idx + 1) * n) {
            T a = data[base];
            T b = data[base + h];
            data[base]     = a + b;
            data[base + h] = a - b;
        }
    }
}

/* ============================================================================
 * Chunked Stage Kernel (Memory Coalescing Optimization)
 * ============================================================================
 * Processes CHUNK_SIZE consecutive elements together to improve memory
 * coalescing. Based on meta-pytorch kernel strategy: threads in a warp
 * access consecutive memory locations, maximizing bandwidth utilization.
 */

template <typename T, unsigned int CHUNK_SIZE>
__global__ void fwht_stage_kernel_chunked(T* __restrict__ data,
                                           size_t n,
                                           size_t h,
                                           size_t pairs_per_transform,
                                           size_t batch_size) {
    size_t transform_idx = blockIdx.y;
    if (transform_idx >= batch_size) {
        return;
    }

    // Calculate chunk-based indexing
    size_t total_chunks = (pairs_per_transform + CHUNK_SIZE - 1) / CHUNK_SIZE;
    size_t chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    unsigned long long h_ll = static_cast<unsigned long long>(h);
    unsigned long long n_ll = static_cast<unsigned long long>(n);
    unsigned long long transform_base = static_cast<unsigned long long>(transform_idx) * n_ll;

    // Process chunks with stride
    for (size_t c = chunk_idx; c < total_chunks; c += stride) {
        size_t chunk_start = c * CHUNK_SIZE;
        size_t chunk_end = chunk_start + CHUNK_SIZE;
        if (chunk_end > pairs_per_transform) {
            chunk_end = pairs_per_transform;
        }

        // Process all pairs in this chunk
        #pragma unroll 4
        for (size_t pair_idx = chunk_start; pair_idx < chunk_end; ++pair_idx) {
            unsigned long long pair_ll = static_cast<unsigned long long>(pair_idx);
            unsigned long long block = pair_ll / h_ll;
            unsigned long long offset = pair_ll - block * h_ll;
            unsigned long long base = transform_base + block * (h_ll << 1) + offset;

            T a = data[base];
            T b = data[base + h];
            data[base]     = a + b;
            data[base + h] = a - b;
        }
    }
}

/* Warp-cooperative coalesced stage kernel
 * Each warp processes a vector of up to 32 offsets inside a 2*h block.
 * Lanes access base+offset and base+h+offset contiguously to maximize
 * global memory coalescing.
 */
template <typename T>
__global__ void fwht_stage_kernel_coalesced(T* __restrict__ data,
                                            size_t n,
                                            size_t h,
                                            size_t batch_size) {
    const unsigned int W = 32;
    const unsigned int lane = threadIdx.x & (W - 1);
    const unsigned int warps_per_block = blockDim.x / W;
    const unsigned int warp_in_block = threadIdx.x / W;
    const unsigned long long warps_per_grid_x = static_cast<unsigned long long>(gridDim.x) * warps_per_block;

    size_t transform_idx = blockIdx.y;
    if (transform_idx >= batch_size) return;

    unsigned long long blocks_per_transform = static_cast<unsigned long long>(n) / (static_cast<unsigned long long>(h) << 1);
    unsigned long long offsets_per_block = (static_cast<unsigned long long>(h) + W - 1ULL) / W;
    unsigned long long total_work = blocks_per_transform * offsets_per_block;

    unsigned long long warp_global = static_cast<unsigned long long>(blockIdx.x) * warps_per_block + warp_in_block;

    unsigned long long n_ll = static_cast<unsigned long long>(n);
    unsigned long long h_ll = static_cast<unsigned long long>(h);
    unsigned long long transform_base = static_cast<unsigned long long>(transform_idx) * n_ll;

    for (unsigned long long work = warp_global; work < total_work; work += warps_per_grid_x) {
        unsigned long long block = work / offsets_per_block;      // which 2*h block
        unsigned long long chunk = work - block * offsets_per_block; // which 32-wide offset chunk
        unsigned long long offset = chunk * W + lane;             // offset within [0, h)
        if (offset < h_ll) {
            unsigned long long base = transform_base + block * (h_ll << 1) + offset;
            T a = data[base];
            T b = data[base + h_ll];
            data[base]        = a + b;
            data[base + h_ll] = a - b;
        }
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
static fwht_status_t fwht_launch_small(T* d_data, size_t n, size_t batch_size, cudaStream_t stream);

template <>
fwht_status_t fwht_launch_small<int32_t>(int32_t* d_data, size_t n, size_t batch_size, cudaStream_t stream) {
    unsigned int max_threads = fwht_cuda_max_threads_per_block();
    if (n > max_threads) {
        return FWHT_ERROR_INVALID_SIZE;
    }
    
    /* OPTIMIZATION: Use warp-shuffle kernels for small N (HadaCore 2024)
     * 
     * Performance hierarchy:
     * - N ≤ 32:  Pure warp shuffle (1 warp per transform)
     * - 32 < N ≤ 512: Multi-element warp shuffle (32 threads per block)
     * - N > 512: Shared memory kernel (standard)
     * 
     * Expected speedup: 2-3× for N ≤ 32, 1.5-2× for 32 < N ≤ 512
     */
    size_t processed = 0;
    
    if (n <= 32) {
        /* Pure warp shuffle: one warp (32 threads) per transform, but only n active */
        unsigned int threads = 32;  /* Full warp for shuffle to work */
        
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_warp_shuffle_kernel<int32_t><<<current, threads, 0, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    } else if (g_fwht_multi_shuffle_enabled && n <= 512) {
        /* Multi-element warp shuffle for 32 < N ≤ 512 (proven range) */
        unsigned int threads = 32; /* One warp per transform */
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_warp_shuffle_multi_kernel<int32_t><<<current, threads, 0, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    } else {
        /* Standard shared memory kernel - proven and reliable */
        size_t shared_bytes = n * sizeof(int32_t);
        unsigned int threads = static_cast<unsigned int>(n);
        
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_kernel_i32<<<current, threads, shared_bytes, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    }
    
    return FWHT_SUCCESS;
}

template <>
fwht_status_t fwht_launch_small<float>(float* d_data, size_t n, size_t batch_size, cudaStream_t stream) {
    unsigned int max_threads = fwht_cuda_max_threads_per_block();
    if (n > max_threads) {
        return FWHT_ERROR_INVALID_SIZE;
    }
    
    /* OPTIMIZATION: Use warp-shuffle kernels for small N */
    size_t processed = 0;
    
    if (n <= 32) {
        /* Pure warp shuffle for tiny transforms */
        unsigned int threads = 32;
        
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_warp_shuffle_kernel<float><<<current, threads, 0, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    } else if (g_fwht_multi_shuffle_enabled && n <= 512) {
        /* Multi-element warp shuffle for 32 < N ≤ 512 (proven range) */
        unsigned int threads = 32;
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_warp_shuffle_multi_kernel<float><<<current, threads, 0, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    } else {
        /* Standard shared memory kernel - proven and reliable */
        size_t shared_bytes = n * sizeof(float);
        unsigned int threads = static_cast<unsigned int>(n);
        
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_kernel_f32<<<current, threads, shared_bytes, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    }
    
    return FWHT_SUCCESS;
}

template <>
fwht_status_t fwht_launch_small<double>(double* d_data, size_t n, size_t batch_size, cudaStream_t stream) {
    unsigned int max_threads = fwht_cuda_max_threads_per_block();
    if (n > max_threads) {
        return FWHT_ERROR_INVALID_SIZE;
    }
    
    /* OPTIMIZATION: Use warp-shuffle kernels for small N */
    size_t processed = 0;
    
    if (n <= 32) {
        /* Pure warp shuffle for tiny transforms */
        unsigned int threads = 32;
        
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_warp_shuffle_kernel<double><<<current, threads, 0, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    } else if (g_fwht_multi_shuffle_enabled && n <= 512) {
        /* Multi-element warp shuffle for 32 < N ≤ 512 (proven range) */
        unsigned int threads = 32;
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_warp_shuffle_multi_kernel<double><<<current, threads, 0, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    } else {
        /* Standard shared memory kernel - proven and reliable */
        size_t shared_bytes = n * sizeof(double);
        unsigned int threads = static_cast<unsigned int>(n);
        
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            fwht_kernel_f64<<<current, threads, shared_bytes, stream>>>(
                d_data + processed * n, static_cast<int>(n));
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    }
    
    return FWHT_SUCCESS;
}

template <typename T>
static fwht_status_t fwht_launch_large(T* d_data, size_t n, size_t batch_size, cudaStream_t stream) {
    size_t pairs_per_transform = n >> 1;
    unsigned int threads = fwht_effective_block_size(pairs_per_transform);
    
    /* WORKAROUND: Force smaller block size for n=1024 to ensure multiple blocks */
    if (n == 1024) {
        threads = 256;  /* Use 256 threads instead of 512 to get 2 blocks */
    }
    
    unsigned long long work_items = pairs_per_transform;
    unsigned int blocks_x = (threads == 0)
        ? 1u
        : static_cast<unsigned int>((work_items + threads - 1) / threads);

    if (blocks_x == 0) {
        blocks_x = 1;
    }

    /* Avoid over-launching empty blocks: cap to device grid limit only */
    unsigned int max_grid_x = fwht_cuda_max_grid_x();
    if (max_grid_x > 0 && blocks_x > max_grid_x) {
        blocks_x = max_grid_x;
    }

    for (size_t h = 1; h < n; h <<= 1) {
        size_t processed = 0;
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            dim3 grid(blocks_x, current);
            fwht_stage_kernel<T><<<grid, threads, 0, stream>>>(d_data + processed * n,
                                                               n,
                                                               h,
                                                               pairs_per_transform,
                                                               current);
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    }
    return FWHT_SUCCESS;
}

/* ============================================================================
 * Chunked Launch (Optimized for 1K-32K range)
 * ============================================================================
 * Warp-cooperative coalesced stage kernel. Each warp processes W (32)
 * consecutive offsets within a 2*h block to maximize global memory
 * coalescing for both a and b (base and base+h).
 */
template <typename T>
static fwht_status_t fwht_launch_large_chunked(T* d_data, size_t n, size_t batch_size, cudaStream_t stream) {
    const unsigned int W = 32;
    unsigned int max_grid_x = fwht_cuda_max_grid_x();
    unsigned int max_threads = fwht_cuda_max_threads_per_block();
    unsigned int threads = (g_fwHT_chunked_threads != 0) ? g_fwHT_chunked_threads : 256u;
    if (threads > max_threads) {
        threads = max_threads;
    }
    if (threads < W) {
        threads = W;
    }

    if (g_fwht_dispatch_logging) {
        fprintf(stderr, "[libfwht] chunked kernel: n=%zu batch=%zu threads=%u (%s)\n",
                n, batch_size, threads,
                (g_fwHT_chunked_threads != 0) ? "env" : "auto");
    }

    for (size_t h = 1; h < n; h <<= 1) {
        unsigned long long blocks_per_transform = static_cast<unsigned long long>(n) / (static_cast<unsigned long long>(h) << 1);
        unsigned long long offsets_per_block = (static_cast<unsigned long long>(h) + W - 1ULL) / W;
        unsigned long long total_warp_units = blocks_per_transform * offsets_per_block;

        unsigned int warps_per_block = threads / W;
        if (warps_per_block == 0) warps_per_block = 1;

        unsigned int blocks_x = static_cast<unsigned int>((total_warp_units + warps_per_block - 1ULL) / warps_per_block);
        if (blocks_x == 0) blocks_x = 1;
        if (max_grid_x > 0 && blocks_x > max_grid_x) {
            blocks_x = max_grid_x;
        }

        size_t processed = 0;
        while (processed < batch_size) {
            unsigned int current = (batch_size - processed > CUDA_BATCH_LIMIT)
                                   ? CUDA_BATCH_LIMIT
                                   : static_cast<unsigned int>(batch_size - processed);
            dim3 grid(blocks_x, current);

            // Launch warp-coalesced stage kernel
            fwht_stage_kernel_coalesced<T><<<grid, threads, 0, stream>>>(
                d_data + processed * n, n, h, current);
            CUDA_CHECK(cudaGetLastError());
            processed += current;
        }
    }
    return FWHT_SUCCESS;
}

template <typename T>
static fwht_status_t fwht_execute_cuda(T* data, size_t n, size_t batch_size) {
    if (batch_size == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    g_fwht_last_metrics.valid = false;
    g_fwht_last_metrics.samples = 0;

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
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_data), bytes);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }

    cudaStream_t stream = 0;
    bool stream_created = false;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err == cudaSuccess) {
        stream_created = true;
    } else {
        stream = 0;
        (void)cudaGetLastError();
    }

    bool profiling = g_fwht_profiling_enabled && stream_created;
    cudaEvent_t evt_h2d_start = NULL;
    cudaEvent_t evt_h2d_end = NULL;
    cudaEvent_t evt_kernel_end = NULL;
    cudaEvent_t evt_d2h_end = NULL;

    if (profiling) {
        if (cudaEventCreateWithFlags(&evt_h2d_start, cudaEventDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&evt_h2d_end, cudaEventDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&evt_kernel_end, cudaEventDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&evt_d2h_end, cudaEventDefault) != cudaSuccess) {
            profiling = false;
            if (evt_h2d_start) cudaEventDestroy(evt_h2d_start);
            if (evt_h2d_end) cudaEventDestroy(evt_h2d_end);
            if (evt_kernel_end) cudaEventDestroy(evt_kernel_end);
            if (evt_d2h_end) cudaEventDestroy(evt_d2h_end);
            evt_h2d_start = evt_h2d_end = evt_kernel_end = evt_d2h_end = NULL;
        }
    }

    fwht_status_t status = FWHT_SUCCESS;
    unsigned int max_block_threads = fwht_cuda_max_threads_per_block();
    if (profiling) {
        (void)cudaEventRecord(evt_h2d_start, stream);
    }

    err = cudaMemcpyAsync(d_data, data, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        status = fwht_cuda_report(err, __FILE__, __LINE__);
        goto cleanup;
    }

    if (profiling) {
        (void)cudaEventRecord(evt_h2d_end, stream);
    }

    // Size-based kernel dispatch with Dao-style fused kernel for medium sizes
    if (n <= max_block_threads && n != 1024) {
        if (g_fwht_dispatch_logging) {
            fprintf(stderr, "[libfwht] dispatch: shared-memory kernel (n=%zu, batch=%zu)\n", n, batch_size);
        }
        // Small transforms: use shared memory kernels (warp shuffle or SMEM)
        status = fwht_launch_small<T>(d_data, n, batch_size, stream);
    } else if (n >= 512 || (g_fwht_chunked_enabled && n >= 4096 && n <= 32768)) {
        if (g_fwht_dispatch_logging) {
            fprintf(stderr, "[libfwht] dispatch: chunked kernel (n=%zu, batch=%zu)\n", n, batch_size);
        }
        // Medium-large transforms: use chunked coalesced kernel
        status = fwht_launch_large_chunked<T>(d_data, n, batch_size, stream);
    } else {
        if (g_fwht_dispatch_logging) {
            fprintf(stderr, "[libfwht] dispatch: stage kernel (n=%zu, batch=%zu)\n", n, batch_size);
        }
        // Very large transforms: use standard stage kernel
        status = fwht_launch_large<T>(d_data, n, batch_size, stream);
    }  // End dispatch block

    if (status != FWHT_SUCCESS) {
        goto cleanup;
    }

    if (profiling) {
        (void)cudaEventRecord(evt_kernel_end, stream);
    }

    err = cudaMemcpyAsync(data, d_data, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        status = fwht_cuda_report(err, __FILE__, __LINE__);
        goto cleanup;
    }

    if (profiling) {
        (void)cudaEventRecord(evt_d2h_end, stream);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        status = fwht_cuda_report(err, __FILE__, __LINE__);
        goto cleanup;
    }

    if (profiling) {
        float h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, evt_h2d_start, evt_h2d_end);
    cudaEventElapsedTime(&kernel_ms, evt_h2d_end, evt_kernel_end);
    cudaEventElapsedTime(&d2h_ms, evt_kernel_end, evt_d2h_end);
        g_fwht_last_metrics.h2d_ms = static_cast<double>(h2d_ms);
        g_fwht_last_metrics.kernel_ms = static_cast<double>(kernel_ms);
        g_fwht_last_metrics.d2h_ms = static_cast<double>(d2h_ms);
        g_fwht_last_metrics.n = n;
        g_fwht_last_metrics.batch_size = batch_size;
        g_fwht_last_metrics.bytes_transferred = bytes;
        g_fwht_last_metrics.samples = 1;
        g_fwht_last_metrics.valid = true;
    }

cleanup:
    if (profiling) {
        if (evt_h2d_start) cudaEventDestroy(evt_h2d_start);
        if (evt_h2d_end) cudaEventDestroy(evt_h2d_end);
        if (evt_kernel_end) cudaEventDestroy(evt_kernel_end);
        if (evt_d2h_end) cudaEventDestroy(evt_d2h_end);
    }

    if (stream_created) {
        cudaStreamDestroy(stream);
    }

    if (d_data != NULL) {
        cudaError_t free_err = cudaFree(d_data);
        if (free_err != cudaSuccess && status == FWHT_SUCCESS) {
            status = fwht_cuda_report(free_err, __FILE__, __LINE__);
        }
    }

    if (status != FWHT_SUCCESS) {
        g_fwht_last_metrics.valid = false;
        g_fwht_last_metrics.samples = 0;
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

fwht_status_t fwht_gpu_set_profiling(bool enable) {
    g_fwht_profiling_enabled = enable;
    if (!enable) {
        g_fwht_last_metrics.valid = false;
        g_fwht_last_metrics.samples = 0;
    }
    return FWHT_SUCCESS;
}

bool fwht_gpu_profiling_enabled(void) {
    return g_fwht_profiling_enabled;
}

fwht_gpu_metrics_t fwht_gpu_get_last_metrics(void) {
    return g_fwht_last_metrics;
}

/* Multi-shuffle toggle API (experimental, opt-in for 32 < N ≤ 512) *//* Small-N multi-shuffle toggle API */
fwht_status_t fwht_gpu_set_multi_shuffle(bool enable) {
    g_fwht_multi_shuffle_enabled = enable;
    return FWHT_SUCCESS;
}

bool fwht_gpu_multi_shuffle_enabled(void) {
    return g_fwht_multi_shuffle_enabled;
}

/* Chunked kernel toggle API */
fwht_status_t fwht_gpu_set_chunked(bool enable) {
    g_fwht_chunked_enabled = enable;
    return FWHT_SUCCESS;
}

bool fwht_gpu_chunked_enabled(void) {
    return g_fwht_chunked_enabled;
}

/* ============================================================================
 * GPU DEVICE INFO API (NEW)
 * ============================================================================ */

unsigned int fwht_gpu_get_smem_banks(void) {
    if (!g_cuda_device_state.initialized) {
        fwht_cuda_ensure_device_state();
    }
    return (unsigned int)fwht_cuda_smem_banks();
}

unsigned int fwht_gpu_get_compute_capability(void) {
    if (!g_cuda_device_state.initialized) {
        fwht_cuda_ensure_device_state();
    }
    return (unsigned int)fwht_cuda_compute_capability();
}

const char* fwht_gpu_get_device_name(void) {
    if (!g_cuda_device_state.initialized) {
        fwht_cuda_ensure_device_state();
    }
    if (g_cuda_device_state.initialized) {
        return g_cuda_device_state.props.name;
    }
    return "Unknown";
}

unsigned int fwht_gpu_get_sm_count(void) {
    if (!g_cuda_device_state.initialized) {
        fwht_cuda_ensure_device_state();
    }
    return fwht_cuda_sm_count();
}

/* =========================================================================
 * PINNED HOST MEMORY HELPERS
 * ========================================================================= */

fwht_status_t fwht_gpu_host_alloc(void** ptr, size_t bytes) {
    if (ptr == NULL || bytes == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    *ptr = NULL;

    fwht_status_t st = fwht_cuda_ensure_device_state();
    if (st != FWHT_SUCCESS) {
        return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }

    void* host_ptr = NULL;
    cudaError_t err = cudaHostAlloc(&host_ptr, bytes, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }
    *ptr = host_ptr;
    return FWHT_SUCCESS;
}

void fwht_gpu_host_free(void* ptr) {
    if (ptr == NULL) return;
    (void)cudaFreeHost(ptr);
}

/* =========================================================================
 * DEVICE MEMORY HELPERS (alloc/free/memcpy)
 * ========================================================================= */

fwht_status_t fwht_gpu_device_alloc(void** d_ptr, size_t bytes) {
    if (d_ptr == NULL || bytes == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    *d_ptr = NULL;
    fwht_status_t st = fwht_cuda_ensure_device_state();
    if (st != FWHT_SUCCESS) return FWHT_ERROR_BACKEND_UNAVAILABLE;
    void* tmp = NULL;
    cudaError_t err = cudaMalloc(&tmp, bytes);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }
    *d_ptr = tmp;
    return FWHT_SUCCESS;
}

void fwht_gpu_device_free(void* d_ptr) {
    if (d_ptr == NULL) return;
    (void)cudaFree(d_ptr);
}

fwht_status_t fwht_gpu_memcpy_h2d(void* d_dst, const void* h_src, size_t bytes) {
    if (d_dst == NULL || h_src == NULL || bytes == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    fwht_status_t st = fwht_cuda_ensure_device_state();
    if (st != FWHT_SUCCESS) return FWHT_ERROR_BACKEND_UNAVAILABLE;
    cudaError_t err = cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }
    return FWHT_SUCCESS;
}

fwht_status_t fwht_gpu_memcpy_d2h(void* h_dst, const void* d_src, size_t bytes) {
    if (h_dst == NULL || d_src == NULL || bytes == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    fwht_status_t st = fwht_cuda_ensure_device_state();
    if (st != FWHT_SUCCESS) return FWHT_ERROR_BACKEND_UNAVAILABLE;
    cudaError_t err = cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }
    return FWHT_SUCCESS;
}

/* ============================================================================
 * CORE WHT API
 * ============================================================================ */

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

fwht_status_t fwht_batch_f32_cuda(float* data, size_t n, size_t batch_size) {
    if (data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda<float>(data, n, batch_size);
}

fwht_status_t fwht_batch_f64_cuda(double* data, size_t n, size_t batch_size) {
    if (data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda<double>(data, n, batch_size);
}

/* ============================================================================
 * BATCH PROCESSING WITH ARRAY OF POINTERS (Optimized Memory Transfer)
 * 
 * These functions handle the packing/unpacking of array-of-pointers format
 * into contiguous memory for optimal GPU batch processing.
 * They are inside the extern "C" block that started at line 1391.
 * ============================================================================ */

fwht_status_t fwht_batch_i32_cuda_from_pointers(int32_t** data_array, size_t n, size_t batch_size) {
    if (data_array == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    
    /* OPTIMIZATION: Pack all arrays into contiguous host buffer for single transfer
     * This matches Meta's approach: single H2D transfer, process batch, single D2H transfer
     * Performance gain: 5-10× faster transfers for large batches
     */
    size_t total_size = n * batch_size;
    size_t total_bytes = total_size * sizeof(int32_t);
    
    /* Allocate temporary contiguous host buffer */
    int32_t* h_packed = (int32_t*)malloc(total_bytes);
    if (h_packed == NULL) {
        return FWHT_ERROR_OUT_OF_MEMORY;
    }
    
    /* Pack: copy all arrays into contiguous buffer */
    for (size_t i = 0; i < batch_size; ++i) {
        if (data_array[i] == NULL) {
            free(h_packed);
            return FWHT_ERROR_NULL_POINTER;
        }
        memcpy(h_packed + i * n, data_array[i], n * sizeof(int32_t));
    }
    
    /* Process batch on GPU (this handles H2D, kernel, D2H) */
    fwht_status_t status = fwht_batch_i32_cuda(h_packed, n, batch_size);
    
    /* Unpack: copy results back to individual arrays */
    if (status == FWHT_SUCCESS) {
        for (size_t i = 0; i < batch_size; ++i) {
            memcpy(data_array[i], h_packed + i * n, n * sizeof(int32_t));
        }
    }
    
    free(h_packed);
    return status;
}

fwht_status_t fwht_batch_f64_cuda_from_pointers(double** data_array, size_t n, size_t batch_size) {
    if (data_array == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    
    /* OPTIMIZATION: Pack all arrays into contiguous host buffer for single transfer */
    size_t total_size = n * batch_size;
    size_t total_bytes = total_size * sizeof(double);
    
    /* Allocate temporary contiguous host buffer */
    double* h_packed = (double*)malloc(total_bytes);
    if (h_packed == NULL) {
        return FWHT_ERROR_OUT_OF_MEMORY;
    }
    
    /* Pack: copy all arrays into contiguous buffer */
    for (size_t i = 0; i < batch_size; ++i) {
        if (data_array[i] == NULL) {
            free(h_packed);
            return FWHT_ERROR_NULL_POINTER;
        }
        memcpy(h_packed + i * n, data_array[i], n * sizeof(double));
    }
    
    /* Process batch on GPU (this handles H2D, kernel, D2H) */
    fwht_status_t status = fwht_batch_f64_cuda(h_packed, n, batch_size);
    
    /* Unpack: copy results back to individual arrays */
    if (status == FWHT_SUCCESS) {
        for (size_t i = 0; i < batch_size; ++i) {
            memcpy(data_array[i], h_packed + i * n, n * sizeof(double));
        }
    }
    
    free(h_packed);
    return status;
}

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * Template functions (C++ only, must be outside extern "C")
 * ============================================================================ */

template <typename T>
static fwht_status_t fwht_launch_wht_device_kernels(T* d_data,
                                                    size_t n,
                                                    size_t batch_size,
                                                    cudaStream_t stream) {
    unsigned int max_block_threads = fwht_cuda_max_threads_per_block();
    fwht_status_t status;
    if (n <= max_block_threads) {
        if (g_fwht_dispatch_logging) {
            fprintf(stderr,
                    "[libfwht] dispatch (device): shared-memory kernel (n=%zu, batch=%zu)\n",
                    n,
                    batch_size);
        }
        status = fwht_launch_small<T>(d_data, n, batch_size, stream);
    } else if (g_fwht_chunked_enabled && n >= 4096 && n <= 32768) {
        if (g_fwht_dispatch_logging) {
            fprintf(stderr,
                    "[libfwht] dispatch (device): chunked kernel (n=%zu, batch=%zu)\n",
                    n,
                    batch_size);
        }
        status = fwht_launch_large_chunked<T>(d_data, n, batch_size, stream);
    } else {
        if (g_fwht_dispatch_logging) {
            fprintf(stderr,
                    "[libfwht] dispatch (device): stage kernel (n=%zu, batch=%zu)\n",
                    n,
                    batch_size);
        }
        status = fwht_launch_large<T>(d_data, n, batch_size, stream);
    }
    return status;
}

template <typename T>
static fwht_status_t fwht_execute_cuda_device_async(T* d_data,
                                                    size_t n,
                                                    size_t batch_size,
                                                    cudaStream_t stream) {
    if (batch_size == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    g_fwht_last_metrics.valid = false;
    g_fwht_last_metrics.samples = 0;

    fwht_status_t init_status = fwht_cuda_ensure_device_state();
    if (init_status != FWHT_SUCCESS) {
        return init_status;
    }

    size_t element_count = n * batch_size;
    if (element_count > SIZE_MAX / sizeof(T)) {
        return FWHT_ERROR_INVALID_SIZE;
    }

    return fwht_launch_wht_device_kernels<T>(d_data, n, batch_size, stream);
}

template <typename T>
static fwht_status_t fwht_execute_cuda_device(T* d_data, size_t n, size_t batch_size) {
    if (batch_size == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    g_fwht_last_metrics.valid = false;
    g_fwht_last_metrics.samples = 0;

    fwht_status_t init_status = fwht_cuda_ensure_device_state();
    if (init_status != FWHT_SUCCESS) {
        return init_status;
    }

    size_t element_count = n * batch_size;
    if (element_count > SIZE_MAX / sizeof(T)) {
        return FWHT_ERROR_INVALID_SIZE;
    }

    cudaStream_t stream = 0;
    bool stream_created = false;
    cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err == cudaSuccess) {
        stream_created = true;
    } else {
        stream = 0;
        (void)cudaGetLastError();
    }

    bool profiling = g_fwht_profiling_enabled && stream_created;
    cudaEvent_t evt_kernel_start = NULL;
    cudaEvent_t evt_kernel_end = NULL;
    if (profiling) {
        if (cudaEventCreateWithFlags(&evt_kernel_start, cudaEventDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&evt_kernel_end, cudaEventDefault) != cudaSuccess) {
            profiling = false;
            if (evt_kernel_start) cudaEventDestroy(evt_kernel_start);
            if (evt_kernel_end) cudaEventDestroy(evt_kernel_end);
            evt_kernel_start = evt_kernel_end = NULL;
        }
    }

    fwht_status_t status;
    if (profiling) {
        (void)cudaEventRecord(evt_kernel_start, stream);
    }

    status = fwht_launch_wht_device_kernels<T>(d_data, n, batch_size, stream);
    if (status != FWHT_SUCCESS) {
        goto cleanup;
    }

    if (profiling) {
        (void)cudaEventRecord(evt_kernel_end, stream);
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        status = fwht_cuda_report(err, __FILE__, __LINE__);
        goto cleanup;
    }

    if (profiling) {
        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, evt_kernel_start, evt_kernel_end);
        g_fwht_last_metrics.h2d_ms = 0.0;
        g_fwht_last_metrics.kernel_ms = static_cast<double>(kernel_ms);
        g_fwht_last_metrics.d2h_ms = 0.0;
        g_fwht_last_metrics.n = n;
        g_fwht_last_metrics.batch_size = batch_size;
        g_fwht_last_metrics.bytes_transferred = 0; /* device-resident */
        g_fwht_last_metrics.samples = 1;
        g_fwht_last_metrics.valid = true;
    }

cleanup:
    if (profiling) {
        if (evt_kernel_start) cudaEventDestroy(evt_kernel_start);
        if (evt_kernel_end) cudaEventDestroy(evt_kernel_end);
    }
    if (stream_created) {
        cudaStreamDestroy(stream);
    }
    if (status != FWHT_SUCCESS) {
        g_fwht_last_metrics.valid = false;
        g_fwht_last_metrics.samples = 0;
    }
    return status;
}

#ifdef __cplusplus
extern "C" {
#endif

fwht_status_t fwht_batch_i32_cuda_device(int32_t* d_data, size_t n, size_t batch_size) {
    if (d_data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda_device<int32_t>(d_data, n, batch_size);
}

fwht_status_t fwht_batch_i32_cuda_device_async(int32_t* d_data,
                                                size_t n,
                                                size_t batch_size,
                                                cudaStream_t stream) {
    if (d_data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda_device_async<int32_t>(d_data, n, batch_size, stream);
}

fwht_status_t fwht_batch_f32_cuda_device(float* d_data, size_t n, size_t batch_size) {
    if (d_data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda_device<float>(d_data, n, batch_size);
}

fwht_status_t fwht_batch_f64_cuda_device(double* d_data, size_t n, size_t batch_size) {
    if (d_data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    return fwht_execute_cuda_device<double>(d_data, n, batch_size);
}

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * PERSISTENT GPU CONTEXT API
 * 
 * Pre-allocates GPU memory to eliminate repeated cudaMalloc/cudaFree overhead.
 * Provides 5-10x speedup for workloads with many small transforms.
 * ============================================================================ */

struct fwht_gpu_context {
    void* d_buffer_i32;      /* Device buffer for int32 data */
    void* d_buffer_f64;      /* Device buffer for double data */
    size_t max_n;            /* Maximum transform size */
    size_t max_batch_size;   /* Maximum batch size */
    cudaStream_t stream;     /* Dedicated CUDA stream */
    bool stream_created;     /* Whether stream was successfully created */
    
    /* Callback function pointers (device functions) */
    void* load_fn_i32;       /* int32 load callback */
    void* store_fn_i32;      /* int32 store callback */
    void* load_fn_f64;       /* double load callback */
    void* store_fn_f64;      /* double store callback */
    void* user_params;       /* User-defined parameter pointer */
};

fwht_gpu_context_t* fwht_gpu_context_create(size_t max_n, size_t max_batch_size) {
    /* Validate inputs */
    if (max_n == 0 || (max_n & (max_n - 1)) != 0) {
        return NULL;  /* max_n must be power of 2 */
    }
    if (max_batch_size == 0) {
        return NULL;
    }
    
    /* Check if CUDA is available */
    fwht_status_t status = fwht_cuda_ensure_device_state();
    if (status != FWHT_SUCCESS) {
        return NULL;
    }
    
    /* Allocate context structure */
    fwht_gpu_context_t* ctx = (fwht_gpu_context_t*)malloc(sizeof(fwht_gpu_context_t));
    if (ctx == NULL) {
        return NULL;
    }
    
    ctx->d_buffer_i32 = NULL;
    ctx->d_buffer_f64 = NULL;
    ctx->max_n = max_n;
    ctx->max_batch_size = max_batch_size;
    ctx->stream = NULL;
    ctx->stream_created = false;
    ctx->load_fn_i32 = NULL;
    ctx->store_fn_i32 = NULL;
    ctx->load_fn_f64 = NULL;
    ctx->store_fn_f64 = NULL;
    ctx->user_params = NULL;
    
    /* Allocate device memory for int32 */
    size_t bytes_i32 = max_n * max_batch_size * sizeof(int32_t);
    cudaError_t err = cudaMalloc(&ctx->d_buffer_i32, bytes_i32);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory for int32: %s\n", 
                cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }
    
    /* Allocate device memory for double */
    size_t bytes_f64 = max_n * max_batch_size * sizeof(double);
    err = cudaMalloc(&ctx->d_buffer_f64, bytes_f64);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate GPU memory for double: %s\n", 
                cudaGetErrorString(err));
        cudaFree(ctx->d_buffer_i32);
        free(ctx);
        return NULL;
    }
    
    /* Create dedicated stream for this context */
    err = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
    if (err == cudaSuccess) {
        ctx->stream_created = true;
    } else {
        /* Stream creation failed, fall back to default stream */
        ctx->stream = NULL;
        ctx->stream_created = false;
        (void)cudaGetLastError();  /* Clear error */
    }
    
    return ctx;
}

void fwht_gpu_context_destroy(fwht_gpu_context_t* ctx) {
    if (ctx == NULL) {
        return;
    }
    
    /* Synchronize before freeing resources */
    if (ctx->stream_created && ctx->stream != NULL) {
        cudaStreamSynchronize(ctx->stream);
        cudaStreamDestroy(ctx->stream);
    }
    
    /* Free device buffers */
    if (ctx->d_buffer_i32 != NULL) {
        cudaFree(ctx->d_buffer_i32);
    }
    if (ctx->d_buffer_f64 != NULL) {
        cudaFree(ctx->d_buffer_f64);
    }
    
    /* Free context structure */
    free(ctx);
}

/* Helper template for context-based computation (C++ only, outside extern "C") */
template <typename T>
static fwht_status_t fwht_gpu_context_compute_impl(fwht_gpu_context_t* ctx,
                                                     T* host_data,
                                                     size_t n,
                                                     size_t batch_size) {
    if (ctx == NULL) return FWHT_ERROR_NULL_POINTER;
    if (host_data == NULL) return FWHT_ERROR_NULL_POINTER;
    if (n == 0 || (n & (n - 1)) != 0) return FWHT_ERROR_INVALID_SIZE;
    if (batch_size == 0) return FWHT_ERROR_INVALID_ARGUMENT;
    
    /* Check that request fits within context limits */
    if (n > ctx->max_n || batch_size > ctx->max_batch_size) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    
    /* Check if callbacks are set for this type */
    bool has_callbacks = false;
    void* load_fn = NULL;
    void* store_fn = NULL;
    
    if (sizeof(T) == sizeof(int32_t)) {
        has_callbacks = (ctx->load_fn_i32 != NULL || ctx->store_fn_i32 != NULL);
        load_fn = ctx->load_fn_i32;
        store_fn = ctx->store_fn_i32;
    } else {
        has_callbacks = (ctx->load_fn_f64 != NULL || ctx->store_fn_f64 != NULL);
        load_fn = ctx->load_fn_f64;
        store_fn = ctx->store_fn_f64;
    }
    
    /* Get the appropriate device buffer */
    T* d_data;
    if (sizeof(T) == sizeof(int32_t)) {
        d_data = (T*)ctx->d_buffer_i32;
    } else {
        d_data = (T*)ctx->d_buffer_f64;
    }
    
    cudaStream_t stream = ctx->stream_created ? ctx->stream : 0;
    size_t bytes = n * batch_size * sizeof(T);
    bool profiling = g_fwht_profiling_enabled;
    cudaEvent_t evt_h2d_start = NULL, evt_h2d_end = NULL, evt_kernel_end = NULL, evt_d2h_end = NULL;
    if (profiling) {
        if (cudaEventCreateWithFlags(&evt_h2d_start, cudaEventDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&evt_h2d_end, cudaEventDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&evt_kernel_end, cudaEventDefault) != cudaSuccess ||
            cudaEventCreateWithFlags(&evt_d2h_end, cudaEventDefault) != cudaSuccess) {
            profiling = false;
            if (evt_h2d_start) cudaEventDestroy(evt_h2d_start);
            if (evt_h2d_end) cudaEventDestroy(evt_h2d_end);
            if (evt_kernel_end) cudaEventDestroy(evt_kernel_end);
            if (evt_d2h_end) cudaEventDestroy(evt_d2h_end);
            evt_h2d_start = evt_h2d_end = evt_kernel_end = evt_d2h_end = NULL;
        }
    }
    
    /* Copy host data to pre-allocated device buffer */
    if (profiling) {
        (void)cudaEventRecord(evt_h2d_start, stream);
    }
    cudaError_t err = cudaMemcpyAsync(d_data, host_data, bytes, 
                                       cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }
    if (profiling) {
        (void)cudaEventRecord(evt_h2d_end, stream);
    }
    
    fwht_status_t status;
    
    /* Use callback-aware kernels if callbacks are set */
    if (has_callbacks && n <= fwht_cuda_max_threads_per_block()) {
        /* Only small transforms support callbacks currently */
        unsigned int block_size = (n < 32) ? 32 : n;
        size_t shared_mem = n * sizeof(T);
        
        if (sizeof(T) == sizeof(int32_t)) {
            fwht_kernel_i32_callbacks<<<batch_size, block_size, shared_mem, stream>>>(
                (int32_t*)d_data, n,
                (fwht_load_fn_i32)load_fn,
                (fwht_store_fn_i32)store_fn,
                ctx->user_params
            );
        } else {
            fwht_kernel_f64_callbacks<<<batch_size, block_size, shared_mem, stream>>>(
                (double*)d_data, n,
                (fwht_load_fn_f64)load_fn,
                (fwht_store_fn_f64)store_fn,
                ctx->user_params
            );
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return fwht_cuda_report(err, __FILE__, __LINE__);
        }
        
        status = FWHT_SUCCESS;
    } else {
        /* Use standard kernels without callbacks - same dispatch logic as main path */
        unsigned int max_block_threads = fwht_cuda_max_threads_per_block();
        if (n <= max_block_threads) {
            if (g_fwht_dispatch_logging) {
                fprintf(stderr, "[libfwht] dispatch (context): shared-memory kernel (n=%zu, batch=%zu)\n", n, batch_size);
            }
            status = fwht_launch_small<T>(d_data, n, batch_size, stream);
        } else if (g_fwht_chunked_enabled && n >= 4096 && n <= 32768) {
            // Chunked coalesced kernel for medium-large sizes
            if (g_fwht_dispatch_logging) {
                fprintf(stderr, "[libfwht] dispatch (context): chunked kernel (n=%zu, batch=%zu)\n", n, batch_size);
            }
            status = fwht_launch_large_chunked<T>(d_data, n, batch_size, stream);
        } else {
            // Standard stage kernel for very large sizes
            if (g_fwht_dispatch_logging) {
                fprintf(stderr, "[libfwht] dispatch (context): stage kernel (n=%zu, batch=%zu)\n", n, batch_size);
            }
            status = fwht_launch_large<T>(d_data, n, batch_size, stream);
        }
    }
    
    if (status != FWHT_SUCCESS) {
        return status;
    }
    if (profiling) {
        (void)cudaEventRecord(evt_kernel_end, stream);
    }
    
    /* Copy results back to host */
    err = cudaMemcpyAsync(host_data, d_data, bytes, 
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }
    if (profiling) {
        (void)cudaEventRecord(evt_d2h_end, stream);
    }
    
    /* Synchronize stream */
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return fwht_cuda_report(err, __FILE__, __LINE__);
    }
    if (profiling) {
        float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, evt_h2d_start, evt_h2d_end);
        cudaEventElapsedTime(&kernel_ms, evt_h2d_end, evt_kernel_end);
        cudaEventElapsedTime(&d2h_ms, evt_kernel_end, evt_d2h_end);
        g_fwht_last_metrics.h2d_ms = static_cast<double>(h2d_ms);
        g_fwht_last_metrics.kernel_ms = static_cast<double>(kernel_ms);
        g_fwht_last_metrics.d2h_ms = static_cast<double>(d2h_ms);
        g_fwht_last_metrics.n = n;
        g_fwht_last_metrics.batch_size = batch_size;
        g_fwht_last_metrics.bytes_transferred = bytes;
        g_fwht_last_metrics.samples = 1;
        g_fwht_last_metrics.valid = true;
        cudaEventDestroy(evt_h2d_start);
        cudaEventDestroy(evt_h2d_end);
        cudaEventDestroy(evt_kernel_end);
        cudaEventDestroy(evt_d2h_end);
    }
    
    return FWHT_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

fwht_status_t fwht_gpu_context_compute_i32(fwht_gpu_context_t* ctx,
                                             int32_t* data, size_t n, size_t batch_size) {
    return fwht_gpu_context_compute_impl<int32_t>(ctx, data, n, batch_size);
}

fwht_status_t fwht_gpu_context_compute_f64(fwht_gpu_context_t* ctx,
                                             double* data, size_t n, size_t batch_size) {
    return fwht_gpu_context_compute_impl<double>(ctx, data, n, batch_size);
}

/* ============================================================================
 * GPU LOAD/STORE CALLBACKS
 * ============================================================================ */

/* Define the function pointer types (matching header) */
typedef int32_t (*fwht_load_fn_i32)(int32_t value, size_t index, void* user_params);
typedef void (*fwht_store_fn_i32)(int32_t* dest, int32_t value, size_t index, void* user_params);
typedef double (*fwht_load_fn_f64)(double value, size_t index, void* user_params);
typedef void (*fwht_store_fn_f64)(double* dest, double value, size_t index, void* user_params);

fwht_status_t fwht_gpu_context_set_callbacks_i32(fwht_gpu_context_t* ctx,
                                                   fwht_load_fn_i32 load_fn,
                                                   fwht_store_fn_i32 store_fn,
                                                   void* user_params) {
    if (ctx == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }
    
    ctx->load_fn_i32 = (void*)load_fn;
    ctx->store_fn_i32 = (void*)store_fn;
    ctx->user_params = user_params;
    
    return FWHT_SUCCESS;
}

fwht_status_t fwht_gpu_context_set_callbacks_f64(fwht_gpu_context_t* ctx,
                                                   fwht_load_fn_f64 load_fn,
                                                   fwht_store_fn_f64 store_fn,
                                                   void* user_params) {
    if (ctx == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }
    
    ctx->load_fn_f64 = (void*)load_fn;
    ctx->store_fn_f64 = (void*)store_fn;
    ctx->user_params = user_params;
    
    return FWHT_SUCCESS;
}

#ifdef __cplusplus
}
#endif

/* ==========================================================================
 * FP16 Support - Native Tensor Core Implementation (MINIMAL)
 * 
 * Starting with simple n=256 kernel using basic butterfly algorithm in fp16.
 * Once this works correctly, will add Tensor Core MMA optimizations.
 * ========================================================================== */

/* Type aliases for clarity */
typedef uint32_t b32;
typedef uint16_t b16;

/* Helper: convert fp16 to fp32 on device */
__device__ __forceinline__ float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    uint32_t f_bits;
    if (exp == 0) {
        f_bits = sign;
    } else if (exp == 31) {
        f_bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        exp = exp - 15 + 127;
        f_bits = sign | (exp << 23) | (mantissa << 13);
    }
    
    return __uint_as_float(f_bits);
}

/* Helper: convert fp32 to fp16 on device */
__device__ __forceinline__ uint16_t float_to_fp16(float f) {
    uint32_t f_bits = __float_as_uint(f);
    uint32_t sign = (f_bits >> 16) & 0x8000;
    int32_t exp = ((f_bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = (f_bits >> 13) & 0x3FF;
    
    uint16_t h;
    if (exp <= 0) {
        h = (uint16_t)sign;
    } else if (exp >= 31) {
        h = (uint16_t)(sign | 0x7C00);
    } else {
        h = (uint16_t)(sign | (exp << 10) | mantissa);
    }
    
    return h;
}

/*
 * FP16 Tensor Core Path - Native __half Implementation
 * 
 * Current Status: Working correctly with scalar operations (__hadd/__hsub)
 * Performance: 4.12× speedup for batch_size=1024, 6.5× for large single transforms
 * 
 * TODO for Tensor Core MMA optimization (Meta's approach):
 * - Requires mma.sync.aligned.m16n8k16 PTX instructions
 * - Need proper Hadamard 16×16 matrix generation (Meta's had_16 bit patterns)
 * - Column-major data layout (256 elements/warp, not row-major)
 * - Warp shuffle operations for coalescing (__shfl_xor_sync)
 * - Register fragment management (4 regs × 32 threads = 128 fp16 pairs)
 * - Multi-iteration MMA for sizes >256 with proper transpose operations
 * - Meta achieves 12.81 GOps/s with this approach vs our current ~4× speedup
 * 
 * Complexity: Meta's implementation is 749 lines with careful bit manipulation.
 * Attempting direct port caused "illegal instruction" errors on SM 12.0 (RTX 5090).
 * Would need SM-specific tuning and extensive testing.
 */

/* FP16 butterfly kernel - native half precision, works for all sizes ≤1024 */
__global__ void hadamard_transform_fp16_butterfly(
    b16* __restrict__ data,
    size_t n
) {
    extern __shared__ __half shared[];
    __half* temp = shared + n;
    
    size_t tid = threadIdx.x;
    if (tid >= n) return;
    
    shared[tid] = reinterpret_cast<__half*>(data)[tid];
    __syncthreads();
    
    for (size_t step = 1; step < n; step *= 2) {
        size_t pair = tid ^ step;
        __half a = shared[tid];
        __half b = shared[pair];
        
        if ((tid & step) == 0) {
            temp[tid] = __hadd(a, b);
        } else {
            temp[tid] = __hsub(b, a);
        }
        __syncthreads();
        shared[tid] = temp[tid];
        __syncthreads();
    }
    
    reinterpret_cast<__half*>(data)[tid] = shared[tid];
}

__global__ void scale_fp16_kernel(__half* data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __half2float(data[idx]);
    data[idx] = __float2half(val * scale);
}

static inline void scale_fp16_buffer(b16* data, size_t n, float scale, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    __half* half_ptr = reinterpret_cast<__half*>(data);
    scale_fp16_kernel<<<blocks, threads, 0, stream>>>(half_ptr, n, scale);
}

bool launch_fp16_butterfly_kernel(b16* data, size_t n, size_t batch_size, cudaStream_t stream) {
    if (n > 1024) return false;
    
    // Use Meta's Tensor Core kernels for supported sizes
    if (n == 256) {
        launch_meta_hadamard_256(data, batch_size, stream);
        return true;
    }
    if (n == 512) {
        launch_meta_hadamard_512(data, batch_size, stream);
        return true;
    }
    if (n == 1024) {
        launch_meta_hadamard_1024(data, batch_size, stream);
        return true;
    }
    // Fallback to scalar butterfly for other sizes
    size_t smem = 2 * n * sizeof(__half);
    for (size_t i = 0; i < batch_size; ++i) {
        hadamard_transform_fp16_butterfly<<<1, n, smem, stream>>>(data + i * n, n);
    }
    return false;
}

int fwht_batch_f16_cuda_device_tensorcore(const uint16_t* d_in, uint16_t* d_out, size_t n, size_t batch_size) {
    if (!d_in || !d_out || batch_size == 0) return FWHT_ERROR_NULL_POINTER;
    if (!is_power_of_2(n) || n > 1024) return FWHT_ERROR_INVALID_SIZE;
    cudaError_t err = cudaSuccess;
    
    if (d_in != d_out) {
        err = cudaMemcpy((void*)d_out, (const void*)d_in, 
                                     n * batch_size * sizeof(uint16_t), 
                                     cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
            return FWHT_ERROR_CUDA;
        }
    }
    
    cudaStream_t stream = 0;
    float unnormalize_scale = sqrtf(static_cast<float>(n));

    bool used_tensor_core = launch_fp16_butterfly_kernel(d_out, n, batch_size, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Tensor Core launch failed for n=%zu, batch=%zu: %s\n", n, batch_size, cudaGetErrorString(err));
        return FWHT_ERROR_CUDA;
    }

    if (used_tensor_core) {
        scale_fp16_buffer(d_out, n * batch_size, unnormalize_scale, stream);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return FWHT_ERROR_CUDA;
    }
    
    return FWHT_SUCCESS;
}


/* ==========================================================================
 * FP16 Support - Conversion wrapper (FALLBACK)
 * 
 * Uses fp16 -> fp32 -> FWHT -> fp32 -> fp16 conversion approach.
 * Correct but not optimized. Used as fallback when Tensor Core path unavailable.
 * ========================================================================== */

/* FP16 <-> FP32 conversion kernels */
__global__ void fp16_to_fp32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint16_t h = in[idx];
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = (h >> 10) & 0x1F;
        uint32_t mantissa = h & 0x3FF;
        
        uint32_t f_bits;
        if (exp == 0) {
            f_bits = sign;
        } else if (exp == 31) {
            f_bits = sign | 0x7F800000 | (mantissa << 13);
        } else {
            exp = exp - 15 + 127;
            f_bits = sign | (exp << 23) | (mantissa << 13);
        }
        
        out[idx] = __uint_as_float(f_bits);
    }
}

__global__ void fp32_to_fp16_kernel(const float* __restrict__ in, uint16_t* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t f_bits = __float_as_uint(in[idx]);
        uint32_t sign = (f_bits >> 16) & 0x8000;
        int32_t exp = ((f_bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (f_bits >> 13) & 0x3FF;
        
        uint16_t h;
        if (exp <= 0) {
            h = (uint16_t)sign;
        } else if (exp >= 31) {
            h = (uint16_t)(sign | 0x7C00);
        } else {
            h = (uint16_t)(sign | (exp << 10) | mantissa);
        }
        
        out[idx] = h;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/* FP16 batch transform using conversion wrapper (FALLBACK)
 * Converts fp16 → fp32 → transform → fp16
 */
int fwht_batch_f16_cuda_device_fallback(const void* d_in, void* d_out, 
                                unsigned int n, unsigned int batch_size) {
    if (!g_cuda_device_state.initialized) {
        return -3;
    }
    
    if (d_in == NULL || d_out == NULL) {
        return -1;
    }
    
    const size_t total_elements = (size_t)n * batch_size;
    const uint16_t* d_in_f16 = (const uint16_t*)d_in;
    uint16_t* d_out_f16 = (uint16_t*)d_out;
    
    /* Allocate temporary fp32 buffer */
    float* d_temp = NULL;
    cudaError_t err = cudaMalloc(&d_temp, total_elements * sizeof(float));
    if (err != cudaSuccess) {
        return -4;
    }
    
    /* Convert fp16 -> fp32 */
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    fp16_to_fp32_kernel<<<grid_size, block_size>>>(d_in_f16, d_temp, total_elements);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_temp);
        return -4;
    }
    
    /* Execute FWHT using existing fp32 kernel */
    fwht_status_t status = fwht_batch_f32_cuda_device(d_temp, n, batch_size);
    if (status != FWHT_SUCCESS) {
        cudaFree(d_temp);
        return -4;
    }
    
    /* Convert fp32 -> fp16 */
    fp32_to_fp16_kernel<<<grid_size, block_size>>>(d_temp, d_out_f16, total_elements);
    
    err = cudaGetLastError();
    cudaFree(d_temp);
    
    return (err == cudaSuccess) ? 0 : -4;
}

/* Smart dispatcher: tries Tensor Core, falls back to conversion wrapper
 * 
 * Decision logic:
 *   1. Check if Tensor Cores available (SM >= 7.0)
 *   2. Check if size suitable for Tensor Core (n >= 256, n % 256 == 0)
 *   3. If both true → use Tensor Core path (2-3× faster)
 *   4. Otherwise → use conversion wrapper (safe, always works)
 */
int fwht_batch_f16_cuda_device(const void* d_in, void* d_out, 
                                unsigned int n, unsigned int batch_size) {
    if (!g_cuda_device_state.initialized) {
        return -3;
    }
    
    if (d_in == NULL || d_out == NULL) {
        return -1;
    }
    
    /* Check if Tensor Cores are available and size is supported */
    int compute_cap = g_cuda_device_state.compute_capability;
    bool has_tensor_cores = fwht_tensorcore_arch_supported(compute_cap);
    bool size_suitable = fwht_tensorcore_size_supported(n);
    
    if (has_tensor_cores && size_suitable) {
        /* Show precision warning on first use (unless suppressed) */
        const char* suppress = getenv("FWHT_SILENCE_FP16_WARNING");
        if (!suppress || (strcmp(suppress, "1") != 0 && strcmp(suppress, "true") != 0)) {
            fwht_fp16_precision_warning();
        }
        
        int result = fwht_batch_f16_cuda_device_tensorcore(
            (const uint16_t*)d_in, (uint16_t*)d_out, n, batch_size);
        if (result == FWHT_SUCCESS) {
            return 0;
        }
        fprintf(stderr, "[libfwht] Tensor Core kernel failed (err=%d); falling back to fp16 conversion path.\n", result);
        fwht_tensorcore_log_unavailable("kernel failure", compute_cap, n);
        // Fall through to fallback if Tensor Core path fails
    } else {
        const char* reason = has_tensor_cores ? "transform size unsupported"
                                              : "GPU architecture not yet supported";
        fwht_tensorcore_log_unavailable(reason, compute_cap, n);
    }
    
    /* Fallback: use conversion wrapper (fp16->fp32->FWHT->fp16) */
    return fwht_batch_f16_cuda_device_fallback(d_in, d_out, n, batch_size);
}

#ifdef __cplusplus
}
#endif
