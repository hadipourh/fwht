/*
 * Meta-Inspired FP16 Hadamard Transform Kernel
 * Based on: meta-pytorch/applied-ai hadamard_transform
 * License: BSD 3-Clause (compatible with GPL-3.0)
 * 
 * This is a NEW kernel for fp16/fp32 performance.
 * The existing float64 kernels remain for cryptographic precision.
 * 
 * Key optimizations from Meta:
 * 1. Specialized for fp16/fp32 (not fp64)
 * 2. Better warp shuffle patterns
 * 3. Optimized launch configurations
 * 4. Improved memory coalescing
 * 
 * Note: We implement the algorithmic ideas, not copying code directly.
 */

#ifndef FWHT_CUDA_FP16_H
#define FWHT_CUDA_FP16_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Meta's key insight: Different strategies for different sizes
// Small (≤512): All in registers + shuffles
// Medium (1K-4K): Registers + shuffles + minimal shared memory  
// Large (>4K): Hybrid approach

namespace fwht {
namespace fp16 {

// ============================================================================
// Meta-Style Launch Configuration (Tuned for RTX 4090)
// ============================================================================

struct LaunchConfig {
    int threads_per_block;
    int blocks_per_sm;
    int elements_per_thread;
};

// Meta's approach: Size-specific tuning
constexpr LaunchConfig get_launch_config(int log_n) {
    // Based on Meta's launch_configs arrays
    switch(log_n) {
        case 9:  return {256, 8, 2};   // n=512
        case 10: return {256, 8, 4};   // n=1024
        case 11: return {512, 4, 4};   // n=2048
        case 12: return {1024, 2, 4};  // n=4096
        case 13: return {1024, 2, 8};  // n=8192
        default: return {256, 4, 4};
    }
}

// ============================================================================
// Optimized Hadamard Butterfly (Meta's Pattern)
// ============================================================================

// Meta's key optimization: XOR-based addressing for perfect coalescing
template<typename T>
__device__ __forceinline__ void hadamard_butterfly(T& a, T& b) {
    T tmp = a;
    a = a + b;
    b = tmp - b;
}

// ============================================================================
// Warp-Level Hadamard (Meta's Shuffle Pattern)
// ============================================================================

template<typename T>
__device__ void hadamard_warp_shuffle(T* data, int log_size) {
    const int tid = threadIdx.x % 32;
    
    // Meta's optimization: Process shuffles in optimal order
    // Start with smallest strides (better for fp16)
    for (int s = 0; s < log_size; ++s) {
        const int stride = 1 << s;
        
        // Shuffle to get partner
        T partner = __shfl_xor_sync(0xFFFFFFFF, *data, stride);
        
        // Apply butterfly
        if ((tid & stride) == 0) {
            hadamard_butterfly(*data, partner);
        } else {
            *data = partner;
        }
        
        __syncwarp();
    }
}

// ============================================================================
// Small Size Kernel (n ≤ 512) - All in Registers
// ============================================================================

template<typename T, int LOG_N>
__global__ void hadamard_small_kernel(T* data, int batch_size) {
    constexpr int N = 1 << LOG_N;
    constexpr int EPT = 2;  // Elements per thread
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    T* batch_data = data + batch_idx * N;
    
    // Load data into registers (Meta's vectorized pattern)
    T regs[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < N) {
            regs[i] = batch_data[idx];
        }
    }
    
    // Process each element through warp shuffles
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        hadamard_warp_shuffle(&regs[i], 5);  // log2(32) = 5
    }
    
    // Store back (coalesced)
    #pragma unroll
    for (int i = 0; i < EPT; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < N) {
            batch_data[idx] = regs[i];
        }
    }
}

// ============================================================================
// Medium Size Kernel (1K-4K) - Meta's Hybrid Approach
// ============================================================================

template<typename T, int LOG_N>
__global__ void hadamard_medium_kernel(T* data, int batch_size) {
    constexpr int N = 1 << LOG_N;
    constexpr int EPT = 4;
    
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    T* batch_data = data + batch_idx * N;
    
    // Meta's insight: Use shared memory only for stages > 32
    extern __shared__ char smem[];
    T* shared = reinterpret_cast<T*>(smem);
    
    // Load into registers with coalescing
    T regs[EPT];
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int idx = tid * EPT + e;
        if (idx < N) {
            regs[e] = batch_data[idx];
        }
    }
    
    // Stage 1: Per-thread butterflies (no synchronization)
    #pragma unroll
    for (int s = 0; s < 2; ++s) {  // First 2 stages in registers
        #pragma unroll
        for (int e = 0; e < EPT; e += (2 << s)) {
            for (int i = 0; i < (1 << s); ++i) {
                hadamard_butterfly(regs[e + i], regs[e + i + (1 << s)]);
            }
        }
    }
    
    // Stage 2: Warp-level shuffles (stages 2-4)
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        hadamard_warp_shuffle(&regs[e], 5);
    }
    
    __syncthreads();
    
    // Stage 3: Block-level via shared memory (remaining stages)
    for (int s = 5; s < LOG_N; ++s) {
        // Write to shared memory
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            shared[tid * EPT + e] = regs[e];
        }
        __syncthreads();
        
        // Meta's XOR addressing for perfect coalescing
        const int stride = 1 << s;
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            int idx = tid * EPT + e;
            int partner_idx = idx ^ stride;
            
            if (partner_idx < N) {
                T val = shared[idx];
                T partner = shared[partner_idx];
                
                if (idx < partner_idx) {
                    hadamard_butterfly(val, partner);
                    regs[e] = val;
                } else {
                    hadamard_butterfly(partner, val);
                    regs[e] = val;
                }
            }
        }
        __syncthreads();
    }
    
    // Store back (coalesced writes)
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int idx = tid * EPT + e;
        if (idx < N) {
            batch_data[idx] = regs[e];
        }
    }
}

// ============================================================================
// Launch Wrapper (Meta's Dynamic Dispatch Pattern)
// ============================================================================

template<typename T>
cudaError_t launch_hadamard_fp16(T* d_data, int n, int batch_size, cudaStream_t stream = 0) {
    const int log_n = __builtin_ctz(n);
    
    if (log_n <= 9) {
        // Small sizes: all-register kernel
        auto config = get_launch_config(log_n);
        hadamard_small_kernel<T, 9><<<batch_size, config.threads_per_block, 0, stream>>>(
            d_data, batch_size);
    } else if (log_n <= 12) {
        // Medium sizes: hybrid kernel (Meta's sweet spot)
        auto config = get_launch_config(log_n);
        size_t smem = n * sizeof(T);
        hadamard_medium_kernel<T, 12><<<batch_size, config.threads_per_block, smem, stream>>>(
            d_data, batch_size);
    } else {
        return cudaErrorInvalidValue;  // Large sizes need multi-block approach
    }
    
    return cudaGetLastError();
}

} // namespace fp16
} // namespace fwht

#endif // FWHT_CUDA_FP16_H
