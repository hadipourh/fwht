#include "../include/fwht.h"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <string.h>

/*
 * GPU helper that builds LAT columns directly from an S-box table.
 * Lessons from the previous attempt:
 *   - Keep this logic isolated from fwht_cuda.cu so fp16 kernels stay untouched.
 *   - Cache device buffers and table uploads to avoid re-copying per column batch.
 *   - Reuse shared-memory tiles so each table row is read once per tile and
 *     fanned out across multiple mask columns (vectorized column synthesis).
 */

struct fwht_cuda_lat_cache {
    uint32_t* d_table;
    size_t table_capacity;
    const uint32_t* host_table; /* last table pointer to detect changes */
    size_t table_size;

    size_t* d_masks;
    size_t mask_capacity;

    int32_t* d_output;
    size_t output_capacity; /* number of int32_t elements */
};

static fwht_cuda_lat_cache g_lat_cache = {
    nullptr,
    0,
    nullptr,
    0,
    nullptr,
    0,
    nullptr,
    0
};

struct fwht_cuda_lat_device_info {
    bool initialized;
    int max_threads_per_block;
    int warp_size;
};

static fwht_cuda_lat_device_info g_lat_device = {false, 1024, 32};

static bool g_lat_env_ready = false;
static unsigned int g_lat_rows_target = 256;  /* threads along rows dimension */
static unsigned int g_lat_masks_target = 4;   /* threads along mask dimension */
static bool g_lat_disable_table_cache = false;
static cudaStream_t g_lat_stream = nullptr;
static bool g_lat_stream_ready = false;

static void fwht_cuda_lat_init_device(void) {
    if (g_lat_device.initialized) {
        return;
    }
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return; /* leave defaults so fallback path still works */
    }
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
        return;
    }
    g_lat_device.max_threads_per_block = props.maxThreadsPerBlock;
    g_lat_device.warp_size = props.warpSize > 0 ? props.warpSize : 32;
    g_lat_device.initialized = true;
}

static unsigned int fwht_cuda_lat_clamp_threads(unsigned int value) {
    fwht_cuda_lat_init_device();
    unsigned int limit = static_cast<unsigned int>(g_lat_device.max_threads_per_block);
    if (value == 0) {
        return 32u;
    }
    return std::max(32u, std::min(value, limit));
}

static unsigned int fwht_cuda_lat_align_multiple(unsigned int value) {
    fwht_cuda_lat_init_device();
    unsigned int warp = static_cast<unsigned int>(g_lat_device.warp_size);
    if (warp == 0) {
        warp = 32u;
    }
    return (value + warp - 1u) / warp * warp;
}

static bool fwht_cuda_lat_env_true(const char* value) {
    if (value == NULL || value[0] == '\0') {
        return false;
    }
    if (strcmp(value, "1") == 0 || strcmp(value, "on") == 0 || strcmp(value, "ON") == 0 ||
        strcmp(value, "true") == 0 || strcmp(value, "TRUE") == 0) {
        return true;
    }
    return false;
}

static void fwht_cuda_lat_parse_env(void) {
    if (g_lat_env_ready) {
        return;
    }
    g_lat_env_ready = true;
    const char* rows_env = getenv("FWHT_CUDA_LAT_ROWS");
    if (rows_env && rows_env[0] != '\0') {
        unsigned long val = strtoul(rows_env, nullptr, 10);
        if (val > 0) {
            g_lat_rows_target = fwht_cuda_lat_clamp_threads(
                fwht_cuda_lat_align_multiple(static_cast<unsigned int>(val))); 
        }
    } else {
        g_lat_rows_target = fwht_cuda_lat_clamp_threads(
            fwht_cuda_lat_align_multiple(g_lat_rows_target));
    }

    const char* masks_env = getenv("FWHT_CUDA_LAT_MASKS");
    if (masks_env && masks_env[0] != '\0') {
        unsigned long val = strtoul(masks_env, nullptr, 10);
        if (val > 0) {
            g_lat_masks_target = static_cast<unsigned int>(val);
        }
    }
    /* keep reasonable default */
    g_lat_masks_target = std::max(1u, std::min(g_lat_masks_target, 32u));

    const char* cache_env = getenv("FWHT_CUDA_LAT_DISABLE_TABLE_CACHE");
    g_lat_disable_table_cache = fwht_cuda_lat_env_true(cache_env);
}

static unsigned int fwht_cuda_lat_pick_rows(size_t size) {
    fwht_cuda_lat_parse_env();
    unsigned int target = g_lat_rows_target;
    target = std::min<unsigned int>(target, static_cast<unsigned int>(fwht_cuda_lat_clamp_threads(target)));
    while (target > size && target > 32u) {
        target >>= 1u;
    }
    size_t limit = size == 0 ? 32u : size;
    size_t clamped = std::min<size_t>(target, limit);
    return static_cast<unsigned int>(std::max<size_t>(static_cast<size_t>(32u), clamped));
}

static unsigned int fwht_cuda_lat_pick_masks(size_t mask_count,
                                             unsigned int rows_per_block) {
    fwht_cuda_lat_parse_env();
    unsigned int masks = g_lat_masks_target;
    if (mask_count < masks) {
        masks = static_cast<unsigned int>(mask_count);
    }
    if (masks == 0) {
        masks = 1u;
    }
    /* ensure total threads stay within device limit */
    unsigned int rows = std::max(32u, rows_per_block);
    unsigned int max_threads = fwht_cuda_lat_clamp_threads(rows);
    while (rows * masks > max_threads && masks > 1u) {
        masks >>= 1u;
    }
    return masks == 0 ? 1u : masks;
}

static cudaStream_t fwht_cuda_lat_get_stream(cudaStream_t user_stream) {
    if (user_stream != nullptr) {
        return user_stream;
    }
    if (!g_lat_stream_ready) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_lat_stream, cudaStreamNonBlocking);
        if (err == cudaSuccess) {
            g_lat_stream_ready = true;
        } else {
            g_lat_stream = nullptr;
            (void)cudaGetLastError();
        }
    }
    return g_lat_stream;
}

static fwht_status_t fwht_cuda_lat_check(cudaError_t err, const char* call_site) {
    if (err == cudaSuccess) {
        return FWHT_SUCCESS;
    }
    if (err == cudaErrorNoDevice) {
        return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }
    fprintf(stderr,
            "[libfwht][cuda-lat] %s failed: %s\n",
            call_site,
            cudaGetErrorString(err));
    return FWHT_ERROR_CUDA;
}

#define CUDA_LAT_TRY(expr) do { \
    fwht_status_t _status = fwht_cuda_lat_check((expr), #expr); \
    if (_status != FWHT_SUCCESS) { \
        return _status; \
    } \
} while (0)

__device__ __forceinline__ int fwht_cuda_lat_parity(size_t bits) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
    return (__popcll(static_cast<unsigned long long>(bits)) & 1);
#else
    bits ^= bits >> 1;
    bits ^= bits >> 2;
    bits = (bits & 0x1111111111111111ULL) * 0x1111111111111111ULL;
    return static_cast<int>((bits >> 60) & 1u);
#endif
}

__global__ void fwht_cuda_lat_kernel(const uint32_t* __restrict__ table,
                                     size_t size,
                                     const size_t* __restrict__ masks,
                                     size_t mask_count,
                                     int32_t* __restrict__ out,
                                     unsigned int rows_per_block,
                                     unsigned int masks_per_block) {
    extern __shared__ uint32_t shared_rows[];

    const size_t row = blockIdx.x * static_cast<size_t>(rows_per_block) + threadIdx.x;
    const size_t mask_tile = blockIdx.y * static_cast<size_t>(masks_per_block) + threadIdx.y;

    if (threadIdx.y == 0 && row < size) {
#if __CUDA_ARCH__ >= 350
        shared_rows[threadIdx.x] = __ldg(&table[row]);
#else
        shared_rows[threadIdx.x] = table[row];
#endif
    }

    __syncthreads();

    if (row >= size || mask_tile >= mask_count) {
        return;
    }

    uint32_t value = shared_rows[threadIdx.x];
#if __CUDA_ARCH__ >= 350
    size_t mask = __ldg(&masks[mask_tile]);
#else
    size_t mask = masks[mask_tile];
#endif
    size_t bits = static_cast<size_t>(value) & mask;
    int parity = fwht_cuda_lat_parity(bits);
    out[mask_tile * size + row] = parity ? -1 : 1;
}

static fwht_status_t fwht_cuda_lat_reserve_table(size_t size) {
    if (size == 0) {
        return FWHT_ERROR_INVALID_SIZE;
    }
    if (g_lat_cache.table_capacity >= size) {
        return FWHT_SUCCESS;
    }
    if (g_lat_cache.d_table) {
        cudaFree(g_lat_cache.d_table);
        g_lat_cache.d_table = nullptr;
        g_lat_cache.table_capacity = 0;
    }
    CUDA_LAT_TRY(cudaMalloc(&g_lat_cache.d_table, size * sizeof(uint32_t)));
    g_lat_cache.table_capacity = size;
    g_lat_cache.host_table = nullptr; /* force upload */
    return FWHT_SUCCESS;
}

static fwht_status_t fwht_cuda_lat_reserve_masks(size_t count) {
    if (count == 0) {
        return FWHT_SUCCESS;
    }
    if (g_lat_cache.mask_capacity >= count) {
        return FWHT_SUCCESS; 
    }
    if (g_lat_cache.d_masks) {
        cudaFree(g_lat_cache.d_masks);
        g_lat_cache.d_masks = nullptr;
        g_lat_cache.mask_capacity = 0;
    }
    CUDA_LAT_TRY(cudaMalloc(&g_lat_cache.d_masks, count * sizeof(size_t)));
    g_lat_cache.mask_capacity = count;
    return FWHT_SUCCESS;
}

static fwht_status_t fwht_cuda_lat_reserve_output(size_t elements) {
    if (elements == 0) {
        return FWHT_SUCCESS;
    }
    if (g_lat_cache.output_capacity >= elements) {
        return FWHT_SUCCESS;
    }
    if (g_lat_cache.d_output) {
        cudaFree(g_lat_cache.d_output);
        g_lat_cache.d_output = nullptr;
        g_lat_cache.output_capacity = 0;
    }
    CUDA_LAT_TRY(cudaMalloc(&g_lat_cache.d_output, elements * sizeof(int32_t)));
    g_lat_cache.output_capacity = elements;
    return FWHT_SUCCESS;
}

static fwht_status_t fwht_cuda_lat_build_internal(const uint32_t* table,
                                                   size_t size,
                                                   const size_t* masks,
                                                   size_t mask_count,
                                                   int32_t* d_out,
                                                   int32_t* h_out,
                                                   cudaStream_t user_stream,
                                                   bool synchronize) {
    if (table == NULL || masks == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }
    if (size == 0 || mask_count == 0) {
        return FWHT_SUCCESS;
    }
    if (mask_count > 65535u) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    if (size > SIZE_MAX / mask_count) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    int device = 0;
    cudaError_t device_err = cudaGetDevice(&device);
    if (device_err == cudaErrorNoDevice) {
        return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }
    if (device_err != cudaSuccess) {
        return fwht_cuda_lat_check(device_err, "cudaGetDevice");
    }

    cudaStream_t stream = fwht_cuda_lat_get_stream(user_stream);

    fwht_status_t status = fwht_cuda_lat_reserve_table(size);
    if (status != FWHT_SUCCESS) {
        return status;
    }
    status = fwht_cuda_lat_reserve_masks(mask_count);
    if (status != FWHT_SUCCESS) {
        return status;
    }
    if (h_out == NULL) {
        /* Device output path */
        if (d_out == NULL) {
            return FWHT_ERROR_NULL_POINTER;
        }
    } else {
        status = fwht_cuda_lat_reserve_output(size * mask_count);
        if (status != FWHT_SUCCESS) {
            return status;
        }
    }

    if (g_lat_disable_table_cache ||
        g_lat_cache.host_table != table || g_lat_cache.table_size != size) {
        CUDA_LAT_TRY(cudaMemcpyAsync(g_lat_cache.d_table,
                                     table,
                                     size * sizeof(uint32_t),
                                     cudaMemcpyHostToDevice,
                                     stream));
        g_lat_cache.host_table = table;
        g_lat_cache.table_size = size;
    }

    CUDA_LAT_TRY(cudaMemcpyAsync(g_lat_cache.d_masks,
                                 masks,
                                 mask_count * sizeof(size_t),
                                 cudaMemcpyHostToDevice,
                                 stream));

    unsigned int rows_per_block = fwht_cuda_lat_pick_rows(size);
    unsigned int masks_per_block = fwht_cuda_lat_pick_masks(mask_count, rows_per_block);
    fwht_cuda_lat_init_device();
    unsigned int max_threads = static_cast<unsigned int>(g_lat_device.max_threads_per_block);
    const unsigned int threads_per_block = rows_per_block * masks_per_block;
    if (threads_per_block == 0 || threads_per_block > max_threads) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    unsigned int grid_x = static_cast<unsigned int>((size + rows_per_block - 1u) / rows_per_block);
    unsigned int grid_y = static_cast<unsigned int>((mask_count + masks_per_block - 1u) / masks_per_block);
    if (grid_y > 65535u) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    dim3 block(rows_per_block, masks_per_block, 1u);
    dim3 grid(grid_x == 0 ? 1u : grid_x, grid_y == 0 ? 1u : grid_y, 1u);
    size_t shared_bytes = static_cast<size_t>(rows_per_block) * sizeof(uint32_t);

    int32_t* kernel_out = (h_out != NULL) ? g_lat_cache.d_output : d_out;

    fwht_cuda_lat_kernel<<<grid, block, shared_bytes, stream>>>(g_lat_cache.d_table,
                                                               size,
                                                               g_lat_cache.d_masks,
                                                               mask_count,
                                                               kernel_out,
                                                               rows_per_block,
                                                               masks_per_block);

    CUDA_LAT_TRY(cudaGetLastError());

    if (h_out != NULL) {
        CUDA_LAT_TRY(cudaMemcpyAsync(h_out,
                                     kernel_out,
                                     size * mask_count * sizeof(int32_t),
                                     cudaMemcpyDeviceToHost,
                                     stream));
    }

    if (synchronize) {
        CUDA_LAT_TRY(cudaStreamSynchronize(stream));
    }

    return FWHT_SUCCESS;
}

extern "C" fwht_status_t fwht_cuda_lat_build_direct(const uint32_t* table,
                                                     size_t size,
                                                     const size_t* masks,
                                                     size_t mask_count,
                                                     int32_t* dst) {
    if (dst == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }
    return fwht_cuda_lat_build_internal(table,
                                        size,
                                        masks,
                                        mask_count,
                                        nullptr,
                                        dst,
                                        nullptr,
                                        true);
}

extern "C" fwht_status_t fwht_cuda_lat_build_device(const uint32_t* table,
                                                     size_t size,
                                                     const size_t* masks,
                                                     size_t mask_count,
                                                     int32_t* d_dst,
                                                     cudaStream_t stream) {
    if (d_dst == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }
    return fwht_cuda_lat_build_internal(table,
                                        size,
                                        masks,
                                        mask_count,
                                        d_dst,
                                        nullptr,
                                        stream,
                                        (stream == nullptr));
}

extern "C" void fwht_cuda_lat_direct_release_cache(void) {
    if (g_lat_cache.d_table) {
        cudaFree(g_lat_cache.d_table);
    }
    if (g_lat_cache.d_masks) {
        cudaFree(g_lat_cache.d_masks);
    }
    if (g_lat_cache.d_output) {
        cudaFree(g_lat_cache.d_output);
    }
    if (g_lat_stream_ready && g_lat_stream != nullptr) {
        cudaStreamDestroy(g_lat_stream);
    }
    g_lat_stream = nullptr;
    g_lat_stream_ready = false;
    g_lat_cache = fwht_cuda_lat_cache{nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0};
}

#endif /* USE_CUDA */
