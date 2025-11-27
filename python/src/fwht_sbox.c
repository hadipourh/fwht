/*
 * Fast Walsh-Hadamard Transform - S-box Analysis Helpers
 *
 * Implements vectorial Boolean (S-box) analysis using existing batch FWHT
 * facilities. Converts lookup tables to Boolean components, computes Walsh
 * spectra, and optionally builds Linear Approximation Tables (LATs).
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

#define _POSIX_C_SOURCE 200809L

#include "fwht.h"
#include "fwht_internal.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE_CUDA
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <cuda_runtime_api.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
#if defined(_OPENMP)
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <omp.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define FWHT_HAVE_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define FWHT_HAVE_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FWHT_HAVE_NEON 1
#endif

#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif

#ifdef USE_CUDA
typedef struct {
    bool ready;
    cudaStream_t stream;
    int32_t* d_components;
    size_t components_capacity;
    int32_t* d_lat_columns;
    size_t lat_capacity;
} fwht_sbox_gpu_workspace_t;

static fwht_sbox_gpu_workspace_t g_sbox_gpu_ws = {false, NULL, NULL, 0, NULL, 0};

static fwht_status_t fwht_sbox_gpu_status(cudaError_t err, const char* what) {
    if (err == cudaSuccess) {
        return FWHT_SUCCESS;
    }
    fprintf(stderr, "[libfwht][sbox][cuda] %s failed: %s\n", what, cudaGetErrorString(err));
    return FWHT_ERROR_CUDA;
}

static fwht_status_t fwht_sbox_gpu_workspace_init(void) {
    if (g_sbox_gpu_ws.ready) {
        return FWHT_SUCCESS;
    }
    cudaError_t err = cudaStreamCreateWithFlags(&g_sbox_gpu_ws.stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        g_sbox_gpu_ws.stream = NULL;
        return fwht_sbox_gpu_status(err, "cudaStreamCreateWithFlags");
    }
    g_sbox_gpu_ws.ready = true;
    return FWHT_SUCCESS;
}

static fwht_status_t fwht_sbox_gpu_reserve_buffer(int32_t** device_ptr,
                                                   size_t* capacity,
                                                   size_t required_elements) {
    if (required_elements == 0) {
        return FWHT_SUCCESS;
    }
    if (*capacity >= required_elements) {
        return FWHT_SUCCESS;
    }
    if (*device_ptr != NULL) {
        cudaFree(*device_ptr);
        *device_ptr = NULL;
        *capacity = 0;
    }
    cudaError_t err = cudaMalloc((void**)device_ptr, required_elements * sizeof(int32_t));
    if (err != cudaSuccess) {
        return fwht_sbox_gpu_status(err, "cudaMalloc");
    }
    *capacity = required_elements;
    return FWHT_SUCCESS;
}

static fwht_status_t fwht_sbox_gpu_transform_components(int32_t* host_data,
                                                         size_t size,
                                                         size_t component_count,
                                                         double* elapsed_ms,
                                                         int profile_timings) {
    if (component_count == 0 || size == 0) {
        return FWHT_SUCCESS;
    }
    fwht_status_t status = fwht_sbox_gpu_workspace_init();
    if (status != FWHT_SUCCESS) {
        return status;
    }
    size_t total_elements = size * component_count;
    status = fwht_sbox_gpu_reserve_buffer(&g_sbox_gpu_ws.d_components,
                                          &g_sbox_gpu_ws.components_capacity,
                                          total_elements);
    if (status != FWHT_SUCCESS) {
        return status;
    }

    size_t bytes = total_elements * sizeof(int32_t);
    bool pinned = false;
    cudaError_t err = cudaHostRegister(host_data, bytes, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        pinned = true;
    }

    cudaEvent_t evt_start = NULL;
    cudaEvent_t evt_end = NULL;
    if (profile_timings) {
        cudaEventCreateWithFlags(&evt_start, cudaEventDefault);
        cudaEventCreateWithFlags(&evt_end, cudaEventDefault);
        if (evt_start && evt_end) {
            cudaEventRecord(evt_start, g_sbox_gpu_ws.stream);
        }
    }

    err = cudaMemcpyAsync(g_sbox_gpu_ws.d_components,
                          host_data,
                          bytes,
                          cudaMemcpyHostToDevice,
                          g_sbox_gpu_ws.stream);
    if (err != cudaSuccess) {
        status = fwht_sbox_gpu_status(err, "cudaMemcpyAsync(H2D)");
        goto components_cleanup;
    }

    status = fwht_batch_i32_cuda_device_async(g_sbox_gpu_ws.d_components,
                                              size,
                                              component_count,
                                              g_sbox_gpu_ws.stream);
    if (status != FWHT_SUCCESS) {
        goto components_cleanup;
    }

    err = cudaMemcpyAsync(host_data,
                          g_sbox_gpu_ws.d_components,
                          bytes,
                          cudaMemcpyDeviceToHost,
                          g_sbox_gpu_ws.stream);
    if (err != cudaSuccess) {
        status = fwht_sbox_gpu_status(err, "cudaMemcpyAsync(D2H)");
        goto components_cleanup;
    }

    if (profile_timings && evt_start && evt_end) {
        cudaEventRecord(evt_end, g_sbox_gpu_ws.stream);
    }

    err = cudaStreamSynchronize(g_sbox_gpu_ws.stream);
    if (err != cudaSuccess) {
        status = fwht_sbox_gpu_status(err, "cudaStreamSynchronize");
        goto components_cleanup;
    }

    if (profile_timings && evt_start && evt_end && elapsed_ms != NULL) {
        cudaEventSynchronize(evt_end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evt_start, evt_end);
        *elapsed_ms += (double)ms / 1000.0;
    }

components_cleanup:
    if (evt_start) cudaEventDestroy(evt_start);
    if (evt_end) cudaEventDestroy(evt_end);
    if (pinned) cudaHostUnregister(host_data);
    return status;
}

static fwht_status_t fwht_sbox_gpu_reserve_lat_buffer(size_t size, size_t batch_capacity) {
    size_t required = size * batch_capacity;
    return fwht_sbox_gpu_reserve_buffer(&g_sbox_gpu_ws.d_lat_columns,
                                        &g_sbox_gpu_ws.lat_capacity,
                                        required);
}

static fwht_status_t fwht_sbox_gpu_process_lat_batch(const uint32_t* table,
                                                     size_t size,
                                                     const size_t* mask_list,
                                                     size_t mask_count,
                                                     int32_t* host_buffer,
                                                     double* lat_column_ms,
                                                     double* lat_fwht_ms,
                                                     int profile_timings) {
    if (mask_count == 0 || size == 0) {
        return FWHT_SUCCESS;
    }
    fwht_status_t status = fwht_sbox_gpu_workspace_init();
    if (status != FWHT_SUCCESS) {
        return status;
    }
    status = fwht_sbox_gpu_reserve_lat_buffer(size, mask_count);
    if (status != FWHT_SUCCESS) {
        return status;
    }

    cudaEvent_t evt_columns_start = NULL;
    cudaEvent_t evt_columns_end = NULL;
    cudaEvent_t evt_fwht_end = NULL;
    cudaError_t err = cudaSuccess;
    if (profile_timings) {
        cudaEventCreateWithFlags(&evt_columns_start, cudaEventDefault);
        cudaEventCreateWithFlags(&evt_columns_end, cudaEventDefault);
        cudaEventCreateWithFlags(&evt_fwht_end, cudaEventDefault);
        if (evt_columns_start) {
            cudaEventRecord(evt_columns_start, g_sbox_gpu_ws.stream);
        }
    }

    status = fwht_cuda_lat_build_device(table,
                                        size,
                                        mask_list,
                                        mask_count,
                                        g_sbox_gpu_ws.d_lat_columns,
                                        g_sbox_gpu_ws.stream);
    if (status != FWHT_SUCCESS) {
        goto lat_gpu_cleanup;
    }

    if (profile_timings && evt_columns_end) {
        cudaEventRecord(evt_columns_end, g_sbox_gpu_ws.stream);
    }

    status = fwht_batch_i32_cuda_device_async(g_sbox_gpu_ws.d_lat_columns,
                                              size,
                                              mask_count,
                                              g_sbox_gpu_ws.stream);
    if (status != FWHT_SUCCESS) {
        goto lat_gpu_cleanup;
    }

    if (profile_timings && evt_fwht_end) {
        cudaEventRecord(evt_fwht_end, g_sbox_gpu_ws.stream);
    }

    err = cudaMemcpyAsync(host_buffer,
                          g_sbox_gpu_ws.d_lat_columns,
                          size * mask_count * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          g_sbox_gpu_ws.stream);
    if (err != cudaSuccess) {
        status = fwht_sbox_gpu_status(err, "cudaMemcpyAsync(lat D2H)");
        goto lat_gpu_cleanup;
    }

    err = cudaStreamSynchronize(g_sbox_gpu_ws.stream);
    if (err != cudaSuccess) {
        status = fwht_sbox_gpu_status(err, "cudaStreamSynchronize(lat)");
        goto lat_gpu_cleanup;
    }

    if (profile_timings && evt_columns_start && evt_columns_end && lat_column_ms != NULL) {
        cudaEventSynchronize(evt_columns_end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evt_columns_start, evt_columns_end);
        *lat_column_ms += (double)ms / 1000.0;
    }
    if (profile_timings && evt_columns_end && evt_fwht_end && lat_fwht_ms != NULL) {
        cudaEventSynchronize(evt_fwht_end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, evt_columns_end, evt_fwht_end);
        *lat_fwht_ms += (double)ms / 1000.0;
    }

lat_gpu_cleanup:
    if (evt_columns_start) cudaEventDestroy(evt_columns_start);
    if (evt_columns_end) cudaEventDestroy(evt_columns_end);
    if (evt_fwht_end) cudaEventDestroy(evt_fwht_end);
    return status;
}
#endif /* USE_CUDA */

static size_t fwht_bit_length_u32(uint32_t value) {
    size_t bits = 0;
    while (value != 0) {
        bits++;
        value >>= 1;
    }
    return bits;
}

static inline int32_t fwht_i32_abs(int32_t v) {
    return (v < 0) ? -v : v;
}

static void fwht_lat_fill_ones(int32_t* dst, size_t size) {
#if defined(FWHT_HAVE_AVX512)
    const __m512i ones512 = _mm512_set1_epi32(1);
    size_t i = 0;
    size_t vec16_end = size & ~(size_t)15;
    for (; i < vec16_end; i += 16) {
        _mm512_storeu_si512((__m512i*)(dst + i), ones512);
    }
    for (; i < size; ++i) {
        dst[i] = 1;
    }
#elif defined(FWHT_HAVE_AVX2)
    const __m256i ones = _mm256_set1_epi32(1);
    size_t i = 0;
    size_t vec_end = size & ~(size_t)7;
    for (; i < vec_end; i += 8) {
        _mm256_storeu_si256((__m256i*)(dst + i), ones);
    }
    for (; i < size; ++i) {
        dst[i] = 1;
    }
#elif defined(FWHT_HAVE_NEON)
    const int32x4_t ones = vdupq_n_s32(1);
    size_t i = 0;
    size_t vec16_end = size & ~(size_t)15;
    for (; i < vec16_end; i += 16) {
        vst1q_s32(dst + i, ones);
        vst1q_s32(dst + i + 4, ones);
        vst1q_s32(dst + i + 8, ones);
        vst1q_s32(dst + i + 12, ones);
    }
    size_t vec4_end = size & ~(size_t)3;
    for (; i < vec4_end; i += 4) {
        vst1q_s32(dst + i, ones);
    }
    for (; i < size; ++i) {
        dst[i] = 1;
    }
#else
    for (size_t i = 0; i < size; ++i) {
        dst[i] = 1;
    }
#endif
}


static inline int32_t fwht_lat_sign_from_table(uint32_t value, size_t mask) {
    size_t bits = ((size_t)value) & mask;
#if defined(__GNUC__) || defined(__clang__)
    return (__builtin_parityll((unsigned long long)bits) ? -1 : 1);
#else
    bits ^= bits >> 1;
    bits ^= bits >> 2;
    bits = (bits & 0x1111111111111111ULL) * 0x1111111111111111ULL;
    return ((bits >> 60) & 1u) ? -1 : 1;
#endif
}

static void fwht_lat_build_from_table(const uint32_t* table,
                                      size_t size,
                                      size_t mask,
                                      int32_t* dst) {
    if (mask == 0) {
        fwht_lat_fill_ones(dst, size);
        return;
    }
    for (size_t i = 0; i < size; ++i) {
        dst[i] = fwht_lat_sign_from_table(table[i], mask);
    }
}

typedef struct {
    size_t size;
    size_t m;
    size_t n;
} fwht_sbox_shape_t;

static fwht_status_t fwht_sbox_resolve_shape(const uint32_t* table,
                                             size_t size,
                                             fwht_sbox_shape_t* shape) {
    if (table == NULL || shape == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }
    if (size == 0 || !fwht_is_power_of_2(size)) {
        return FWHT_ERROR_INVALID_SIZE;
    }

    uint32_t max_value = 0;
    for (size_t i = 0; i < size; ++i) {
        if (table[i] > max_value) {
            max_value = table[i];
        }
    }

    if (max_value >= size) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    size_t component_count = fwht_bit_length_u32(max_value);
    if (component_count == 0) {
        component_count = 1;
    }
    if (component_count > (size_t)INT_MAX) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    shape->size = size;
    shape->m = (size_t)fwht_log2(size);
    shape->n = component_count;
    return FWHT_SUCCESS;
}

static double fwht_now_seconds(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq;
    LARGE_INTEGER counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

fwht_status_t fwht_sbox_analyze_components(const uint32_t* table,
                                            size_t size,
                                            const fwht_sbox_component_request_t* request,
                                            fwht_sbox_component_metrics_t* metrics) {
    if (metrics == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }

    fwht_sbox_shape_t shape;
    fwht_status_t status = fwht_sbox_resolve_shape(table, size, &shape);
    if (status != FWHT_SUCCESS) {
        return status;
    }

    fwht_sbox_component_metrics_t tmp = {0};
    tmp.size = shape.size;
    tmp.m = shape.m;
    tmp.n = shape.n;

    fwht_backend_t backend = (request != NULL) ? request->backend : FWHT_BACKEND_AUTO;
    if (backend < FWHT_BACKEND_AUTO || backend > FWHT_BACKEND_GPU) {
        backend = FWHT_BACKEND_AUTO;
    }
    const bool profile_timings = (request != NULL && request->profile_timings);
#ifdef USE_CUDA
    const bool prefer_gpu_backend = (backend == FWHT_BACKEND_GPU);
#else
    const bool prefer_gpu_backend = false;
#endif
    const bool require_context = (!prefer_gpu_backend && backend != FWHT_BACKEND_AUTO);

    size_t component_count = shape.n;
    if (component_count != 0 && shape.size > SIZE_MAX / component_count) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    size_t component_total = component_count * shape.size;
    if (component_total > SIZE_MAX / sizeof(int32_t)) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    int32_t* component_data = NULL;
    bool component_owned = false;
    if (request != NULL && request->spectra != NULL) {
        component_data = request->spectra;
    } else {
        component_data = (int32_t*)malloc(component_total * sizeof(int32_t));
        if (component_data == NULL) {
            return FWHT_ERROR_OUT_OF_MEMORY;
        }
        component_owned = true;
    }

    for (size_t bit = 0; bit < component_count; ++bit) {
        int32_t* row = component_data + bit * shape.size;
        for (size_t i = 0; i < shape.size; ++i) {
            row[i] = ((table[i] >> bit) & 1u) ? -1 : 1;
        }
    }

    int32_t** component_ptrs = (int32_t**)malloc(component_count * sizeof(int32_t*));
    if (component_ptrs == NULL) {
        if (component_owned) {
            free(component_data);
        }
        return FWHT_ERROR_OUT_OF_MEMORY;
    }
    for (size_t bit = 0; bit < component_count; ++bit) {
        component_ptrs[bit] = component_data + bit * shape.size;
    }

    fwht_context_t* ctx = NULL;
    if (require_context) {
        fwht_config_t cfg = fwht_default_config();
        cfg.backend = backend;
        ctx = fwht_create_context(&cfg);
        if (ctx == NULL) {
            free(component_ptrs);
            if (component_owned) {
                free(component_data);
            }
            return FWHT_ERROR_OUT_OF_MEMORY;
        }
    }

    double component_fwht_seconds = 0.0;
#ifdef USE_CUDA
    if (prefer_gpu_backend) {
        status = fwht_sbox_gpu_transform_components(component_data,
                                                    shape.size,
                                                    component_count,
                                                    &component_fwht_seconds,
                                                    profile_timings);
    } else
#endif
    {
        double component_start = profile_timings ? fwht_now_seconds() : 0.0;
        status = fwht_batch_i32(ctx, component_ptrs, shape.size, (int)component_count);
        if (profile_timings) {
            component_fwht_seconds += fwht_now_seconds() - component_start;
        }
    }

    if (status != FWHT_SUCCESS) {
        if (ctx) {
            fwht_destroy_context(ctx);
        }
        free(component_ptrs);
        if (component_owned) {
            free(component_data);
        }
        return status;
    }

    int32_t global_max_walsh = 0;
    double min_nonlinearity = (double)shape.size * 0.5;
    for (size_t bit = 0; bit < component_count; ++bit) {
        int32_t bit_max = 0;
        int32_t* spectrum = component_data + bit * shape.size;
        for (size_t i = 0; i < shape.size; ++i) {
            int32_t abs_val = fwht_i32_abs(spectrum[i]);
            if (abs_val > bit_max) {
                bit_max = abs_val;
            }
        }
        if (bit_max > global_max_walsh) {
            global_max_walsh = bit_max;
        }
        double bit_nl = (double)shape.size * 0.5 - 0.5 * (double)bit_max;
        if (bit_nl < min_nonlinearity) {
            min_nonlinearity = bit_nl;
        }
    }

    tmp.max_walsh = global_max_walsh;
    tmp.min_nonlinearity = (component_count > 0) ? min_nonlinearity : 0.0;
    tmp.fwht_ms = profile_timings ? component_fwht_seconds * 1000.0 : 0.0;

    if (ctx) {
        fwht_destroy_context(ctx);
    }
    free(component_ptrs);
    if (component_owned) {
        free(component_data);
    }

    *metrics = tmp;
    return FWHT_SUCCESS;
}

fwht_status_t fwht_sbox_analyze_lat(const uint32_t* table,
                                     size_t size,
                                     const fwht_sbox_lat_request_t* request,
                                     fwht_sbox_lat_metrics_t* metrics) {
    if (metrics == NULL) {
        return FWHT_ERROR_NULL_POINTER;
    }

    fwht_sbox_shape_t shape;
    fwht_status_t status = fwht_sbox_resolve_shape(table, size, &shape);
    if (status != FWHT_SUCCESS) {
        return status;
    }

    fwht_sbox_lat_metrics_t tmp = {0};
    tmp.size = shape.size;
    tmp.m = shape.m;
    tmp.n = shape.n;

    const size_t bit_capacity = sizeof(size_t) * CHAR_BIT;
    if (shape.n >= bit_capacity) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }
    size_t lat_cols = (size_t)1 << shape.n;
    if (lat_cols == 0) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    fwht_backend_t backend = (request != NULL) ? request->backend : FWHT_BACKEND_AUTO;
    if (backend < FWHT_BACKEND_AUTO || backend > FWHT_BACKEND_GPU) {
        backend = FWHT_BACKEND_AUTO;
    }
    const bool profile_timings = (request != NULL && request->profile_timings);
    int32_t* lat_output = (request != NULL) ? request->lat : NULL;
    if (lat_output != NULL) {
        if (shape.size > SIZE_MAX / lat_cols) {
            return FWHT_ERROR_INVALID_ARGUMENT;
        }
    }

    const size_t lat_stride = lat_cols;
    int32_t lat_max_abs = 0;

#ifdef USE_CUDA
    const bool prefer_gpu_backend = (backend == FWHT_BACKEND_GPU);
    bool lat_use_gpu_columns = prefer_gpu_backend;
#else
    const bool prefer_gpu_backend = false;
#endif
    const bool require_context = (!prefer_gpu_backend && backend != FWHT_BACKEND_AUTO);

    size_t lat_batch_capacity;
    if (prefer_gpu_backend) {
        const size_t gpu_batch_hint = 256;
        const size_t gpu_batch_cap = 1024;
        const size_t gpu_batch_target_bytes = 64ull * 1024ull * 1024ull;
        lat_batch_capacity = gpu_batch_hint;
        size_t bytes_per_column = (shape.size > SIZE_MAX / sizeof(int32_t))
                                      ? SIZE_MAX
                                      : shape.size * sizeof(int32_t);
        if (bytes_per_column != 0) {
            size_t bytes_limit = gpu_batch_target_bytes / bytes_per_column;
            if (bytes_limit == 0) {
                bytes_limit = 1;
            }
            if (bytes_limit < lat_batch_capacity) {
                lat_batch_capacity = bytes_limit;
            }
        }
        if (lat_batch_capacity > gpu_batch_cap) {
            lat_batch_capacity = gpu_batch_cap;
        }
    } else {
        lat_batch_capacity = 8;
    }
    if (lat_batch_capacity == 0) {
        lat_batch_capacity = 1;
    }
    if (lat_batch_capacity > lat_cols) {
        lat_batch_capacity = lat_cols;
    }
    if (lat_batch_capacity > (size_t)INT_MAX) {
        lat_batch_capacity = (size_t)INT_MAX;
    }
    if (shape.size > 0 && lat_batch_capacity > SIZE_MAX / shape.size) {
        return FWHT_ERROR_INVALID_ARGUMENT;
    }

    fwht_context_t* ctx = NULL;
    double lat_column_seconds = 0.0;
    double lat_fwht_seconds = 0.0;
    fwht_status_t result = FWHT_SUCCESS;

    int32_t* lat_batch_buffer = NULL;
    int32_t** lat_batch_ptrs = NULL;
#ifdef USE_CUDA
    size_t* lat_mask_list = NULL;
    bool lat_buffer_pinned = false;
#endif

    lat_batch_buffer = (int32_t*)malloc(lat_batch_capacity * shape.size * sizeof(int32_t));
    lat_batch_ptrs = (int32_t**)malloc(lat_batch_capacity * sizeof(int32_t*));
    if (lat_batch_buffer == NULL || lat_batch_ptrs == NULL) {
        result = FWHT_ERROR_OUT_OF_MEMORY;
        goto lat_cleanup;
    }
#ifdef USE_CUDA
    if (lat_use_gpu_columns) {
        lat_mask_list = (size_t*)malloc(lat_batch_capacity * sizeof(size_t));
        if (lat_mask_list == NULL) {
            lat_use_gpu_columns = false;
        }
    }
#endif

    if (require_context) {
        fwht_config_t cfg = fwht_default_config();
        cfg.backend = backend;
        ctx = fwht_create_context(&cfg);
        if (ctx == NULL) {
            result = FWHT_ERROR_OUT_OF_MEMORY;
            goto lat_cleanup;
        }
    }

    for (size_t i = 0; i < lat_batch_capacity; ++i) {
        lat_batch_ptrs[i] = lat_batch_buffer + i * shape.size;
    }

#ifdef USE_CUDA
    if (lat_use_gpu_columns) {
        size_t host_bytes = lat_batch_capacity * shape.size * sizeof(int32_t);
        if (cudaHostRegister(lat_batch_buffer, host_bytes, cudaHostRegisterPortable) == cudaSuccess) {
            lat_buffer_pinned = true;
        }
        fwht_status_t ws_status = fwht_sbox_gpu_workspace_init();
        if (ws_status != FWHT_SUCCESS ||
            fwht_sbox_gpu_reserve_lat_buffer(shape.size, lat_batch_capacity) != FWHT_SUCCESS) {
            lat_use_gpu_columns = false;
            if (lat_buffer_pinned) {
                cudaHostUnregister(lat_batch_buffer);
                lat_buffer_pinned = false;
            }
        }
    }
#endif

    for (size_t base = 0; base < lat_cols; base += lat_batch_capacity) {
        size_t current = lat_batch_capacity;
        if (base + current > lat_cols) {
            current = lat_cols - base;
        }

        bool combined_on_gpu = false;
#ifdef USE_CUDA
        if (lat_use_gpu_columns) {
            for (size_t col = 0; col < current; ++col) {
                lat_mask_list[col] = base + col;
            }
            fwht_status_t gpu_status = fwht_sbox_gpu_process_lat_batch(table,
                                                                        shape.size,
                                                                        lat_mask_list,
                                                                        current,
                                                                        lat_batch_buffer,
                                                                        profile_timings ? &lat_column_seconds : NULL,
                                                                        profile_timings ? &lat_fwht_seconds : NULL,
                                                                        profile_timings);
            if (gpu_status == FWHT_SUCCESS) {
                combined_on_gpu = true;
            } else if (gpu_status == FWHT_ERROR_BACKEND_UNAVAILABLE) {
                lat_use_gpu_columns = false;
            } else {
                result = gpu_status;
                goto lat_cleanup;
            }
        }
#endif
        if (!combined_on_gpu) {
            double combine_start = profile_timings ? fwht_now_seconds() : 0.0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (size_t col = 0; col < current; ++col) {
                size_t mask = base + col;
                int32_t* dst = lat_batch_ptrs[col];
                fwht_lat_build_from_table(table, shape.size, mask, dst);
            }
            if (profile_timings) {
                lat_column_seconds += fwht_now_seconds() - combine_start;
            }

            double lat_fwht_start = profile_timings ? fwht_now_seconds() : 0.0;
            fwht_status_t lat_status = fwht_batch_i32(ctx, lat_batch_ptrs, shape.size, (int)current);
            if (profile_timings) {
                lat_fwht_seconds += fwht_now_seconds() - lat_fwht_start;
            }
            if (lat_status != FWHT_SUCCESS) {
                result = lat_status;
                goto lat_cleanup;
            }
        }

        for (size_t col = 0; col < current; ++col) {
            size_t mask = base + col;
            int32_t* column_data = lat_batch_ptrs[col];
            for (size_t row = 0; row < shape.size; ++row) {
                int32_t value = column_data[row];
                /* Exclude LAT[0,0] from max computation (always trivial maximum) */
                if (row == 0 && mask == 0) {
                    if (lat_output != NULL) {
                        lat_output[row * lat_stride + mask] = value;
                    }
                    continue;
                }
                int32_t abs_val = fwht_i32_abs(value);
                if (abs_val > lat_max_abs) {
                    lat_max_abs = abs_val;
                }
                if (lat_output != NULL) {
                    lat_output[row * lat_stride + mask] = value;
                }
            }
        }
    }

    tmp.lat_max = lat_max_abs;
    tmp.lat_max_bias = (lat_max_abs > 0) ? ((double)lat_max_abs / (double)shape.size) : 0.0;
    if (profile_timings) {
        tmp.column_ms = lat_column_seconds * 1000.0;
        tmp.fwht_ms = lat_fwht_seconds * 1000.0;
    }

lat_cleanup:
#ifdef USE_CUDA
    if (lat_mask_list) {
        free(lat_mask_list);
    }
    if (lat_buffer_pinned) {
        cudaHostUnregister(lat_batch_buffer);
    }
#endif
    if (lat_batch_ptrs) {
        free(lat_batch_ptrs);
    }
    if (lat_batch_buffer) {
        free(lat_batch_buffer);
    }
    if (ctx) {
        fwht_destroy_context(ctx);
    }

    if (result == FWHT_SUCCESS) {
        *metrics = tmp;
    }
    return result;
}
