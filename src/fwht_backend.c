/*
 * Fast Walsh-Hadamard Transform - Backend Dispatcher
 *
 * Routes calls to the appropriate backend (CPU, OpenMP, or CUDA).
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
#include <stdbool.h>

/* Forward declarations for CUDA functions */
#ifdef USE_CUDA
extern fwht_status_t fwht_i32_cuda(int32_t* data, size_t n);
extern fwht_status_t fwht_f64_cuda(double* data, size_t n);
#endif

/* Forward declarations for CPU functions */
extern fwht_status_t fwht_i32_cpu(int32_t* data, size_t n);
extern fwht_status_t fwht_f64_cpu(double* data, size_t n);

/* Runtime backend availability */
bool fwht_has_gpu(void) {
#ifdef USE_CUDA
    return true;
#else
    return false;
#endif
}

bool fwht_has_openmp(void) {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

const char* fwht_backend_name(fwht_backend_t backend) {
    switch (backend) {
        case FWHT_BACKEND_AUTO:      return "auto";
        case FWHT_BACKEND_CPU:       return "cpu";
        case FWHT_BACKEND_CPU_SAFE:  return "cpu_safe";
        case FWHT_BACKEND_OPENMP:    return "openmp";
        case FWHT_BACKEND_GPU:       return "gpu";
        default:                     return "unknown";
    }
}

const char* fwht_error_string(fwht_status_t status) {
    switch (status) {
        case FWHT_SUCCESS:                    return "success";
        case FWHT_ERROR_INVALID_SIZE:         return "invalid size (must be power of 2)";
        case FWHT_ERROR_NULL_POINTER:         return "null pointer argument";
        case FWHT_ERROR_BACKEND_UNAVAILABLE:  return "backend not available";
        case FWHT_ERROR_OUT_OF_MEMORY:        return "out of memory";
        case FWHT_ERROR_INVALID_ARGUMENT:     return "invalid argument";
        case FWHT_ERROR_CUDA:                 return "CUDA error";
        case FWHT_ERROR_OVERFLOW:             return "integer overflow detected";
        default:                              return "unknown error";
    }
}

/* Recommend backend based on size */
fwht_backend_t fwht_recommend_backend(size_t n) {
    /* Use GPU for large transforms if available */
    if (n >= 1024 && fwht_has_gpu()) {
        return FWHT_BACKEND_GPU;
    }
    
    /* Use OpenMP for medium transforms if available */
    if (n >= 256 && fwht_has_openmp()) {
        return FWHT_BACKEND_OPENMP;
    }
    
    /* Use CPU for small transforms */
    return FWHT_BACKEND_CPU;
}

/* Get version */
const char* fwht_version(void) {
    return FWHT_VERSION;
}

/* Check if power of 2 */
bool fwht_is_power_of_2(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* Compute log2 */
int fwht_log2(size_t n) {
    if (!fwht_is_power_of_2(n)) return -1;
    int log = 0;
    while (n > 1) {
        n >>= 1;
        log++;
    }
    return log;
}
