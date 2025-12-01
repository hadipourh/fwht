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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define FWHT_DEFAULT_OPENMP_THRESHOLD 8192u
#define FWHT_DEFAULT_GPU_THRESHOLD (1u << 20)
#define FWHT_THRESHOLD_CONFIG "meta/backend_threshold.json"

static size_t fwht_openmp_threshold = FWHT_DEFAULT_OPENMP_THRESHOLD;
static size_t fwht_gpu_threshold = FWHT_DEFAULT_GPU_THRESHOLD;
static bool fwht_thresholds_loaded = false;

static void fwht_parse_threshold(const char* text, const char* key, size_t* out, size_t minimum) {
    const char* pos = strstr(text, key);
    if (!pos) {
        return;
    }
    pos = strchr(pos, ':');
    if (!pos) {
        return;
    }
    pos++;
    while (*pos == ' ' || *pos == '\"') {
        pos++;
    }
    char* endptr = NULL;
    unsigned long long value = strtoull(pos, &endptr, 10);
    if (endptr == pos || value < minimum) {
        return;
    }
    *out = (size_t)value;
}

static void fwht_try_load_thresholds(void) {
    if (fwht_thresholds_loaded) {
        return;
    }
    fwht_thresholds_loaded = true;

    FILE* fp = fopen(FWHT_THRESHOLD_CONFIG, "rb");
    if (!fp) {
        return;
    }

    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return;
    }
    long length = ftell(fp);
    if (length <= 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return;
    }

    char* buffer = (char*)malloc((size_t)length + 1);
    if (!buffer) {
        fclose(fp);
        return;
    }

    size_t read = fread(buffer, 1, (size_t)length, fp);
    fclose(fp);
    buffer[read] = '\0';

    fwht_parse_threshold(buffer, "openmp_threshold", &fwht_openmp_threshold, 64u);
    fwht_parse_threshold(buffer, "gpu_threshold", &fwht_gpu_threshold, 1024u);

    free(buffer);
}

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
    fwht_try_load_thresholds();

    /* Use GPU for large transforms if available */
    if (n >= fwht_gpu_threshold && fwht_has_gpu()) {
        return FWHT_BACKEND_GPU;
    }
    
    /* Use OpenMP for medium transforms if available */
    if (n >= fwht_openmp_threshold && fwht_has_openmp()) {
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
