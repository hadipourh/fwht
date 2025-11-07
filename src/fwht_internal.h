/*
 * Fast Walsh-Hadamard Transform - Internal Declarations
 * 
 * This header is for internal use only and not part of the public API.
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

#ifndef FWHT_INTERNAL_H
#define FWHT_INTERNAL_H

#include "../include/fwht.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CUDA Backend Functions (implemented in fwht_cuda.cu)
 * ============================================================================ */

#ifdef __NVCC__
/* CUDA implementations */
fwht_status_t fwht_i32_cuda(int32_t* data, size_t n);
fwht_status_t fwht_f64_cuda(double* data, size_t n);
fwht_status_t fwht_batch_i32_cuda(int32_t* data, size_t n, size_t batch_size);
fwht_status_t fwht_batch_f64_cuda(double* data, size_t n, size_t batch_size);

/* CUDA context functions */
fwht_context_t* fwht_create_context_cuda(size_t max_n, fwht_backend_t backend);
void fwht_destroy_context_cuda(fwht_context_t* ctx);
fwht_status_t fwht_transform_i32_cuda(fwht_context_t* ctx, int32_t* data, size_t n);
fwht_status_t fwht_transform_f64_cuda(fwht_context_t* ctx, double* data, size_t n);
#endif

#ifdef __cplusplus
}
#endif

#endif /* FWHT_INTERNAL_H */
