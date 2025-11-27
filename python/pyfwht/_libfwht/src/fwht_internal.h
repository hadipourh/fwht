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

#ifdef USE_CUDA
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <cuda_runtime_api.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

fwht_status_t fwht_i32_cuda(int32_t* data, size_t n);
fwht_status_t fwht_f64_cuda(double* data, size_t n);
fwht_status_t fwht_batch_i32_cuda(int32_t* data, size_t n, size_t batch_size);
fwht_status_t fwht_batch_f64_cuda(double* data, size_t n, size_t batch_size);
fwht_status_t fwht_batch_i32_cuda_device(int32_t* d_data, size_t n, size_t batch_size);
fwht_status_t fwht_batch_i32_cuda_device_async(int32_t* d_data,
											   size_t n,
											   size_t batch_size,
											   cudaStream_t stream);
fwht_status_t fwht_boolean_packed_cuda(const uint64_t* packed_bits,
											 int32_t* wht_out,
											 size_t n);

fwht_status_t fwht_cuda_lat_build_direct(const uint32_t* table,
										 size_t size,
										 const size_t* masks,
										 size_t mask_count,
										 int32_t* dst);
fwht_status_t fwht_cuda_lat_build_device(const uint32_t* table,
										 size_t size,
										 const size_t* masks,
										 size_t mask_count,
										 int32_t* d_dst,
										 cudaStream_t stream);
void fwht_cuda_lat_direct_release_cache(void);
#endif /* USE_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* FWHT_INTERNAL_H */
