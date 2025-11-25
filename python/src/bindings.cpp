/*
 * pyfwht - Python bindings for libfwht
 * 
 * This file wraps the C library API using pybind11 for seamless NumPy integration.
 * 
 * Copyright (C) 2025 Hosein Hadipour
 * License: GPL-3.0-or-later
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#ifdef USE_CUDA
#include <dlpack/dlpack.h>
#include <cuda_runtime.h>  // For cudaMalloc, cudaMemcpy, cudaFree
#include <cuda_fp16.h>  // C++ header, must be outside extern "C"
#endif

extern "C" {
    #include "fwht.h"
}

#ifdef USE_CUDA
// Forward declarations for fp16/fp32 kernels (not in fwht.h yet)
// These are C++ functions due to __half type, so they're outside extern "C"
extern "C" {
    fwht_status_t fwht_batch_f32_cuda(float* d_data, size_t n, size_t batch_size);
    // FP16 function uses void* to avoid __half linkage issues
    int fwht_batch_f16_cuda_device(const void* d_in, void* d_out, 
                                    unsigned int n, unsigned int batch_size);
}
#endif

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// Exception wrapper to convert C error codes to Python exceptions
static void check_status(fwht_status_t status, const char* operation) {
    if (status == FWHT_SUCCESS) {
        return;
    }
    
    const char* error_msg = fwht_error_string(status);
    std::string full_msg = std::string(operation) + ": " + error_msg;
    
    switch (status) {
        case FWHT_ERROR_INVALID_SIZE:
        case FWHT_ERROR_INVALID_ARGUMENT:
            throw std::invalid_argument(full_msg);
        case FWHT_ERROR_NULL_POINTER:
            throw std::runtime_error(full_msg);
        case FWHT_ERROR_BACKEND_UNAVAILABLE:
            throw std::runtime_error(full_msg);
        case FWHT_ERROR_OUT_OF_MEMORY:
            throw std::bad_alloc();
        case FWHT_ERROR_CUDA:
            throw std::runtime_error(full_msg);
        default:
            throw std::runtime_error(full_msg);
    }
}

// =============================================================================
// CORE TRANSFORMS - IN-PLACE
// =============================================================================

void py_fwht_i32(py::array_t<int32_t> data) {
    auto buf = data.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    int32_t* ptr = static_cast<int32_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    fwht_status_t status = fwht_i32(ptr, n);
    check_status(status, "fwht_i32");
}

void py_fwht_f64(py::array_t<double> data) {
    auto buf = data.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];
    
    fwht_status_t status = fwht_f64(ptr, n);
    check_status(status, "fwht_f64");
}

void py_fwht_i8(py::array_t<int8_t> data) {
    auto buf = data.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    int8_t* ptr = static_cast<int8_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    fwht_status_t status = fwht_i8(ptr, n);
    check_status(status, "fwht_i8");
}

void py_fwht_i32_safe(py::array_t<int32_t> data) {
    auto buf = data.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    int32_t* ptr = static_cast<int32_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    fwht_status_t status = fwht_i32_safe(ptr, n);
    check_status(status, "fwht_i32_safe");
}

// =============================================================================
// VECTORIZED BATCH API - SIMD-optimized CPU batch processing
// =============================================================================

void py_fwht_i32_batch(py::list data_list, size_t n) {
    size_t batch_size = data_list.size();
    if (batch_size == 0) {
        throw std::invalid_argument("Empty batch");
    }
    
    // Convert Python list of arrays to array of pointers
    std::vector<int32_t*> ptrs(batch_size);
    std::vector<py::array_t<int32_t>> arrays;  // Keep references alive
    arrays.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; i++) {
        py::array_t<int32_t> arr = data_list[i].cast<py::array_t<int32_t>>();
        auto buf = arr.request();
        
        if (buf.ndim != 1) {
            throw std::invalid_argument("All arrays must be 1-dimensional");
        }
        if (static_cast<size_t>(buf.shape[0]) != n) {
            throw std::invalid_argument("All arrays must have the same size n");
        }
        
        ptrs[i] = static_cast<int32_t*>(buf.ptr);
        arrays.push_back(std::move(arr));
    }
    
    fwht_status_t status = fwht_i32_batch(ptrs.data(), n, batch_size);
    check_status(status, "fwht_i32_batch");
}

void py_fwht_f64_batch(py::list data_list, size_t n) {
    size_t batch_size = data_list.size();
    if (batch_size == 0) {
        throw std::invalid_argument("Empty batch");
    }
    
    // Convert Python list of arrays to array of pointers
    std::vector<double*> ptrs(batch_size);
    std::vector<py::array_t<double>> arrays;  // Keep references alive
    arrays.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; i++) {
        py::array_t<double> arr = data_list[i].cast<py::array_t<double>>();
        auto buf = arr.request();
        
        if (buf.ndim != 1) {
            throw std::invalid_argument("All arrays must be 1-dimensional");
        }
        if (static_cast<size_t>(buf.shape[0]) != n) {
            throw std::invalid_argument("All arrays must have the same size n");
        }
        
        ptrs[i] = static_cast<double*>(buf.ptr);
        arrays.push_back(std::move(arr));
    }
    
    fwht_status_t status = fwht_f64_batch(ptrs.data(), n, batch_size);
    check_status(status, "fwht_f64_batch");
}

// =============================================================================
// BACKEND CONTROL
// =============================================================================

void py_fwht_i32_backend(py::array_t<int32_t> data, fwht_backend_t backend) {
    auto buf = data.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    int32_t* ptr = static_cast<int32_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    fwht_status_t status = fwht_i32_backend(ptr, n, backend);
    check_status(status, "fwht_i32_backend");
}

void py_fwht_f64_backend(py::array_t<double> data, fwht_backend_t backend) {
    auto buf = data.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    size_t n = buf.shape[0];
    
    fwht_status_t status = fwht_f64_backend(ptr, n, backend);
    check_status(status, "fwht_f64_backend");
}

// =============================================================================
// OUT-OF-PLACE TRANSFORMS
// =============================================================================

py::array_t<int32_t> py_fwht_compute_i32(py::array_t<int32_t> input) {
    auto buf = input.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    int32_t* result = fwht_compute_i32(ptr, n);
    if (result == nullptr) {
        throw std::runtime_error("fwht_compute_i32 failed");
    }
    
    // Create NumPy array that owns the data
    // Use std::free for aligned memory deallocation
    py::capsule free_when_done(result, [](void* p) {
        std::free(p);
    });
    
    return py::array_t<int32_t>(
        {static_cast<py::ssize_t>(n)},
        {sizeof(int32_t)},
        result,
        free_when_done
    );
}

py::array_t<double> py_fwht_compute_f64(py::array_t<double> input) {
    auto buf = input.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const double* ptr = static_cast<const double*>(buf.ptr);
    size_t n = buf.shape[0];
    
    double* result = fwht_compute_f64(ptr, n);
    if (result == nullptr) {
        throw std::runtime_error("fwht_compute_f64 failed");
    }
    
    // Create NumPy array that owns the data
    // Use std::free for aligned memory deallocation
    py::capsule free_when_done(result, [](void* p) {
        std::free(p);
    });
    
    return py::array_t<double>(
        {static_cast<py::ssize_t>(n)},
        {sizeof(double)},
        result,
        free_when_done
    );
}

py::array_t<int32_t> py_fwht_compute_i32_backend(py::array_t<int32_t> input, fwht_backend_t backend) {
    auto buf = input.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    int32_t* result = fwht_compute_i32_backend(ptr, n, backend);
    if (result == nullptr) {
        throw std::runtime_error("fwht_compute_i32_backend failed");
    }
    
    py::capsule free_when_done(result, [](void* p) {
        std::free(p);
    });
    
    return py::array_t<int32_t>(
        {static_cast<py::ssize_t>(n)},
        {sizeof(int32_t)},
        result,
        free_when_done
    );
}

py::array_t<double> py_fwht_compute_f64_backend(py::array_t<double> input, fwht_backend_t backend) {
    auto buf = input.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const double* ptr = static_cast<const double*>(buf.ptr);
    size_t n = buf.shape[0];
    
    double* result = fwht_compute_f64_backend(ptr, n, backend);
    if (result == nullptr) {
        throw std::runtime_error("fwht_compute_f64_backend failed");
    }
    
    py::capsule free_when_done(result, [](void* p) {
        std::free(p);
    });
    
    return py::array_t<double>(
        {static_cast<py::ssize_t>(n)},
        {sizeof(double)},
        result,
        free_when_done
    );
}

// =============================================================================
// BOOLEAN FUNCTION API
// =============================================================================

py::array_t<int32_t> py_fwht_from_bool(py::array_t<uint8_t> bool_func, bool signed_rep) {
    auto buf = bool_func.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    // Allocate output array
    auto result = py::array_t<int32_t>(n);
    auto result_buf = result.request();
    int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
    
    fwht_status_t status = fwht_from_bool(ptr, result_ptr, n, signed_rep);
    check_status(status, "fwht_from_bool");
    
    return result;
}

py::array_t<double> py_fwht_correlations(py::array_t<uint8_t> bool_func) {
    auto buf = bool_func.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);
    size_t n = buf.shape[0];
    
    // Allocate output array
    auto result = py::array_t<double>(n);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    
    fwht_status_t status = fwht_correlations(ptr, result_ptr, n);
    check_status(status, "fwht_correlations");
    
    return result;
}

// Bit-sliced Boolean WHT (packed representation)
py::array_t<int32_t> py_fwht_boolean_packed(py::array_t<uint64_t> packed_bits, size_t n) {
    auto buf = packed_bits.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const uint64_t* ptr = static_cast<const uint64_t*>(buf.ptr);
    size_t n_words_expected = (n + 63) / 64;
    
    if (static_cast<size_t>(buf.shape[0]) < n_words_expected) {
        throw std::invalid_argument("Packed array too small for specified n");
    }
    
    // Allocate output array
    auto result = py::array_t<int32_t>(n);
    auto result_buf = result.request();
    int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
    
    fwht_status_t status = fwht_boolean_packed(ptr, result_ptr, n);
    check_status(status, "fwht_boolean_packed");
    
    return result;
}

// Bit-sliced Boolean WHT with backend selection
py::array_t<int32_t> py_fwht_boolean_packed_backend(py::array_t<uint64_t> packed_bits, 
                                                      size_t n, fwht_backend_t backend) {
    auto buf = packed_bits.request();
    if (buf.ndim != 1) {
        throw std::invalid_argument("Input must be 1-dimensional array");
    }
    
    const uint64_t* ptr = static_cast<const uint64_t*>(buf.ptr);
    size_t n_words_expected = (n + 63) / 64;
    
    if (static_cast<size_t>(buf.shape[0]) < n_words_expected) {
        throw std::invalid_argument("Packed array too small for specified n");
    }
    
    // Allocate output array
    auto result = py::array_t<int32_t>(n);
    auto result_buf = result.request();
    int32_t* result_ptr = static_cast<int32_t*>(result_buf.ptr);
    
    fwht_status_t status = fwht_boolean_packed_backend(ptr, result_ptr, n, backend);
    check_status(status, "fwht_boolean_packed_backend");
    
    return result;
}

// Batch bit-sliced Boolean WHT (for S-box cryptanalysis)
py::list py_fwht_boolean_batch(py::list packed_list, size_t n) {
    size_t batch_size = packed_list.size();
    if (batch_size == 0) {
        throw std::invalid_argument("Empty batch");
    }
    
    // Convert Python list to array of pointers
    std::vector<const uint64_t*> packed_ptrs(batch_size);
    std::vector<int32_t*> wht_ptrs(batch_size);
    std::vector<py::array_t<uint64_t>> packed_arrays;
    std::vector<py::array_t<int32_t>> wht_arrays;
    
    packed_arrays.reserve(batch_size);
    wht_arrays.reserve(batch_size);
    
    size_t n_words = (n + 63) / 64;
    
    for (size_t i = 0; i < batch_size; i++) {
        // Get input array
        py::array_t<uint64_t> arr = packed_list[i].cast<py::array_t<uint64_t>>();
        auto buf = arr.request();
        
        if (buf.ndim != 1) {
            throw std::invalid_argument("All arrays must be 1-dimensional");
        }
        if (static_cast<size_t>(buf.shape[0]) < n_words) {
            throw std::invalid_argument("Packed array too small for specified n");
        }
        
        packed_ptrs[i] = static_cast<const uint64_t*>(buf.ptr);
        packed_arrays.push_back(std::move(arr));
        
        // Allocate output array
        auto wht = py::array_t<int32_t>(n);
        auto wht_buf = wht.request();
        wht_ptrs[i] = static_cast<int32_t*>(wht_buf.ptr);
        wht_arrays.push_back(std::move(wht));
    }
    
    fwht_status_t status = fwht_boolean_batch(packed_ptrs.data(), wht_ptrs.data(), n, batch_size);
    check_status(status, "fwht_boolean_batch");
    
    // Return list of output arrays
    py::list result;
    for (auto& arr : wht_arrays) {
        result.append(arr);
    }
    
    return result;
}

// =============================================================================
// CONTEXT API
// =============================================================================

class PyFWHTContext {
private:
    fwht_context_t* ctx_;

public:
    PyFWHTContext(const fwht_config_t& config) {
        ctx_ = fwht_create_context(&config);
        if (ctx_ == nullptr) {
            throw std::runtime_error("Failed to create FWHT context");
        }
    }
    
    ~PyFWHTContext() {
        if (ctx_ != nullptr) {
            fwht_destroy_context(ctx_);
        }
    }
    
    // Disable copy
    PyFWHTContext(const PyFWHTContext&) = delete;
    PyFWHTContext& operator=(const PyFWHTContext&) = delete;
    
    void transform_i32(py::array_t<int32_t> data) {
        auto buf = data.request();
        if (buf.ndim != 1) {
            throw std::invalid_argument("Input must be 1-dimensional array");
        }
        
        int32_t* ptr = static_cast<int32_t*>(buf.ptr);
        size_t n = buf.shape[0];
        
        fwht_status_t status = fwht_transform_i32(ctx_, ptr, n);
        check_status(status, "fwht_transform_i32");
    }
    
    void transform_f64(py::array_t<double> data) {
        auto buf = data.request();
        if (buf.ndim != 1) {
            throw std::invalid_argument("Input must be 1-dimensional array");
        }
        
        double* ptr = static_cast<double*>(buf.ptr);
        size_t n = buf.shape[0];
        
        fwht_status_t status = fwht_transform_f64(ctx_, ptr, n);
        check_status(status, "fwht_transform_f64");
    }
    
    void close() {
        if (ctx_ != nullptr) {
            fwht_destroy_context(ctx_);
            ctx_ = nullptr;
        }
    }
};

// =============================================================================
// GPU-SPECIFIC API
// =============================================================================

#ifdef USE_CUDA

// GPU Device Info
class PyGPUInfo {
public:
    static unsigned int get_smem_banks() {
        return fwht_gpu_get_smem_banks();
    }
    
    static unsigned int get_compute_capability() {
        return fwht_gpu_get_compute_capability();
    }
    
    static std::string get_device_name() {
        const char* name = fwht_gpu_get_device_name();
        return name ? std::string(name) : std::string("");
    }
    
    static unsigned int get_sm_count() {
        return fwht_gpu_get_sm_count();
    }
};

// GPU Profiling
class PyGPUProfiling {
public:
    static void set_profiling(bool enable) {
        fwht_status_t status = fwht_gpu_set_profiling(enable);
        check_status(status, "fwht_gpu_set_profiling");
    }
    
    static bool profiling_enabled() {
        return fwht_gpu_profiling_enabled();
    }
    
    static py::dict get_last_metrics() {
        fwht_gpu_metrics_t metrics = fwht_gpu_get_last_metrics();
        
        py::dict result;
        if (metrics.valid) {
            result["h2d_ms"] = metrics.h2d_ms;
            result["kernel_ms"] = metrics.kernel_ms;
            result["d2h_ms"] = metrics.d2h_ms;
            result["total_ms"] = metrics.h2d_ms + metrics.kernel_ms + metrics.d2h_ms;
            result["n"] = metrics.n;
            result["batch_size"] = metrics.batch_size;
            result["bytes_transferred"] = metrics.bytes_transferred;
            result["samples"] = metrics.samples;
            result["valid"] = true;
        } else {
            result["valid"] = false;
        }
        
        return result;
    }
};

// GPU Batch Processing
void py_fwht_batch_i32_cuda(py::array_t<int32_t> data, size_t n, size_t batch_size) {
    auto buf = data.request();
    
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::invalid_argument("Input must be 1D or 2D array");
    }
    
    size_t total_elements = buf.shape[0];
    if (buf.ndim == 2) {
        total_elements = buf.shape[0] * buf.shape[1];
    }
    
    if (total_elements != n * batch_size) {
        throw std::invalid_argument("Array size must equal n * batch_size");
    }
    
    int32_t* ptr = static_cast<int32_t*>(buf.ptr);
    fwht_status_t status = fwht_batch_i32_cuda(ptr, n, batch_size);
    check_status(status, "fwht_batch_i32_cuda");
}

void py_fwht_batch_f64_cuda(py::array_t<double> data, size_t n, size_t batch_size) {
    auto buf = data.request();
    
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::invalid_argument("Input must be 1D or 2D array");
    }
    
    size_t total_elements = buf.shape[0];
    if (buf.ndim == 2) {
        total_elements = buf.shape[0] * buf.shape[1];
    }
    
    if (total_elements != n * batch_size) {
        throw std::invalid_argument("Array size must equal n * batch_size");
    }
    
    double* ptr = static_cast<double*>(buf.ptr);
    fwht_status_t status = fwht_batch_f64_cuda(ptr, n, batch_size);
    check_status(status, "fwht_batch_f64_cuda");
}

void py_fwht_batch_f32_cuda(py::array_t<float> data, size_t n, size_t batch_size) {
    auto buf = data.request();
    
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::invalid_argument("Input must be 1D or 2D array");
    }
    
    size_t total_elements = buf.shape[0];
    if (buf.ndim == 2) {
        total_elements = buf.shape[0] * buf.shape[1];
    }
    
    if (total_elements != n * batch_size) {
        throw std::invalid_argument("Array size must equal n * batch_size");
    }
    
    float* ptr = static_cast<float*>(buf.ptr);
    fwht_status_t status = fwht_batch_f32_cuda(ptr, n, batch_size);
    check_status(status, "fwht_batch_f32_cuda");
}

// FP16 host memory wrapper - converts to __half, allocates GPU memory, processes, converts back
void py_fwht_batch_f16_cuda(py::array_t<uint16_t> data, size_t n, size_t batch_size) {
    auto buf = data.request();
    
    if (buf.ndim != 1 && buf.ndim != 2) {
        throw std::invalid_argument("Input must be 1D or 2D array");
    }
    
    size_t total_elements = buf.shape[0];
    if (buf.ndim == 2) {
        total_elements = buf.shape[0] * buf.shape[1];
    }
    
    if (total_elements != n * batch_size) {
        throw std::invalid_argument("Array size must equal n * batch_size");
    }
    
    // NumPy float16 is stored as uint16 in memory (IEEE 754 binary16)
    uint16_t* ptr = static_cast<uint16_t*>(buf.ptr);
    
    // Allocate device memory for input and output
    void* d_in = nullptr;
    void* d_out = nullptr;
    size_t bytes = total_elements * sizeof(uint16_t);
    
    cudaError_t err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc for d_in failed: ") + cudaGetErrorString(err));
    }
    
    err = cudaMalloc(&d_out, bytes);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        throw std::runtime_error(std::string("cudaMalloc for d_out failed: ") + cudaGetErrorString(err));
    }
    
    // Copy host→device
    err = cudaMemcpy(d_in, ptr, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err));
    }
    
    // Process on device using Tensor Cores (function signature uses void*)
    int status = fwht_batch_f16_cuda_device(d_in, d_out, 
                                             static_cast<unsigned int>(n), 
                                             static_cast<unsigned int>(batch_size));
    
    if (status == 0) {
        // Copy device→host
        err = cudaMemcpy(ptr, d_out, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_in);
            cudaFree(d_out);
            throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err));
        }
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
    
    if (status != 0) {
        throw std::runtime_error("fwht_batch_f16_cuda_device failed with error code: " + std::to_string(status));
    }
}

// GPU Context
class PyGPUContext {
private:
    fwht_gpu_context_t* ctx_;

public:
    PyGPUContext(size_t max_n, size_t max_batch_size) {
        ctx_ = fwht_gpu_context_create(max_n, max_batch_size);
        if (ctx_ == nullptr) {
            throw std::runtime_error("Failed to create GPU context");
        }
    }
    
    ~PyGPUContext() {
        if (ctx_ != nullptr) {
            fwht_gpu_context_destroy(ctx_);
        }
    }
    
    // Disable copy
    PyGPUContext(const PyGPUContext&) = delete;
    PyGPUContext& operator=(const PyGPUContext&) = delete;
    
    void compute_i32(py::array_t<int32_t> data, size_t n, size_t batch_size) {
        auto buf = data.request();
        int32_t* ptr = static_cast<int32_t*>(buf.ptr);
        
        fwht_status_t status = fwht_gpu_context_compute_i32(ctx_, ptr, n, batch_size);
        check_status(status, "fwht_gpu_context_compute_i32");
    }
    
    void compute_f64(py::array_t<double> data, size_t n, size_t batch_size) {
        auto buf = data.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        fwht_status_t status = fwht_gpu_context_compute_f64(ctx_, ptr, n, batch_size);
        check_status(status, "fwht_gpu_context_compute_f64");
    }
    
    void close() {
        if (ctx_ != nullptr) {
            fwht_gpu_context_destroy(ctx_);
            ctx_ = nullptr;
        }
    }
};

// GPU DLPack Support - Zero-copy interop with PyTorch, CuPy, JAX, etc.
namespace {
inline void consume_dlpack_capsule(py::capsule capsule) {
    PyObject* raw = capsule.ptr();
    if (raw == nullptr) {
        return;
    }

    const char* name = PyCapsule_GetName(raw);
    if (name == nullptr) {
        return;  // Already consumed or invalid
    }
    
    // Check if already consumed
    if (std::strcmp(name, "used_dltensor") == 0) {
        return;
    }
    
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(raw, name));
    if (tensor != nullptr && tensor->deleter != nullptr) {
        tensor->deleter(tensor);
    }
    
    // Mark as consumed by changing name only (don't set pointer to null)
    PyCapsule_SetName(raw, "used_dltensor");
}
} // namespace

void py_fwht_batch_f64_dlpack(py::capsule dlpack_tensor, size_t n, size_t batch_size) {
    // Import DLManagedTensor from capsule
    auto dlm_tensor = dlpack_tensor.get_pointer<DLManagedTensor>();
    
    // Validate tensor properties
    if (dlm_tensor->dl_tensor.ndim != 2) {
        throw std::invalid_argument("DLPack tensor must be 2-dimensional (batch_size, n)");
    }
    if (dlm_tensor->dl_tensor.dtype.code != kDLFloat || dlm_tensor->dl_tensor.dtype.bits != 64) {
        throw std::invalid_argument("DLPack tensor must be float64");
    }
    if (dlm_tensor->dl_tensor.device.device_type != kDLCUDA) {
        throw std::invalid_argument("DLPack tensor must be on CUDA device");
    }
    
    // Verify shape matches
    if (static_cast<size_t>(dlm_tensor->dl_tensor.shape[0]) != batch_size ||
        static_cast<size_t>(dlm_tensor->dl_tensor.shape[1]) != n) {
        throw std::invalid_argument("DLPack tensor shape must be (batch_size, n)");
    }
    
    // Get pointer to GPU data (already on device!)
    double* d_ptr = static_cast<double*>(dlm_tensor->dl_tensor.data);
    
    // Call CUDA kernel directly - NO H2D/D2H transfers!
    fwht_status_t status = fwht_batch_f64_cuda_device(d_ptr, n, batch_size);
    
    // Consume capsule first to avoid SystemError if check_status throws
    consume_dlpack_capsule(dlpack_tensor);
    check_status(status, "fwht_batch_f64_cuda_device");
}

void py_fwht_batch_i32_dlpack(py::capsule dlpack_tensor, size_t n, size_t batch_size) {
    auto dlm_tensor = dlpack_tensor.get_pointer<DLManagedTensor>();
    
    if (dlm_tensor->dl_tensor.ndim != 2) {
        throw std::invalid_argument("DLPack tensor must be 2-dimensional (batch_size, n)");
    }
    if (dlm_tensor->dl_tensor.dtype.code != kDLInt || dlm_tensor->dl_tensor.dtype.bits != 32) {
        throw std::invalid_argument("DLPack tensor must be int32");
    }
    if (dlm_tensor->dl_tensor.device.device_type != kDLCUDA) {
        throw std::invalid_argument("DLPack tensor must be on CUDA device");
    }
    
    if (static_cast<size_t>(dlm_tensor->dl_tensor.shape[0]) != batch_size ||
        static_cast<size_t>(dlm_tensor->dl_tensor.shape[1]) != n) {
        throw std::invalid_argument("DLPack tensor shape must be (batch_size, n)");
    }
    
    int32_t* d_ptr = static_cast<int32_t*>(dlm_tensor->dl_tensor.data);
    
    fwht_status_t status = fwht_batch_i32_cuda_device(d_ptr, n, batch_size);
    
    // Consume capsule first to avoid SystemError if check_status throws
    consume_dlpack_capsule(dlpack_tensor);
    check_status(status, "fwht_batch_i32_cuda_device");
}

// FP32 DLPack support (Meta-inspired high-speed kernel)
void py_fwht_batch_f32_dlpack(py::capsule dlpack_tensor, size_t n, size_t batch_size) {
    auto dlm_tensor = dlpack_tensor.get_pointer<DLManagedTensor>();
    
    if (dlm_tensor->dl_tensor.ndim != 2) {
        throw std::invalid_argument("DLPack tensor must be 2-dimensional (batch_size, n)");
    }
    if (dlm_tensor->dl_tensor.dtype.code != kDLFloat || dlm_tensor->dl_tensor.dtype.bits != 32) {
        throw std::invalid_argument("DLPack tensor must be float32");
    }
    if (dlm_tensor->dl_tensor.device.device_type != kDLCUDA) {
        throw std::invalid_argument("DLPack tensor must be on CUDA device");
    }
    
    if (static_cast<size_t>(dlm_tensor->dl_tensor.shape[0]) != batch_size ||
        static_cast<size_t>(dlm_tensor->dl_tensor.shape[1]) != n) {
        throw std::invalid_argument("DLPack tensor shape must be (batch_size, n)");
    }
    
    float* d_ptr = static_cast<float*>(dlm_tensor->dl_tensor.data);
    
    // Call Meta-inspired fp32 kernel using device pointer
    fwht_status_t status = fwht_batch_f32_cuda_device(d_ptr, n, batch_size);
    
    // Consume capsule first to avoid SystemError if check_status throws
    consume_dlpack_capsule(dlpack_tensor);
    check_status(status, "fwht_batch_f32_cuda_device");
}

// FP16 DLPack support (Meta-inspired maximum speed)
// Note: Uses void* to match C API signature, performs in-place transform
void py_fwht_batch_f16_dlpack(py::capsule dlpack_tensor, size_t n, size_t batch_size) {
    auto dlm_tensor = dlpack_tensor.get_pointer<DLManagedTensor>();
    
    if (dlm_tensor->dl_tensor.ndim != 2) {
        throw std::invalid_argument("DLPack tensor must be 2-dimensional (batch_size, n)");
    }
    if (dlm_tensor->dl_tensor.dtype.code != kDLFloat || dlm_tensor->dl_tensor.dtype.bits != 16) {
        throw std::invalid_argument("DLPack tensor must be float16");
    }
    if (dlm_tensor->dl_tensor.device.device_type != kDLCUDA) {
        throw std::invalid_argument("DLPack tensor must be on CUDA device");
    }
    
    if (static_cast<size_t>(dlm_tensor->dl_tensor.shape[0]) != batch_size ||
        static_cast<size_t>(dlm_tensor->dl_tensor.shape[1]) != n) {
        throw std::invalid_argument("DLPack tensor shape must be (batch_size, n)");
    }
    
    // DLPack gives us device pointer - do in-place transform (d_in == d_out)
    void* d_ptr = dlm_tensor->dl_tensor.data;
    
    // Call fp16 kernel with in-place transform (same pointer for input and output)
    int status = fwht_batch_f16_cuda_device(d_ptr, d_ptr, 
                                             static_cast<unsigned int>(n), 
                                             static_cast<unsigned int>(batch_size));
    
    // Consume capsule first to avoid SystemError if check fails
    consume_dlpack_capsule(dlpack_tensor);
    
    if (status != 0) {
        throw std::runtime_error("fwht_batch_f16_cuda_device failed with error code: " + std::to_string(status));
    }
}

// GPU Toggles
class PyGPUToggles {
public:
    static void set_multi_shuffle(bool enable) {
        fwht_status_t status = fwht_gpu_set_multi_shuffle(enable);
        check_status(status, "fwht_gpu_set_multi_shuffle");
    }
    
    static bool multi_shuffle_enabled() {
        return fwht_gpu_multi_shuffle_enabled();
    }
    
    static void set_block_size(unsigned int block_size) {
        fwht_status_t status = fwht_gpu_set_block_size(block_size);
        check_status(status, "fwht_gpu_set_block_size");
    }
    
    static unsigned int get_block_size() {
        return fwht_gpu_get_block_size();
    }
};

#endif  // USE_CUDA

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

bool py_fwht_is_power_of_2(size_t n) {
    return fwht_is_power_of_2(n);
}

int py_fwht_log2(size_t n) {
    return fwht_log2(n);
}

// =============================================================================
// MODULE DEFINITION
// =============================================================================

PYBIND11_MODULE(_pyfwht, m) {
    m.doc() = "Python bindings for libfwht - Fast Walsh-Hadamard Transform";
    
    // Enums
    py::enum_<fwht_backend_t>(m, "Backend", "Backend selection for FWHT computation")
        .value("AUTO", FWHT_BACKEND_AUTO, "Automatic backend selection based on size")
        .value("CPU", FWHT_BACKEND_CPU, "Single-threaded CPU (SIMD-optimized)")
        .value("OPENMP", FWHT_BACKEND_OPENMP, "Multi-threaded CPU (OpenMP)")
        .value("GPU", FWHT_BACKEND_GPU, "GPU-accelerated (CUDA)")
        .export_values();
    
    // Configuration struct
    py::class_<fwht_config_t>(m, "Config", "Configuration for FWHT context")
        .def(py::init<>())
        .def_readwrite("backend", &fwht_config_t::backend)
        .def_readwrite("num_threads", &fwht_config_t::num_threads)
        .def_readwrite("gpu_device", &fwht_config_t::gpu_device)
        .def_readwrite("normalize", &fwht_config_t::normalize);
    
    // Default config factory
    m.def("default_config", &fwht_default_config, "Get default FWHT configuration");
    
    // Core in-place transforms
    m.def("fwht_i32", &py_fwht_i32, py::arg("data"),
          "In-place Walsh-Hadamard Transform for int32 array");
    m.def("fwht_i32_safe", &py_fwht_i32_safe, py::arg("data"),
          "In-place WHT for int32 with overflow detection (5-10% slower but safe)");
    m.def("fwht_f64", &py_fwht_f64, py::arg("data"),
          "In-place Walsh-Hadamard Transform for float64 array");
    m.def("fwht_i8", &py_fwht_i8, py::arg("data"),
          "In-place Walsh-Hadamard Transform for int8 array (may overflow)");
    
    // Backend control
    m.def("fwht_i32_backend", &py_fwht_i32_backend, 
          py::arg("data"), py::arg("backend"),
          "In-place WHT for int32 with explicit backend selection");
    m.def("fwht_f64_backend", &py_fwht_f64_backend,
          py::arg("data"), py::arg("backend"),
          "In-place WHT for float64 with explicit backend selection");
    
    // Vectorized CPU batch processing (SIMD-optimized)
    m.def("fwht_i32_batch", &py_fwht_i32_batch,
          py::arg("data_list"), py::arg("n"),
          "Vectorized batch WHT for list of int32 arrays (3-5× faster for n≤256)");
    m.def("fwht_f64_batch", &py_fwht_f64_batch,
          py::arg("data_list"), py::arg("n"),
          "Vectorized batch WHT for list of float64 arrays (3-5× faster for n≤256)");
    
    // Out-of-place transforms
    m.def("fwht_compute_i32", &py_fwht_compute_i32, py::arg("input"),
          "Compute WHT for int32 (returns new array)");
    m.def("fwht_compute_f64", &py_fwht_compute_f64, py::arg("input"),
          "Compute WHT for float64 (returns new array)");
    m.def("fwht_compute_i32_backend", &py_fwht_compute_i32_backend,
          py::arg("input"), py::arg("backend"),
          "Compute WHT for int32 with backend selection (returns new array)");
    m.def("fwht_compute_f64_backend", &py_fwht_compute_f64_backend,
          py::arg("input"), py::arg("backend"),
          "Compute WHT for float64 with backend selection (returns new array)");
    
    // Boolean function API
    m.def("fwht_from_bool", &py_fwht_from_bool,
          py::arg("bool_func"), py::arg("signed_rep") = true,
          "Compute WHT from Boolean function (0/1 array)");
    m.def("fwht_correlations", &py_fwht_correlations, py::arg("bool_func"),
          "Compute correlations for Boolean function");
    
    // Bit-sliced Boolean WHT (packed representation)
    m.def("fwht_boolean_packed", &py_fwht_boolean_packed,
          py::arg("packed_bits"), py::arg("n"),
          "Compute WHT from bit-packed Boolean function (memory-efficient)");
    m.def("fwht_boolean_packed_backend", &py_fwht_boolean_packed_backend,
          py::arg("packed_bits"), py::arg("n"), py::arg("backend"),
          "Compute WHT from bit-packed Boolean function with backend selection");
    m.def("fwht_boolean_batch", &py_fwht_boolean_batch,
          py::arg("packed_list"), py::arg("n"),
          "Batch WHT for list of bit-packed Boolean functions (S-box cryptanalysis)");
    
    // Context API
    py::class_<PyFWHTContext>(m, "Context", "FWHT computation context for repeated calls")
        .def(py::init<const fwht_config_t&>(), py::arg("config"))
        .def("transform_i32", &PyFWHTContext::transform_i32, py::arg("data"),
             "Transform int32 array using context")
        .def("transform_f64", &PyFWHTContext::transform_f64, py::arg("data"),
             "Transform float64 array using context")
        .def("close", &PyFWHTContext::close,
             "Close context and release resources");
    
    // Utility functions
    m.def("is_power_of_2", &py_fwht_is_power_of_2, py::arg("n"),
          "Check if n is a power of 2");
    m.def("log2", &py_fwht_log2, py::arg("n"),
          "Compute log2(n) for power of 2 (returns -1 if not)");
    m.def("recommend_backend", &fwht_recommend_backend, py::arg("n"),
          "Get recommended backend for given size");
    
    // Backend availability
    m.def("has_openmp", &fwht_has_openmp, "Check if OpenMP support is available");
    m.def("has_gpu", &fwht_has_gpu, "Check if GPU/CUDA support is available");
    m.def("backend_name", &fwht_backend_name, py::arg("backend"),
          "Get name string for backend");
    
    // Version info
    m.def("version", &fwht_version, "Get library version string");
    m.attr("__version__") = fwht_version();
    
#ifdef USE_CUDA
    // GPU submodule
    py::module_ gpu = m.def_submodule("gpu", "GPU-specific functions and classes");
    
    // GPU Info
    py::class_<PyGPUInfo>(gpu, "Info", "GPU device information")
        .def_static("get_smem_banks", &PyGPUInfo::get_smem_banks,
                   "Get shared memory bank count (16 or 32)")
        .def_static("get_compute_capability", &PyGPUInfo::get_compute_capability,
                   "Get compute capability (e.g., 90 for SM 9.0)")
        .def_static("get_device_name", &PyGPUInfo::get_device_name,
                   "Get GPU device name")
        .def_static("get_sm_count", &PyGPUInfo::get_sm_count,
                   "Get streaming multiprocessor count");
    
    // GPU Profiling
    py::class_<PyGPUProfiling>(gpu, "Profiling", "GPU profiling controls")
        .def_static("set_profiling", &PyGPUProfiling::set_profiling, py::arg("enable"),
                   "Enable or disable GPU profiling")
        .def_static("profiling_enabled", &PyGPUProfiling::profiling_enabled,
                   "Check if GPU profiling is enabled")
        .def_static("get_last_metrics", &PyGPUProfiling::get_last_metrics,
                   "Get last profiling metrics (H2D/Kernel/D2H times)");
    
    // GPU Batch Processing
    gpu.def("batch_i32", &py_fwht_batch_i32_cuda,
            py::arg("data"), py::arg("n"), py::arg("batch_size"),
            "Batch WHT for int32 on GPU (data must be flat array of n * batch_size)");
    gpu.def("batch_f64", &py_fwht_batch_f64_cuda,
            py::arg("data"), py::arg("n"), py::arg("batch_size"),
            "Batch WHT for float64 on GPU (data must be flat array of n * batch_size)");
    gpu.def("batch_f32", &py_fwht_batch_f32_cuda,
            py::arg("data"), py::arg("n"), py::arg("batch_size"),
            "Batch WHT for float32 on GPU (uses Tensor Cores on sm_70+, ~2× faster than fp64)");
    gpu.def("batch_f16", &py_fwht_batch_f16_cuda,
            py::arg("data"), py::arg("n"), py::arg("batch_size"),
            "Batch WHT for float16 on GPU (uses Tensor Cores on sm_70+, achieves 1115 GOps/s @ n=4096 with PyTorch DLPack)");
    
    // DLPack-based zero-copy batch processing
    gpu.def("batch_f64_dlpack", &py_fwht_batch_f64_dlpack,
            py::arg("dlpack_tensor"), py::arg("n"), py::arg("batch_size"),
            "Zero-copy batch WHT for float64 via DLPack (PyTorch/CuPy/JAX tensors on GPU)");
    gpu.def("batch_i32_dlpack", &py_fwht_batch_i32_dlpack,
            py::arg("dlpack_tensor"), py::arg("n"), py::arg("batch_size"),
            "Zero-copy batch WHT for int32 via DLPack (PyTorch/CuPy/JAX tensors on GPU)");
    gpu.def("batch_f32_dlpack", &py_fwht_batch_f32_dlpack,
            py::arg("dlpack_tensor"), py::arg("n"), py::arg("batch_size"),
            "Zero-copy batch WHT for float32 via DLPack (Meta-inspired high-speed kernel, 30× faster than fp64)");
    gpu.def("batch_f16_dlpack", &py_fwht_batch_f16_dlpack,
            py::arg("dlpack_tensor"), py::arg("n"), py::arg("batch_size"),
            "Zero-copy batch WHT for float16 via DLPack (Meta-inspired maximum-speed kernel, up to 54× faster than fp64, 1115 GOps/s)");
    
    // GPU Context
    py::class_<PyGPUContext>(gpu, "Context", "Persistent GPU context for repeated transforms")
        .def(py::init<size_t, size_t>(), py::arg("max_n"), py::arg("max_batch_size"),
             "Create GPU context with pre-allocated memory")
        .def("compute_i32", &PyGPUContext::compute_i32,
             py::arg("data"), py::arg("n"), py::arg("batch_size"),
             "Compute WHT using context (int32)")
        .def("compute_f64", &PyGPUContext::compute_f64,
             py::arg("data"), py::arg("n"), py::arg("batch_size"),
             "Compute WHT using context (float64)")
        .def("close", &PyGPUContext::close,
             "Close context and release GPU resources");
    
    // GPU Toggles
    py::class_<PyGPUToggles>(gpu, "Toggles", "GPU kernel configuration and toggles")
        .def_static("set_multi_shuffle", &PyGPUToggles::set_multi_shuffle, py::arg("enable"),
                   "Enable multi-element warp-shuffle kernel (32 < N ≤ 512, experimental)")
        .def_static("multi_shuffle_enabled", &PyGPUToggles::multi_shuffle_enabled,
                   "Check if multi-shuffle kernel is enabled")
        .def_static("set_block_size", &PyGPUToggles::set_block_size, py::arg("block_size"),
                   "Set CUDA block size (power-of-2 in [1, 1024], or 0 for auto)")
        .def_static("get_block_size", &PyGPUToggles::get_block_size,
                   "Get current CUDA block size configuration");
#endif
}
