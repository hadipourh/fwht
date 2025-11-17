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

extern "C" {
    #include "fwht.h"
}

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
    py::class_<PyGPUToggles>(gpu, "Toggles", "GPU kernel variant toggles")
        .def_static("set_multi_shuffle", &PyGPUToggles::set_multi_shuffle, py::arg("enable"),
                   "Enable multi-element warp-shuffle kernel (32 < N ≤ 512, experimental)")
        .def_static("multi_shuffle_enabled", &PyGPUToggles::multi_shuffle_enabled,
                   "Check if multi-shuffle kernel is enabled");
#endif
}
