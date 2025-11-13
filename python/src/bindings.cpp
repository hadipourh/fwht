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
    // Use fwht_free for aligned memory deallocation
    py::capsule free_when_done(result, [](void* p) {
        fwht_free(p);
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
    // Use fwht_free for aligned memory deallocation
    py::capsule free_when_done(result, [](void* p) {
        fwht_free(p);
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
        fwht_free(p);
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
        fwht_free(p);
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
}
