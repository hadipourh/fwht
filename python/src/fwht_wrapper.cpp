/**
 * @file fwht_wrapper.cpp
 * @brief Wrapper to compile C library sources as C++ for Python bindings.
 * 
 * This file includes the C library implementation files and compiles them
 * as C++. This avoids compilation flag conflicts between C and C++ in the
 * Python extension build process.
 */

extern "C" {
    // Include C implementation files from local c_src directory
    #include "../c_src/fwht_core.c"
    #include "../c_src/fwht_backend.c"
}
