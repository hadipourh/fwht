/**
 * @file fwht_wrapper.cpp
 * @brief Wrapper to compile C library sources as C++ for Python bindings.
 * 
 * This file includes the C library implementation files and compiles them
 * as C++. This avoids compilation flag conflicts between C and C++ in the
 * Python extension build process.
 */

extern "C" {
    // C implementation files are provided via the compiler's include path
    #include "fwht_core.c"
    #include "fwht_backend.c"
    #include "fwht_batch.c"
    #include "fwht_sbox.c"
}
