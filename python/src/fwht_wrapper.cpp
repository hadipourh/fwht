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
    /*
     * Use angle-bracket includes so the compiler resolves these files via
     * the include search path (configured to point at either the repo's
     * src/ directory or the vendored fallback). This avoids silently
     * picking up stale copies in python/src/ when developing from source.
     */
    #include <fwht_core.c>
    #include <fwht_backend.c>
    #include <fwht_batch.c>
    #include <fwht_sbox.c>
    #include <fwht_simd_avx2.c>
    #include <fwht_simd_neon.c>
}
