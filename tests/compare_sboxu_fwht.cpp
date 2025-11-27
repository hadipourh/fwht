#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <chrono>
#include <map>
#include <algorithm>
#include <string>
#include <thread>
#include <fstream>


static double seconds_between(std::chrono::steady_clock::time_point a,
                              std::chrono::steady_clock::time_point b);


struct BenchResult {
    size_t n;
    unsigned int threads;
    size_t iterations;
    double sboxu_total_s;
    double libfwht_total_s;
    double libfwht_gpu_unpacked_s; // GPU timing after host unpack (always available if GPU)
    double libfwht_gpu_packed_s;   // GPU timing using packed bits (n <= 64K)
    double libfwht_gpu_device_s;   // GPU timing with device-resident buffer (kernel only)
    bool correctness_ok;           // True if all methods agree
};

static std::vector<BenchResult> g_bench_results;

#include "fwht.h"
#include "../sboxU/sboxU/sboxU_cython/sboxu_cpp.hpp"
#include "../sboxU/sboxU/sboxU_cython/sboxu_cpp_diff_lin.hpp"

// Simple assert helper
static void die(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

// Generate random Boolean truth table of size n (0/1 values)
static std::vector<uint8_t> random_bool_table(size_t n) {
    static std::mt19937_64 rng(1234567);
    std::bernoulli_distribution dist(0.5);
    std::vector<uint8_t> f(n);
    for (size_t i = 0; i < n; ++i) {
        f[i] = dist(rng) ? 1u : 0u;
    }
    return f;
}

// Pack Boolean table (0/1) into uint64_t words: bit i of word j is f[j*64 + i]
static std::vector<uint64_t> pack_bits(const std::vector<uint8_t>& f) {
    const size_t n = f.size();
    const size_t words = (n + 63) / 64;
    std::vector<uint64_t> packed(words, 0);
    for (size_t i = 0; i < n; ++i) {
        if (f[i]) {
            size_t w = i / 64;
            size_t b = i & 63u;
            packed[w] |= (uint64_t(1) << b);
        }
    }
    return packed;
}

// Build SboxU Sbox from Boolean truth table: here we interpret f(x) as a 0/1 output.
// For Walsh on a single Boolean function, SboxU expects an Sbox of outputs, we use 0/1.
static Sbox truth_table_to_sbox(const std::vector<uint8_t>& f) {
    Sbox s(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
        s[i] = static_cast<BinWord>(f[i]);
    }
    return s;
}

static std::vector<int32_t> truth_table_to_signed(const std::vector<uint8_t>& f) {
    std::vector<int32_t> out(f.size());
    for (size_t i = 0; i < f.size(); ++i) {
        out[i] = f[i] ? -1 : 1;
    }
    return out;
}

static double benchmark_gpu_device_kernel(const std::vector<int32_t>& seed,
                                          size_t n,
                                          size_t iters) {
#ifdef USE_CUDA
    if (!fwht_has_gpu()) {
        return 0.0;
    }

    const size_t bytes = n * sizeof(int32_t);
    int32_t* d_data = NULL;
    int32_t* d_seed = NULL;
    fwht_status_t status = fwht_gpu_device_alloc(reinterpret_cast<void**>(&d_data), bytes);
    if (status != FWHT_SUCCESS) {
        return 0.0;
    }

    status = fwht_gpu_device_alloc(reinterpret_cast<void**>(&d_seed), bytes);
    if (status != FWHT_SUCCESS) {
        fwht_gpu_device_free(d_data);
        return 0.0;
    }

    status = fwht_gpu_memcpy_h2d(d_seed, seed.data(), bytes);
    if (status != FWHT_SUCCESS) {
        fwht_gpu_device_free(d_data);
        fwht_gpu_device_free(d_seed);
        return 0.0;
    }

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        // Restore seed data on device
        status = fwht_gpu_memcpy_h2d(d_data, seed.data(), bytes);
        if (status != FWHT_SUCCESS) {
            fwht_gpu_device_free(d_data);
            fwht_gpu_device_free(d_seed);
            return 0.0;
        }
        
        status = fwht_batch_i32_cuda_device(d_data, n, 1);
        if (status != FWHT_SUCCESS) {
            fwht_gpu_device_free(d_data);
            fwht_gpu_device_free(d_seed);
            return 0.0;
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    std::vector<int32_t> scratch(n);
    status = fwht_gpu_memcpy_d2h(scratch.data(), d_data, bytes);
    if (status != FWHT_SUCCESS) {
        fwht_gpu_device_free(d_data);
        fwht_gpu_device_free(d_seed);
        return 0.0;
    }

    fwht_gpu_device_free(d_data);
    fwht_gpu_device_free(d_seed);
    return seconds_between(t0, t1);
#else
    (void)seed;
    (void)n;
    (void)iters;
    return 0.0;
#endif
}

static bool use_bitpacked_path(size_t n) {
    return n <= (1u << 16);
}

template <typename T>
static void print_spectrum(const char* label,
                           const char* function_name,
                           const std::vector<T>& values) {
    std::printf("  %s (%s): [", label, function_name);
    for (size_t i = 0; i < values.size(); ++i) {
        long long v = static_cast<long long>(values[i]);
        std::printf("%lld", v);
        if (i + 1 != values.size()) {
            std::printf(", ");
        }
    }
    std::printf("]\n");
}

static void check_correctness(size_t print_limit_n, size_t max_n) {
    std::printf("[check] verifying SboxU vs libfwht spectra up to n=%zu...\n", max_n);

    const int trials_per_size = 6;
    for (size_t n = 2; n <= max_n; n <<= 1) {
        for (int trial = 0; trial < trials_per_size; ++trial) {
            auto f = random_bool_table(n);
            auto packed = pack_bits(f);
            auto signed_data = truth_table_to_signed(f);
            Sbox s = truth_table_to_sbox(f);

            std::vector<Integer> w_sboxu = walsh_spectrum_coord(s);
            std::vector<int32_t> w_fwht(n);
            fwht_status_t st;
            if (use_bitpacked_path(n)) {
                st = fwht_boolean_packed(packed.data(), w_fwht.data(), n);
            } else {
                w_fwht = signed_data;
                st = fwht_i32(w_fwht.data(), n);
            }
            if (st != FWHT_SUCCESS) {
                std::fprintf(stderr, "fwht_boolean_packed failed: %s\n", fwht_error_string(st));
                std::exit(1);
            }

            if (w_sboxu.size() != w_fwht.size()) {
                die("Output length mismatch between libraries");
            }

            for (size_t i = 0; i < n; ++i) {
                if (w_sboxu[i] != static_cast<Integer>(w_fwht[i])) {
                    std::fprintf(stderr,
                                 "Mismatch at n=%zu trial=%d index=%zu: SboxU=%lld, libfwht=%d\n",
                                 n, trial, i,
                                 static_cast<long long>(w_sboxu[i]), static_cast<int>(w_fwht[i]));
                    std::exit(1);
                }
            }

            if (trial == 0 && n <= print_limit_n) {
                std::printf("Example spectra for n=%zu using walsh_spectrum_coord (SboxU) and fwht_boolean_packed (libfwht):\n", n);
                print_spectrum("SboxU spectrum", "walsh_spectrum_coord", w_sboxu);
                print_spectrum("libfwht spectrum", "fwht_boolean_packed", w_fwht);
            }
        }
    }

    std::printf("[ok] correctness confirmed up to n=%zu.\n", max_n);
}

static double seconds_between(std::chrono::steady_clock::time_point a,
                              std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static int log2_size(size_t n) {
    int k = 0;
    while ((size_t(1) << k) < n) {
        ++k;
    }
    return k;
}

static std::vector<unsigned int> threads_for_size(size_t n,
                                                  unsigned int hw_threads) {
    const bool prefer_multi_only = (n > (1u << 15)) && (hw_threads > 1);
    std::vector<unsigned int> out;

    if (!prefer_multi_only || hw_threads == 1) {
        out.push_back(1);
    }
    if (hw_threads > 1) {
        out.push_back(hw_threads);
    }
    return out;
}

static size_t recommended_iters(size_t n) {
    if (n >= (1u << 26)) {
        return 10;
    }
    return 100;
}

static void write_results_csv(const std::vector<BenchResult>& results,
                              const std::string& path) {
    if (results.empty()) {
        return;
    }
    std::ofstream file(path);
    if (!file) {
        std::fprintf(stderr, "Failed to open %s for writing benchmark results.\n", path.c_str());
        return;
    }
    file << "n,threads,iters,";
    file << "sboxu_total_s,libfwht_total_s,";
    file << "libfwht_gpu_unpacked_s,libfwht_gpu_packed_s,libfwht_gpu_device_s,";
    file << "sboxu_us_per_iter,libfwht_us_per_iter,";
    file << "libfwht_gpu_unpacked_us_per_iter,libfwht_gpu_packed_us_per_iter,";
    file << "libfwht_gpu_device_us_per_iter,";
    file << "speedup_cpu,speedup_gpu_unpacked,speedup_gpu_packed,speedup_gpu_device,";
    file << "correctness\n";
    for (const auto& r : results) {
        const double sboxu_us = 1e6 * r.sboxu_total_s / r.iterations;
        const double libfwht_us = 1e6 * r.libfwht_total_s / r.iterations;
        const double libfwht_gpu_unpacked_us = (r.libfwht_gpu_unpacked_s > 0.0)
                                                    ? 1e6 * r.libfwht_gpu_unpacked_s / r.iterations
                                                    : 0.0;
        const double libfwht_gpu_packed_us = (r.libfwht_gpu_packed_s > 0.0)
                                                  ? 1e6 * r.libfwht_gpu_packed_s / r.iterations
                                                  : 0.0;
        const double libfwht_gpu_device_us = (r.libfwht_gpu_device_s > 0.0)
                                                   ? 1e6 * r.libfwht_gpu_device_s / r.iterations
                                                   : 0.0;
        const double speedup_cpu = (r.libfwht_total_s > 0.0) ? r.sboxu_total_s / r.libfwht_total_s : 0.0;
        const double speedup_gpu_unpacked = (r.libfwht_gpu_unpacked_s > 0.0)
                                                  ? r.sboxu_total_s / r.libfwht_gpu_unpacked_s
                                                  : 0.0;
        const double speedup_gpu_packed = (r.libfwht_gpu_packed_s > 0.0)
                                                ? r.sboxu_total_s / r.libfwht_gpu_packed_s
                                                : 0.0;
        const double speedup_gpu_device = (r.libfwht_gpu_device_s > 0.0)
                                                ? r.sboxu_total_s / r.libfwht_gpu_device_s
                                                : 0.0;

        file << r.n << ',' << r.threads << ',' << r.iterations << ','
             << r.sboxu_total_s << ',' << r.libfwht_total_s << ','
             << r.libfwht_gpu_unpacked_s << ',' << r.libfwht_gpu_packed_s << ','
             << r.libfwht_gpu_device_s << ','
             << sboxu_us << ',' << libfwht_us << ','
             << libfwht_gpu_unpacked_us << ',' << libfwht_gpu_packed_us << ','
             << libfwht_gpu_device_us << ','
             << speedup_cpu << ',' << speedup_gpu_unpacked << ','
             << speedup_gpu_packed << ',' << speedup_gpu_device << ','
             << (r.correctness_ok ? "PASS" : "FAIL") << '\n';
    }
    std::printf("\n[info] Wrote benchmark CSV to %s (%zu rows).\n",
                path.c_str(), results.size());
}

static void display_result(const BenchResult& result) {
    const double sboxu_us = 1e6 * result.sboxu_total_s / result.iterations;
    const double libfwht_us = 1e6 * result.libfwht_total_s / result.iterations;
    const double speedup = (result.libfwht_total_s > 0.0)
                               ? result.sboxu_total_s / result.libfwht_total_s
                               : 0.0;
    const bool have_gpu_unpacked = result.libfwht_gpu_unpacked_s > 0.0;
    const bool have_gpu_packed = result.libfwht_gpu_packed_s > 0.0;
    const bool have_gpu_device = result.libfwht_gpu_device_s > 0.0;
    const double libfwht_gpu_unpacked_us = have_gpu_unpacked
                                               ? 1e6 * result.libfwht_gpu_unpacked_s / result.iterations
                                               : 0.0;
    const double libfwht_gpu_packed_us = have_gpu_packed
                                             ? 1e6 * result.libfwht_gpu_packed_s / result.iterations
                                             : 0.0;
    const double libfwht_gpu_device_us = have_gpu_device
                                             ? 1e6 * result.libfwht_gpu_device_s / result.iterations
                                             : 0.0;
    const double speedup_gpu_unpacked = have_gpu_unpacked
                                            ? result.sboxu_total_s / result.libfwht_gpu_unpacked_s
                                            : 0.0;
    const double speedup_gpu_packed = have_gpu_packed
                                          ? result.sboxu_total_s / result.libfwht_gpu_packed_s
                                          : 0.0;
    const double speedup_gpu_device = have_gpu_device
                                           ? result.sboxu_total_s / result.libfwht_gpu_device_s
                                           : 0.0;

    const char* correctness_mark = result.correctness_ok ? "✓" : "✗";
    std::printf("\n[bench] n=%zu (2^%d), threads=%u, iters=%zu [%s]\n",
                result.n, log2_size(result.n), result.threads, result.iterations, correctness_mark);
    std::printf("    %-15s | %12s | %14s\n", "method", "total (s)", "per-iter (us)");
    std::printf("    %-15s | %12.6f | %14.2f\n", "SboxU coord",
                result.sboxu_total_s, sboxu_us);
    std::printf("    %-15s | %12.6f | %14.2f\n", "libfwht",
                result.libfwht_total_s, libfwht_us);
    if (have_gpu_unpacked) {
        std::printf("    %-15s | %12.6f | %14.2f\n", "GPU (unpacked)",
                    result.libfwht_gpu_unpacked_s, libfwht_gpu_unpacked_us);
    }
    if (have_gpu_packed) {
        std::printf("    %-15s | %12.6f | %14.2f\n", "GPU (packed)",
                    result.libfwht_gpu_packed_s, libfwht_gpu_packed_us);
    }
    if (have_gpu_device) {
        std::printf("    %-15s | %12.6f | %14.2f\n", "GPU (device)",
                    result.libfwht_gpu_device_s, libfwht_gpu_device_us);
    }
    double best_time = result.libfwht_total_s;
    const char* best_label = "libfwht";
    if (have_gpu_unpacked && result.libfwht_gpu_unpacked_s > 0.0 &&
        result.libfwht_gpu_unpacked_s < best_time) {
        best_time = result.libfwht_gpu_unpacked_s;
        best_label = "GPU (unpacked)";
    }
    if (have_gpu_packed && result.libfwht_gpu_packed_s > 0.0 &&
        result.libfwht_gpu_packed_s < best_time) {
        best_time = result.libfwht_gpu_packed_s;
        best_label = "GPU (packed)";
    }
    if (have_gpu_device && result.libfwht_gpu_device_s > 0.0 &&
        result.libfwht_gpu_device_s < best_time) {
        best_time = result.libfwht_gpu_device_s;
        best_label = "GPU (device)";
    }
    if (best_time > 0.0) {
        const double best_speedup = result.sboxu_total_s / best_time;
        std::printf("    %-15s | %12.2f | %s\n", "speedup (best)", best_speedup, best_label);
    }
}


static BenchResult benchmark(size_t n, unsigned int n_threads, size_t iters) {
    auto f = random_bool_table(n);
    Sbox s = truth_table_to_sbox(f);
    auto packed = pack_bits(f);
    auto fwht_seed = truth_table_to_signed(f);
    std::vector<int32_t> fwht_work;

    // Correctness verification: compare all methods
    bool correctness_ok = true;
    std::vector<Integer> ref_spectrum = walsh_spectrum_coord(s);
    std::vector<int32_t> verify_cpu(n);
    std::vector<int32_t> verify_gpu_unpacked(n);
    std::vector<int32_t> verify_gpu_packed(n);

    // CPU path
    if (use_bitpacked_path(n)) {
        fwht_status_t st = fwht_boolean_packed(packed.data(), verify_cpu.data(), n);
        if (st != FWHT_SUCCESS) {
            std::fprintf(stderr, "[warn] CPU verification failed\n");
            correctness_ok = false;
        }
    } else {
        verify_cpu = fwht_seed;
        fwht_status_t st = fwht_i32(verify_cpu.data(), n);
        if (st != FWHT_SUCCESS) {
            std::fprintf(stderr, "[warn] CPU verification failed\n");
            correctness_ok = false;
        }
    }

    // GPU unpacked path
    if (fwht_has_gpu() && correctness_ok) {
        verify_gpu_unpacked = fwht_seed;
        fwht_status_t st = fwht_i32_backend(verify_gpu_unpacked.data(), n, FWHT_BACKEND_GPU);
        if (st == FWHT_SUCCESS) {
            for (size_t i = 0; i < n && correctness_ok; ++i) {
                if (verify_gpu_unpacked[i] != verify_cpu[i]) {
                    std::fprintf(stderr, "[warn] GPU unpacked mismatch at index %zu\n", i);
                    correctness_ok = false;
                }
            }
        }
    }

    // GPU packed path (n <= 64K)
    if (fwht_has_gpu() && use_bitpacked_path(n) && correctness_ok) {
        fwht_status_t st = fwht_boolean_packed_backend(
            packed.data(), verify_gpu_packed.data(), n, FWHT_BACKEND_GPU);
        if (st == FWHT_SUCCESS) {
            for (size_t i = 0; i < n && correctness_ok; ++i) {
                if (verify_gpu_packed[i] != verify_cpu[i]) {
                    std::fprintf(stderr, "[warn] GPU packed mismatch at index %zu\n", i);
                    correctness_ok = false;
                }
            }
        }
    }

    // Compare with SboxU reference
    if (correctness_ok) {
        for (size_t i = 0; i < n && correctness_ok; ++i) {
            if (ref_spectrum[i] != static_cast<Integer>(verify_cpu[i])) {
                std::fprintf(stderr, "[warn] CPU vs SboxU mismatch at index %zu\n", i);
                correctness_ok = false;
            }
        }
    }

    // Warmup
    if (n <= (1u << 14)) {
        (void)walsh_spectrum_coord(s);
    }
    fwht_status_t warmup_status;
    std::vector<int32_t> w_fwht;
    if (use_bitpacked_path(n)) {
        w_fwht.assign(n, 0);
        warmup_status = fwht_boolean_packed(packed.data(), w_fwht.data(), n);
    } else {
        fwht_work = fwht_seed;
        warmup_status = fwht_i32(fwht_work.data(), n);
    }
    if (warmup_status != FWHT_SUCCESS) {
        std::fprintf(stderr, "libfwht warmup failed: %s\n", fwht_error_string(warmup_status));
        std::exit(1);
    }

    // SboxU coord (scalar)
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        auto spec = walsh_spectrum_coord(s);
        (void)spec;
    }
    auto t1 = std::chrono::steady_clock::now();
    double sboxu_sec = seconds_between(t0, t1);

    // libfwht (bit-packed for small n, fwht_i32 for larger)
    t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        if (use_bitpacked_path(n)) {
            fwht_status_t st = fwht_boolean_packed(packed.data(), w_fwht.data(), n);
            if (st != FWHT_SUCCESS) die("fwht_boolean_packed failed in benchmark");
        } else {
            fwht_work = fwht_seed;
            fwht_status_t st = fwht_i32(fwht_work.data(), n);
            if (st != FWHT_SUCCESS) die("fwht_i32 failed in benchmark");
        }
    }
    t1 = std::chrono::steady_clock::now();
    double fwht_sec = seconds_between(t0, t1);

    double fwht_gpu_unpacked_sec = 0.0;
    double fwht_gpu_packed_sec = 0.0;
    double fwht_gpu_device_sec = 0.0;
    const bool gpu_available = fwht_has_gpu();
    if (gpu_available) {
        // Measure GPU time when data is already unpacked (int32 ±1)
        bool gpu_failed = false;
        t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) {
            std::vector<int32_t> gpu_unpacked_buffer = fwht_seed;
            fwht_status_t st = fwht_i32_backend(gpu_unpacked_buffer.data(), n, FWHT_BACKEND_GPU);
            if (st != FWHT_SUCCESS) {
                fprintf(stderr, "[warn] fwht_i32_backend GPU failed for n=%zu: %s\n",
                        n, fwht_error_string(st));
                gpu_failed = true;
                break;
            }
        }
        if (!gpu_failed) {
            t1 = std::chrono::steady_clock::now();
            fwht_gpu_unpacked_sec = seconds_between(t0, t1);
        }

        // Measure GPU time when input remains packed (n <= 64K only)
        if (!gpu_failed && use_bitpacked_path(n)) {
            std::vector<int32_t> w_fwht_gpu_packed(n, 0);
            t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < iters; ++i) {
                fwht_status_t st = fwht_boolean_packed_backend(
                    packed.data(), w_fwht_gpu_packed.data(), n, FWHT_BACKEND_GPU);
                if (st != FWHT_SUCCESS) {
                    fprintf(stderr, "[warn] fwht_boolean_packed_backend GPU failed for n=%zu: %s\n",
                            n, fwht_error_string(st));
                    gpu_failed = true;
                    break;
                }
            }
            if (!gpu_failed) {
                t1 = std::chrono::steady_clock::now();
                fwht_gpu_packed_sec = seconds_between(t0, t1);
            }
        }

        if (!gpu_failed) {
            double device_time = benchmark_gpu_device_kernel(fwht_seed, n, iters);
            if (device_time > 0.0) {
                fwht_gpu_device_sec = device_time;
            }
        }
    }

    BenchResult result{n, n_threads, iters, sboxu_sec, fwht_sec,
                       fwht_gpu_unpacked_sec, fwht_gpu_packed_sec,
                       fwht_gpu_device_sec, correctness_ok};
    display_result(result);
    return result;
}

int main() {
    const size_t print_limit_n = 16;  // print full spectra for n <= 16
    const size_t correctness_max_n = 1u << 13; // 8192
    check_correctness(print_limit_n, correctness_max_n);

    const std::vector<size_t> sizes = {
        1u << 10,
        1u << 12,
        1u << 13,
        1u << 14,
        1u << 15,
        1u << 16,
        1u << 17,
        1u << 18,
        1u << 19,
        1u << 20,
        1u << 21,
        1u << 22,
        1u << 23,
        1u << 24,
        1u << 25,
        1u << 26,
        1u << 27
    };

    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) {
        hw_threads = 4;
    }

    if (hw_threads == 1) {
        std::printf("\n[info] Benchmark thread counts: 1 (single-thread only).\n");
    } else {
        std::printf("\n[info] Benchmark thread counts: 1 for n <= 2^15, and %u for larger sizes.\n",
                    hw_threads);
    }

    for (size_t n : sizes) {
        auto thread_modes = threads_for_size(n, hw_threads);
        for (unsigned int threads : thread_modes) {
            g_bench_results.push_back(benchmark(n, threads, recommended_iters(n)));
        }
    }

    write_results_csv(g_bench_results, "build/compare_sboxu_fwht.csv");

    return 0;
}
