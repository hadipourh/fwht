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

struct BenchResult {
    size_t n;
    unsigned int threads;
    size_t iterations;
    double sboxu_total_s;
    double libfwht_total_s;
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
    file << "n,threads,iters,sboxu_total_s,libfwht_total_s,sboxu_us_per_iter,libfwht_us_per_iter,speedup\n";
    for (const auto& r : results) {
        const double sboxu_us = 1e6 * r.sboxu_total_s / r.iterations;
        const double libfwht_us = 1e6 * r.libfwht_total_s / r.iterations;
        const double speedup = (r.libfwht_total_s > 0.0)
                                   ? r.sboxu_total_s / r.libfwht_total_s
                                   : 0.0;
        file << r.n << ',' << r.threads << ',' << r.iterations << ','
             << r.sboxu_total_s << ',' << r.libfwht_total_s << ','
             << sboxu_us << ',' << libfwht_us << ',' << speedup << '\n';
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

    std::printf("\n[bench] n=%zu (2^%d), threads=%u, iters=%zu\n",
                result.n, log2_size(result.n), result.threads, result.iterations);
    std::printf("    %-15s | %12s | %14s\n", "method", "total (s)", "per-iter (us)");
    std::printf("    %-15s | %12.6f | %14.2f\n", "SboxU coord",
                result.sboxu_total_s, sboxu_us);
    std::printf("    %-15s | %12.6f | %14.2f\n", "libfwht",
                result.libfwht_total_s, libfwht_us);
    if (speedup > 0.0) {
        std::printf("    %-15s | %12.2f | %-14s\n",
                    "speedup", speedup, "libfwht faster");
    }
}

static BenchResult benchmark(size_t n, unsigned int n_threads, size_t iters) {
    auto f = random_bool_table(n);
    Sbox s = truth_table_to_sbox(f);
    auto packed = pack_bits(f);
    auto fwht_seed = truth_table_to_signed(f);
    std::vector<int32_t> fwht_work;

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

    BenchResult result{n, n_threads, iters, sboxu_sec, fwht_sec};
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
