#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../include/fwht.h"

extern fwht_backend_t fwht_recommend_backend(size_t n);

typedef struct {
    size_t n;
    int threads;
    size_t iterations;
    fwht_backend_t libfwht_backend;
    double libfwht_setup_s;
    double libfwht_exec_s;
    double fftw_plan_s;
    double fftw_exec_s;
    double max_diff;
    bool correctness_ok;
} bench_result_t;

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int ilog2_size(size_t n) {
    int bits = 0;
    while (n > 1) {
        n >>= 1;
        ++bits;
    }
    return bits;
}

static uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void fill_random_f64(double* data, size_t n, uint64_t seed) {
    uint64_t state = seed;
    for (size_t i = 0; i < n; ++i) {
        uint64_t value = splitmix64_next(&state);
        data[i] = (double)((int64_t)(value % 8192ULL) - 4096LL);
    }
}

static void direct_wht(const double* input, double* output, size_t n) {
    for (size_t u = 0; u < n; ++u) {
        double total = 0.0;
        for (size_t x = 0; x < n; ++x) {
            int parity = __builtin_parityll((unsigned long long)(u & x));
            total += parity ? -input[x] : input[x];
        }
        output[u] = total;
    }
}

static double max_abs_diff(const double* left, const double* right, size_t n) {
    double max_value = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = fabs(left[i] - right[i]);
        if (diff > max_value) {
            max_value = diff;
        }
    }
    return max_value;
}

static size_t recommended_iters(size_t n) {
    if (n <= ((size_t)1 << 12)) {
        return 2000;
    }
    if (n <= ((size_t)1 << 16)) {
        return 400;
    }
    if (n <= ((size_t)1 << 20)) {
        return 50;
    }
    if (n <= ((size_t)1 << 23)) {
        return 10;
    }
    if (n <= ((size_t)1 << 25)) {
        return 5;
    }
    if (n <= ((size_t)1 << 27)) {
        return 3;
    }
    return 1;
}

static int detected_thread_count(void) {
    long value = sysconf(_SC_NPROCESSORS_ONLN);
    if (value < 1) {
        return 1;
    }
    if (value > 256) {
        return 256;
    }
    return (int)value;
}

static double gib_from_bytes(size_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static void print_vector(const char* label, const double* values, size_t n) {
    printf("%s[", label);
    for (size_t i = 0; i < n; ++i) {
        printf("%s%.0f", (i == 0 ? "" : ", "), values[i]);
    }
    printf("]\n");
}

static fwht_backend_t resolved_libfwht_backend(size_t n, fwht_backend_t backend) {
    if (backend == FWHT_BACKEND_AUTO) {
        return fwht_recommend_backend(n);
    }
    return backend;
}

static double* allocate_work_buffer(size_t n, const char* label) {
    size_t bytes = n * sizeof(double);
    double* buffer = (double*)fftw_malloc(bytes);
    if (buffer == NULL) {
        fprintf(stderr,
                "[compare_fftw_fwht] %s allocation failed for n=%zu (%.2f GiB)\n",
                label,
                n,
                gib_from_bytes(bytes));
    }
    return buffer;
}

static bool benchmark_libfwht(double* work,
                              size_t n,
                              size_t iterations,
                              uint64_t seed,
                              fwht_backend_t backend,
                              int threads,
                              fwht_backend_t* resolved_backend,
                              double* setup_s,
                              double* exec_s) {
    fwht_config_t config = fwht_default_config();
    config.backend = backend;
    config.num_threads = threads;

    if (resolved_backend != NULL) {
        *resolved_backend = resolved_libfwht_backend(n, backend);
    }

    double t0 = monotonic_seconds();
    fwht_context_t* ctx = fwht_create_context(&config);
    double t1 = monotonic_seconds();
    if (ctx == NULL) {
        fprintf(stderr, "[compare_fftw_fwht] fwht_create_context failed for n=%zu\n", n);
        return false;
    }
    *setup_s = t1 - t0;

    fill_random_f64(work, n, seed);
    if (fwht_transform_f64(ctx, work, n) != FWHT_SUCCESS) {
        fprintf(stderr, "[compare_fftw_fwht] libfwht transform failed for n=%zu\n", n);
        fwht_destroy_context(ctx);
        return false;
    }

    double total = 0.0;
    for (size_t it = 0; it < iterations; ++it) {
        fill_random_f64(work, n, seed);
        t0 = monotonic_seconds();
        if (fwht_transform_f64(ctx, work, n) != FWHT_SUCCESS) {
            fprintf(stderr, "[compare_fftw_fwht] libfwht transform failed during benchmark for n=%zu\n", n);
            fwht_destroy_context(ctx);
            return false;
        }
        t1 = monotonic_seconds();
        total += (t1 - t0);
    }

    *exec_s = total / (double)iterations;

    fwht_destroy_context(ctx);
    return true;
}

static bool benchmark_fftw_dht(double* work,
                               size_t n,
                               size_t iterations,
                               uint64_t seed,
                               int threads,
                               bool threads_available,
                               double* plan_s,
                               double* exec_s) {
    int rank = ilog2_size(n);
    int* dims = (int*)malloc((size_t)rank * sizeof(int));
    fftw_r2r_kind* kinds = (fftw_r2r_kind*)malloc((size_t)rank * sizeof(fftw_r2r_kind));
    if (dims == NULL || kinds == NULL) {
        fprintf(stderr, "[compare_fftw_fwht] FFTW metadata allocation failed for n=%zu\n", n);
        free(dims);
        free(kinds);
        return false;
    }

    for (int i = 0; i < rank; ++i) {
        dims[i] = 2;
        kinds[i] = FFTW_DHT;
    }

#ifdef USE_FFTW_THREADS
    if (threads > 1 && threads_available) {
        fftw_plan_with_nthreads(threads);
    } else {
        fftw_plan_with_nthreads(1);
    }
#else
    (void)threads;
    (void)threads_available;
#endif

    fill_random_f64(work, n, seed);
    double t0 = monotonic_seconds();
    fftw_plan plan = fftw_plan_r2r(rank, dims, work, work, kinds, FFTW_MEASURE);
    double t1 = monotonic_seconds();
    if (plan == NULL) {
        fprintf(stderr, "[compare_fftw_fwht] FFTW plan creation failed for n=%zu\n", n);
        free(dims);
        free(kinds);
        return false;
    }
    *plan_s = t1 - t0;

    fill_random_f64(work, n, seed);
    fftw_execute(plan);

    double total = 0.0;
    for (size_t it = 0; it < iterations; ++it) {
        fill_random_f64(work, n, seed);
        t0 = monotonic_seconds();
        fftw_execute(plan);
        t1 = monotonic_seconds();
        total += (t1 - t0);
    }

    *exec_s = total / (double)iterations;

    fftw_destroy_plan(plan);
    free(dims);
    free(kinds);
    return true;
}

static bool benchmark_mode(size_t n,
                           size_t iterations,
                           uint64_t seed,
                           fwht_backend_t backend,
                           int threads,
                           bool fftw_threads_available,
                           bench_result_t* result) {
    double* lib_work = allocate_work_buffer(n, "libfwht work buffer");
    double* fftw_work = allocate_work_buffer(n, "FFTW work buffer");
    if (lib_work == NULL || fftw_work == NULL) {
        fftw_free(lib_work);
        fftw_free(fftw_work);
        return false;
    }

    result->n = n;
    result->threads = threads;
    result->iterations = iterations;

    if (!benchmark_libfwht(lib_work,
                           n,
                           iterations,
                           seed,
                           backend,
                           threads,
                           &result->libfwht_backend,
                           &result->libfwht_setup_s,
                           &result->libfwht_exec_s) ||
        !benchmark_fftw_dht(fftw_work,
                            n,
                            iterations,
                            seed,
                            threads,
                            fftw_threads_available,
                            &result->fftw_plan_s,
                            &result->fftw_exec_s)) {
        fftw_free(lib_work);
        fftw_free(fftw_work);
        return false;
    }

    result->max_diff = max_abs_diff(lib_work, fftw_work, n);
    result->correctness_ok = result->max_diff <= 1e-9;

    fftw_free(lib_work);
    fftw_free(fftw_work);
    return true;
}

static bool verify_correctness(int threaded_threads, bool fftw_threads_available) {
    printf("[check] verifying libfwht and FFTW DHT against a direct WHT up to n=4096...\n");
    for (size_t n = 2; n <= ((size_t)1 << 12); n <<= 1) {
        uint64_t seed = 1234u + (uint64_t)n;
        double* input = (double*)malloc(n * sizeof(double));
        double* reference = (double*)malloc(n * sizeof(double));
        double* lib_out = (double*)fftw_malloc(n * sizeof(double));
        double* fftw_out = (double*)fftw_malloc(n * sizeof(double));
        if (input == NULL || reference == NULL || lib_out == NULL || fftw_out == NULL) {
            fprintf(stderr, "[compare_fftw_fwht] correctness allocation failed for n=%zu\n", n);
            free(input);
            free(reference);
            fftw_free(lib_out);
            fftw_free(fftw_out);
            return false;
        }

        fill_random_f64(input, n, seed);
        direct_wht(input, reference, n);

        double setup_s = 0.0;
        double exec_s = 0.0;
        fwht_backend_t resolved_backend = FWHT_BACKEND_CPU;
        if (!benchmark_libfwht(lib_out,
                               n,
                               1,
                               seed,
                               FWHT_BACKEND_CPU,
                               1,
                               &resolved_backend,
                               &setup_s,
                               &exec_s)) {
            free(input);
            free(reference);
            fftw_free(lib_out);
            fftw_free(fftw_out);
            return false;
        }
        if (max_abs_diff(reference, lib_out, n) > 1e-9) {
            fprintf(stderr, "[compare_fftw_fwht] libfwht CPU mismatch for n=%zu\n", n);
            free(input);
            free(reference);
            fftw_free(lib_out);
            fftw_free(fftw_out);
            return false;
        }

        if (!benchmark_fftw_dht(fftw_out, n, 1, seed, 1, fftw_threads_available, &setup_s, &exec_s)) {
            free(input);
            free(reference);
            fftw_free(lib_out);
            fftw_free(fftw_out);
            return false;
        }
        if (max_abs_diff(reference, fftw_out, n) > 1e-9) {
            fprintf(stderr, "[compare_fftw_fwht] FFTW DHT mismatch for n=%zu\n", n);
            free(input);
            free(reference);
            fftw_free(lib_out);
            fftw_free(fftw_out);
            return false;
        }

        if (threaded_threads > 1 && fwht_has_openmp()) {
            if (!benchmark_libfwht(lib_out,
                                   n,
                                   1,
                                   seed,
                                   FWHT_BACKEND_OPENMP,
                                   threaded_threads,
                                   &resolved_backend,
                                   &setup_s,
                                   &exec_s)) {
                free(input);
                free(reference);
                fftw_free(lib_out);
                fftw_free(fftw_out);
                return false;
            }
            if (max_abs_diff(reference, lib_out, n) > 1e-9) {
                fprintf(stderr, "[compare_fftw_fwht] libfwht OpenMP mismatch for n=%zu\n", n);
                free(input);
                free(reference);
                fftw_free(lib_out);
                fftw_free(fftw_out);
                return false;
            }
        }

        if (threaded_threads > 1 && fftw_threads_available) {
            if (!benchmark_fftw_dht(fftw_out,
                                    n,
                                    1,
                                    seed,
                                    threaded_threads,
                                    fftw_threads_available,
                                    &setup_s,
                                    &exec_s)) {
                free(input);
                free(reference);
                fftw_free(lib_out);
                fftw_free(fftw_out);
                return false;
            }
            if (max_abs_diff(reference, fftw_out, n) > 1e-9) {
                fprintf(stderr, "[compare_fftw_fwht] FFTW threaded DHT mismatch for n=%zu\n", n);
                free(input);
                free(reference);
                fftw_free(lib_out);
                fftw_free(fftw_out);
                return false;
            }
        }

        if (n <= 8) {
            print_vector("  input      = ", input, n);
            print_vector("  direct WHT = ", reference, n);
            print_vector("  FFTW DHT   = ", fftw_out, n);
        }

        free(input);
        free(reference);
        fftw_free(lib_out);
        fftw_free(fftw_out);
    }
    printf("[ok] correctness confirmed through n=4096.\n");
    return true;
}

static void print_header(const char* label) {
    printf("\n%s\n", label);
    printf("%10s  %7s  %7s  %14s  %14s  %12s  %10s  %14s  %11s\n",
           "Size",
           "threads",
           "iters",
           "libfwht_s",
           "fftw_exec_s",
           "fftw_plan_s",
           "backend",
           "winner",
           "max|diff|");
}

static void print_result(const bench_result_t* result) {
    char winner[64];
    if (result->libfwht_exec_s < result->fftw_exec_s) {
        double ratio = result->fftw_exec_s / result->libfwht_exec_s;
        snprintf(winner, sizeof(winner), "libfwht x %.2f", ratio);
    } else {
        double ratio = result->libfwht_exec_s / result->fftw_exec_s;
        snprintf(winner, sizeof(winner), "FFTW x %.2f", ratio);
    }
        printf("%10zu  %7d  %7zu  %14.9f  %14.9f  %12.6f  %10s  %14s  %11.3e%s\n",
           result->n,
           result->threads,
           result->iterations,
           result->libfwht_exec_s,
           result->fftw_exec_s,
           result->fftw_plan_s,
            fwht_backend_name(result->libfwht_backend),
           winner,
           result->max_diff,
           result->correctness_ok ? "" : "  FAIL");
}

int main(void) {
    const size_t sizes[] = {
        ((size_t)1 << 8),  ((size_t)1 << 9),  ((size_t)1 << 10), ((size_t)1 << 11),
        ((size_t)1 << 12), ((size_t)1 << 13), ((size_t)1 << 14), ((size_t)1 << 15),
        ((size_t)1 << 16), ((size_t)1 << 17), ((size_t)1 << 18), ((size_t)1 << 19),
        ((size_t)1 << 20), ((size_t)1 << 21), ((size_t)1 << 22), ((size_t)1 << 23),
        ((size_t)1 << 24), ((size_t)1 << 25), ((size_t)1 << 26), ((size_t)1 << 27),
        ((size_t)1 << 28), ((size_t)1 << 29), ((size_t)1 << 30)
    };
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int max_threads = detected_thread_count();

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

#ifdef USE_FFTW_THREADS
    bool fftw_threads_available = fftw_init_threads() != 0;
#else
    bool fftw_threads_available = false;
#endif

    printf("libfwht vs FFTW DHT (double-precision CPU comparison)\n");
    printf("FFTW route: rank-n size-2 FFTW_DHT plan reused across iterations\n");
    printf("Planning cost is reported separately from steady-state execution time\n");
    printf("Largest sweep size : 2^30 (%.2f GiB per work buffer)\n", gib_from_bytes(((size_t)1 << 30) * sizeof(double)));
    printf("Threaded section: both libfwht and FFTW forced to use OpenMP for all sizes (apple-to-apple).\n");
    printf("OpenMP available : %s\n", fwht_has_openmp() ? "yes" : "no");
    printf("FFTW threads     : %s\n", fftw_threads_available ? "yes" : "no");
    printf("Max threads      : %d\n", max_threads);

    if (!verify_correctness(max_threads, fftw_threads_available)) {
#ifdef USE_FFTW_THREADS
        if (fftw_threads_available) {
            fftw_cleanup_threads();
        }
#endif
        return EXIT_FAILURE;
    }

    print_header("Mode: single-thread execution");
    for (size_t i = 0; i < num_sizes; ++i) {
        bench_result_t result;
        size_t n = sizes[i];
        size_t iterations = recommended_iters(n);
        if (!benchmark_mode(n,
                            iterations,
                            2026u + (uint64_t)n,
                            FWHT_BACKEND_CPU,
                            1,
                            fftw_threads_available,
                            &result)) {
#ifdef USE_FFTW_THREADS
            if (fftw_threads_available) {
                fftw_cleanup_threads();
            }
#endif
            return EXIT_FAILURE;
        }
        print_result(&result);
    }

    if (max_threads > 1 && fwht_has_openmp() && fftw_threads_available) {
        print_header("Mode: forced OpenMP threading with max-thread request (apple-to-apple)");
        for (size_t i = 0; i < num_sizes; ++i) {
            bench_result_t result;
            size_t n = sizes[i];
            size_t iterations = recommended_iters(n);
            if (!benchmark_mode(n,
                                iterations,
                                4096u + (uint64_t)n,
                                FWHT_BACKEND_OPENMP,
                                max_threads,
                                fftw_threads_available,
                                &result)) {
#ifdef USE_FFTW_THREADS
                fftw_cleanup_threads();
#endif
                return EXIT_FAILURE;
            }
            print_result(&result);
        }
    } else if (max_threads > 1) {
        printf("\n[info] Skipping matched multi-thread benchmark because one runtime lacks thread support.\n");
    }

#ifdef USE_FFTW_THREADS
    if (fftw_threads_available) {
        fftw_cleanup_threads();
    }
#endif
    return EXIT_SUCCESS;
}
