#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#include "../include/fwht.h"
#include "../otherlibs/FFHT/fht.h"

static double timestamp_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

static void fill_random_f64(double *data, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        data[i] = (double)(rand() % 65536 - 32768);
    }
}

static int verify_equal_doubles(const double *a, const double *b, size_t n, double epsilon) {
    for (size_t i = 0; i < n; ++i) {
        double diff = fabs(a[i] - b[i]);
        if (diff > epsilon) {
            return 0;
        }
    }
    return 1;
}

static int verify_small_sizes(void) {
    unsigned int seed = 12345U;
    double epsilon = 1e-9;
    
    for (size_t n = 2; n <= 1u << 12; n <<= 1) {
        double *buf_fwht = (double *)malloc(n * sizeof(double));
        double *buf_ffht = (double *)malloc(n * sizeof(double));
        if (!buf_fwht || !buf_ffht) {
            fprintf(stderr, "[compare_ffht_fwht_fp64] alloc failed for n=%zu\n", n);
            free(buf_fwht);
            free(buf_ffht);
            return 0;
        }

        fill_random_f64(buf_fwht, n, seed);
        memcpy(buf_ffht, buf_fwht, n * sizeof(double));

        if (fwht_f64_backend(buf_fwht, n, FWHT_BACKEND_CPU) != FWHT_SUCCESS) {
            fprintf(stderr, "[compare_ffht_fwht_fp64] fwht_f64_backend failed for n=%zu\n", n);
            free(buf_fwht);
            free(buf_ffht);
            return 0;
        }

        int log_n = 0;
        size_t tmp = n;
        while (tmp > 1) {
            tmp >>= 1;
            log_n++;
        }

        if (fht_double(buf_ffht, log_n) != 0) {
            fprintf(stderr, "[compare_ffht_fwht_fp64] fht_double failed for n=%zu\n", n);
            free(buf_fwht);
            free(buf_ffht);
            return 0;
        }

        if (!verify_equal_doubles(buf_fwht, buf_ffht, n, epsilon)) {
            fprintf(stderr, "[compare_ffht_fwht_fp64] mismatch between libfwht and FFHT for n=%zu\n", n);
            free(buf_fwht);
            free(buf_ffht);
            return 0;
        }

        free(buf_fwht);
        free(buf_ffht);
        seed += 7U;
    }
    return 1;
}

static void benchmark_size(size_t n, size_t iters, unsigned int seed) {
    double *buf_fwht = (double *)malloc(n * sizeof(double));
    double *buf_ffht = (double *)malloc(n * sizeof(double));
    if (!buf_fwht || !buf_ffht) {
        fprintf(stderr, "[compare_ffht_fwht_fp64] alloc failed for n=%zu\n", n);
        free(buf_fwht);
        free(buf_ffht);
        return;
    }

    double t_fwht_cpu = 0.0;
    double t_ffht = 0.0;

    for (size_t it = 0; it < iters; ++it) {
        fill_random_f64(buf_fwht, n, seed + (unsigned int)it);
        memcpy(buf_ffht, buf_fwht, n * sizeof(double));

        double *work_fwht = (double *)malloc(n * sizeof(double));
        double *work_ffht = (double *)malloc(n * sizeof(double));
        if (!work_fwht || !work_ffht) {
            fprintf(stderr, "[compare_ffht_fwht_fp64] alloc failed (work) for n=%zu\n", n);
            free(work_fwht);
            free(work_ffht);
            break;
        }

        memcpy(work_fwht, buf_fwht, n * sizeof(double));
        double t0 = timestamp_seconds();
        fwht_status_t st = fwht_f64_backend(work_fwht, n, FWHT_BACKEND_CPU);
        double t1 = timestamp_seconds();
        if (st != FWHT_SUCCESS) {
            fprintf(stderr, "[compare_ffht_fwht_fp64] fwht_f64_backend failed in bench for n=%zu\n", n);
            free(work_fwht);
            free(work_ffht);
            break;
        }
        t_fwht_cpu += (t1 - t0);

        memcpy(work_ffht, buf_ffht, n * sizeof(double));
        int log_n = 0;
        size_t tmp = n;
        while (tmp > 1) {
            tmp >>= 1;
            log_n++;
        }
        double t2 = timestamp_seconds();
        if (fht_double(work_ffht, log_n) != 0) {
            fprintf(stderr, "[compare_ffht_fwht_fp64] fht_double failed in bench for n=%zu\n", n);
            free(work_fwht);
            free(work_ffht);
            break;
        }
        double t3 = timestamp_seconds();
        t_ffht += (t3 - t2);

        free(work_fwht);
        free(work_ffht);
    }

    if (t_fwht_cpu > 0.0 && t_ffht > 0.0) {
        double avg_fwht_cpu = t_fwht_cpu / (double)iters;
        double avg_ffht = t_ffht / (double)iters;
        double log2n = log2((double)n);
        double ops = (double)n * log2n;
        double gops_fwht_cpu = ops / avg_fwht_cpu / 1e9;
        double gops_ffht = ops / avg_ffht / 1e9;
        double speedup = avg_fwht_cpu / avg_ffht;
        printf("%10zu  %10.6f  %10.6f  %10.3f  %10.3f  %8.3fx\n",
               n,
               avg_fwht_cpu, avg_ffht,
               gops_fwht_cpu, gops_ffht,
               speedup);
    }

    free(buf_fwht);
    free(buf_ffht);
}

int main(void) {
    if (!verify_small_sizes()) {
        fprintf(stderr, "[compare_ffht_fwht_fp64] correctness check failed; aborting.\n");
        return EXIT_FAILURE;
    }

    const size_t sizes[] = {
        1u << 8,  1u << 9,  1u << 10,
        1u << 11, 1u << 12, 1u << 13,
        1u << 14, 1u << 15, 1u << 16,
        1u << 17, 1u << 18, 1u << 19,
        1u << 20, 1u << 21, 1u << 22,
        1u << 23, 1u << 24, 1u << 25
    };
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t iters = 30;
    const unsigned int seed = 2025U;

    printf("libfwht fp64 vs FFHT double (single-threaded CPU comparison)\n");
    printf("%10s  %10s  %10s  %10s  %10s  %9s\n",
           "Size", "libfwht_s", "FFHT_s",
           "libfwht_GO/s", "FFHT_GO/s",
           "Speedup");

    for (size_t i = 0; i < num_sizes; ++i) {
        benchmark_size(sizes[i], iters, seed);
    }

    return EXIT_SUCCESS;
}
