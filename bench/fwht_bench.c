/*
 * Fast Walsh-Hadamard Transform - Benchmark Tool
 *
 * Performance benchmarking utility for the FWHT library.
 *
 * Copyright (C) 2025 Hosein Hadipour
 *
 * Author: Hosein Hadipour <hsn.hadipour@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>

#include "../include/fwht.h"

#ifndef ARRAY_LEN
#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0]))
#endif

static double timestamp_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill_random(int32_t* data, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        data[i] = (int32_t)(rand() % 65536 - 32768);
    }
}

typedef struct {
    const size_t* sizes;
    size_t count;
    int repeats;
    int warmup;
    fwht_backend_t backend;
    unsigned int seed;
} bench_config_t;

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "Options:\n"
            "  --backend=auto|cpu|gpu   Select backend (default: auto)\n"
            "  --sizes=comma,list       Transform sizes (power of two)\n"
            "  --repeats=N              Repetitions per size (default: 25)\n"
            "  --warmup=N               Unmeasured warm-up iterations per size (default: 0, GPU defaults to 1)\n"
            "  --seed=N                 Random seed (default: 12345)\n"
            "  --help                   Show this help\n",
            prog);
}

static int parse_backend(const char* value, fwht_backend_t* backend) {
    if (strcmp(value, "auto") == 0) {
        *backend = FWHT_BACKEND_AUTO;
    } else if (strcmp(value, "cpu") == 0) {
        *backend = FWHT_BACKEND_CPU;
    } else if (strcmp(value, "gpu") == 0) {
        *backend = FWHT_BACKEND_GPU;
    } else if (strcmp(value, "openmp") == 0) {
        *backend = FWHT_BACKEND_OPENMP;
    } else {
        return -1;
    }
    return 0;
}

static int parse_sizes(const char* value, size_t** out_sizes, size_t* out_count) {
    char* copy = strdup(value);
    if (!copy) return -1;

    size_t capacity = 8;
    size_t count = 0;
    size_t* sizes = (size_t*)malloc(capacity * sizeof(size_t));
    if (!sizes) {
        free(copy);
        return -1;
    }

    char* token = strtok(copy, ",");
    while (token != NULL) {
        size_t n = (size_t)strtoull(token, NULL, 10);
        if (n == 0 || (n & (n - 1)) != 0) {
            fprintf(stderr, "Invalid size '%s' (must be power of two)\n", token);
            free(copy);
            free(sizes);
            return -1;
        }
        if (count == capacity) {
            capacity *= 2;
            size_t* tmp = (size_t*)realloc(sizes, capacity * sizeof(size_t));
            if (!tmp) {
                free(copy);
                free(sizes);
                return -1;
            }
            sizes = tmp;
        }
        sizes[count++] = n;
        token = strtok(NULL, ",");
    }

    free(copy);
    *out_sizes = sizes;
    *out_count = count;
    return 0;
}

static void print_mode_header(const char* label) {
    printf("Mode: %s\n", label);
    printf("%10s  %12s  %12s\n", "Size", "Mean (ms)", "StdDev (ms)");
    printf("%10s  %12s  %12s\n", "----------", "------------", "------------");
}

static int ensure_capacity(int32_t** data_ptr, size_t* capacity_ptr, size_t n) {
    if (*capacity_ptr >= n) {
        return 0;
    }

    size_t bytes = n * sizeof(int32_t);
    int32_t* tmp = (int32_t*)realloc(*data_ptr, bytes);
    if (!tmp) {
        fprintf(stderr, "Allocation failed for size %zu\n", n);
        return -1;
    }

    *data_ptr = tmp;
    *capacity_ptr = n;
    return 0;
}

typedef struct {
    double mean_ms;
    double stddev_ms;
    bool   valid;
} bench_result_t;

static bench_result_t measure_backend(const bench_config_t* cfg,
                                      fwht_backend_t backend,
                                      int32_t* data,
                                      size_t n) {
    bench_result_t result = {0.0, 0.0, false};
    double mean = 0.0;
    double m2 = 0.0;
    int total_runs = cfg->warmup + cfg->repeats;

    for (int run = 0; run < total_runs; ++run) {
        fill_random(data, n, cfg->seed + (unsigned int)run);

        double start = timestamp_seconds();
        fwht_status_t status = fwht_i32_backend(data, n, backend);
        double end = timestamp_seconds();

        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "FWHT failed for size %zu: %s\n", n, fwht_error_string(status));
            return result;
        }

        if (run < cfg->warmup) {
            continue;
        }

        int sample_index = run - cfg->warmup;
        double elapsed_ms = (end - start) * 1000.0;
        double delta = elapsed_ms - mean;
        mean += delta / (double)(sample_index + 1);
        double delta2 = elapsed_ms - mean;
        m2 += delta * delta2;
    }

    double variance = (cfg->repeats > 1) ? (m2 / (double)(cfg->repeats - 1)) : 0.0;
    result.mean_ms = mean;
    result.stddev_ms = (variance > 0.0) ? sqrt(variance) : 0.0;
    result.valid = true;
    return result;
}

static int benchmark_backend_mode(const bench_config_t* cfg,
                                  fwht_backend_t backend,
                                  const char* mode_label,
                                  int32_t** data_ptr,
                                  size_t* capacity_ptr) {
    print_mode_header(mode_label);

    for (size_t i = 0; i < cfg->count; ++i) {
        size_t n = cfg->sizes[i];
        if (ensure_capacity(data_ptr, capacity_ptr, n) != 0) {
            return -1;
        }

        bench_result_t res = measure_backend(cfg, backend, *data_ptr, n);
        if (!res.valid) {
            return -1;
        }

        printf("%10zu  %12.3f  %12.3f\n", n, res.mean_ms, res.stddev_ms);
    }

    printf("\n");
    return 0;
}

static int run_benchmark(const bench_config_t* cfg) {
    printf("FWHT Benchmark\n");
    printf("Requested backend : %s\n", fwht_backend_name(cfg->backend));
    printf("Repeats : %d\n", cfg->repeats);
    printf("Warmup  : %d\n", cfg->warmup);
    printf("Seed    : %u\n", cfg->seed);
    printf("GPU available    : %s\n", fwht_has_gpu() ? "yes" : "no");
    printf("OpenMP available : %s\n\n", fwht_has_openmp() ? "yes" : "no");

    int32_t* data = NULL;
    size_t data_capacity = 0;
    int rc = 0;

    if (cfg->backend == FWHT_BACKEND_CPU) {
        const char* mode_labels[] = {
            "cpu (single-threaded)",
            "openmp (multi-threaded)",
            "auto (runtime selection)"
        };
        const fwht_backend_t mode_backends[] = {
            FWHT_BACKEND_CPU,
            FWHT_BACKEND_OPENMP,
            FWHT_BACKEND_AUTO
        };
        size_t mode_count = sizeof(mode_backends) / sizeof(mode_backends[0]);

        for (size_t i = 0; i < mode_count; ++i) {
            if (mode_backends[i] == FWHT_BACKEND_OPENMP && !fwht_has_openmp()) {
                printf("Mode: %s\n", mode_labels[i]);
                printf("  Skipped (OpenMP backend not available; rebuild with `make openmp` to enable)\n\n");
                continue;
            }

            rc = benchmark_backend_mode(cfg, mode_backends[i], mode_labels[i], &data, &data_capacity);
            if (rc != 0) {
                goto cleanup;
            }
        }
    } else {
        const char* label = fwht_backend_name(cfg->backend);
        rc = benchmark_backend_mode(cfg, cfg->backend, label, &data, &data_capacity);
    }

cleanup:
    free(data);
    return rc;
}

int main(int argc, char** argv) {
    const size_t default_sizes[] = {256, 512, 1024, 4096, 16384};

    bench_config_t cfg;
    cfg.sizes = default_sizes;
    cfg.count = ARRAY_LEN(default_sizes);
    cfg.repeats = 25;
    cfg.warmup = 0;
    cfg.backend = FWHT_BACKEND_AUTO;
    cfg.seed = 12345U;

    size_t* dynamic_sizes = NULL;

    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--backend=", 10) == 0) {
            if (parse_backend(argv[i] + 10, &cfg.backend) != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--sizes=", 8) == 0) {
            if (parse_sizes(argv[i] + 8, &dynamic_sizes, &cfg.count) != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            cfg.sizes = dynamic_sizes;
        } else if (strncmp(argv[i], "--repeats=", 10) == 0) {
            cfg.repeats = atoi(argv[i] + 10);
            if (cfg.repeats <= 0) {
                fprintf(stderr, "Invalid repeats value\n");
                usage(argv[0]);
                free(dynamic_sizes);
                return EXIT_FAILURE;
            }
        } else if (strncmp(argv[i], "--seed=", 7) == 0) {
            cfg.seed = (unsigned int)strtoul(argv[i] + 7, NULL, 10);
        } else if (strncmp(argv[i], "--warmup=", 9) == 0) {
            cfg.warmup = atoi(argv[i] + 9);
            if (cfg.warmup < 0) {
                fprintf(stderr, "Invalid warmup value\n");
                usage(argv[0]);
                free(dynamic_sizes);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            free(dynamic_sizes);
            return EXIT_SUCCESS;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            free(dynamic_sizes);
            return EXIT_FAILURE;
        }
    }

    if (cfg.backend == FWHT_BACKEND_GPU && cfg.warmup == 0) {
        cfg.warmup = 1;
    }

    int rc = run_benchmark(&cfg);
    free(dynamic_sizes);
    return (rc == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
