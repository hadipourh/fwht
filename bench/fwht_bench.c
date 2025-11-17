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
/* No direct CUDA headers here; use fwht.h helper APIs when CUDA is enabled */

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

#ifdef USE_CUDA
static int gpu_profile_requested(void) {
    static int cached = -1;
    if (cached != -1) {
        return cached;
    }

    const char* env = getenv("FWHT_GPU_PROFILE");
    if (env && env[0] != '\0' && env[0] != '0') {
        if (fwht_gpu_set_profiling(true) == FWHT_SUCCESS) {
            cached = 1;
        } else {
            fprintf(stderr, "[fwht_bench] Failed to enable GPU profiling; continuing without breakdowns.\n");
            cached = 0;
        }
    } else {
        cached = 0;
    }
    return cached;
}
#endif

typedef struct {
    const size_t* sizes;
    size_t count;
    int repeats;
    int warmup;
    fwht_backend_t backend;
    unsigned int seed;
    size_t batch_size;     /* Number of transforms to compute per measurement */
    int use_context;       /* When GPU: reuse a persistent context across repeats */
    int pinned;            /* When GPU: use pinned (page-locked) host memory for H2D/D2H */
    int device_resident;   /* When GPU: keep data on device, measure kernel-only */
    int multi_shuffle;     /* When GPU: -1 = leave library default, 0 = force disable, 1 = force enable */
    int profile;           /* When GPU: enable profiling breakdown (H2D/Kernel/D2H) */
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
            "  --batch=N                Batch size (default: 1). For GPU, uses batched kernel; for CPU, uses SIMD batch.\n"
            "  --use-context            When GPU, reuse a persistent context across repeats (reduces alloc overhead).\n"
            "  --pinned                 When GPU, use page-locked host buffers to speed up H2D/D2H.\n"
            "  --device-resident        When GPU, keep data on device and run kernel-only (no H2D/D2H).\n"
            "  --multi-shuffle=on|off   When GPU, force enable/disable medium-N multi-shuffle optimization.\n"
            "  --profile                When GPU, enable profiling (report H2D/Kernel/D2H breakdown).\n"
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
    printf("%10s  %12s  %12s  %12s\n", "Size", "Mean (ms)", "StdDev (ms)", "GOps/s");
    printf("%10s  %12s  %12s  %12s\n", "----------", "------------", "------------", "------------");
}

static int ensure_capacity(int32_t** data_ptr, size_t* capacity_ptr, size_t n,
                           bool want_pinned, bool* is_pinned_ptr) {
    if (*capacity_ptr >= n && (*is_pinned_ptr == want_pinned)) {
        return 0;
    }

    /* Free existing buffer if type changes or capacity insufficient */
    if (*data_ptr) {
        if (*is_pinned_ptr) {
#ifdef USE_CUDA
            fwht_gpu_host_free(*data_ptr);
#else
            /* Should not happen: pinned requested without CUDA */
            free(*data_ptr);
#endif
        } else {
            free(*data_ptr);
        }
        *data_ptr = NULL;
        *capacity_ptr = 0;
        *is_pinned_ptr = false;
    }

    size_t bytes = n * sizeof(int32_t);
#ifdef USE_CUDA
    if (want_pinned) {
        void* tmp = NULL;
        if (fwht_gpu_host_alloc(&tmp, bytes) == FWHT_SUCCESS && tmp != NULL) {
            *data_ptr = (int32_t*)tmp;
            *capacity_ptr = n;
            *is_pinned_ptr = true;
            return 0;
        } else {
            fprintf(stderr, "[fwht_bench] Pinned allocation unavailable; falling back to pageable.\n");
        }
    }
#endif
    int32_t* tmp = (int32_t*)malloc(bytes);
    if (!tmp) {
        fprintf(stderr, "Allocation failed for size %zu\n", n);
        return -1;
    }
    *data_ptr = tmp;
    *capacity_ptr = n;
    *is_pinned_ptr = false;
    return 0;
}

typedef struct {
    double mean_ms;
    double stddev_ms;
    bool   valid;
    double gpu_h2d_ms;
    double gpu_kernel_ms;
    double gpu_d2h_ms;
    size_t gpu_bytes;
    size_t gpu_batch;
    size_t gpu_n;
    int    gpu_samples;
    bool   gpu_metrics_valid;
} bench_result_t;

static bench_result_t measure_backend(const bench_config_t* cfg,
                                      fwht_backend_t backend,
                                      int32_t* data,
                                      size_t n) {
    bench_result_t result = {0.0, 0.0, false, 0.0, 0.0, 0.0, 0u, 0u, 0u, 0, false};
    double mean = 0.0;
    double m2 = 0.0;
    int total_runs = cfg->warmup + cfg->repeats;
#ifdef USE_CUDA
    double gpu_h2d_sum = 0.0;
    double gpu_kernel_sum = 0.0;
    double gpu_d2h_sum = 0.0;
    size_t gpu_last_bytes = 0;
    size_t gpu_last_batch = 0;
    size_t gpu_last_n = 0;
    int gpu_samples = 0;
    int collect_gpu_metrics = 0;
    fwht_gpu_context_t* gpu_ctx = NULL;
    void* d_buffer = NULL;
    if (backend == FWHT_BACKEND_GPU) {
        /* Enable profiling if requested via CLI, else consult environment */
        if (cfg->profile) {
            if (fwht_gpu_set_profiling(true) == FWHT_SUCCESS) {
                collect_gpu_metrics = 1;
            }
        } else {
            collect_gpu_metrics = gpu_profile_requested();
        }
        if (cfg->device_resident) {
            /* Allocate device buffer for kernel-only timing */
            size_t bytes = n * (cfg->batch_size > 0 ? cfg->batch_size : 1) * sizeof(int32_t);
            if (fwht_gpu_device_alloc(&d_buffer, bytes) != FWHT_SUCCESS) {
                fprintf(stderr, "[fwht_bench] Failed to allocate device buffer (%zu bytes)\n", bytes);
                d_buffer = NULL;
                return result;
            }
            /* Preload random data once */
            if (cfg->batch_size <= 1) {
                fill_random(data, n, cfg->seed);
            } else {
                for (size_t b = 0; b < cfg->batch_size; ++b) {
                    fill_random(data + b * n, n, cfg->seed + (unsigned int)b);
                }
            }
            if (fwht_gpu_memcpy_h2d(d_buffer, data, bytes) != FWHT_SUCCESS) {
                fprintf(stderr, "[fwht_bench] Failed to copy initial data to device\n");
                fwht_gpu_device_free(d_buffer);
                d_buffer = NULL;
                return result;
            }
        } else if (cfg->use_context && cfg->batch_size > 0) {
            gpu_ctx = fwht_gpu_context_create(n, cfg->batch_size);
            if (!gpu_ctx) {
                fprintf(stderr, "[fwht_bench] Failed to create GPU context (n=%zu, batch=%zu)\n", n, cfg->batch_size);
            }
        }
    }
#endif

    /* Prepare pointer array for CPU batch API when batch_size > 1 */
    int use_batch = (cfg->batch_size > 1);
    int32_t** data_ptrs = NULL;
    if (use_batch) {
        data_ptrs = (int32_t**)malloc(sizeof(int32_t*) * cfg->batch_size);
        if (!data_ptrs) {
            fprintf(stderr, "Allocation failed for data_ptrs (batch=%zu)\n", cfg->batch_size);
            return result;
        }
        for (size_t b = 0; b < cfg->batch_size; ++b) {
            data_ptrs[b] = data + b * n;
        }
    }

    for (int run = 0; run < total_runs; ++run) {
        /* Fill input(s) with random data */
        if (!(backend == FWHT_BACKEND_GPU && cfg->device_resident)) {
            if (!use_batch) {
                fill_random(data, n, cfg->seed + (unsigned int)run);
            } else {
                for (size_t b = 0; b < cfg->batch_size; ++b) {
                    fill_random(data_ptrs[b], n, cfg->seed + (unsigned int)run + (unsigned int)b);
                }
            }
        }

        double start = timestamp_seconds();
        fwht_status_t status = FWHT_SUCCESS;
        if (!use_batch) {
            status = fwht_i32_backend(data, n, backend);
        } else {
            if (backend == FWHT_BACKEND_GPU) {
#ifdef USE_CUDA
                if (cfg->device_resident && d_buffer) {
                    status = fwht_batch_i32_cuda_device((int32_t*)d_buffer, n, cfg->batch_size);
                } else if (cfg->use_context && gpu_ctx) {
                    status = fwht_gpu_context_compute_i32(gpu_ctx, data, n, cfg->batch_size);
                } else {
                    status = fwht_batch_i32_cuda(data, n, cfg->batch_size);
                }
#else
                status = FWHT_ERROR_BACKEND_UNAVAILABLE;
#endif
            } else {
                /* CPU/OpenMP path: use SIMD batch API */
                status = fwht_i32_batch(data_ptrs, n, cfg->batch_size);
            }
        }
        double end = timestamp_seconds();

        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "FWHT failed for size %zu: %s\n", n, fwht_error_string(status));
#ifdef USE_CUDA
            if (gpu_ctx) fwht_gpu_context_destroy(gpu_ctx);
#endif
            if (data_ptrs) free(data_ptrs);
            return result;
        }

        if (run >= cfg->warmup) {
            int sample_index = run - cfg->warmup;
            double elapsed_ms = (end - start) * 1000.0;
            double delta = elapsed_ms - mean;
            mean += delta / (double)(sample_index + 1);
            double delta2 = elapsed_ms - mean;
            m2 += delta * delta2;
#ifdef USE_CUDA
            if (backend == FWHT_BACKEND_GPU && collect_gpu_metrics) {
                fwht_gpu_metrics_t metrics = fwht_gpu_get_last_metrics();
                if (metrics.valid) {
                    gpu_h2d_sum += metrics.h2d_ms;
                    gpu_kernel_sum += metrics.kernel_ms;
                    gpu_d2h_sum += metrics.d2h_ms;
                    gpu_last_bytes = metrics.bytes_transferred;
                    gpu_last_batch = metrics.batch_size;
                    gpu_last_n = metrics.n;
                    gpu_samples += 1;
                }
            }
#endif
        }
    }

    double variance = (cfg->repeats > 1) ? (m2 / (double)(cfg->repeats - 1)) : 0.0;
    result.mean_ms = mean;
    result.stddev_ms = (variance > 0.0) ? sqrt(variance) : 0.0;
    result.valid = true;
#ifdef USE_CUDA
    if (gpu_ctx) {
        fwht_gpu_context_destroy(gpu_ctx);
    }
    if (d_buffer) {
        fwht_gpu_device_free(d_buffer);
    }
    if (gpu_samples > 0) {
        result.gpu_h2d_ms = gpu_h2d_sum / (double)gpu_samples;
        result.gpu_kernel_ms = gpu_kernel_sum / (double)gpu_samples;
        result.gpu_d2h_ms = gpu_d2h_sum / (double)gpu_samples;
        result.gpu_bytes = gpu_last_bytes;
        result.gpu_batch = gpu_last_batch;
        result.gpu_n = gpu_last_n;
        result.gpu_samples = gpu_samples;
        result.gpu_metrics_valid = true;
    }
#endif
    if (data_ptrs) free(data_ptrs);
    return result;
}

static int benchmark_backend_mode(const bench_config_t* cfg,
                                  fwht_backend_t backend,
                                  const char* mode_label,
                                  int32_t** data_ptr,
                                  size_t* capacity_ptr,
                                  bool* data_is_pinned) {
    print_mode_header(mode_label);

    for (size_t i = 0; i < cfg->count; ++i) {
        size_t n = cfg->sizes[i];
        size_t need = n * (cfg->batch_size > 0 ? cfg->batch_size : 1);
        bool want_pinned = (backend == FWHT_BACKEND_GPU) && (cfg->pinned != 0);
        if (ensure_capacity(data_ptr, capacity_ptr, need, want_pinned, data_is_pinned) != 0) {
            return -1;
        }

        bench_result_t res = measure_backend(cfg, backend, *data_ptr, n);
        if (!res.valid) {
            return -1;
        }
        /* Throughput estimate: treat each butterfly as 2 ops; there are (n * log2(n))/2 butterflies. */
        double log2n = log2((double)n);
        double ops = (double)cfg->batch_size * (double)n * log2n; /* proxy ops count */
        double gops = (res.mean_ms > 0.0) ? (ops / (res.mean_ms / 1000.0) / 1e9) : 0.0;
        printf("%10zu  %12.3f  %12.3f  %12.2f\n", n, res.mean_ms, res.stddev_ms, gops);
#ifdef USE_CUDA
    if (backend == FWHT_BACKEND_GPU && res.gpu_metrics_valid) {
        printf("            [GPU breakdown] H2D %.3f ms | Kernel %.3f ms | D2H %.3f ms"
           " (batch=%zu, bytes=%zu)\n",
           res.gpu_h2d_ms,
           res.gpu_kernel_ms,
           res.gpu_d2h_ms,
           res.gpu_batch,
           res.gpu_bytes);
    }
#endif
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
    printf("Batch   : %zu\n", cfg->batch_size);
#ifdef USE_CUDA
    if (cfg->backend == FWHT_BACKEND_GPU) {
        printf("GPU ctx : %s\n", cfg->use_context ? "persistent" : "none");
        printf("Pinned  : %s\n", cfg->pinned ? "yes" : "no");
        printf("Profile : %s\n", cfg->profile ? "on" : "off");
        /* Print device info & apply multi-shuffle override if requested */
        printf("Device  : %s\n", fwht_gpu_get_device_name());
        unsigned int cc = fwht_gpu_get_compute_capability();
        printf("Compute capability: %u (sm_%u)\n", cc, cc);
        printf("SM count: %u\n", fwht_gpu_get_sm_count());
        printf("SMEM banks: %u\n", fwht_gpu_get_smem_banks());
        if (cfg->multi_shuffle >= 0) {
            fwht_gpu_set_multi_shuffle(cfg->multi_shuffle != 0);
        }
        printf("Multi-shuffle: %s (override=%s)\n",
               fwht_gpu_multi_shuffle_enabled() ? "enabled" : "disabled",
               (cfg->multi_shuffle < 0) ? "none" : (cfg->multi_shuffle ? "on" : "off"));
    }
#endif
    printf("GPU available    : %s\n", fwht_has_gpu() ? "yes" : "no");
    printf("OpenMP available : %s\n\n", fwht_has_openmp() ? "yes" : "no");

    int32_t* data = NULL;
    size_t data_capacity = 0;
    bool data_is_pinned = false;
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

            rc = benchmark_backend_mode(cfg, mode_backends[i], mode_labels[i], &data, &data_capacity, &data_is_pinned);
            if (rc != 0) {
                goto cleanup;
            }
        }
    } else {
        const char* label = fwht_backend_name(cfg->backend);
        rc = benchmark_backend_mode(cfg, cfg->backend, label, &data, &data_capacity, &data_is_pinned);
    }

cleanup:
    if (data) {
        if (data_is_pinned) {
#ifdef USE_CUDA
            fwht_gpu_host_free(data);
#else
            free(data);
#endif
        } else {
            free(data);
        }
    }
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
    cfg.batch_size = 1;
    cfg.use_context = 0;
    cfg.pinned = 0;
    cfg.device_resident = 0;
    cfg.multi_shuffle = -1;
    cfg.profile = 0;

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
        } else if (strncmp(argv[i], "--batch=", 8) == 0) {
            long b = strtol(argv[i] + 8, NULL, 10);
            if (b <= 0) {
                fprintf(stderr, "Invalid batch value\n");
                usage(argv[0]);
                free(dynamic_sizes);
                return EXIT_FAILURE;
            }
            cfg.batch_size = (size_t)b;
        } else if (strcmp(argv[i], "--use-context") == 0) {
            cfg.use_context = 1;
        } else if (strcmp(argv[i], "--pinned") == 0) {
            cfg.pinned = 1;
        } else if (strcmp(argv[i], "--device-resident") == 0) {
            cfg.device_resident = 1;
        } else if (strncmp(argv[i], "--multi-shuffle=", 16) == 0) {
            const char* val = argv[i] + 16;
            if (strcmp(val, "on") == 0 || strcmp(val, "enable") == 0) {
                cfg.multi_shuffle = 1;
            } else if (strcmp(val, "off") == 0 || strcmp(val, "disable") == 0) {
                cfg.multi_shuffle = 0;
            } else {
                fprintf(stderr, "Invalid value for --multi-shuffle: %s (expected on|off)\n", val);
                usage(argv[0]);
                free(dynamic_sizes);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--profile") == 0) {
            cfg.profile = 1;
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
