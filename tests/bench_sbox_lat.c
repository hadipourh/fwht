#include "fwht.h"

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define MIN_SBOX_BITS 4
#define MAX_SBOX_BITS 17
#define RUNS_PER_SIZE 1
#define SBOX_DIR "build/sboxes"

static int ensure_directory(const char *path) {
    if (path == NULL) {
        return -1;
    }
#if defined(_WIN32)
    int rc = _mkdir(path);
#else
    int rc = mkdir(path, 0755);
#endif
    if (rc == 0) {
        return 0;
    }
    if (errno == EEXIST) {
        return 0;
    }
    fprintf(stderr, "Error: cannot create directory '%s' (%s)\n", path, strerror(errno));
    return -1;
}

static void generate_random_sbox(uint32_t *table, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        table[i] = (uint32_t)i;
    }
    if (size <= 1) {
        return;
    }
    for (size_t i = size - 1; i > 0; --i) {
        size_t j = (size_t)(rand() % (int)(i + 1));
        uint32_t tmp = table[i];
        table[i] = table[j];
        table[j] = tmp;
    }
}

static int write_sbox_file(const char *path, const uint32_t *table, size_t size) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error: cannot open '%s' for writing (%s)\n", path, strerror(errno));
        return -1;
    }
    for (size_t i = 0; i < size; ++i) {
        if (fprintf(fp, "%u\n", table[i]) < 0) {
            fprintf(stderr, "Error: failed while writing '%s' (%s)\n", path, strerror(errno));
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static int run_backend(size_t bits,
                       size_t size,
                       const uint32_t *table,
                       int run_id,
                       fwht_backend_t backend) {
    fwht_sbox_lat_request_t request;
    memset(&request, 0, sizeof(request));
    request.backend = backend;
    request.profile_timings = true;

    fwht_sbox_lat_metrics_t metrics;
    const char *name = fwht_backend_name(backend);
    if (name == NULL) {
        name = "unknown";
    }

    fwht_status_t status = fwht_sbox_analyze_lat(table, size, &request, &metrics);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr,
                "Error: backend %s failed for %zu-bit S-box (%s)\n",
                name,
                bits,
                fwht_error_string(status));
        return -1;
    }

        double total_ms = metrics.column_ms + metrics.fwht_ms;

    printf("%4zu %3d %-8s %14.3f %14.3f %14.3f %10d %13.6f\n",
           bits,
           run_id,
           name,
            metrics.column_ms,
            metrics.fwht_ms,
           total_ms,
           metrics.lat_max,
           metrics.lat_max_bias);
    return 0;
}

int main(void) {
    srand((unsigned int)time(NULL));

    if (ensure_directory("build") != 0) {
        return EXIT_FAILURE;
    }
    if (ensure_directory(SBOX_DIR) != 0) {
        return EXIT_FAILURE;
    }

    bool gpu_available = fwht_has_gpu();

        printf("FWHT S-box LAT benchmark (C harness)\n");
        printf("Writing random S-boxes to %s\n", SBOX_DIR);
        printf("GPU backend: %s\n", gpu_available ? "available" : "unavailable");
        printf("Runs per size: %d\n\n", RUNS_PER_SIZE);
        printf("Columns (in compute order): lat_build_ms (construct LAT columns) -> lat_fwht_ms (LAT FWHT) -> total_ms=sum, lat_max (peak bias), lat_bias=lat_max/size.\n\n");
        printf("%4s %3s %-8s %14s %14s %14s %10s %13s\n",
           "bits",
           "run",
           "backend",
           "lat_build_ms",
           "lat_fwht_ms",
            "total_ms",
           "lat_max",
           "lat_bias");
        printf("-------------------------------------------------------------------------------------------------------------------\n");

    for (size_t bits = MIN_SBOX_BITS; bits <= MAX_SBOX_BITS; ++bits) {
        size_t size = (size_t)1 << bits;
        uint32_t *table = (uint32_t *)malloc(size * sizeof(uint32_t));
        if (table == NULL) {
            fprintf(stderr, "Error: out of memory for %zu-bit S-box\n", bits);
            return EXIT_FAILURE;
        }

        for (int run = 0; run < RUNS_PER_SIZE; ++run) {
            int run_id = run + 1;
            generate_random_sbox(table, size);

            char path[256];
            int written = snprintf(path,
                                   sizeof(path),
                                   "%s/rand_sbox_%zubits_run%02d.txt",
                                   SBOX_DIR,
                                   bits,
                                   run_id);
            if (written < 0 || (size_t)written >= sizeof(path)) {
                fprintf(stderr, "Error: S-box path truncated for %zu-bit table\n", bits);
                free(table);
                return EXIT_FAILURE;
            }
            if (write_sbox_file(path, table, size) != 0) {
                free(table);
                return EXIT_FAILURE;
            }

            if (run_backend(bits, size, table, run_id, FWHT_BACKEND_CPU) != 0) {
                free(table);
                return EXIT_FAILURE;
            }
            if (gpu_available) {
                if (run_backend(bits, size, table, run_id, FWHT_BACKEND_GPU) != 0) {
                    free(table);
                    return EXIT_FAILURE;
                }
            }
        }

        free(table);

        printf("-------------------------------------------------------------------------------------------------------------------\n");
    }

    return EXIT_SUCCESS;
}
