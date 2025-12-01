#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#include "fwht.h"

#define DEFAULT_MIN_POWER 6
#define DEFAULT_MAX_POWER 24
#define DEFAULT_WARMUPS 2
#define DEFAULT_REPEATS 8
#define DEFAULT_SPEEDUP 1.20
#define DEFAULT_GPU_THRESHOLD (1u << 20)
#define OUTPUT_PATH "meta/backend_threshold.json"

typedef struct {
    size_t n;
    double cpu_ms;
    double omp_ms;
} measurement_t;

static double now_seconds(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t t = mach_absolute_time();
    double nanos = (double)t * (double)info.numer / (double)info.denom;
    return nanos * 1e-9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void fill_random(int32_t* data, size_t n) {
    uint64_t state = 0x6d98302c94b95b3dULL;
    for (size_t i = 0; i < n; ++i) {
        state = state * 6364136223846793005ULL + 1ULL;
        data[i] = (int32_t)(state >> 32);
    }
}

static double benchmark_backend(fwht_backend_t backend,
                                const int32_t* input,
                                int32_t* scratch,
                                size_t n,
                                int warmups,
                                int repeats) {
    fwht_status_t status;
    for (int i = 0; i < warmups; ++i) {
        memcpy(scratch, input, n * sizeof(int32_t));
        status = fwht_i32_backend(scratch, n, backend);
        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "FWHT warmup failed: %s\n", fwht_error_string(status));
            exit(EXIT_FAILURE);
        }
    }

    double best = HUGE_VAL;
    for (int i = 0; i < repeats; ++i) {
        memcpy(scratch, input, n * sizeof(int32_t));
        double start = now_seconds();
        status = fwht_i32_backend(scratch, n, backend);
        double elapsed = now_seconds() - start;
        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "FWHT benchmark failed: %s\n", fwht_error_string(status));
            exit(EXIT_FAILURE);
        }
        if (elapsed < best) {
            best = elapsed;
        }
    }
    return best * 1e3;  // ms
}

static size_t find_threshold(const measurement_t* rows, size_t count, double min_speedup) {
    for (size_t i = 0; i < count; ++i) {
        if (rows[i].omp_ms <= 0.0) {
            continue;
        }
        double speedup = rows[i].cpu_ms / rows[i].omp_ms;
        if (speedup >= min_speedup) {
            return rows[i].n;
        }
    }
    return 0;
}

static void ensure_meta_directory(void) {
    struct stat st;
    if (stat("meta", &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            fprintf(stderr, "meta exists but is not a directory\n");
            exit(EXIT_FAILURE);
        }
        return;
    }
    if (mkdir("meta", 0755) != 0) {
        perror("mkdir(meta)");
        exit(EXIT_FAILURE);
    }
}

static void write_json(const char* path,
                       size_t openmp_threshold,
                       size_t gpu_threshold,
                       double min_speedup,
                       int min_power,
                       int max_power) {
    ensure_meta_directory();
    FILE* fp = fopen(path, "w");
    if (!fp) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));

    fprintf(fp,
            "{\n"
            "  \"generated_at\": \"%s\",\n"
            "  \"min_power\": %d,\n"
            "  \"max_power\": %d,\n"
            "  \"min_speedup\": %.2f,\n"
            "  \"openmp_threshold\": %zu,\n"
            "  \"gpu_threshold\": %zu\n"
            "}\n",
            timestamp,
            min_power,
            max_power,
            min_speedup,
            openmp_threshold,
            gpu_threshold);

    fclose(fp);
    printf("\nWrote tuning data to %s\n", path);
}

static void usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s [--min-power k] [--max-power k] [--repeats n] [--warmups n]\\n"
            "          [--speedup f] [--output path]\n",
            prog);
}

int main(int argc, char** argv) {
    int min_power = DEFAULT_MIN_POWER;
    int max_power = DEFAULT_MAX_POWER;
    int warmups = DEFAULT_WARMUPS;
    int repeats = DEFAULT_REPEATS;
    double min_speedup = DEFAULT_SPEEDUP;
    const char* output_path = OUTPUT_PATH;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--min-power") == 0 && i + 1 < argc) {
            min_power = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-power") == 0 && i + 1 < argc) {
            max_power = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            warmups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--speedup") == 0 && i + 1 < argc) {
            min_speedup = atof(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (!fwht_has_openmp()) {
        fprintf(stderr, "OpenMP backend is not available in this build.\n");
        return 1;
    }

    if (min_power > max_power) {
        fprintf(stderr, "min-power must be <= max-power\n");
        return 1;
    }

    size_t total = (size_t)(max_power - min_power + 1);
    measurement_t* rows = calloc(total, sizeof(measurement_t));
    if (!rows) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    printf("FWHT backend tuning (CPU vs OpenMP)\n");
    printf("Range: 2^%d .. 2^%d, repeats=%d, warmups=%d, min speedup=%.2f\n\n",
           min_power,
           max_power,
           repeats,
           warmups,
           min_speedup);
    printf("%6s %12s %12s %10s\n", "2^k", "CPU (ms)", "OpenMP (ms)", "speedup");
    printf("---------------------------------------------------------\n");

    for (int power = min_power; power <= max_power; ++power) {
        size_t n = (size_t)1 << power;
        rows[power - min_power].n = n;

        int32_t* input = malloc(n * sizeof(int32_t));
        int32_t* scratch = malloc(n * sizeof(int32_t));
        if (!input || !scratch) {
            fprintf(stderr, "Allocation failed for size %zu\n", n);
            free(input);
            free(scratch);
            free(rows);
            return 1;
        }
        fill_random(input, n);

        double cpu_ms = benchmark_backend(FWHT_BACKEND_CPU, input, scratch, n, warmups, repeats);
        rows[power - min_power].cpu_ms = cpu_ms;

        double omp_ms = benchmark_backend(FWHT_BACKEND_OPENMP, input, scratch, n, warmups, repeats);
        rows[power - min_power].omp_ms = omp_ms;

        double speedup = cpu_ms / omp_ms;
        printf("2^%2d %12.6f %12.6f %10.2f\n", power, cpu_ms, omp_ms, speedup);

        free(input);
        free(scratch);
    }

    size_t openmp_threshold = find_threshold(rows, total, min_speedup);
    if (openmp_threshold == 0) {
        openmp_threshold = (size_t)1 << DEFAULT_MIN_POWER;
        fprintf(stderr,
                "Warning: OpenMP never met the requested speedup; using 2^%d as threshold.\n",
                DEFAULT_MIN_POWER);
    }

    int threshold_power = 0;
    size_t tmp = openmp_threshold;
    while ((tmp >>= 1) != 0) {
        threshold_power++;
    }

    printf("\nSuggested OpenMP threshold: n = %zu (2^%d)\n",
           openmp_threshold,
           threshold_power);

    write_json(output_path, openmp_threshold, DEFAULT_GPU_THRESHOLD, min_speedup, min_power, max_power);

    free(rows);
    return 0;
}
