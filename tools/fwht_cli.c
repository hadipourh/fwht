/*
 * Fast Walsh-Hadamard Transform - Command-Line Interface
 *
 * CLI tool for computing Walsh-Hadamard spectra from the terminal.
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

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fwht.h"

#define INITIAL_CAPACITY 64
#define MAX_PRECISION 12

typedef enum {
    INPUT_BOOL,
    INPUT_SIGNED,
    INPUT_FLOAT
} input_format_t;

typedef enum {
    DTYPE_I32,
    DTYPE_F64
} dtype_t;

typedef enum {
    MODE_WHT = 0,
    MODE_SBOX
} cli_mode_t;

typedef struct {
    double *data;
    size_t length;
    size_t capacity;
} value_buffer_t;

static const char *dtype_name(dtype_t dtype) {
    return (dtype == DTYPE_I32) ? "int32" : "float64";
}

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  --input <file>          Read values from text file (whitespace or comma separated).\n"
            "  --values <list>         Comma/space separated list of inline values.\n"
            "  --backend <name>        Backend: auto (default), cpu, cpu-safe, openmp, gpu.\n"
            "  --dtype <i32|f64>       Choose int32 (default) or float64 transforms.\n"
            "  --batch-size <n>        Interpret input as n equal-length transforms.\n"
            "  --input-format <mode>   bool (default), signed, or float (float requires --dtype f64).\n"
            "                        Boolean mode keeps truth tables bit-packed on the GPU for n ≤ 65536.\n"
            "  --safe                  Enable overflow-checked CPU backend (int32 only).\n"
            "  --normalize             Print coefficients divided by sqrt(n).\n"
            "  --precision <digits>    Decimal precision for floating output (0-%d, default 6).\n"
            "  --gpu-profile           Enable CUDA profiling metrics (requires USE_CUDA).\n"
            "  --gpu-block-size <pow2> Override CUDA block size (advanced).\n"
            "  --no-index              Omit the index column from the output.\n"
            "  --quiet                 Suppress header metadata.\n"
            "  --sbox                  Analyze LUT as an S-box (ignores FWHT output options).\n"
            "  --sbox-lat <file>       Write full LAT to file (implies LAT computation).\n"
            "  --sbox-components <file> Write component Walsh spectra to file.\n"
            "  --sbox-lat-stats        Compute LAT statistics even without dumping the matrix.\n"
            "  --sbox-lat-only         Skip component analysis (requires --sbox-lat or --sbox-lat-stats).\n"
            "  --sbox-profile          Print per-phase timing breakdown for S-box analysis.\n"
            "  --help                  Show this message.\n"
            "\n"
            "Examples:\n"
            "  %s --values 0,1,2,7,6,4,5,3 --backend cpu\n"
            "  %s --input walsh.txt --backend gpu --normalize\n"
            "  %s --values spectra.txt --dtype f64 --input-format float --batch-size 4\n",
            prog, MAX_PRECISION, prog, prog, prog);
}

static int ensure_capacity(value_buffer_t *buffer, size_t needed) {
    if (needed <= buffer->capacity) {
        return 0;
    }
    size_t new_cap = buffer->capacity ? buffer->capacity : INITIAL_CAPACITY;
    while (new_cap < needed) {
        new_cap *= 2;
    }
    double *tmp = realloc(buffer->data, new_cap * sizeof(double));
    if (!tmp) {
        fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
        return -1;
    }
    buffer->data = tmp;
    buffer->capacity = new_cap;
    return 0;
}

static int append_value(value_buffer_t *buffer, double value) {
    if (ensure_capacity(buffer, buffer->length + 1) != 0) {
        return -1;
    }
    buffer->data[buffer->length++] = value;
    return 0;
}

static size_t bit_length_u32(uint32_t value) {
    size_t bits = 0;
    while (value != 0) {
        bits++;
        value >>= 1;
    }
    return bits;
}

static int write_component_file(const char* path, const int32_t* spectra,
                                size_t n_bits, size_t size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' for writing (%s).\n", path, strerror(errno));
        return -1;
    }
    fprintf(fp, "# Component spectra (n=%zu, size=%zu)\n", n_bits, size);
    for (size_t bit = 0; bit < n_bits; ++bit) {
        for (size_t idx = 0; idx < size; ++idx) {
            size_t offset = bit * size + idx;
            fprintf(fp, "%zu %zu %d\n", bit, idx, spectra[offset]);
        }
    }
    fclose(fp);
    return 0;
}

static int write_lat_file(const char* path, const int32_t* lat,
                          size_t rows, size_t cols,
                          size_t m_bits, size_t n_bits) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' for writing (%s).\n", path, strerror(errno));
        return -1;
    }
    fprintf(fp, "# LAT matrix (rows=2^%zu, cols=2^%zu)\n", m_bits, n_bits);
    for (size_t a = 0; a < rows; ++a) {
        fprintf(fp, "%zu", a);
        for (size_t b = 0; b < cols; ++b) {
            fprintf(fp, " %d", lat[a * cols + b]);
        }
        fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int run_sbox_mode(const value_buffer_t* buffer,
                         fwht_backend_t backend,
                         int quiet,
                         const char* lat_path,
                         const char* component_path,
                         int lat_stats,
                         int profile_timings,
                         int lat_only) {
    size_t size = buffer->length;
    if (size == 0) {
        fprintf(stderr, "Error: S-box mode requires at least one value.\n");
        return 1;
    }
    if (!fwht_is_power_of_2(size)) {
        fprintf(stderr, "Error: S-box table length (%zu) must be a power of two.\n", size);
        return 1;
    }

    uint32_t* table = (uint32_t*)malloc(size * sizeof(uint32_t));
    if (!table) {
        fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
        return 1;
    }

    int exit_code = 1;
    int32_t* component_buf = NULL;
    int32_t* lat_buf = NULL;

    uint32_t max_entry = 0;
    for (size_t i = 0; i < size; ++i) {
        double value = buffer->data[i];
        double int_part;
        if (modf(value, &int_part) != 0.0) {
            fprintf(stderr, "Error: S-box values must be integers (got %.6f).\n", value);
            goto sbox_cleanup;
        }
        if (int_part < 0.0 || int_part > (double)UINT32_MAX) {
            fprintf(stderr, "Error: S-box value %.0f out of uint32 range.\n", int_part);
            goto sbox_cleanup;
        }
        table[i] = (uint32_t)int_part;
        if (table[i] > max_entry) {
            max_entry = table[i];
        }
    }

    if (max_entry >= size) {
        fprintf(stderr, "Error: S-box outputs must be < 2^m (max=%u, size=%zu).\n",
                max_entry, size);
        goto sbox_cleanup;
    }

    size_t n_bits = bit_length_u32(max_entry);
    if (n_bits == 0) {
        n_bits = 1;
    }

    const size_t bit_capacity = sizeof(size_t) * CHAR_BIT;
    int need_lat = (lat_path != NULL) || lat_stats;
    if (need_lat && n_bits >= bit_capacity) {
        fprintf(stderr, "Error: LAT output requires n < %zu bits.\n", bit_capacity);
        goto sbox_cleanup;
    }

    size_t lat_cols = (n_bits >= bit_capacity) ? 0 : ((size_t)1 << n_bits);
    if (lat_only) {
        need_lat = 1;
    }

    if (component_path) {
        if (lat_only) {
            fprintf(stderr, "Error: --sbox-lat-only cannot write component spectra.\n");
            goto sbox_cleanup;
        }
        if (n_bits != 0 && size > SIZE_MAX / n_bits) {
            fprintf(stderr, "Error: component buffer would overflow.\n");
            goto sbox_cleanup;
        }
        size_t total = n_bits * size;
        if (total > SIZE_MAX / sizeof(int32_t)) {
            fprintf(stderr, "Error: component buffer too large.\n");
            goto sbox_cleanup;
        }
        component_buf = (int32_t*)malloc(total * sizeof(int32_t));
        if (!component_buf) {
            fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
            goto sbox_cleanup;
        }
    }

    if (lat_path) {
        if (lat_cols != 0 && size > SIZE_MAX / lat_cols) {
            fprintf(stderr, "Error: LAT buffer would overflow.\n");
            goto sbox_cleanup;
        }
        size_t total = size * lat_cols;
        if (total > SIZE_MAX / sizeof(int32_t)) {
            fprintf(stderr, "Error: LAT buffer too large.\n");
            goto sbox_cleanup;
        }
        lat_buf = (int32_t*)malloc(total * sizeof(int32_t));
        if (!lat_buf) {
            fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
            goto sbox_cleanup;
        }
    }

    fwht_sbox_component_metrics_t comp_metrics = {0};
    fwht_status_t status = FWHT_SUCCESS;
    if (!lat_only) {
        fwht_sbox_component_request_t comp_request = {
            .backend = backend,
            .profile_timings = profile_timings ? true : false,
            .spectra = component_buf
        };
        status = fwht_sbox_analyze_components(table, size, &comp_request, &comp_metrics);
        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "Error: component analysis failed (%s).\n", fwht_error_string(status));
            goto sbox_cleanup;
        }
    }

    fwht_sbox_lat_metrics_t lat_metrics;
    int have_lat_metrics = 0;
    if (need_lat) {
        fwht_sbox_lat_request_t lat_request = {
            .backend = backend,
            .profile_timings = profile_timings ? true : false,
            .lat = lat_buf
        };
        status = fwht_sbox_analyze_lat(table, size, &lat_request, &lat_metrics);
        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "Error: LAT analysis failed (%s).\n", fwht_error_string(status));
            goto sbox_cleanup;
        }
        have_lat_metrics = 1;
    }

    if (!quiet) {
        printf("# S-box analysis\n");
    }
    if (!lat_only) {
        printf("m_bits: %zu\n", comp_metrics.m);
        printf("n_bits: %zu\n", comp_metrics.n);
        printf("size: %zu\n", comp_metrics.size);
        printf("component_max_walsh: %d\n", comp_metrics.max_walsh);
        printf("component_min_nonlinearity: %.6f\n", comp_metrics.min_nonlinearity);
    }
    if (have_lat_metrics) {
        printf("lat_max: %d\n", lat_metrics.lat_max);
        printf("lat_max_bias: %.6f\n", lat_metrics.lat_max_bias);
    }

    if (profile_timings) {
        if (!lat_only) {
            printf("component_fwht_ms: %.3f\n", comp_metrics.fwht_ms);
        }
        if (have_lat_metrics) {
            printf("lat_column_ms: %.3f\n", lat_metrics.column_ms);
            printf("lat_fwht_ms: %.3f\n", lat_metrics.fwht_ms);
        }
    }

    if (!lat_only && component_path && component_buf) {
        if (write_component_file(component_path, component_buf, comp_metrics.n, comp_metrics.size) != 0) {
            fprintf(stderr, "Error: failed to write component spectra.\n");
            goto sbox_cleanup;
        }
    }

    if (lat_path && lat_buf) {
        size_t cols = (size_t)1 << lat_metrics.n;
        if (write_lat_file(lat_path, lat_buf, lat_metrics.size, cols, lat_metrics.m, lat_metrics.n) != 0) {
            fprintf(stderr, "Error: failed to write LAT output.\n");
            goto sbox_cleanup;
        }
    }

    exit_code = 0;

sbox_cleanup:
    free(lat_buf);
    free(component_buf);
    free(table);
    return exit_code;
}

static int append_token(const char *token, input_format_t format, value_buffer_t *buffer) {
    char *endptr = NULL;
    if (format == INPUT_FLOAT) {
        double value = strtod(token, &endptr);
        if (endptr == token || *endptr != '\0') {
            fprintf(stderr, "Error: invalid float token '%s'.\n", token);
            return -1;
        }
        return append_value(buffer, value);
    }
    long value = strtol(token, &endptr, 10);
    if (endptr == token || *endptr != '\0') {
        fprintf(stderr, "Error: invalid integer token '%s'.\n", token);
        return -1;
    }
    if (format == INPUT_BOOL && value != 0 && value != 1) {
        fprintf(stderr, "Error: token '%s' is not 0 or 1 (bool input).\n", token);
        return -1;
    }
    return append_value(buffer, (double)value);
}

static int parse_line(char *line, input_format_t format, value_buffer_t *buffer) {
    char *comment = strchr(line, '#');
    if (comment) {
        *comment = '\0';
    }
    const char *delims = ", \t\r\n";
    char *token = strtok(line, delims);
    while (token) {
        if (append_token(token, format, buffer) != 0) {
            return -1;
        }
        token = strtok(NULL, delims);
    }
    return 0;
}

static int read_stream(FILE *stream, input_format_t format, value_buffer_t *buffer) {
    char line[4096];
    while (fgets(line, sizeof(line), stream)) {
        if (parse_line(line, format, buffer) != 0) {
            return -1;
        }
    }
    if (ferror(stream)) {
        fprintf(stderr, "Error: failed while reading input (%s).\n", strerror(errno));
        return -1;
    }
    return 0;
}

static int read_file(const char *path, input_format_t format, value_buffer_t *buffer) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' (%s).\n", path, strerror(errno));
        return -1;
    }
    int rc = read_stream(fp, format, buffer);
    fclose(fp);
    return rc;
}

static int read_values_arg(const char *values, input_format_t format, value_buffer_t *buffer) {
    size_t len = strlen(values);
    char *copy = malloc(len + 1);
    if (!copy) {
        fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
        return -1;
    }
    memcpy(copy, values, len + 1);
    int rc = parse_line(copy, format, buffer);
    free(copy);
    return rc;
}

static fwht_backend_t parse_backend(const char *arg) {
    if (strcmp(arg, "cpu") == 0) {
        return FWHT_BACKEND_CPU;
    } else if (strcmp(arg, "gpu") == 0) {
        return FWHT_BACKEND_GPU;
    } else if (strcmp(arg, "openmp") == 0) {
        return FWHT_BACKEND_OPENMP;
    } else if (strcmp(arg, "cpu-safe") == 0) {
        return FWHT_BACKEND_CPU_SAFE;
    } else if (strcmp(arg, "auto") == 0) {
        return FWHT_BACKEND_AUTO;
    }
    return (fwht_backend_t)-1;
}

#ifdef USE_CUDA
static void print_gpu_metrics(void) {
    fwht_gpu_metrics_t metrics = fwht_gpu_get_last_metrics();
    if (!metrics.valid) {
        return;
    }
    printf("# GPU metrics: n=%zu batch=%zu h2d=%.3f ms kernel=%.3f ms d2h=%.3f ms (samples=%d)\n",
           metrics.n, metrics.batch_size, metrics.h2d_ms, metrics.kernel_ms, metrics.d2h_ms,
           metrics.samples);
}
#else
static void print_gpu_metrics(void) {
    (void)0;
}
#endif

static fwht_status_t run_batch_i32(int32_t *data, size_t n, size_t batch_size,
                                   fwht_backend_t backend, fwht_backend_t *resolved_backend,
                                   int *used_gpu) {
    if (used_gpu) {
        *used_gpu = 0;
    }
    if (backend == FWHT_BACKEND_CPU_SAFE) {
        for (size_t b = 0; b < batch_size; ++b) {
            fwht_status_t st = fwht_i32_safe(data + b * n, n);
            if (st != FWHT_SUCCESS) {
                return st;
            }
        }
        *resolved_backend = FWHT_BACKEND_CPU_SAFE;
        return FWHT_SUCCESS;
    }

#ifdef USE_CUDA
    int try_gpu = (backend == FWHT_BACKEND_GPU) ||
                  (backend == FWHT_BACKEND_AUTO && fwht_has_gpu() &&
                   fwht_recommend_backend(n) == FWHT_BACKEND_GPU);
    if (try_gpu) {
        fwht_status_t st = fwht_batch_i32_cuda(data, n, batch_size);
        if (st == FWHT_SUCCESS) {
            *resolved_backend = FWHT_BACKEND_GPU;
            if (used_gpu) {
                *used_gpu = 1;
            }
        }
        return st;
    }
#else
    if (backend == FWHT_BACKEND_GPU) {
        fprintf(stderr, "Error: CLI built without CUDA support; drop --backend gpu or rebuild with USE_CUDA=1.\n");
        return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }
#endif

    int32_t **ptrs = malloc(batch_size * sizeof(int32_t *));
    if (!ptrs) {
        fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
        return FWHT_ERROR_OUT_OF_MEMORY;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        ptrs[b] = data + b * n;
    }

    fwht_status_t status = fwht_i32_batch(ptrs, n, batch_size);
    free(ptrs);
    *resolved_backend = (backend == FWHT_BACKEND_AUTO) ? FWHT_BACKEND_CPU : backend;
    return status;
}

static fwht_status_t run_batch_f64(double *data, size_t n, size_t batch_size,
                                   fwht_backend_t backend, fwht_backend_t *resolved_backend,
                                   int *used_gpu) {
    if (used_gpu) {
        *used_gpu = 0;
    }
#ifdef USE_CUDA
    int try_gpu = (backend == FWHT_BACKEND_GPU) ||
                  (backend == FWHT_BACKEND_AUTO && fwht_has_gpu() &&
                   fwht_recommend_backend(n) == FWHT_BACKEND_GPU);
    if (try_gpu) {
        fwht_status_t st = fwht_batch_f64_cuda(data, n, batch_size);
        if (st == FWHT_SUCCESS) {
            *resolved_backend = FWHT_BACKEND_GPU;
            if (used_gpu) {
                *used_gpu = 1;
            }
        }
        return st;
    }
#else
    if (backend == FWHT_BACKEND_GPU) {
        fprintf(stderr, "Error: CLI built without CUDA support; drop --backend gpu or rebuild with USE_CUDA=1.\n");
        return FWHT_ERROR_BACKEND_UNAVAILABLE;
    }
#endif

    double **ptrs = malloc(batch_size * sizeof(double *));
    if (!ptrs) {
        fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
        return FWHT_ERROR_OUT_OF_MEMORY;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        ptrs[b] = data + b * n;
    }

    fwht_status_t status = fwht_f64_batch(ptrs, n, batch_size);
    free(ptrs);
    *resolved_backend = (backend == FWHT_BACKEND_AUTO) ? FWHT_BACKEND_CPU : backend;
    return status;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *values_arg = NULL;
    fwht_backend_t backend = FWHT_BACKEND_AUTO;
    input_format_t format = INPUT_BOOL;
    dtype_t dtype = DTYPE_I32;
    size_t batch_size = 1;
    int normalize = 0;
    int show_index = 1;
    int quiet = 0;
    int precision = 6;
    int safe_flag = 0;
    int gpu_profile = 0;
    int gpu_block_size = 0;
    cli_mode_t mode = MODE_WHT;
    const char* sbox_lat_path = NULL;
    const char* sbox_component_path = NULL;
    int sbox_lat_stats = 0;
    int sbox_profile = 0;
    int sbox_lat_only = 0;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(arg, "--input") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --input requires a path.\n");
                return 1;
            }
            input_path = argv[++i];
        } else if (strncmp(arg, "--input=", 8) == 0) {
            input_path = arg + 8;
        } else if (strcmp(arg, "--values") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --values requires a list.\n");
                return 1;
            }
            values_arg = argv[++i];
        } else if (strncmp(arg, "--values=", 9) == 0) {
            values_arg = arg + 9;
        } else if (strcmp(arg, "--backend") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --backend requires a value.\n");
                return 1;
            }
            backend = parse_backend(argv[++i]);
            if (backend == (fwht_backend_t)-1) {
                fprintf(stderr, "Error: unknown backend '%s'.\n", argv[i]);
                return 1;
            }
        } else if (strncmp(arg, "--backend=", 10) == 0) {
            backend = parse_backend(arg + 10);
            if (backend == (fwht_backend_t)-1) {
                fprintf(stderr, "Error: unknown backend '%s'.\n", arg + 10);
                return 1;
            }
        } else if (strcmp(arg, "--dtype") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --dtype requires a value.\n");
                return 1;
            }
            ++i;
            if (strcmp(argv[i], "i32") == 0) {
                dtype = DTYPE_I32;
            } else if (strcmp(argv[i], "f64") == 0) {
                dtype = DTYPE_F64;
            } else {
                fprintf(stderr, "Error: unknown dtype '%s'.\n", argv[i]);
                return 1;
            }
        } else if (strncmp(arg, "--dtype=", 8) == 0) {
            const char *mode = arg + 8;
            if (strcmp(mode, "i32") == 0) {
                dtype = DTYPE_I32;
            } else if (strcmp(mode, "f64") == 0) {
                dtype = DTYPE_F64;
            } else {
                fprintf(stderr, "Error: unknown dtype '%s'.\n", mode);
                return 1;
            }
        } else if (strcmp(arg, "--batch-size") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --batch-size requires a value.\n");
                return 1;
            }
            batch_size = strtoull(argv[++i], NULL, 10);
        } else if (strncmp(arg, "--batch-size=", 13) == 0) {
            batch_size = strtoull(arg + 13, NULL, 10);
        } else if (strcmp(arg, "--input-format") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --input-format requires a value.\n");
                return 1;
            }
            ++i;
            if (strcmp(argv[i], "bool") == 0) {
                format = INPUT_BOOL;
            } else if (strcmp(argv[i], "signed") == 0) {
                format = INPUT_SIGNED;
            } else if (strcmp(argv[i], "float") == 0) {
                format = INPUT_FLOAT;
            } else {
                fprintf(stderr, "Error: unknown input format '%s'.\n", argv[i]);
                return 1;
            }
        } else if (strncmp(arg, "--input-format=", 15) == 0) {
            const char *mode = arg + 15;
            if (strcmp(mode, "bool") == 0) {
                format = INPUT_BOOL;
            } else if (strcmp(mode, "signed") == 0) {
                format = INPUT_SIGNED;
            } else if (strcmp(mode, "float") == 0) {
                format = INPUT_FLOAT;
            } else {
                fprintf(stderr, "Error: unknown input format '%s'.\n", mode);
                return 1;
            }
        } else if (strcmp(arg, "--safe") == 0) {
            safe_flag = 1;
        } else if (strcmp(arg, "--normalize") == 0) {
            normalize = 1;
        } else if (strcmp(arg, "--precision") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --precision requires a value.\n");
                return 1;
            }
            precision = atoi(argv[++i]);
        } else if (strncmp(arg, "--precision=", 12) == 0) {
            precision = atoi(arg + 12);
        } else if (strcmp(arg, "--gpu-profile") == 0) {
            gpu_profile = 1;
        } else if (strcmp(arg, "--gpu-block-size") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --gpu-block-size requires a value.\n");
                return 1;
            }
            gpu_block_size = atoi(argv[++i]);
        } else if (strncmp(arg, "--gpu-block-size=", 18) == 0) {
            gpu_block_size = atoi(arg + 18);
        } else if (strcmp(arg, "--no-index") == 0) {
            show_index = 0;
        } else if (strcmp(arg, "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(arg, "--sbox") == 0) {
            mode = MODE_SBOX;
            format = INPUT_SIGNED;
        } else if (strcmp(arg, "--sbox-lat") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --sbox-lat requires a path.\n");
                return 1;
            }
            sbox_lat_path = argv[++i];
        } else if (strncmp(arg, "--sbox-lat=", 11) == 0) {
            sbox_lat_path = arg + 11;
        } else if (strcmp(arg, "--sbox-components") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --sbox-components requires a path.\n");
                return 1;
            }
            sbox_component_path = argv[++i];
        } else if (strncmp(arg, "--sbox-components=", 19) == 0) {
            sbox_component_path = arg + 19;
        } else if (strcmp(arg, "--sbox-lat-stats") == 0) {
            sbox_lat_stats = 1;
        } else if (strcmp(arg, "--sbox-profile") == 0) {
            sbox_profile = 1;
        } else if (strcmp(arg, "--sbox-lat-only") == 0) {
            sbox_lat_only = 1;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'.\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (precision < 0 || precision > MAX_PRECISION) {
        fprintf(stderr, "Error: precision must be between 0 and %d.\n", MAX_PRECISION);
        return 1;
    }
    if (batch_size == 0) {
        fprintf(stderr, "Error: --batch-size must be >= 1.\n");
        return 1;
    }
    if (sbox_lat_only) {
        if (mode != MODE_SBOX) {
            fprintf(stderr, "Error: --sbox-lat-only is only valid together with --sbox.\n");
            return 1;
        }
        if (sbox_component_path) {
            fprintf(stderr, "Error: --sbox-lat-only cannot be combined with --sbox-components.\n");
            return 1;
        }
        if (!sbox_lat_path && !sbox_lat_stats) {
            fprintf(stderr, "Error: --sbox-lat-only requires --sbox-lat or --sbox-lat-stats.\n");
            return 1;
        }
    }
    if (dtype == DTYPE_I32 && format == INPUT_FLOAT) {
        fprintf(stderr, "Error: floating input requires --dtype f64.\n");
        return 1;
    }
    if (safe_flag) {
        if (dtype != DTYPE_I32) {
            fprintf(stderr, "Error: --safe is only valid with --dtype i32.\n");
            return 1;
        }
        backend = FWHT_BACKEND_CPU_SAFE;
    }

    value_buffer_t buffer = {0};
    int32_t *i32_data = NULL;
    double *f64_data = NULL;
    int exit_code = 0;
    uint64_t *packed_bits = NULL;
    size_t packed_words = 0;
    int pack_boolean_bits = 0;

    if (!input_path && !values_arg) {
        fprintf(stderr, "Reading values from stdin (Ctrl+D to finish).\n");
    }

    if (input_path) {
        if (read_file(input_path, format, &buffer) != 0) {
            exit_code = 1;
            goto cleanup;
        }
    }
    if (values_arg) {
        if (read_values_arg(values_arg, format, &buffer) != 0) {
            exit_code = 1;
            goto cleanup;
        }
    }
    if (!input_path && !values_arg) {
        if (read_stream(stdin, format, &buffer) != 0) {
            exit_code = 1;
            goto cleanup;
        }
    }

    if (buffer.length == 0) {
        fprintf(stderr, "Error: no data provided.\n");
        exit_code = 1;
        goto cleanup;
    }

    size_t total = buffer.length;
    if (total % batch_size != 0) {
        fprintf(stderr, "Error: total samples (%zu) not divisible by batch-size (%zu).\n",
                total, batch_size);
        exit_code = 1;
        goto cleanup;
    }
    size_t n = total / batch_size;
    if (!fwht_is_power_of_2(n) || n == 0) {
        fprintf(stderr, "Error: transform size (%zu) is not a power of two.\n", n);
        exit_code = 1;
        goto cleanup;
    }

    if (format == INPUT_BOOL && batch_size == 1) {
        packed_words = (n + 63u) / 64u;
        packed_bits = calloc(packed_words, sizeof(uint64_t));
        if (!packed_bits) {
            fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
            exit_code = 1;
            goto cleanup;
        }
        pack_boolean_bits = 1;
    }

    if (mode == MODE_SBOX) {
        if (batch_size != 1) {
            fprintf(stderr, "Error: --sbox mode does not support batch processing.\n");
            exit_code = 1;
            goto cleanup;
        }
        exit_code = run_sbox_mode(&buffer, backend, quiet,
                       sbox_lat_path, sbox_component_path,
                       sbox_lat_stats, sbox_profile,
                       sbox_lat_only);
        goto cleanup;
    }

    if (backend == FWHT_BACKEND_GPU && !fwht_has_gpu()) {
        fprintf(stderr, "Error: GPU backend requested but no CUDA-capable device detected.\n");
        exit_code = 1;
        goto cleanup;
    }

    if (dtype == DTYPE_I32) {
        i32_data = malloc(total * sizeof(int32_t));
        if (!i32_data) {
            fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
            exit_code = 1;
            goto cleanup;
        }
        for (size_t i = 0; i < total; ++i) {
            double value = buffer.data[i];
            if (format == INPUT_BOOL) {
                int bit = (value != 0.0);
                if (pack_boolean_bits && bit) {
                    size_t word_idx = i / 64u;
                    size_t bit_idx = i % 64u;
                    packed_bits[word_idx] |= (1ULL << bit_idx);
                }
                i32_data[i] = bit ? -1 : 1;
                continue;
            }
            if (value < (double)INT32_MIN || value > (double)INT32_MAX) {
                fprintf(stderr, "Error: value %.0f out of int32 range.\n", value);
                exit_code = 1;
                goto cleanup;
            }
            int32_t v = (int32_t)value;
            if ((double)v != value) {
                fprintf(stderr, "Error: non-integer token encountered; use --dtype f64 or --input-format float.\n");
                exit_code = 1;
                goto cleanup;
            }
            i32_data[i] = v;
        }
    } else {
        f64_data = malloc(total * sizeof(double));
        if (!f64_data) {
            fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
            exit_code = 1;
            goto cleanup;
        }
        for (size_t i = 0; i < total; ++i) {
            double value = buffer.data[i];
            if (format == INPUT_BOOL) {
                value = (value == 0.0) ? 1.0 : -1.0;
            }
            f64_data[i] = value;
        }
    }

    free(buffer.data);
    buffer.data = NULL;

#ifdef USE_CUDA
    if (gpu_block_size > 0) {
        fwht_status_t st = fwht_gpu_set_block_size((unsigned int)gpu_block_size);
        if (st != FWHT_SUCCESS) {
            fprintf(stderr, "Error: failed to set GPU block size (%s).\n", fwht_error_string(st));
            exit_code = 1;
            goto cleanup;
        }
    }
    if (gpu_profile) {
        fwht_status_t st = fwht_gpu_set_profiling(true);
        if (st != FWHT_SUCCESS) {
            fprintf(stderr, "Error: failed to enable GPU profiling (%s).\n", fwht_error_string(st));
            exit_code = 1;
            goto cleanup;
        }
    }
#else
    if (gpu_block_size > 0 || gpu_profile) {
        fprintf(stderr, "Error: GPU tuning flags require a CUDA build (recompile with USE_CUDA=1).\n");
        exit_code = 1;
        goto cleanup;
    }
#endif

    fwht_status_t status = FWHT_SUCCESS;
    fwht_backend_t resolved_backend = backend;
    int used_gpu_backend = 0;
    int used_bitpacked_gpu = 0;

    if (pack_boolean_bits && dtype == DTYPE_I32 && packed_bits != NULL &&
        n <= 65536 && backend != FWHT_BACKEND_CPU_SAFE) {
#ifdef USE_CUDA
        int gpu_available = fwht_has_gpu();
#else
        int gpu_available = 0;
#endif
        int want_gpu = (backend == FWHT_BACKEND_GPU) ||
                       (backend == FWHT_BACKEND_AUTO && gpu_available);
        if (want_gpu && gpu_available) {
            status = fwht_boolean_packed_backend(packed_bits, i32_data, n, FWHT_BACKEND_GPU);
            if (status != FWHT_SUCCESS) {
                fprintf(stderr, "Error: bit-packed GPU backend failed (%s).\n", fwht_error_string(status));
                exit_code = 1;
                goto cleanup;
            }
            resolved_backend = FWHT_BACKEND_GPU;
            used_gpu_backend = 1;
            used_bitpacked_gpu = 1;
        }
    }

    if (!used_bitpacked_gpu && batch_size > 1) {
        if (dtype == DTYPE_I32) {
            status = run_batch_i32(i32_data, n, batch_size, backend, &resolved_backend,
                                   &used_gpu_backend);
        } else {
            status = run_batch_f64(f64_data, n, batch_size, backend, &resolved_backend,
                                   &used_gpu_backend);
        }
    } else if (!used_bitpacked_gpu) {
        if (dtype == DTYPE_I32) {
            if (backend == FWHT_BACKEND_AUTO) {
                status = fwht_i32(i32_data, n);
                resolved_backend = fwht_recommend_backend(n);
            } else if (backend == FWHT_BACKEND_CPU_SAFE) {
                status = fwht_i32_safe(i32_data, n);
                resolved_backend = FWHT_BACKEND_CPU_SAFE;
            } else {
                status = fwht_i32_backend(i32_data, n, backend);
                resolved_backend = backend;
            }
        } else {
            if (backend == FWHT_BACKEND_AUTO) {
                status = fwht_f64(f64_data, n);
                resolved_backend = fwht_recommend_backend(n);
            } else {
                status = fwht_f64_backend(f64_data, n, backend);
                resolved_backend = backend;
            }
        }
        if (resolved_backend == FWHT_BACKEND_GPU) {
            used_gpu_backend = 1;
        }
    }

    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "Error: FWHT failed (%s).\n", fwht_error_string(status));
        exit_code = 1;
        goto cleanup;
    }

    if (!quiet) {
        int logn = fwht_log2(n);
        printf("# FWHT coefficients\n");
        printf("# Transform size: %zu (2^%d)\n", n, logn);
        if (batch_size > 1) {
            printf("# Batch size: %zu\n", batch_size);
        }
        printf("# Backend: %s\n", fwht_backend_name(resolved_backend));
        printf("# DType: %s\n", dtype_name(dtype));
        printf("# Format: %s\n", normalize ? "normalized" :
                                            ((dtype == DTYPE_I32) ? "integer" : "float"));
    }

    double scale = normalize ? (1.0 / sqrt((double)n)) : 1.0;
    size_t total_elems = n * batch_size;
    if (dtype == DTYPE_I32 && !normalize) {
        for (size_t i = 0; i < total_elems; ++i) {
            if (show_index) {
                printf("%zu %d\n", i, i32_data[i]);
            } else {
                printf("%d\n", i32_data[i]);
            }
        }
    } else if (dtype == DTYPE_I32) {
        for (size_t i = 0; i < total_elems; ++i) {
            double value = (double)i32_data[i] * scale;
            if (show_index) {
                printf("%zu %.*f\n", i, precision, value);
            } else {
                printf("%.*f\n", precision, value);
            }
        }
    } else {
        for (size_t i = 0; i < total_elems; ++i) {
            double value = f64_data[i] * scale;
            if (show_index) {
                printf("%zu %.*f\n", i, precision, value);
            } else {
                printf("%.*f\n", precision, value);
            }
        }
    }

    if (gpu_profile && used_gpu_backend) {
        print_gpu_metrics();
    }

cleanup:
    free(buffer.data);
    free(i32_data);
    free(f64_data);
    free(packed_bits);
    return exit_code;
}
