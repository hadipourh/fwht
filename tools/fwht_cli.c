/*
 * Fast Walsh-Hadamard Transform - Command-Line Interface
 * 
 * CLI tool for computing WHT without writing C code.
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fwht.h"

#define INITIAL_CAPACITY 64

typedef enum {
    INPUT_BOOL,
    INPUT_SIGNED
} input_format_t;

typedef struct {
    int32_t *data;
    size_t length;
    size_t capacity;
} int_buffer_t;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "Options:\n"
            "  --input <file>          Read Boolean/signed values from text file (whitespace or comma separated).\n"
            "  --values <list>         Comma/space separated list of values provided inline.\n"
            "  --backend <name>        Backend: auto (default), cpu, openmp, gpu.\n"
            "  --input-format <mode>   Input mode: bool (default, expects 0/1) or signed (expects integers).\n"
            "  --normalize             Print normalized coefficients (value / sqrt(n)).\n"
            "  --precision <digits>    Decimal precision when using --normalize (default: 6).\n"
            "  --no-index              Print coefficients only (omit index column).\n"
            "  --quiet                 Suppress header metadata.\n"
            "  --help                  Show this message.\n"
            "\n"
            "Examples:\n"
            "  %s --values 0,1,1,0,1,0,0,1 --backend cpu\n"
            "  %s --input walsh_input.txt --backend gpu --normalize\n",
            prog, prog, prog);
}

static int ensure_capacity(int_buffer_t *buffer, size_t needed) {
    if (needed <= buffer->capacity) {
        return 0;
    }
    size_t new_cap = buffer->capacity ? buffer->capacity : INITIAL_CAPACITY;
    while (new_cap < needed) {
        new_cap *= 2;
    }
    int32_t *tmp = realloc(buffer->data, new_cap * sizeof(int32_t));
    if (!tmp) {
        fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
        return -1;
    }
    buffer->data = tmp;
    buffer->capacity = new_cap;
    return 0;
}

static int append_token(const char *token, input_format_t format, int_buffer_t *buffer) {
    char *endptr = NULL;
    long value = strtol(token, &endptr, 10);
    if (endptr == token || *endptr != '\0') {
        fprintf(stderr, "Error: invalid token '%s'.\n", token);
        return -1;
    }
    if (format == INPUT_BOOL) {
        if (value != 0 && value != 1) {
            fprintf(stderr, "Error: token '%s' is not 0 or 1 (bool input).\n", token);
            return -1;
        }
    }
    if (ensure_capacity(buffer, buffer->length + 1) != 0) {
        return -1;
    }
    buffer->data[buffer->length++] = (int32_t)value;
    return 0;
}

static int parse_line(char *line, input_format_t format, int_buffer_t *buffer) {
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

static int read_stream(FILE *stream, input_format_t format, int_buffer_t *buffer) {
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

static int read_file(const char *path, input_format_t format, int_buffer_t *buffer) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s' (%s).\n", path, strerror(errno));
        return -1;
    }
    int rc = read_stream(fp, format, buffer);
    fclose(fp);
    return rc;
}

static int read_values_arg(const char *values, input_format_t format, int_buffer_t *buffer) {
    char *copy = strdup(values);
    if (!copy) {
        fprintf(stderr, "Error: memory allocation failed (%s).\n", strerror(errno));
        return -1;
    }
    int rc = parse_line(copy, format, buffer);
    free(copy);
    return rc;
}

static int32_t *convert_bool_to_signed(int32_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (data[i] == 0) {
            data[i] = 1;
        } else {
            data[i] = -1;
        }
    }
    return data;
}

static fwht_backend_t parse_backend(const char *arg) {
    if (strcmp(arg, "cpu") == 0) {
        return FWHT_BACKEND_CPU;
    } else if (strcmp(arg, "gpu") == 0) {
        return FWHT_BACKEND_GPU;
    } else if (strcmp(arg, "openmp") == 0) {
        return FWHT_BACKEND_OPENMP;
    } else if (strcmp(arg, "auto") == 0) {
        return FWHT_BACKEND_AUTO;
    }
    return (fwht_backend_t)-1;
}

int main(int argc, char **argv) {
    const char *input_path = NULL;
    const char *values_arg = NULL;
    fwht_backend_t backend = FWHT_BACKEND_AUTO;
    input_format_t format = INPUT_BOOL;
    int normalize = 0;
    int show_index = 1;
    int quiet = 0;
    int precision = 6;

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
            } else {
                fprintf(stderr, "Error: unknown input format '%s'.\n", mode);
                return 1;
            }
        } else if (strcmp(arg, "--normalize") == 0) {
            normalize = 1;
        } else if (strcmp(arg, "--precision") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --precision requires a value.\n");
                return 1;
            }
            precision = atoi(argv[++i]);
            if (precision < 0 || precision > 12) {
                fprintf(stderr, "Error: precision must be between 0 and 12.\n");
                return 1;
            }
        } else if (strncmp(arg, "--precision=", 12) == 0) {
            precision = atoi(arg + 12);
            if (precision < 0 || precision > 12) {
                fprintf(stderr, "Error: precision must be between 0 and 12.\n");
                return 1;
            }
        } else if (strcmp(arg, "--no-index") == 0) {
            show_index = 0;
        } else if (strcmp(arg, "--quiet") == 0) {
            quiet = 1;
        } else {
            fprintf(stderr, "Error: unknown argument '%s'.\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!input_path && !values_arg) {
        fprintf(stderr, "Reading values from stdin (Ctrl+D to finish).\n");
    }

    int_buffer_t buffer = {.data = NULL, .length = 0, .capacity = 0};

    if (input_path) {
        if (read_file(input_path, format, &buffer) != 0) {
            free(buffer.data);
            return 1;
        }
    }
    if (values_arg) {
        if (read_values_arg(values_arg, format, &buffer) != 0) {
            free(buffer.data);
            return 1;
        }
    }
    if (!input_path && !values_arg) {
        if (read_stream(stdin, format, &buffer) != 0) {
            free(buffer.data);
            return 1;
        }
    }

    if (buffer.length == 0) {
        fprintf(stderr, "Error: no data provided.\n");
        free(buffer.data);
        return 1;
    }

    size_t n = buffer.length;
    if (!fwht_is_power_of_2(n)) {
        fprintf(stderr, "Error: number of samples (%zu) is not a power of two.\n", n);
        free(buffer.data);
        return 1;
    }

    if (format == INPUT_BOOL) {
        convert_bool_to_signed(buffer.data, n);
    }

    if (backend == FWHT_BACKEND_GPU && !fwht_has_gpu()) {
        fprintf(stderr, "Error: GPU backend requested but no CUDA-capable device detected.\n");
        free(buffer.data);
        return 1;
    }

    fwht_status_t status;
    if (backend == FWHT_BACKEND_AUTO) {
        status = fwht_i32(buffer.data, n);
    } else {
        status = fwht_i32_backend(buffer.data, n, backend);
    }

    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "Error: FWHT failed (%s).\n", fwht_error_string(status));
        free(buffer.data);
        return 1;
    }

    if (!quiet) {
        int logn = fwht_log2(n);
        fwht_backend_t actual_backend = (backend == FWHT_BACKEND_AUTO) ? fwht_recommend_backend(n) : backend;
        printf("# FWHT coefficients\n");
        printf("# Size: %zu (2^%d)\n", n, logn);
        printf("# Backend: %s\n", fwht_backend_name(actual_backend));
        printf("# Format: %s\n", normalize ? "normalized" : "integer");
    }

    if (normalize) {
        double scale = 1.0 / sqrt((double)n);
        for (size_t i = 0; i < n; ++i) {
            double value = (double)buffer.data[i] * scale;
            if (show_index) {
                printf("%zu %.*f\n", i, precision, value);
            } else {
                printf("%.*f\n", precision, value);
            }
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (show_index) {
                printf("%zu %d\n", i, buffer.data[i]);
            } else {
                printf("%d\n", buffer.data[i]);
            }
        }
    }

    free(buffer.data);
    return 0;
}
