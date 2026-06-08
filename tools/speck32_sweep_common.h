#ifndef LIBFWHT_TOOLS_SPECK32_SWEEP_COMMON_H
#define LIBFWHT_TOOLS_SPECK32_SWEEP_COMMON_H

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    uint32_t mask;
    double value;
    double score;
} speck32_sweep_top_entry_t;

static int speck32_sweep_parse_u64_hex(const char* text, uint64_t* out_value) {
    char* end = NULL;
    unsigned long long parsed;

    if (text == NULL || out_value == NULL) {
        return 0;
    }

    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }

    *out_value = (uint64_t)parsed;
    return 1;
}

static int speck32_sweep_parse_unsigned_size(const char* text, size_t* out_value) {
    char* end = NULL;
    unsigned long long parsed;

    if (text == NULL || out_value == NULL) {
        return 0;
    }

    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }

    *out_value = (size_t)parsed;
    return 1;
}

static int speck32_sweep_parse_rounds(const char* text, unsigned int* out_rounds) {
    size_t parsed;

    if (!speck32_sweep_parse_unsigned_size(text, &parsed) || parsed > UINT32_MAX) {
        return 0;
    }

    *out_rounds = (unsigned int)parsed;
    return 1;
}

static int speck32_sweep_read_os_random_u64(uint64_t* out_value) {
    FILE* handle;
    size_t read_count;

    if (out_value == NULL) {
        return 0;
    }

    handle = fopen("/dev/urandom", "rb");
    if (handle == NULL) {
        return 0;
    }

    read_count = fread(out_value, 1u, sizeof(*out_value), handle);
    fclose(handle);
    return read_count == sizeof(*out_value);
}

static uint64_t speck32_sweep_splitmix64_next(uint64_t* state) {
    uint64_t z;

    *state += UINT64_C(0x9e3779b97f4a7c15);
    z = *state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static int speck32_sweep_prepare_random_seed(bool key_count_is_set,
                                             bool seed_is_set,
                                             uint64_t* seed) {
    if (!key_count_is_set || seed_is_set) {
        return 1;
    }
    if (!speck32_sweep_read_os_random_u64(seed)) {
        fprintf(stderr, "Error: unable to read a random seed from /dev/urandom.\n");
        return 0;
    }
    return 1;
}

static uint64_t speck32_sweep_detect_physical_memory_bytes(void) {
#if defined(__APPLE__)
    uint64_t memory_bytes = 0u;
    size_t size = sizeof(memory_bytes);

    if (sysctlbyname("hw.memsize", &memory_bytes, &size, NULL, 0) == 0) {
        return memory_bytes;
    }
    return 0u;
#elif defined(__linux__)
    long page_size = sysconf(_SC_PAGESIZE);
    long page_count = sysconf(_SC_PHYS_PAGES);

    if (page_size <= 0 || page_count <= 0) {
        return 0u;
    }
    return (uint64_t)page_size * (uint64_t)page_count;
#else
    return 0u;
#endif
}

static double speck32_sweep_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static size_t speck32_sweep_choose_auto_chunk_size(size_t selector_count,
                                                   size_t selector_bytes,
                                                   size_t reserved_bytes,
                                                   size_t auto_cap) {
    const uint64_t physical_memory_bytes = speck32_sweep_detect_physical_memory_bytes();
    size_t chunk_size = 1u;

    if (selector_count == 0u) {
        return 0u;
    }
    if (physical_memory_bytes != 0u && selector_bytes != 0u) {
        const uint64_t available_bytes = (physical_memory_bytes > (uint64_t)reserved_bytes)
            ? (physical_memory_bytes - (uint64_t)reserved_bytes)
            : 0u;
        const uint64_t safety_bytes = available_bytes - (available_bytes / 4u);

        if (safety_bytes >= (uint64_t)selector_bytes) {
            chunk_size = (size_t)(safety_bytes / (uint64_t)selector_bytes);
        }
    }

    if (chunk_size == 0u) {
        chunk_size = 1u;
    }
    if (chunk_size > selector_count) {
        chunk_size = selector_count;
    }
    if (auto_cap != 0u && chunk_size > auto_cap) {
        chunk_size = auto_cap;
    }
    return chunk_size;
}

static void speck32_sweep_format_signed_power_of_two(double value,
                                                     char* buffer,
                                                     size_t buffer_size) {
    const char* sign;
    double exponent;

    if (buffer == NULL || buffer_size == 0u) {
        return;
    }
    if (value == 0.0 || !isfinite(value)) {
        snprintf(buffer, buffer_size, "0");
        return;
    }

    sign = (value < 0.0) ? "-" : "+";
    exponent = log2(fabs(value));
    if (fabs(exponent) < 0.00005) {
        exponent = 0.0;
    }
    snprintf(buffer, buffer_size, "%s2^(%.4f)", sign, exponent);
}

static void speck32_sweep_format_abs_power_of_two(double value,
                                                  char* buffer,
                                                  size_t buffer_size) {
    double exponent;

    if (buffer == NULL || buffer_size == 0u) {
        return;
    }
    if (value == 0.0 || !isfinite(value)) {
        snprintf(buffer, buffer_size, "0");
        return;
    }

    exponent = log2(fabs(value));
    if (fabs(exponent) < 0.00005) {
        exponent = 0.0;
    }
    snprintf(buffer, buffer_size, "2^(%.4f)", exponent);
}

static int speck32_sweep_insert_top_entry(speck32_sweep_top_entry_t* entries,
                                          size_t entry_count,
                                          uint32_t mask,
                                          double value) {
    size_t index;
    const double score = fabs(value);

    if (entry_count == 0u) {
        return 0;
    }
    if (score <= entries[entry_count - 1u].score) {
        return 0;
    }

    entries[entry_count - 1u].mask = mask;
    entries[entry_count - 1u].value = value;
    entries[entry_count - 1u].score = score;

    for (index = entry_count - 1u; index > 0u; --index) {
        if (entries[index].score <= entries[index - 1u].score) {
            break;
        }

        {
            speck32_sweep_top_entry_t tmp = entries[index - 1u];
            entries[index - 1u] = entries[index];
            entries[index] = tmp;
        }
    }

    return 1;
}

static void speck32_sweep_initialize_top_entries(speck32_sweep_top_entry_t* entries,
                                                 size_t entry_count) {
    size_t index;

    for (index = 0u; index < entry_count; ++index) {
        entries[index].mask = 0u;
        entries[index].value = 0.0;
        entries[index].score = -1.0;
    }
}

static void speck32_sweep_merge_top_entries(speck32_sweep_top_entry_t* dst,
                                            size_t entry_count,
                                            const speck32_sweep_top_entry_t* src) {
    size_t index;

    for (index = 0u; index < entry_count; ++index) {
        if (src[index].score < 0.0) {
            break;
        }
        speck32_sweep_insert_top_entry(dst, entry_count, src[index].mask, src[index].value);
    }
}

static void speck32_sweep_collect_top_entries(const double* values,
                                              size_t value_count,
                                              size_t top_k,
                                              speck32_sweep_top_entry_t* entries) {
    speck32_sweep_top_entry_t* local_entries = NULL;
    size_t index;

    speck32_sweep_initialize_top_entries(entries, top_k);

#ifdef _OPENMP
    if (top_k > 0u && value_count >= (size_t)(1u << 20)) {
        const int thread_count = omp_get_max_threads();

        local_entries = (speck32_sweep_top_entry_t*)malloc((size_t)thread_count * top_k * sizeof(*local_entries));
        if (local_entries != NULL) {
            for (index = 0u; index < (size_t)thread_count; ++index) {
                speck32_sweep_initialize_top_entries(local_entries + index * top_k, top_k);
            }

            #pragma omp parallel
            {
                speck32_sweep_top_entry_t* thread_entries = local_entries + (size_t)omp_get_thread_num() * top_k;
                size_t value_index;

                #pragma omp for schedule(static)
                for (value_index = 0u; value_index < value_count; ++value_index) {
                    speck32_sweep_insert_top_entry(thread_entries,
                                                   top_k,
                                                   (uint32_t)value_index,
                                                   values[value_index]);
                }
            }

            for (index = 0u; index < (size_t)thread_count; ++index) {
                speck32_sweep_merge_top_entries(entries, top_k, local_entries + index * top_k);
            }
        }
    }
#endif

    if (local_entries == NULL) {
        for (index = 0u; index < value_count; ++index) {
            speck32_sweep_insert_top_entry(entries, top_k, (uint32_t)index, values[index]);
        }
    }

    free(local_entries);
}

#endif