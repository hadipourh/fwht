#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "../ciphers/speck/speck32_exact.h"

typedef enum {
    SPECK32_LINEAR_FIXED_INPUT_MASK = 0,
    SPECK32_LINEAR_FIXED_OUTPUT_MASK = 1
} speck32_linear_mode_t;

typedef struct {
    unsigned int rounds;
    uint64_t key;
    uint64_t seed;
    uint32_t mask;
    size_t top_k;
    size_t key_count;
    speck32_linear_mode_t mode;
    bool key_is_set;
    bool key_count_is_set;
    bool seed_is_set;
    bool mode_is_set;
    bool use_codebook;
    bool force;
    bool unsafe_memory;
    bool dry_run;
    bool allow_any_mask;
} speck32_linear_options_t;

typedef struct {
    uint32_t mask;
    double correlation;
    double score;
} speck32_linear_top_entry_t;

static void print_usage(const char* program) {
    fprintf(stderr,
            "Usage: %s --rounds <n> (--key <hex64> | --num-keys <n> [--seed <u64>]) "
            "(--input-mask <hex32> | --output-mask <hex32>) [options]\n"
            "\n"
            "Options:\n"
            "  --rounds <n>          Number of Speck32 rounds (1-22).\n"
            "  --key <hex64>        One 64-bit master key in hexadecimal, for example 0x1918111009080100.\n"
            "  --num-keys <n>       Use n uniformly random 64-bit master keys and report RMS over keys.\n"
            "  --seed <u64>         Seed for the random master-key generator in multi-key mode.\n"
            "  --input-mask <hex32> Fix one input mask and compute all output-mask correlations.\n"
            "  --output-mask <hex32> Fix one output mask and compute all input-mask correlations.\n"
            "  --top <n>            Print the strongest n masks by absolute correlation (default: 16).\n"
            "  --codebook           Precompute the needed codebook instead of streaming. Adds about 16 GiB.\n"
            "  --allow-any-mask     Bypass the Hamming-weight 1 or 2 guideline check.\n"
            "  --dry-run            Print the planned analysis and memory estimate, then exit.\n"
            "  --force              Execute the exact analysis after the memory warning.\n"
            "  --unsafe-memory      Allow execution even when requested memory exceeds detected RAM.\n"
            "  --help               Show this message.\n"
            "\n"
            "Notes:\n"
            "  Exact Speck32 linear analysis allocates a 2^32 double spectrum, about 32 GiB.\n"
            "  Multi-key RMS mode needs one more 2^32 double accumulator, so about 64 GiB before any codebook.\n"
            "  Without --force the tool only reports the plan and exits safely.\n"
            "\n"
            "Examples:\n"
            "  %s --rounds 7 --key 0x1918111009080100 --input-mask 0x00000001 --dry-run\n"
            "  %s --rounds 7 --num-keys 256 --seed 0x1234 --output-mask 0x00010000 --top 8 --dry-run\n"
            "  %s --rounds 7 --key 0x1918111009080100 --output-mask 0x00010000 --top 8 --force\n",
            program,
            program,
            program,
            program);
}

static int parse_u64_hex(const char* text, uint64_t* out_value) {
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

static int parse_u32_hex(const char* text, uint32_t* out_value) {
    uint64_t parsed;

    if (!parse_u64_hex(text, &parsed) || parsed > UINT32_MAX) {
        return 0;
    }

    *out_value = (uint32_t)parsed;
    return 1;
}

static int parse_unsigned_size(const char* text, size_t* out_value) {
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

static int parse_rounds(const char* text, unsigned int* out_rounds) {
    size_t parsed;

    if (!parse_unsigned_size(text, &parsed) || parsed > UINT32_MAX) {
        return 0;
    }

    *out_rounds = (unsigned int)parsed;
    return 1;
}

static int read_os_random_u64(uint64_t* out_value) {
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

static uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z;

    *state += UINT64_C(0x9e3779b97f4a7c15);
    z = *state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static int prepare_random_key_seed(speck32_linear_options_t* options) {
    if (options == NULL || !options->key_count_is_set) {
        return 1;
    }
    if (options->seed_is_set) {
        return 1;
    }
    if (!read_os_random_u64(&options->seed)) {
        fprintf(stderr, "Error: unable to read a random seed from /dev/urandom.\n");
        return 0;
    }
    options->seed_is_set = true;
    return 1;
}

static uint64_t detect_physical_memory_bytes(void) {
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

static double bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static const char* mode_name(speck32_linear_mode_t mode) {
    return (mode == SPECK32_LINEAR_FIXED_INPUT_MASK)
        ? "fixed input mask -> all output masks"
        : "fixed output mask -> all input masks";
}

static const char* mask_label(speck32_linear_mode_t mode) {
    return (mode == SPECK32_LINEAR_FIXED_INPUT_MASK) ? "input mask" : "output mask";
}

static void print_plan(const speck32_linear_options_t* options,
                       size_t spectrum_bytes,
                       size_t accumulator_bytes,
                       size_t codebook_bytes) {
    const uint64_t physical_memory_bytes = detect_physical_memory_bytes();
    const uint64_t total_bytes = (uint64_t)spectrum_bytes + (uint64_t)accumulator_bytes + (uint64_t)codebook_bytes;

    printf("Speck32 exact linear analysis\n");
    printf("  mode: %s\n", mode_name(options->mode));
    printf("  rounds: %u\n", options->rounds);
    if (options->key_is_set) {
        printf("  key: 0x%016" PRIx64 "\n", options->key);
    } else {
        printf("  master keys: %zu uniformly random keys\n", options->key_count);
        printf("  seed: 0x%016" PRIx64 "\n", options->seed);
        if (options->key_count > 1u) {
            printf("  aggregation: RMS over keys\n");
        }
    }
    printf("  %s: 0x%08" PRIx32 "\n", mask_label(options->mode), options->mask);
    printf("  spectrum memory: %.2f GiB\n", bytes_to_gib((uint64_t)spectrum_bytes));
    if (accumulator_bytes != 0u) {
        printf("  sum-squared memory: %.2f GiB\n", bytes_to_gib((uint64_t)accumulator_bytes));
    }
    if (options->use_codebook) {
        printf("  codebook memory: %.2f GiB\n", bytes_to_gib((uint64_t)codebook_bytes));
    } else {
        printf("  codebook memory: 0.00 GiB (streaming mode)\n");
    }
    printf("  minimum total memory: %.2f GiB\n", bytes_to_gib(total_bytes));
    if (physical_memory_bytes != 0u) {
        printf("  detected physical memory: %.2f GiB\n", bytes_to_gib(physical_memory_bytes));
        if (total_bytes > physical_memory_bytes) {
            printf("  feasibility: exceeds detected physical memory\n");
        }
    }
    printf("  top results to print: %zu\n", options->top_k);
}

static void format_signed_power_of_two(double value, char* buffer, size_t buffer_size) {
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

static void format_abs_power_of_two(double value, char* buffer, size_t buffer_size) {
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

static int insert_top_entry(speck32_linear_top_entry_t* entries,
                            size_t entry_count,
                            uint32_t mask,
                            double correlation) {
    size_t index;
    const double score = fabs(correlation);

    if (entry_count == 0u) {
        return 0;
    }
    if (score <= entries[entry_count - 1u].score) {
        return 0;
    }

    entries[entry_count - 1u].mask = mask;
    entries[entry_count - 1u].correlation = correlation;
    entries[entry_count - 1u].score = score;

    for (index = entry_count - 1u; index > 0u; --index) {
        if (entries[index].score <= entries[index - 1u].score) {
            break;
        }

        {
            speck32_linear_top_entry_t tmp = entries[index - 1u];
            entries[index - 1u] = entries[index];
            entries[index] = tmp;
        }
    }

    return 1;
}

static void initialize_top_entries(speck32_linear_top_entry_t* entries, size_t entry_count) {
    size_t index;

    for (index = 0u; index < entry_count; ++index) {
        entries[index].mask = 0u;
        entries[index].correlation = 0.0;
        entries[index].score = -1.0;
    }
}

static void merge_top_entries(speck32_linear_top_entry_t* dst,
                              size_t entry_count,
                              const speck32_linear_top_entry_t* src) {
    size_t index;

    for (index = 0u; index < entry_count; ++index) {
        if (src[index].score < 0.0) {
            break;
        }
        insert_top_entry(dst, entry_count, src[index].mask, src[index].correlation);
    }
}

static void print_top_results(const speck32_linear_options_t* options,
                              const double* spectrum,
                              size_t spectrum_length) {
    speck32_linear_top_entry_t* entries;
    speck32_linear_top_entry_t* local_entries = NULL;
    const bool use_rms = options->key_count_is_set && options->key_count > 1u;
    char signed_buffer[64];
    char abs_buffer[64];
    size_t index;

    entries = (speck32_linear_top_entry_t*)malloc(options->top_k * sizeof(*entries));
    if (entries == NULL) {
        fprintf(stderr, "Error: unable to allocate the result buffer (%s).\n", strerror(errno));
        return;
    }

    initialize_top_entries(entries, options->top_k);

#ifdef _OPENMP
    if (options->top_k > 0u && spectrum_length >= (size_t)(1u << 20)) {
        const int thread_count = omp_get_max_threads();

        local_entries = (speck32_linear_top_entry_t*)malloc((size_t)thread_count * options->top_k * sizeof(*local_entries));
        if (local_entries == NULL) {
            fprintf(stderr,
                    "Warning: unable to allocate the OpenMP result buffers (%s). Falling back to serial top-k scan.\n",
                    strerror(errno));
        } else {
            for (index = 0u; index < (size_t)thread_count; ++index) {
                initialize_top_entries(local_entries + index * options->top_k, options->top_k);
            }

            #pragma omp parallel
            {
                speck32_linear_top_entry_t* thread_entries = local_entries + (size_t)omp_get_thread_num() * options->top_k;
                size_t spectrum_index;

                #pragma omp for schedule(static)
                for (spectrum_index = 0u; spectrum_index < spectrum_length; ++spectrum_index) {
                    insert_top_entry(thread_entries,
                                     options->top_k,
                                     (uint32_t)spectrum_index,
                                     spectrum[spectrum_index]);
                }
            }

            for (index = 0u; index < (size_t)thread_count; ++index) {
                merge_top_entries(entries, options->top_k, local_entries + index * options->top_k);
            }
        }
    }
#endif

    if (local_entries == NULL) {
        for (index = 0u; index < spectrum_length; ++index) {
            insert_top_entry(entries, options->top_k, (uint32_t)index, spectrum[index]);
        }
    }

    if (use_rms) {
        printf("Top %zu RMS correlations by absolute value\n", options->top_k);
    } else {
        printf("Top %zu correlations by absolute value\n", options->top_k);
    }
    for (index = 0u; index < options->top_k; ++index) {
        if (entries[index].score < 0.0) {
            break;
        }
        format_abs_power_of_two(entries[index].correlation, abs_buffer, sizeof(abs_buffer));
        if (use_rms) {
            printf("  %2zu. mask=0x%08" PRIx32 " rms=%s\n",
                   index + 1u,
                   entries[index].mask,
                   abs_buffer);
        } else {
            format_signed_power_of_two(entries[index].correlation, signed_buffer, sizeof(signed_buffer));
            printf("  %2zu. mask=0x%08" PRIx32 " correlation=%s abs=%s\n",
                   index + 1u,
                   entries[index].mask,
                   signed_buffer,
                   abs_buffer);
        }
    }

    free(local_entries);
    free(entries);
}

static int parse_options(int argc, char** argv, speck32_linear_options_t* options) {
    int index;

    memset(options, 0, sizeof(*options));
    options->top_k = 16u;

    for (index = 1; index < argc; ++index) {
        const char* arg = argv[index];

        if (strcmp(arg, "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (strcmp(arg, "--rounds") == 0) {
            if (++index >= argc || !parse_rounds(argv[index], &options->rounds)) {
                fprintf(stderr, "Error: --rounds expects an integer in [1, 22].\n");
                return -1;
            }
            continue;
        }
        if (strcmp(arg, "--key") == 0) {
            if (++index >= argc || !parse_u64_hex(argv[index], &options->key)) {
                fprintf(stderr, "Error: --key expects a 64-bit hexadecimal value.\n");
                return -1;
            }
            options->key_is_set = true;
            continue;
        }
        if (strcmp(arg, "--num-keys") == 0) {
            if (++index >= argc || !parse_unsigned_size(argv[index], &options->key_count) || options->key_count == 0u) {
                fprintf(stderr, "Error: --num-keys expects a positive integer.\n");
                return -1;
            }
            options->key_count_is_set = true;
            continue;
        }
        if (strcmp(arg, "--seed") == 0) {
            if (++index >= argc || !parse_u64_hex(argv[index], &options->seed)) {
                fprintf(stderr, "Error: --seed expects a 64-bit hexadecimal or decimal value.\n");
                return -1;
            }
            options->seed_is_set = true;
            continue;
        }
        if (strcmp(arg, "--input-mask") == 0) {
            if (++index >= argc || !parse_u32_hex(argv[index], &options->mask)) {
                fprintf(stderr, "Error: --input-mask expects a 32-bit hexadecimal value.\n");
                return -1;
            }
            options->mode = SPECK32_LINEAR_FIXED_INPUT_MASK;
            options->mode_is_set = true;
            continue;
        }
        if (strcmp(arg, "--output-mask") == 0) {
            if (++index >= argc || !parse_u32_hex(argv[index], &options->mask)) {
                fprintf(stderr, "Error: --output-mask expects a 32-bit hexadecimal value.\n");
                return -1;
            }
            options->mode = SPECK32_LINEAR_FIXED_OUTPUT_MASK;
            options->mode_is_set = true;
            continue;
        }
        if (strcmp(arg, "--top") == 0) {
            if (++index >= argc || !parse_unsigned_size(argv[index], &options->top_k) || options->top_k == 0u) {
                fprintf(stderr, "Error: --top expects a positive integer.\n");
                return -1;
            }
            continue;
        }
        if (strcmp(arg, "--codebook") == 0) {
            options->use_codebook = true;
            continue;
        }
        if (strcmp(arg, "--force") == 0) {
            options->force = true;
            continue;
        }
        if (strcmp(arg, "--unsafe-memory") == 0) {
            options->unsafe_memory = true;
            continue;
        }
        if (strcmp(arg, "--dry-run") == 0) {
            options->dry_run = true;
            continue;
        }
        if (strcmp(arg, "--allow-any-mask") == 0) {
            options->allow_any_mask = true;
            continue;
        }

        fprintf(stderr, "Error: unknown option '%s'.\n", arg);
        return -1;
    }

    if (options->key_is_set == options->key_count_is_set) {
        fprintf(stderr, "Error: choose exactly one of --key or --num-keys.\n");
        return -1;
    }
    if (options->seed_is_set && !options->key_count_is_set) {
        fprintf(stderr, "Error: --seed is only valid together with --num-keys.\n");
        return -1;
    }
    if (!options->mode_is_set) {
        fprintf(stderr, "Error: choose exactly one of --input-mask or --output-mask.\n");
        return -1;
    }
    if (!speck32_exact_rounds_supported(options->rounds)) {
        fprintf(stderr, "Error: --rounds must be in [1, 22].\n");
        return -1;
    }
    if (!options->allow_any_mask && !speck32_exact_has_hamming_weight_one_or_two(options->mask)) {
        fprintf(stderr,
                "Error: the fixed mask must have Hamming weight 1 or 2 under the current guideline.\n"
                "Use --allow-any-mask to bypass this check.\n");
        return -1;
    }

    return 0;
}

int main(int argc, char** argv) {
    speck32_linear_options_t options;
    speck32_exact_key_t schedule;
    fwht_status_t status;
    size_t spectrum_length;
    size_t spectrum_bytes;
    size_t accumulator_bytes;
    size_t codebook_bytes;
    size_t effective_key_count;
    size_t key_index;
    uint64_t random_state = 0u;
    uint64_t physical_memory_bytes;
    uint64_t total_bytes;
    double* spectrum = NULL;
    double* sum_squared = NULL;
    uint32_t* codebook = NULL;
    int parse_status;
    bool use_rms;

    parse_status = parse_options(argc, argv, &options);
    if (parse_status > 0) {
        return 0;
    }
    if (parse_status < 0) {
        print_usage(argv[0]);
        return 1;
    }

    if (!prepare_random_key_seed(&options)) {
        return 1;
    }

    use_rms = options.key_count_is_set && options.key_count > 1u;

    spectrum_length = speck32_exact_domain_size();
    spectrum_bytes = speck32_exact_required_spectrum_bytes();
    accumulator_bytes = (options.key_count_is_set && options.key_count > 1u) ? spectrum_bytes : 0u;
    codebook_bytes = options.use_codebook ? speck32_exact_required_codebook_bytes() : 0u;
    effective_key_count = options.key_count_is_set ? options.key_count : 1u;
    total_bytes = (uint64_t)spectrum_bytes + (uint64_t)accumulator_bytes + (uint64_t)codebook_bytes;
    physical_memory_bytes = detect_physical_memory_bytes();
    if (spectrum_length == 0u || spectrum_bytes == 0u) {
        fprintf(stderr, "Error: exact Speck32 analysis is unavailable on this platform.\n");
        return 1;
    }

    print_plan(&options, spectrum_bytes, accumulator_bytes, codebook_bytes);
    if (options.dry_run) {
        return 0;
    }
    if (!options.force) {
        fprintf(stderr,
                "Refusing to run without --force because exact Speck32 linear analysis needs a huge buffer.\n");
        return 2;
    }
    if (physical_memory_bytes != 0u && total_bytes > physical_memory_bytes && !options.unsafe_memory) {
        fprintf(stderr,
                "Refusing to run because the requested memory (%.2f GiB) exceeds detected physical memory (%.2f GiB).\n"
                "Use --dry-run to inspect the plan, or add --unsafe-memory if you intentionally want to oversubscribe RAM.\n",
                bytes_to_gib(total_bytes),
                bytes_to_gib(physical_memory_bytes));
        return 2;
    }

    spectrum = (double*)malloc(spectrum_bytes);
    if (spectrum == NULL) {
        fprintf(stderr,
                "Error: unable to allocate the spectrum buffer (%.2f GiB, %s).\n",
                bytes_to_gib((uint64_t)spectrum_bytes),
                strerror(errno));
        return 1;
    }

    if (options.key_count_is_set && options.key_count > 1u) {
        sum_squared = (double*)calloc(spectrum_length, sizeof(*sum_squared));
        if (sum_squared == NULL) {
            fprintf(stderr,
                    "Error: unable to allocate the RMS accumulator buffer (%.2f GiB, %s).\n",
                    bytes_to_gib((uint64_t)accumulator_bytes),
                    strerror(errno));
            free(spectrum);
            return 1;
        }
    }

    if (options.use_codebook) {
        codebook = (uint32_t*)malloc(codebook_bytes);
        if (codebook == NULL) {
            fprintf(stderr,
                    "Error: unable to allocate the codebook buffer (%.2f GiB, %s).\n",
                    bytes_to_gib((uint64_t)codebook_bytes),
                    strerror(errno));
            free(sum_squared);
            free(spectrum);
            return 1;
        }
    }

    if (options.key_count_is_set) {
        random_state = options.seed;
    }

    for (key_index = 0u; key_index < effective_key_count; ++key_index) {
        const uint64_t master_key = options.key_is_set ? options.key : splitmix64_next(&random_state);

        if (!speck32_exact_init_key_le64(&schedule, master_key, options.rounds)) {
            fprintf(stderr, "Error: unable to initialize the Speck32 key schedule.\n");
            free(codebook);
            free(sum_squared);
            free(spectrum);
            return 1;
        }

        if (options.use_codebook) {
            if (options.mode == SPECK32_LINEAR_FIXED_INPUT_MASK) {
                status = speck32_exact_build_inverse_codebook(&schedule, codebook, spectrum_length);
            } else {
                status = speck32_exact_build_forward_codebook(&schedule, codebook, spectrum_length);
            }
            if (status != FWHT_SUCCESS) {
                fprintf(stderr, "Error: codebook construction failed: %s.\n", fwht_error_string(status));
                free(codebook);
                free(sum_squared);
                free(spectrum);
                return 1;
            }
        }

        if (options.mode == SPECK32_LINEAR_FIXED_INPUT_MASK) {
            status = speck32_exact_linear_output_spectrum_from_input_mask(&schedule,
                                                                          options.mask,
                                                                          codebook,
                                                                          NULL,
                                                                          spectrum,
                                                                          spectrum_length);
        } else {
            status = speck32_exact_linear_input_spectrum_from_output_mask(&schedule,
                                                                          options.mask,
                                                                          codebook,
                                                                          NULL,
                                                                          spectrum,
                                                                          spectrum_length);
        }
        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "Error: linear analysis failed: %s.\n", fwht_error_string(status));
            free(codebook);
            free(sum_squared);
            free(spectrum);
            return 1;
        }

        if (options.key_count_is_set && options.key_count > 1u) {
            speck32_exact_accumulate_squared_values(sum_squared, spectrum, spectrum_length);
        }
    }

    if (use_rms) {
        speck32_exact_finish_rms(spectrum, sum_squared, spectrum_length, effective_key_count);
    }

    print_top_results(&options, spectrum, spectrum_length);

    free(codebook);
    free(sum_squared);
    free(spectrum);
    return 0;
}