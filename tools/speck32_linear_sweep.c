#include "speck32_sweep_common.h"

#include "../ciphers/speck/speck32_exact.h"

#define SPECK32_LINEAR_SWEEP_AUTO_CHUNK_CAP 4u

typedef enum {
    SPECK32_LINEAR_SWEEP_FIXED_INPUT_MASK = 0,
    SPECK32_LINEAR_SWEEP_FIXED_OUTPUT_MASK = 1
} speck32_linear_sweep_mode_t;

typedef struct {
    unsigned int rounds;
    uint64_t key;
    uint64_t seed;
    const char* csv_output_path;
    size_t top_k;
    size_t key_count;
    size_t requested_chunk_size;
    size_t chunk_size;
    speck32_linear_sweep_mode_t mode;
    bool key_is_set;
    bool key_count_is_set;
    bool seed_is_set;
    bool mode_is_set;
    bool use_codebook;
    bool force;
    bool unsafe_memory;
    bool dry_run;
} speck32_linear_sweep_options_t;

static const char* speck32_linear_sweep_mode_name(speck32_linear_sweep_mode_t mode) {
    return (mode == SPECK32_LINEAR_SWEEP_FIXED_INPUT_MASK)
        ? "fixed input masks -> all output masks"
        : "fixed output masks -> all input masks";
}

static const char* speck32_linear_sweep_fixed_side_name(speck32_linear_sweep_mode_t mode) {
    return (mode == SPECK32_LINEAR_SWEEP_FIXED_INPUT_MASK) ? "input" : "output";
}

static const char* speck32_linear_sweep_key_mode_name(const speck32_linear_sweep_options_t* options) {
    return options->key_count_is_set ? "random_keys" : "single_key";
}

static void speck32_linear_sweep_print_usage(const char* program) {
    fprintf(stderr,
            "Usage: %s --rounds <n> (--key <hex64> | --num-keys <n> [--seed <u64>]) \
--fixed-side <input|output> --csv-output <path> [options]\n"
            "\n"
            "Options:\n"
            "  --rounds <n>          Number of Speck32 rounds (1-22).\n"
            "  --key <hex64>         One 64-bit master key in hexadecimal.\n"
            "  --num-keys <n>        Use n uniformly random 64-bit master keys and report RMS over keys.\n"
            "  --seed <u64>          Seed for the random master-key generator in multi-key mode.\n"
            "  --fixed-side <side>   Sweep fixed masks on the input or output side.\n"
            "  --csv-output <path>   Output CSV file path.\n"
            "  --top <n>             Top results to store per fixed mask (default: 5).\n"
            "  --chunk-size <n>      Fixed masks to keep in memory at once. Default: auto.\n"
            "  --codebook            Precompute one codebook per key and reuse it across the chunk.\n"
            "  --dry-run             Print the planned analysis and memory estimate, then exit.\n"
            "  --force               Execute after the memory warning.\n"
            "  --unsafe-memory       Allow execution even when requested memory exceeds detected RAM.\n"
            "  --help                Show this message.\n"
            "\n"
            "Notes:\n"
            "  This backend sweeps all Hamming-weight 1 and 2 masks in one process.\n"
            "  It keeps a small chunk of full 2^32 spectra resident so that one key schedule and one\n"
            "  optional codebook can be reused across multiple fixed masks before moving on.\n"
            "\n"
            "Examples:\n"
            "  %s --rounds 5 --key 0x1918111009080100 --fixed-side output \
--csv-output linear.csv --dry-run\n"
            "  %s --rounds 5 --num-keys 256 --seed 0x1234 --fixed-side input \
--csv-output linear.csv --chunk-size 2 --dry-run\n",
            program,
            program,
            program);
}

static void speck32_linear_sweep_print_plan(const speck32_linear_sweep_options_t* options,
                                            size_t selector_count,
                                            size_t spectrum_bytes,
                                            size_t accumulator_bytes,
                                            size_t codebook_bytes) {
    const uint64_t physical_memory_bytes = speck32_sweep_detect_physical_memory_bytes();
    const uint64_t total_bytes = (uint64_t)options->chunk_size *
        ((uint64_t)spectrum_bytes + (uint64_t)accumulator_bytes) +
        (uint64_t)codebook_bytes;

    printf("Speck32 exact linear sweep\n");
    printf("  mode: %s\n", speck32_linear_sweep_mode_name(options->mode));
    printf("  rounds: %u\n", options->rounds);
    printf("  total fixed masks: %zu\n", selector_count);
    if (options->key_is_set) {
        printf("  key: 0x%016" PRIx64 "\n", options->key);
    } else {
        printf("  master keys: %zu uniformly random keys\n", options->key_count);
        printf("  seed: 0x%016" PRIx64 "\n", options->seed);
        if (options->key_count > 1u) {
            printf("  aggregation: RMS over keys\n");
            printf("  note: the same random key sample is reused for every fixed mask\n");
        }
    }
    if (options->requested_chunk_size == 0u) {
        printf("  chunk size: %zu (auto)\n", options->chunk_size);
    } else {
        printf("  chunk size: %zu\n", options->chunk_size);
    }
    printf("  chunk spectrum memory: %.2f GiB\n",
           speck32_sweep_bytes_to_gib((uint64_t)options->chunk_size * (uint64_t)spectrum_bytes));
    if (accumulator_bytes != 0u) {
        printf("  chunk sum-squared memory: %.2f GiB\n",
               speck32_sweep_bytes_to_gib((uint64_t)options->chunk_size * (uint64_t)accumulator_bytes));
    }
    if (options->use_codebook) {
        printf("  codebook memory: %.2f GiB\n",
               speck32_sweep_bytes_to_gib((uint64_t)codebook_bytes));
    } else {
        printf("  codebook memory: 0.00 GiB (streaming mode)\n");
    }
    printf("  minimum working memory: %.2f GiB\n", speck32_sweep_bytes_to_gib(total_bytes));
    if (physical_memory_bytes != 0u) {
        printf("  detected physical memory: %.2f GiB\n",
               speck32_sweep_bytes_to_gib(physical_memory_bytes));
        if (total_bytes > physical_memory_bytes) {
            printf("  feasibility: exceeds detected physical memory\n");
        }
    }
    printf("  top results per mask: %zu\n", options->top_k);
    printf("  csv output: %s\n", options->csv_output_path);
}

static fwht_status_t speck32_linear_sweep_compute_spectrum(const speck32_exact_key_t* schedule,
                                                           const speck32_linear_sweep_options_t* options,
                                                           uint32_t fixed_mask,
                                                           const uint32_t* codebook,
                                                           fwht_context_t* ctx,
                                                           double* spectrum,
                                                           size_t spectrum_length) {
    if (options->mode == SPECK32_LINEAR_SWEEP_FIXED_INPUT_MASK) {
        return speck32_exact_linear_output_spectrum_from_input_mask(schedule,
                                                                    fixed_mask,
                                                                    codebook,
                                                                    ctx,
                                                                    spectrum,
                                                                    spectrum_length);
    }

    return speck32_exact_linear_input_spectrum_from_output_mask(schedule,
                                                                fixed_mask,
                                                                codebook,
                                                                ctx,
                                                                spectrum,
                                                                spectrum_length);
}

static void speck32_linear_sweep_write_csv_header(FILE* output) {
    fputs("rounds,key_mode,key_hex,num_keys,seed_hex,fixed_side,fixed_mask_hex,fixed_mask_weight,rank,free_mask_hex,metric_name,metric_pow2,abs_metric_pow2\n",
          output);
}

static void speck32_linear_sweep_write_rows(FILE* output,
                                            const speck32_linear_sweep_options_t* options,
                                            const uint32_t* fixed_masks,
                                            const double* spectra,
                                            size_t chunk_size,
                                            size_t spectrum_length,
                                            speck32_sweep_top_entry_t* entries) {
    const bool use_rms = options->key_count_is_set && options->key_count > 1u;
    const char* metric_name = use_rms ? "rms" : "correlation";
    const char* fixed_side = speck32_linear_sweep_fixed_side_name(options->mode);
    char key_hex_buffer[32];
    char num_keys_buffer[32];
    char seed_hex_buffer[32];
    const char* key_hex = "";
    const char* num_keys_text = "";
    const char* seed_hex = "";
    size_t chunk_index;

    if (options->key_is_set) {
        snprintf(key_hex_buffer, sizeof(key_hex_buffer), "0x%016" PRIx64, options->key);
        key_hex = key_hex_buffer;
    }
    if (options->key_count_is_set) {
        snprintf(num_keys_buffer, sizeof(num_keys_buffer), "%zu", options->key_count);
        snprintf(seed_hex_buffer, sizeof(seed_hex_buffer), "0x%016" PRIx64, options->seed);
        num_keys_text = num_keys_buffer;
        seed_hex = seed_hex_buffer;
    }

    for (chunk_index = 0u; chunk_index < chunk_size; ++chunk_index) {
        const uint32_t fixed_mask = fixed_masks[chunk_index];
        const double* spectrum = spectra + chunk_index * spectrum_length;
        size_t rank;

        speck32_sweep_collect_top_entries(spectrum, spectrum_length, options->top_k, entries);

        for (rank = 0u; rank < options->top_k; ++rank) {
            char metric_buffer[64];
            char abs_buffer[64];

            if (entries[rank].score < 0.0) {
                break;
            }

            speck32_sweep_format_abs_power_of_two(entries[rank].value, abs_buffer, sizeof(abs_buffer));
            if (use_rms) {
                strncpy(metric_buffer, abs_buffer, sizeof(metric_buffer));
                metric_buffer[sizeof(metric_buffer) - 1u] = '\0';
            } else {
                speck32_sweep_format_signed_power_of_two(entries[rank].value,
                                                         metric_buffer,
                                                         sizeof(metric_buffer));
            }

            fprintf(output,
                    "%u,%s,%s,%s,%s,%s,0x%08" PRIx32 ",%u,%zu,0x%08" PRIx32 ",%s,%s,%s\n",
                    options->rounds,
                    speck32_linear_sweep_key_mode_name(options),
                    key_hex,
                    num_keys_text,
                    seed_hex,
                    fixed_side,
                    fixed_mask,
                    speck32_exact_hamming_weight(fixed_mask),
                    rank + 1u,
                    entries[rank].mask,
                    metric_name,
                    metric_buffer,
                    abs_buffer);
        }
    }
}

static int speck32_linear_sweep_parse_options(int argc,
                                              char** argv,
                                              speck32_linear_sweep_options_t* options) {
    int index;

    memset(options, 0, sizeof(*options));
    options->top_k = 5u;

    for (index = 1; index < argc; ++index) {
        const char* arg = argv[index];

        if (strcmp(arg, "--help") == 0) {
            speck32_linear_sweep_print_usage(argv[0]);
            return 1;
        }
        if (strcmp(arg, "--rounds") == 0) {
            if (++index >= argc || !speck32_sweep_parse_rounds(argv[index], &options->rounds)) {
                fprintf(stderr, "Error: --rounds expects an integer in [1, 22].\n");
                return -1;
            }
            continue;
        }
        if (strcmp(arg, "--key") == 0) {
            if (++index >= argc || !speck32_sweep_parse_u64_hex(argv[index], &options->key)) {
                fprintf(stderr, "Error: --key expects a 64-bit hexadecimal value.\n");
                return -1;
            }
            options->key_is_set = true;
            continue;
        }
        if (strcmp(arg, "--num-keys") == 0) {
            if (++index >= argc ||
                !speck32_sweep_parse_unsigned_size(argv[index], &options->key_count) ||
                options->key_count == 0u) {
                fprintf(stderr, "Error: --num-keys expects a positive integer.\n");
                return -1;
            }
            options->key_count_is_set = true;
            continue;
        }
        if (strcmp(arg, "--seed") == 0) {
            if (++index >= argc || !speck32_sweep_parse_u64_hex(argv[index], &options->seed)) {
                fprintf(stderr, "Error: --seed expects a 64-bit hexadecimal or decimal value.\n");
                return -1;
            }
            options->seed_is_set = true;
            continue;
        }
        if (strcmp(arg, "--fixed-side") == 0) {
            if (++index >= argc) {
                fprintf(stderr, "Error: --fixed-side expects 'input' or 'output'.\n");
                return -1;
            }
            if (strcmp(argv[index], "input") == 0) {
                options->mode = SPECK32_LINEAR_SWEEP_FIXED_INPUT_MASK;
            } else if (strcmp(argv[index], "output") == 0) {
                options->mode = SPECK32_LINEAR_SWEEP_FIXED_OUTPUT_MASK;
            } else {
                fprintf(stderr, "Error: --fixed-side expects 'input' or 'output'.\n");
                return -1;
            }
            options->mode_is_set = true;
            continue;
        }
        if (strcmp(arg, "--csv-output") == 0) {
            if (++index >= argc) {
                fprintf(stderr, "Error: --csv-output expects a path.\n");
                return -1;
            }
            options->csv_output_path = argv[index];
            continue;
        }
        if (strcmp(arg, "--top") == 0) {
            if (++index >= argc ||
                !speck32_sweep_parse_unsigned_size(argv[index], &options->top_k) ||
                options->top_k == 0u) {
                fprintf(stderr, "Error: --top expects a positive integer.\n");
                return -1;
            }
            continue;
        }
        if (strcmp(arg, "--chunk-size") == 0) {
            if (++index >= argc ||
                !speck32_sweep_parse_unsigned_size(argv[index], &options->requested_chunk_size) ||
                options->requested_chunk_size == 0u) {
                fprintf(stderr, "Error: --chunk-size expects a positive integer.\n");
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
        fprintf(stderr, "Error: --fixed-side is required.\n");
        return -1;
    }
    if (options->csv_output_path == NULL) {
        fprintf(stderr, "Error: --csv-output is required.\n");
        return -1;
    }
    if (!speck32_exact_rounds_supported(options->rounds)) {
        fprintf(stderr, "Error: --rounds must be in [1, 22].\n");
        return -1;
    }

    return 0;
}

int main(int argc, char** argv) {
    speck32_linear_sweep_options_t options;
    speck32_exact_key_t schedule;
    fwht_context_t* ctx = NULL;
    fwht_config_t config;
    fwht_status_t status;
    uint32_t* fixed_masks = NULL;
    uint32_t* codebook = NULL;
    uint64_t* master_keys = NULL;
    double* spectra = NULL;
    double* sum_squared = NULL;
    speck32_sweep_top_entry_t* entries = NULL;
    FILE* output = NULL;
    size_t selector_count;
    size_t spectrum_length;
    size_t spectrum_bytes;
    size_t accumulator_bytes;
    size_t selector_bytes;
    size_t codebook_bytes;
    size_t effective_key_count;
    size_t chunk_start;
    uint64_t physical_memory_bytes;
    uint64_t total_bytes;
    int parse_status;
    bool use_rms;

    parse_status = speck32_linear_sweep_parse_options(argc, argv, &options);
    if (parse_status > 0) {
        return 0;
    }
    if (parse_status < 0) {
        speck32_linear_sweep_print_usage(argv[0]);
        return 1;
    }

    if (!speck32_sweep_prepare_random_seed(options.key_count_is_set, options.seed_is_set, &options.seed)) {
        return 1;
    }
    if (options.key_count_is_set) {
        options.seed_is_set = true;
    }

    selector_count = speck32_exact_hamming_weight_one_two_count();
    spectrum_length = speck32_exact_domain_size();
    spectrum_bytes = speck32_exact_required_spectrum_bytes();
    effective_key_count = options.key_count_is_set ? options.key_count : 1u;
    use_rms = effective_key_count > 1u;
    accumulator_bytes = use_rms ? spectrum_bytes : 0u;
    selector_bytes = spectrum_bytes + accumulator_bytes;
    codebook_bytes = options.use_codebook ? speck32_exact_required_codebook_bytes() : 0u;
    options.chunk_size = (options.requested_chunk_size != 0u)
        ? options.requested_chunk_size
        : speck32_sweep_choose_auto_chunk_size(selector_count,
                                               selector_bytes,
                                               codebook_bytes,
                                               SPECK32_LINEAR_SWEEP_AUTO_CHUNK_CAP);
    if (options.chunk_size > selector_count) {
        options.chunk_size = selector_count;
    }

    total_bytes = (uint64_t)options.chunk_size * (uint64_t)selector_bytes + (uint64_t)codebook_bytes;
    physical_memory_bytes = speck32_sweep_detect_physical_memory_bytes();

    speck32_linear_sweep_print_plan(&options,
                                    selector_count,
                                    spectrum_bytes,
                                    accumulator_bytes,
                                    codebook_bytes);
    if (options.dry_run) {
        return 0;
    }
    if (!options.force) {
        fprintf(stderr,
                "Refusing to run without --force because the exact Speck32 linear sweep needs huge buffers.\n");
        return 2;
    }
    if (physical_memory_bytes != 0u && total_bytes > physical_memory_bytes && !options.unsafe_memory) {
        fprintf(stderr,
                "Refusing to run because the requested memory (%.2f GiB) exceeds detected physical memory (%.2f GiB).\n"
                "Use --dry-run to inspect the plan, or add --unsafe-memory if you intentionally want to oversubscribe RAM.\n",
                speck32_sweep_bytes_to_gib(total_bytes),
                speck32_sweep_bytes_to_gib(physical_memory_bytes));
        return 2;
    }

    fixed_masks = (uint32_t*)malloc(selector_count * sizeof(*fixed_masks));
    if (fixed_masks == NULL) {
        fprintf(stderr, "Error: unable to allocate the fixed-mask list (%s).\n", strerror(errno));
        return 1;
    }
    if (speck32_exact_enumerate_hamming_weight_one_two(fixed_masks, selector_count) != selector_count) {
        fprintf(stderr, "Error: unable to enumerate the fixed masks.\n");
        free(fixed_masks);
        return 1;
    }

    spectra = (double*)malloc(options.chunk_size * spectrum_bytes);
    if (spectra == NULL) {
        fprintf(stderr,
                "Error: unable to allocate the chunk spectrum buffer (%.2f GiB, %s).\n",
                speck32_sweep_bytes_to_gib((uint64_t)options.chunk_size * (uint64_t)spectrum_bytes),
                strerror(errno));
        free(fixed_masks);
        return 1;
    }

    if (use_rms) {
        sum_squared = (double*)calloc(options.chunk_size * spectrum_length, sizeof(*sum_squared));
        if (sum_squared == NULL) {
            fprintf(stderr,
                    "Error: unable to allocate the chunk RMS buffer (%.2f GiB, %s).\n",
                    speck32_sweep_bytes_to_gib((uint64_t)options.chunk_size * (uint64_t)accumulator_bytes),
                    strerror(errno));
            free(spectra);
            free(fixed_masks);
            return 1;
        }
    }

    if (options.use_codebook) {
        codebook = (uint32_t*)malloc(codebook_bytes);
        if (codebook == NULL) {
            fprintf(stderr,
                    "Error: unable to allocate the codebook buffer (%.2f GiB, %s).\n",
                    speck32_sweep_bytes_to_gib((uint64_t)codebook_bytes),
                    strerror(errno));
            free(sum_squared);
            free(spectra);
            free(fixed_masks);
            return 1;
        }
    }

    entries = (speck32_sweep_top_entry_t*)malloc(options.top_k * sizeof(*entries));
    if (entries == NULL) {
        fprintf(stderr, "Error: unable to allocate the top-k buffer (%s).\n", strerror(errno));
        free(codebook);
        free(sum_squared);
        free(spectra);
        free(fixed_masks);
        return 1;
    }

    output = fopen(options.csv_output_path, "w");
    if (output == NULL) {
        fprintf(stderr,
                "Error: unable to open %s for writing (%s).\n",
                options.csv_output_path,
                strerror(errno));
        free(entries);
        free(codebook);
        free(sum_squared);
        free(spectra);
        free(fixed_masks);
        return 1;
    }
    speck32_linear_sweep_write_csv_header(output);

    if (options.key_count_is_set) {
        size_t key_index;
        uint64_t random_state = options.seed;

        master_keys = (uint64_t*)malloc(effective_key_count * sizeof(*master_keys));
        if (master_keys == NULL) {
            fprintf(stderr, "Error: unable to allocate the master-key list (%s).\n", strerror(errno));
            fclose(output);
            free(entries);
            free(codebook);
            free(sum_squared);
            free(spectra);
            free(fixed_masks);
            return 1;
        }

        for (key_index = 0u; key_index < effective_key_count; ++key_index) {
            master_keys[key_index] = speck32_sweep_splitmix64_next(&random_state);
        }
    }

    config = fwht_default_config();
    if (fwht_has_openmp()) {
        config.backend = FWHT_BACKEND_OPENMP;
    }
    ctx = fwht_create_context(&config);

    if (effective_key_count == 1u) {
        const uint64_t master_key = options.key_is_set ? options.key : master_keys[0u];

        if (!speck32_exact_init_key_le64(&schedule, master_key, options.rounds)) {
            fprintf(stderr, "Error: unable to initialize the Speck32 key schedule.\n");
            fwht_destroy_context(ctx);
            free(master_keys);
            fclose(output);
            free(entries);
            free(codebook);
            free(sum_squared);
            free(spectra);
            free(fixed_masks);
            return 1;
        }

        if (options.use_codebook) {
            if (options.mode == SPECK32_LINEAR_SWEEP_FIXED_INPUT_MASK) {
                status = speck32_exact_build_inverse_codebook(&schedule, codebook, spectrum_length);
            } else {
                status = speck32_exact_build_forward_codebook(&schedule, codebook, spectrum_length);
            }
            if (status != FWHT_SUCCESS) {
                fprintf(stderr, "Error: codebook construction failed: %s.\n", fwht_error_string(status));
                fwht_destroy_context(ctx);
                free(master_keys);
                fclose(output);
                free(entries);
                free(codebook);
                free(sum_squared);
                free(spectra);
                free(fixed_masks);
                return 1;
            }
        }

        for (chunk_start = 0u; chunk_start < selector_count; chunk_start += options.chunk_size) {
            const size_t current_chunk = (selector_count - chunk_start < options.chunk_size)
                ? (selector_count - chunk_start)
                : options.chunk_size;
            size_t chunk_index;

            fprintf(stderr,
                    "[%s %zu-%zu/%zu]\n",
                    speck32_linear_sweep_fixed_side_name(options.mode),
                    chunk_start + 1u,
                    chunk_start + current_chunk,
                    selector_count);

            for (chunk_index = 0u; chunk_index < current_chunk; ++chunk_index) {
                status = speck32_linear_sweep_compute_spectrum(&schedule,
                                                               &options,
                                                               fixed_masks[chunk_start + chunk_index],
                                                               codebook,
                                                               ctx,
                                                               spectra + chunk_index * spectrum_length,
                                                               spectrum_length);
                if (status != FWHT_SUCCESS) {
                    fprintf(stderr, "Error: linear analysis failed: %s.\n", fwht_error_string(status));
                    fwht_destroy_context(ctx);
                    free(master_keys);
                    fclose(output);
                    free(entries);
                    free(codebook);
                    free(sum_squared);
                    free(spectra);
                    free(fixed_masks);
                    return 1;
                }
            }

            speck32_linear_sweep_write_rows(output,
                                            &options,
                                            fixed_masks + chunk_start,
                                            spectra,
                                            current_chunk,
                                            spectrum_length,
                                            entries);
        }
    } else {
        for (chunk_start = 0u; chunk_start < selector_count; chunk_start += options.chunk_size) {
            const size_t current_chunk = (selector_count - chunk_start < options.chunk_size)
                ? (selector_count - chunk_start)
                : options.chunk_size;
            size_t key_index;

            memset(sum_squared, 0, current_chunk * spectrum_bytes);
            fprintf(stderr,
                    "[%s %zu-%zu/%zu]\n",
                    speck32_linear_sweep_fixed_side_name(options.mode),
                    chunk_start + 1u,
                    chunk_start + current_chunk,
                    selector_count);

            for (key_index = 0u; key_index < effective_key_count; ++key_index) {
                size_t chunk_index;

                if (!speck32_exact_init_key_le64(&schedule, master_keys[key_index], options.rounds)) {
                    fprintf(stderr, "Error: unable to initialize the Speck32 key schedule.\n");
                    fwht_destroy_context(ctx);
                    free(master_keys);
                    fclose(output);
                    free(entries);
                    free(codebook);
                    free(sum_squared);
                    free(spectra);
                    free(fixed_masks);
                    return 1;
                }

                if (options.use_codebook) {
                    if (options.mode == SPECK32_LINEAR_SWEEP_FIXED_INPUT_MASK) {
                        status = speck32_exact_build_inverse_codebook(&schedule, codebook, spectrum_length);
                    } else {
                        status = speck32_exact_build_forward_codebook(&schedule, codebook, spectrum_length);
                    }
                    if (status != FWHT_SUCCESS) {
                        fprintf(stderr, "Error: codebook construction failed: %s.\n", fwht_error_string(status));
                        fwht_destroy_context(ctx);
                        free(master_keys);
                        fclose(output);
                        free(entries);
                        free(codebook);
                        free(sum_squared);
                        free(spectra);
                        free(fixed_masks);
                        return 1;
                    }
                }

                for (chunk_index = 0u; chunk_index < current_chunk; ++chunk_index) {
                    double* spectrum = spectra + chunk_index * spectrum_length;
                    double* squared = sum_squared + chunk_index * spectrum_length;

                    status = speck32_linear_sweep_compute_spectrum(&schedule,
                                                                   &options,
                                                                   fixed_masks[chunk_start + chunk_index],
                                                                   codebook,
                                                                   ctx,
                                                                   spectrum,
                                                                   spectrum_length);
                    if (status != FWHT_SUCCESS) {
                        fprintf(stderr, "Error: linear analysis failed: %s.\n", fwht_error_string(status));
                        fwht_destroy_context(ctx);
                        free(master_keys);
                        fclose(output);
                        free(entries);
                        free(codebook);
                        free(sum_squared);
                        free(spectra);
                        free(fixed_masks);
                        return 1;
                    }

                    speck32_exact_accumulate_squared_values(squared, spectrum, spectrum_length);
                }
            }

            {
                size_t chunk_index;
                for (chunk_index = 0u; chunk_index < current_chunk; ++chunk_index) {
                    speck32_exact_finish_rms(spectra + chunk_index * spectrum_length,
                                             sum_squared + chunk_index * spectrum_length,
                                             spectrum_length,
                                             effective_key_count);
                }
            }

            speck32_linear_sweep_write_rows(output,
                                            &options,
                                            fixed_masks + chunk_start,
                                            spectra,
                                            current_chunk,
                                            spectrum_length,
                                            entries);
        }
    }

    fwht_destroy_context(ctx);
    free(master_keys);
    fclose(output);
    free(entries);
    free(codebook);
    free(sum_squared);
    free(spectra);
    free(fixed_masks);
    printf("wrote %s\n", options.csv_output_path);
    return 0;
}