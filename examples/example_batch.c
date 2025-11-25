/*
 * Fast Walsh-Hadamard Transform - Batch and Safety Examples
 *
 * Demonstrates overflow-safe transforms, explicit backend selection,
 * SIMD batch APIs for int32/double data, and bit-packed Boolean batches.
 * Build with: make examples (from repo root)
 */

#include <fwht.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static uint8_t parity_u32(uint32_t x) {
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x &= 0xF;
    return (uint8_t)((0x6996 >> x) & 1U);
}

static void fill_alternating_int32(int32_t* data, size_t n, int phase) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = ((i + phase) % 3 == 0) ? 1 : -1;
    }
}

static void pack_truth_table(const uint8_t* truth, uint64_t* packed, size_t n) {
    size_t words = (n + 63) / 64;
    memset(packed, 0, words * sizeof(uint64_t));
    for (size_t i = 0; i < n; ++i) {
        if (truth[i]) {
            packed[i / 64] |= (1ULL << (i % 64));
        }
    }
}

int main(void) {
    printf("============================================================\n");
    printf("Batch / Safety API Showcase\n");
    printf("============================================================\n\n");

    /* ------------------------------------------------------------ */
    /* Example 1: Overflow-safe transform with backend selection    */
    /* ------------------------------------------------------------ */
    printf("Example 1: fwht_i32_safe + fwht_i32_backend\n");

    int32_t risky[8];
    for (size_t i = 0; i < 8; ++i) {
        risky[i] = (int32_t)(1u << 29);  /* Guaranteed to overflow */
    }

    fwht_status_t status = fwht_i32_safe(risky, 8);
    if (status == FWHT_ERROR_OVERFLOW) {
        printf("  ✓ Overflow detected as expected at n=8\n");
    } else if (status != FWHT_SUCCESS) {
        fprintf(stderr, "  × Unexpected failure: %s\n", fwht_error_string(status));
        return 1;
    }

    for (size_t i = 0; i < 8; ++i) {
        risky[i] = (i % 2 == 0) ? 1 : -1;
    }

    status = fwht_i32_backend(risky, 8, FWHT_BACKEND_CPU);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "  × fwht_i32_backend failed: %s\n", fwht_error_string(status));
        return 1;
    }

    printf("  ✓ CPU backend result: [%d, %d, %d, %d, ...]\n\n",
           risky[0], risky[1], risky[2], risky[3]);

    /* ------------------------------------------------------------ */
    /* Example 2: SIMD vectorized batch transforms (int32/f64)      */
    /* ------------------------------------------------------------ */
    printf("Example 2: fwht_i32_batch / fwht_f64_batch\n");

    const size_t n_batch = 256;
    const size_t batch_size = 4;

    int32_t* signals_i32[batch_size];
    double* signals_f64[batch_size];

    for (size_t b = 0; b < batch_size; ++b) {
        signals_i32[b] = (int32_t*)malloc(n_batch * sizeof(int32_t));
        signals_f64[b] = (double*)malloc(n_batch * sizeof(double));
        if (!signals_i32[b] || !signals_f64[b]) {
            fprintf(stderr, "Allocation failed for batch entry %zu\n", b);
            return 1;
        }
        fill_alternating_int32(signals_i32[b], n_batch, (int)b);
        for (size_t i = 0; i < n_batch; ++i) {
            signals_f64[b][i] = (i % 2 == 0) ? 1.0 : -1.0;
        }
    }

    status = fwht_i32_batch(signals_i32, n_batch, batch_size);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "  × fwht_i32_batch failed: %s\n", fwht_error_string(status));
        return 1;
    }

    status = fwht_f64_batch(signals_f64, n_batch, batch_size);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "  × fwht_f64_batch failed: %s\n", fwht_error_string(status));
        return 1;
    }

    printf("  ✓ Processed %zu int32 and %zu float64 transforms in parallel\n",
           batch_size, batch_size);
    printf("  • int32 batch WHT[0] (first two): %d, %d\n",
           signals_i32[0][0], signals_i32[1][0]);
    printf("  • float64 batch WHT[0] (first two): %.1f, %.1f\n\n",
           signals_f64[0][0], signals_f64[1][0]);

    for (size_t b = 0; b < batch_size; ++b) {
        free(signals_i32[b]);
        free(signals_f64[b]);
    }

    /* ------------------------------------------------------------ */
    /* Example 3: Bit-packed Boolean batch processing               */
    /* ------------------------------------------------------------ */
    printf("Example 3: fwht_boolean_batch for S-box components\n");

    const size_t n_bool = 256;
    const size_t bool_components = 8;
    const size_t bool_words = (n_bool + 63) / 64;

    uint64_t* packed_bits[bool_components];
    const uint64_t* packed_inputs[bool_components];
    int32_t* spectra[bool_components];

    uint8_t* temp_truth = (uint8_t*)malloc(n_bool * sizeof(uint8_t));
    if (!temp_truth) {
        fprintf(stderr, "Allocation failed for truth table buffer\n");
        return 1;
    }

    for (size_t comp = 0; comp < bool_components; ++comp) {
        packed_bits[comp] = (uint64_t*)malloc(bool_words * sizeof(uint64_t));
        spectra[comp] = (int32_t*)malloc(n_bool * sizeof(int32_t));
        if (!packed_bits[comp] || !spectra[comp]) {
            fprintf(stderr, "Allocation failed for component %zu\n", comp);
            return 1;
        }

        /* Synthetic component: parity of (comp+1) masked bits */
        uint8_t mask = (uint8_t)(comp + 1);
        for (size_t i = 0; i < n_bool; ++i) {
            uint8_t parity = parity_u32((uint32_t)(i & mask));
            temp_truth[i] = (uint8_t)(parity ^ (uint8_t)(comp & 1U));
        }
        pack_truth_table(temp_truth, packed_bits[comp], n_bool);
        packed_inputs[comp] = packed_bits[comp];
    }
    free(temp_truth);

    status = fwht_boolean_batch(packed_inputs, spectra, n_bool, bool_components);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "  × fwht_boolean_batch failed: %s\n", fwht_error_string(status));
        return 1;
    }

    printf("  ✓ Processed %zu bit-packed Boolean functions\n", bool_components);
    printf("  • Component 0 WHT[0] = %d\n", spectra[0][0]);
    printf("  • Component 1 WHT[3] = %d\n\n", spectra[1][3]);

    for (size_t comp = 0; comp < bool_components; ++comp) {
        free((void*)packed_inputs[comp]);
        free(spectra[comp]);
    }

    printf("All batch examples completed successfully.\n");
    return 0;
}
