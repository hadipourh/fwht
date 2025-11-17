/*
 * Fast Walsh-Hadamard Transform - Bit-Packed Boolean Example
 *
 * Demonstrates computing the WHT of a Boolean function provided in a
 * bit-packed representation (uint64 words). Shows packing, transform,
 * and verification vs the unpacked convenience API.
 *
 * Copyright (C) 2025 Hosein Hadipour
 * License: GPL-3.0-or-later
 */

#include <fwht.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Pack uint8 truth table into uint64 bit-packed representation */
static void pack_truth_table(const uint8_t* truth_table, uint64_t* packed, size_t n) {
    size_t n_words = (n + 63) / 64;
    memset(packed, 0, n_words * sizeof(uint64_t));
    for (size_t i = 0; i < n; ++i) {
        if (truth_table[i]) {
            size_t word_idx = i / 64;
            size_t bit_idx = i % 64;
            packed[word_idx] |= (1ULL << bit_idx);
        }
    }
}

int main(void) {
    printf("============================================================\n");
    printf("Bit-Packed Boolean WHT Example\n");
    printf("============================================================\n\n");

    /* Example: f(x0,x1,x2) = x0 ⊕ x1 ⊕ x2 → truth table of length 8 */
    const size_t n = 8;
    uint8_t truth_table[8] = {0, 1, 1, 0, 1, 0, 0, 1};

    /* Compute reference WHT (unpacked helper) */
    int32_t wht_ref[n];
    fwht_status_t st = fwht_from_bool(truth_table, wht_ref, n, true);
    if (st != FWHT_SUCCESS) {
        fprintf(stderr, "fwht_from_bool failed: %s\n", fwht_error_string(st));
        return 1;
    }

    /* Pack to uint64 words and compute bit-packed WHT */
    uint64_t packed[(n + 63) / 64];
    pack_truth_table(truth_table, packed, n);

    int32_t wht_packed[n];
    st = fwht_boolean_packed(packed, wht_packed, n);
    if (st != FWHT_SUCCESS) {
        fprintf(stderr, "fwht_boolean_packed failed: %s\n", fwht_error_string(st));
        return 1;
    }

    /* Print and verify */
    printf("Truth table: ");
    for (size_t i = 0; i < n; ++i) printf("%u ", (unsigned)truth_table[i]);
    printf("\n\nWHT coefficients (ref vs packed):\n");
    int ok = 1;
    for (size_t i = 0; i < n; ++i) {
        printf("  W[%zu] = %4d  |  %4d\n", i, wht_ref[i], wht_packed[i]);
        if (wht_ref[i] != wht_packed[i]) ok = 0;
    }

    if (ok) {
        printf("\n✓ Packed and unpacked results match.\n");
    } else {
        printf("\n× Mismatch detected.\n");
        return 2;
    }

    /* Larger example: random function of size 256 */
    const size_t n2 = 256;
    uint8_t* tt = (uint8_t*)malloc(n2 * sizeof(uint8_t));
    if (!tt) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < n2; ++i) tt[i] = (uint8_t)(i * 2654435761u % 2);

    int32_t* wht2_ref = (int32_t*)malloc(n2 * sizeof(int32_t));
    int32_t* wht2_pk  = (int32_t*)malloc(n2 * sizeof(int32_t));
    uint64_t* pk2 = (uint64_t*)malloc(((n2 + 63) / 64) * sizeof(uint64_t));
    if (!wht2_ref || !wht2_pk || !pk2) { fprintf(stderr, "alloc failed\n"); return 1; }

    st = fwht_from_bool(tt, wht2_ref, n2, true);
    if (st != FWHT_SUCCESS) { fprintf(stderr, "from_bool failed\n"); return 1; }
    pack_truth_table(tt, pk2, n2);
    st = fwht_boolean_packed(pk2, wht2_pk, n2);
    if (st != FWHT_SUCCESS) { fprintf(stderr, "boolean_packed failed\n"); return 1; }

    for (size_t i = 0; i < n2; ++i) {
        if (wht2_ref[i] != wht2_pk[i]) {
            fprintf(stderr, "Mismatch at %zu: %d vs %d\n", i, wht2_ref[i], wht2_pk[i]);
            free(tt); free(wht2_ref); free(wht2_pk); free(pk2);
            return 2;
        }
    }
    printf("\n✓ Verified match for n=256 as well.\n");

    free(tt); free(wht2_ref); free(wht2_pk); free(pk2);
    printf("\nAll done.\n");
    return 0;
}
