/*
 * Fast Walsh-Hadamard Transform - Basic Example
 * 
 * Demonstrates computing WHT of a simple Boolean function
 * and finding its best linear approximation.
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

#include <fwht.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("=================================================================\n");
    printf("FWHT Library - Basic Example\n");
    printf("=================================================================\n\n");
    
    /* Example 1: Linear Function f(x) = x_0 (first bit) */
    printf("Example 1: Linear function f(x) = x_0\n");
    printf("-----------------------------------------------------------------\n");
    
    uint8_t bool_func[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    int32_t wht[8];
    
    /* Compute WHT with signed representation */
    fwht_status_t status = fwht_from_bool(bool_func, wht, 8, true);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "Error: %s\n", fwht_error_string(status));
        return 1;
    }
    
    printf("Boolean function: ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", bool_func[i]);
    }
    printf("\n\n");
    
    printf("WHT coefficients:\n");
    for (int u = 0; u < 8; u++) {
        printf("  WHT[%d] = %5d", u, wht[u]);
        if (wht[u] != 0) {
            double corr = (double)wht[u] / 8.0;
            printf("  (correlation: %+.3f)", corr);
        }
        printf("\n");
    }
    
    printf("\n✓ Perfect correlation with u=1 (Cor = 1.0)\n");
    printf("  This confirms f(x) = x_0\n\n");
    
    /* Example 2: Arbitrary Boolean Function */
    printf("Example 2: Finding best linear approximation\n");
    printf("-----------------------------------------------------------------\n");
    
    uint8_t f[16] = {0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0};
    double correlations[16];
    
    status = fwht_correlations(f, correlations, 16);
    if (status != FWHT_SUCCESS) {
        fprintf(stderr, "Error: %s\n", fwht_error_string(status));
        return 1;
    }
    
    printf("Boolean function (hex): ");
    for (int i = 0; i < 16; i++) {
        printf("%x", f[i]);
    }
    printf("\n\n");
    
    /* Find maximum absolute correlation */
    int best_u = 0;
    double best_corr = 0.0;
    for (int u = 0; u < 16; u++) {
        if (fabs(correlations[u]) > fabs(best_corr)) {
            best_corr = correlations[u];
            best_u = u;
        }
    }
    
    printf("Best linear approximation:\n");
    printf("  Mask u = %d (binary: %04b)\n", best_u, best_u);
    printf("  Correlation: %+.6f\n", best_corr);
    printf("  WHT coefficient: %+.1f\n", best_corr * 16.0);
    
    if (fabs(best_corr) == 1.0) {
        printf("\n✓ Perfect correlation! f is a linear function.\n");
    } else if (fabs(best_corr) > 0.5) {
        printf("\n✓ Strong correlation. Good linear approximation.\n");
    } else {
        printf("\n  Moderate correlation. Nonlinear function.\n");
    }
    
    printf("\n");
    
    /* Example 3: Performance Demonstration */
    printf("Example 3: Performance on larger functions\n");
    printf("-----------------------------------------------------------------\n");
    
    size_t sizes[] = {256, 1024, 4096, 16384};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Computing WHT for various sizes:\n\n");
    
    for (int i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];
        int32_t* data = (int32_t*)malloc(n * sizeof(int32_t));
        if (data == NULL) {
            fprintf(stderr, "Allocation failed\n");
            return 1;
        }
        
        /* Fill with test pattern */
        for (size_t j = 0; j < n; j++) {
            data[j] = (j % 3 == 0) ? 1 : -1;
        }
        
        /* Compute WHT */
        status = fwht_i32(data, n);
        if (status != FWHT_SUCCESS) {
            fprintf(stderr, "Error at n=%zu: %s\n", n, fwht_error_string(status));
            free(data);
            return 1;
        }
        
        printf("  n = %5zu: ", n);
        printf("✓ WHT computed successfully\n");
        
        free(data);
    }
    
    printf("\n");
    printf("=================================================================\n");
    printf("All examples completed successfully!\n");
    printf("=================================================================\n");
    
    return 0;
}
