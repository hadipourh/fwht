# API Design Analysis: Current vs Needed for Crypto

**Date:** 2025-11-16  
**Purpose:** Gap analysis between current libfwht API and crypto requirements from state-of-the-art papers

---

## Current State: What We Have ✅

### 1. Core Integer Transforms
- ✅ `fwht_i32(data, n)` — exact integer WHT (primary crypto use-case)
- ✅ `fwht_i32_safe(data, n)` — with overflow detection
- ✅ `fwht_i8(data, n)` — memory-efficient for small n
- ✅ `fwht_f64(data, n)` — floating-point variant

**Assessment:** Perfect foundation for Boolean crypto. Exact integer arithmetic is essential.

### 2. GPU Acceleration
- ✅ CUDA backend with `fwht_batch_i32_cuda(data, n, batch)`
- ✅ Persistent GPU context: `fwht_gpu_context_t` with pre-allocated memory
- ✅ GPU load/store callbacks for kernel fusion
- ✅ Profiling support: `fwht_gpu_metrics_t`

**Assessment:** Solid GPU foundation but missing crypto-specific optimizations.

### 3. Batch Processing
- ✅ `fwht_i32_batch(data_array, n, batch_size)` — SIMD vectorized (AVX2/NEON)
- ✅ GPU batch via persistent context
- ✅ 3-5× speedup for n ≤ 256 (perfect for S-boxes!)

**Assessment:** Excellent for crypto workloads (S-box analysis).

### 4. Boolean Utilities
- ✅ `fwht_from_bool(bool_func, wht_out, n, signed_rep)` — 0/1 → ±1 conversion
- ✅ `fwht_correlations(bool_func, corr_out, n)` — normalized correlations

**Assessment:** Basic Boolean support exists.

---

## Gaps: What's Missing Based on State-of-the-Art ❌

### Gap 1: Bit-Packed Boolean WHT (CRITICAL)

**What papers say:**
- Crypto applications use ±1 inputs exclusively
- Most efficient: bit-pack truth tables, use popcount for correlation sums
- Expected: 32-64× speedup vs unpacked

**What we have:**
- `fwht_from_bool` converts 0/1 array → ±1 → runs standard int32 WHT
- No bit-packed input API
- No popcount-based kernel

**What we need:**
```c
/* Bit-packed Boolean WHT (CPU) */
fwht_status_t fwht_bool_i32(const uint32_t* packed_truth_table, 
                             int32_t* spectrum, 
                             int n_vars);  // n = 2^n_vars

fwht_status_t fwht_bool_i64(const uint64_t* packed_truth_table,
                             int64_t* spectrum,
                             int n_vars);

/* GPU variant with bit-sliced execution */
fwht_status_t fwht_bool_i32_cuda(const uint32_t* packed_truth_table,
                                  int32_t* spectrum,
                                  int n_vars);
```

**Priority:** HIGH — This is the core crypto use-case.

---

### Gap 2: Crypto Metrics Utilities (HIGH PRIORITY)

**What papers say:**
- Nonlinearity = 2^(n-1) - 0.5 × max|Walsh| (Meier & Staffelbach)
- Correlation immunity order = largest m where Walsh[α]=0 for all wt(α)≤m (Siegenthaler)
- Bent property = all |Walsh| = 2^(n/2) (Rothaus)
- Perfect nonlinearity for S-boxes (Nyberg)

**What we have:**
- Nothing — user must write custom code to extract these metrics

**What we need:**
```c
typedef struct {
    int32_t max_walsh_abs;        /* max |W(α)| */
    int32_t walsh_abs_sum;         /* Σ |W(α)| */
    int nonlinearity;              /* 2^(n-1) - max_walsh_abs/2 */
    int correlation_immunity;      /* CI order */
    int algebraic_degree;          /* deg(f) if computable */
    bool is_bent;                  /* all |W| = 2^(n/2) */
    bool is_balanced;              /* W(0) = 0 */
} fwht_crypto_metrics_t;

/* Analyze single Boolean function */
fwht_crypto_metrics_t fwht_analyze_boolean(const int32_t* spectrum, int n_vars);

/* Analyze vectorial Boolean function (S-box) */
typedef struct {
    int n_vars;                    /* Input variables */
    int m_vars;                    /* Output variables */
    int min_nonlinearity;          /* min NL across components */
    int max_nonlinearity;          /* max NL across components */
    double avg_nonlinearity;       /* average NL */
    int differential_uniformity;   /* max δ for DDT */
    bool is_APN;                   /* δ = 2 (almost perfect nonlinear) */
} fwht_sbox_metrics_t;

fwht_sbox_metrics_t fwht_analyze_sbox(const uint8_t* sbox_lut, 
                                       int n_vars, int m_vars);
```

**Priority:** HIGH — Makes library actually useful for crypto researchers.

---

### Gap 3: GPU Bank-Conflict Tuning (MEDIUM PRIORITY)

**What Andrade 2014 says:**
- Bank conflicts dominate perf for shared-memory kernels
- Must tune radix to match GPU SMEM bank count (16 or 32)
- Auto-detect and select radix variant

**What we have:**
- Generic CUDA kernels (no bank-aware tuning)
- No device-specific radix selection

**What we need:**
```c
/* Internal: auto-tune CUDA kernels based on GPU architecture */
fwht_status_t fwht_gpu_auto_tune(fwht_gpu_context_t* ctx);

/* Query selected kernel variant */
const char* fwht_gpu_kernel_info(fwht_gpu_context_t* ctx);
```

**Priority:** MEDIUM — 1.5-2× perf gain for GPU int32 kernels.

---

### Gap 4: Warp-Shuffle Small-N Kernels (MEDIUM PRIORITY)

**What HadaCore says:**
- For N ≤ 1024: pure register butterfly using `__shfl_xor_sync()`
- Zero SMEM traffic, minimal latency
- Expected: 2-3× faster than SMEM-based

**What we have:**
- SMEM-based kernels for all sizes
- No warp-shuffle path

**What we need:**
```c
/* Internal: warp-shuffle kernel selection threshold */
#define FWHT_GPU_WARP_SHUFFLE_MAX_N 1024

/* Transparent: auto-select warp-shuffle for small N */
```

**Priority:** MEDIUM — Improves latency for small crypto transforms.

---

### Gap 5: Tensor Core Path (OPTIONAL for Crypto)

**What HadaCore + Markovic say:**
- FP16/BF16 via 16×16 MMA tiles
- 1.1-3.5× speedup on Ampere/Hopper
- Two algorithms: Cooley-Tukey, constant geometry

**What we have:**
- Nothing (no FP16/BF16, no Tensor Cores)

**What we need:**
```c
/* Tensor Core WHT (FP16/BF16 only) */
typedef enum {
    FWHT_TC_ALGO_COOLEY_TUKEY,
    FWHT_TC_ALGO_CONSTANT_GEOMETRY
} fwht_tc_algorithm_t;

fwht_status_t fwht_gpu_set_tensor_core(fwht_gpu_context_t* ctx, bool enable);
fwht_status_t fwht_gpu_set_tc_algorithm(fwht_gpu_context_t* ctx, 
                                         fwht_tc_algorithm_t algo);

/* FP16/BF16 types (requires CUDA __half) */
fwht_status_t fwht_f16_cuda(half* data, size_t n);
fwht_status_t fwht_bf16_cuda(nv_bfloat16* data, size_t n);
```

**Priority:** LOW — Not needed for exact Boolean crypto; only if you have float workloads.

---

## Proposed API Changes: Keep It Simple

### Principle: Minimal Surface, Maximum Utility

**DO NOT:**
- ❌ Add 10+ new top-level functions
- ❌ Expose internal tuning knobs to users
- ❌ Create separate Boolean vs integer vs float hierarchies

**DO:**
- ✅ Add 2-3 high-level crypto functions
- ✅ Auto-select optimal backend/kernel internally
- ✅ Keep existing API unchanged (backward compatible)

---

## Recommended New API (Simple & Clean)

### Add to `fwht.h`:

```c
/* ============================================================================
 * CRYPTOGRAPHY API
 * 
 * High-level functions for Boolean function analysis.
 * ============================================================================ */

/*
 * Compute Walsh spectrum of Boolean function from bit-packed truth table.
 * 
 * Parameters:
 *   truth_table - Packed truth table (ceil(2^n_vars / 32) uint32_t words)
 *   spectrum    - Output spectrum (2^n_vars int32_t values)
 *   n_vars      - Number of input variables (k) where truth table size = 2^k
 * 
 * Input format:
 *   Bit i of truth_table encodes f(i): 0 → +1, 1 → -1
 *   LSB of truth_table[0] = f(0), bit 1 = f(1), etc.
 * 
 * Output:
 *   spectrum[u] = Σ_{x=0}^{2^k-1} (-1)^{f(x) ⊕ u·x}
 * 
 * Performance:
 *   CPU: 10-20× faster than fwht_from_bool() due to bit-packing
 *   GPU: 32-64× faster via popcount-based correlation (when batch_size > 100)
 * 
 * Returns: FWHT_SUCCESS or error code
 */
fwht_status_t fwht_boolean(const uint32_t* truth_table,
                            int32_t* spectrum,
                            int n_vars);

/*
 * Analyze cryptographic properties of Boolean function.
 * 
 * Call after fwht_boolean() or fwht_i32() to extract crypto metrics.
 * 
 * Parameters:
 *   spectrum - Walsh spectrum (2^n_vars int32_t values)
 *   n_vars   - Number of input variables
 * 
 * Returns:
 *   Struct containing nonlinearity, correlation immunity order,
 *   bent/balanced properties, max Walsh coefficient, etc.
 */
typedef struct {
    int32_t max_walsh_abs;
    int nonlinearity;
    int correlation_immunity;
    bool is_bent;
    bool is_balanced;
} fwht_crypto_metrics_t;

fwht_crypto_metrics_t fwht_crypto_analyze(const int32_t* spectrum, int n_vars);

/*
 * Analyze vectorial Boolean function (S-box).
 * 
 * Computes WHT for all 2^m - 1 non-trivial component functions and
 * aggregates metrics (min/max/avg nonlinearity, differential uniformity).
 * 
 * Parameters:
 *   sbox   - S-box lookup table (2^n_vars entries, each m_vars bits)
 *   n_vars - Input size (k where input is k bits)
 *   m_vars - Output size (m where output is m bits)
 * 
 * Returns:
 *   Struct with min/max/avg nonlinearity, differential uniformity, APN flag
 */
typedef struct {
    int n_vars;
    int m_vars;
    int min_nonlinearity;
    int max_nonlinearity;
    double avg_nonlinearity;
    int differential_uniformity;
    bool is_APN;
} fwht_sbox_metrics_t;

fwht_sbox_metrics_t fwht_sbox_analyze(const uint8_t* sbox, 
                                       int n_vars, int m_vars);
```

### Internal Changes (No User-Facing API):

1. **GPU kernels:**
   - Add bank-conflict-aware radix selection
   - Add warp-shuffle path for N ≤ 1024
   - Auto-detect and select at runtime

2. **CPU:**
   - Add bit-packed Boolean kernel using popcount
   - Integrate with existing SIMD batch path

3. **Metrics:**
   - Implement crypto analysis functions
   - GPU reduction kernels for max|Walsh|

---

## Migration Path: No Breaking Changes

### Existing users:
- ✅ All current API functions remain unchanged
- ✅ Behavior identical (same results, same performance)
- ✅ No recompilation required

### New users (crypto):
- Use 3 new functions: `fwht_boolean()`, `fwht_crypto_analyze()`, `fwht_sbox_analyze()`
- Ignore everything else unless they need it

### Example workflow (new):
```c
#include <fwht.h>

int main(void) {
    /* Boolean function: n=8 variables */
    uint32_t truth_table[8] = {/* ... */};  /* 256 bits = 8 words */
    int32_t spectrum[256];
    
    /* Compute Walsh spectrum (auto-selects CPU/GPU) */
    fwht_boolean(truth_table, spectrum, 8);
    
    /* Analyze crypto properties */
    fwht_crypto_metrics_t metrics = fwht_crypto_analyze(spectrum, 8);
    
    printf("Nonlinearity: %d\n", metrics.nonlinearity);
    printf("CI order: %d\n", metrics.correlation_immunity);
    printf("Bent: %s\n", metrics.is_bent ? "yes" : "no");
    
    return 0;
}
```

Clean, simple, 10 lines of code.

---

## Summary: Gap Analysis

| Feature | Status | Priority | Impact |
|---------|--------|----------|--------|
| Exact int32 WHT | ✅ Have | — | Core crypto |
| SIMD batch | ✅ Have | — | 3-5× for S-boxes |
| GPU persistent context | ✅ Have | — | 5-10× for repeated |
| **Bit-packed Boolean** | ❌ Missing | **HIGH** | 10-64× speedup |
| **Crypto metrics** | ❌ Missing | **HIGH** | Usability for crypto |
| Bank-conflict tuning | ❌ Missing | MEDIUM | 1.5-2× GPU int32 |
| Warp-shuffle small-N | ❌ Missing | MEDIUM | 2-3× latency |
| Tensor Cores FP16/BF16 | ❌ Missing | LOW | Not for crypto |

---

## Recommended Implementation Order

1. **fwht_boolean()** — Bit-packed Boolean WHT (CPU + GPU)
   - Highest crypto impact
   - Reuses existing infrastructure
   - 10-64× speedup for primary use-case

2. **fwht_crypto_analyze()** + **fwht_sbox_analyze()** — Metrics utilities
   - Makes library useful for crypto researchers
   - Small implementation (1-2 days)
   - Immediate usability gain

3. **GPU bank-conflict tuning** — Andrade 2014 insights
   - 1.5-2× GPU int32 performance
   - No API changes (internal optimization)
   - Benefits existing users too

4. **Warp-shuffle small-N** — HadaCore insights
   - 2-3× latency for small transforms
   - No API changes (auto-selected)
   - Complements bank-conflict work

5. **Tensor Cores** (optional) — HadaCore + Markovic
   - Only if float workloads exist
   - Skip for pure crypto focus
   - Large implementation effort

---

## Final API: 3 New Functions, Zero Breaking Changes

```c
/* New crypto API (3 functions) */
fwht_status_t fwht_boolean(const uint32_t* truth_table, int32_t* spectrum, int n_vars);
fwht_crypto_metrics_t fwht_crypto_analyze(const int32_t* spectrum, int n_vars);
fwht_sbox_metrics_t fwht_sbox_analyze(const uint8_t* sbox, int n_vars, int m_vars);

/* All existing functions unchanged */
fwht_status_t fwht_i32(int32_t* data, size_t n);  /* Still works */
/* ... 20+ other functions still work exactly the same ... */
```

**Result:** Clean, simple, crypto-focused without API bloat.
