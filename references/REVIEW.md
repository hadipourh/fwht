# Reference Review and Actionable Insights

**Review date:** 2025-11-15  
**Focus:** Cryptographic WHT for Boolean/vectorial Boolean functions

---

## Summary: Relevance Assessment

### GPU Implementation Papers (3 entries) — HIGHLY RELEVANT ✓

All three GPU papers directly address WHT/FWHT performance optimization on NVIDIA hardware:

1. **Agarwal et al. (2024, arXiv) — HadaCore**
   - **Relevance:** HIGH — Tensor Core acceleration via 16×16 MMA blocks
   - **Application:** FP16/BF16 WHT; not directly for Boolean, but architectural insights transfer
   - **What we learn:** Warp shuffles, shared-memory transposes, scaling beyond 256, in-place optimization

2. **Markovic et al. (2025, JRTIP) — Tensor Cores + algorithmic variants**
   - **Relevance:** HIGH — Compares Cooley–Tukey vs constant geometry on Tensor Cores
   - **Application:** WHT spectra computation; constant geometry reported faster
   - **What we learn:** Multiple algorithmic paths can coexist; need auto-selection by size/type

3. **Andrade et al. (2014, Parallel Computing) — Bank-conflict tuning**
   - **Relevance:** HIGH — Shared-memory bank conflicts; radix-n tuning for 16/32-bank GPUs
   - **Application:** Non-Tensor-Core integer/float paths
   - **What we learn:** Radix scheduling must align with SMEM bank count; auto-detect and adapt

### Cryptography Papers (4 entries) — CONTEXT ONLY ⚠️

These define *why* WHT matters in crypto but offer **no implementation guidance**:

4. **Rothaus (1976) — Bent functions**
   - **Relevance:** CONTEXT — Defines bent functions via Walsh spectrum properties
   - **Application:** Validates that WHT is the core tool for computing nonlinearity
   - **What we learn:** User wants max |Walsh coefficient|; we must support exact integer WHT for ±1 inputs

5. **Siegenthaler (1984) — Correlation immunity**
   - **Relevance:** CONTEXT — Correlation immunity = specific Walsh spectrum zero patterns
   - **Application:** Users check Walsh[α] = 0 for certain α to verify CI order
   - **What we learn:** Need efficient batch WHT for checking many α positions

6. **Meier & Staffelbach (1990) — Nonlinearity criteria**
   - **Relevance:** CONTEXT — Nonlinearity = 2^(n-1) - 0.5 * max|Walsh|
   - **Application:** Main crypto metric derived from WHT output
   - **What we learn:** User needs max-abs reduction after WHT; we should provide utility function

7. **Nyberg (1991) — Perfect nonlinear S-boxes**
   - **Relevance:** CONTEXT — Vectorial Boolean functions; PN property via component Walsh spectra
   - **Application:** Batch WHT across all component functions of an S-box
   - **What we learn:** Support batched transforms for vectorial functions (m outputs, each n→1)

### Recommendation: REMOVE cryptography theory papers from papers.bib

**Rationale:**
- They provide **mathematical context** but **zero algorithmic or implementation details**.
- Users who need these already know them; users who don't need them won't benefit from citations in a performance library.
- Keeping them clutters the bibliography and implies implementation relevance where none exists.

**Action:** Move Rothaus, Siegenthaler, Meier & Staffelbach, Nyberg to a separate `references/crypto-background.md` file labeled "Background on WHT in Cryptography" so interested users can see *why* Boolean WHT matters, but it won't be confused with implementation references.

---

## What We Learn: Concrete Library Improvements

### From GPU Papers

#### 1. Tensor Core Path (HadaCore + Markovic)

**What to implement:**
- FP16/BF16 WHT kernels using 16×16 MMA tiles (WMMA API or inline PTX)
- Two algorithmic variants:
  - **Cooley–Tukey decomposition** (HadaCore's transpose-based scaling)
  - **Constant geometry** (Markovic reports this is faster; need to investigate exact tiling)
- Size handling:
  - ≤256: in-register via two MMA ops + transpose
  - 512–2048: warp shuffles for coalescing/register remapping
  - ≥4096: shared-memory staging + transposes
- Non-power-of-16: factorize d = 2^m · 16^n; H16 iterations + final 2^m diagonal tiling
- BF16 mode: accumulate in FP32, convert back to BF16

**API additions:**
```c
// Opt-in Tensor Core backend
void fwht_gpu_set_tensor_core_mode(fwht_gpu_context *ctx, int enable);
void fwht_gpu_set_tensor_core_algorithm(fwht_gpu_context *ctx, 
                                         fwht_tc_algorithm alg); // COOLEY_TUKEY | CONSTANT_GEOMETRY
```

**Testing:**
- Validate precision: compare FP16/BF16 results vs FP64 reference; measure max relative error
- Benchmark vs current float kernels; expect 1.1–3.5× speedup on A100/H100 for sizes ≥256

**Constraints:**
- Tensor Cores require Ampere+ (sm_80+)
- FP16/BF16 only (not for exact integer crypto; those stay on INT32 path)
- Occupancy tuning: warps-per-block, chunks-per-warp

#### 2. Bank-Conflict-Aware Non-TC Path (Andrade)

**What to implement:**
- Auto-detect GPU shared-memory bank count (16 or 32) via device query
- Radix scheduling: choose radix-16 stages for 16-bank GPUs, radix-32 for 32-bank GPUs
- Shared-memory layout: stride accesses to align with bank count and minimize conflicts
- Thread-to-element mapping: ensure coalesced global loads and conflict-free SMEM access

**API additions:**
```c
// Auto-tune kernel selection by GPU architecture
void fwht_gpu_auto_tune(fwht_gpu_context *ctx); // queries device, sets radix/block size
```

**Testing:**
- Profile SMEM bank conflicts using `nvprof --metrics shared_load_transactions_per_request`
- Measure achieved throughput vs peak theoretical (should be >80% for well-tuned kernels)

**Constraints:**
- Applies to INT32 and non-Tensor-Core float paths
- May need separate kernel templates per radix variant; use C preprocessor or CUDA templates

#### 3. Warp-Level Small-N Kernels (HadaCore insight)

**What to implement:**
- For N ≤ 1024 (fits in warp registers): pure warp-shuffle butterfly with no SMEM
- Thread i holds elements at positions matching lane ID; shuffle to exchange pairs/quads/octets across butterfly stages
- Single warp processes one transform; threadblock = multiple independent warps for batching

**API:**
- Transparent; auto-selected when N ≤ 1024 and batch size allows occupancy

**Testing:**
- Validate correctness vs reference
- Benchmark latency and throughput; expect near-peak for small N

**Constraints:**
- Register pressure: ensure no spills (use `--ptxas-options=-v` to check)
- Limited to sizes fitting in 32 registers per thread (≤1024 for int32; ≤512 for int64)

---

### From Crypto Papers (Context → API Design)

#### 4. Exact Integer WHT for ±1 Inputs

**What crypto papers tell us:**
- Walsh spectrum entries are sums of ±1 products → range is [-2^n, +2^n]
- For n=16: max magnitude 65536 (fits int32); n=32: max 2^32 (need int64)
- Users need **exact** results for Boolean analysis (no rounding errors)

**What to implement:**
- Dedicated `fwht_boolean()` API: input is bit-packed uint32/uint64, output is int32/int64 spectrum
- Bit-sliced execution: use `__popc()` to compute XOR-popcount for correlation sums
- Warp-level reduction for batch processing of many Boolean functions in parallel

**API:**
```c
// Boolean-specific WHT (±1 encoding, exact integer output)
void fwht_boolean_i32(const uint32_t *truth_table_packed, int n, int32_t *spectrum);
void fwht_boolean_i64(const uint64_t *truth_table_packed, int n, int64_t *spectrum);

// Batch variant for vectorial Boolean functions (S-box analysis)
void fwht_boolean_batch(const uint32_t **truth_tables, int n, int m, int32_t **spectra);
```

**Testing:**
- Validate against known bent functions (max|Walsh| = 2^(n/2) for all entries)
- Check correlation-immune functions (specific Walsh entries = 0)

#### 5. Utility Functions for Crypto Metrics

**What crypto papers tell us:**
- Nonlinearity = 2^(n-1) - 0.5 * max|Walsh|
- Correlation immunity order = largest m such that Walsh[α] = 0 for all wt(α) ≤ m
- Bent property = all |Walsh| = 2^(n/2)

**What to implement:**
- Post-WHT metric extraction functions (CPU-side or GPU reduction kernels)

**API:**
```c
typedef struct {
    int32_t max_walsh_abs;       // max |W(α)|
    int nonlinearity;             // 2^(n-1) - 0.5*max_walsh_abs
    int correlation_immunity;     // CI order
    bool is_bent;                 // all |W| equal
    bool is_balanced;             // W(0) == 0
} fwht_crypto_metrics;

fwht_crypto_metrics fwht_analyze_boolean(const int32_t *spectrum, int n);
```

**Testing:**
- Verify against published examples from Rothaus, Siegenthaler, etc.

---

## Action Plan

### Immediate (keep in papers.bib):
1. **agarwal2024hadacore** ✓
2. **markovic2025walshHadamardTensorCores** ✓
3. **andrade2014optimizedFWHTGPU** ✓

### Move to crypto-background.md:
4. **rothaus1976bent** → background context
5. **siegenthaler1984correlationImmunity** → background context
6. **meier1990nonlinearity** → background context
7. **nyberg1991perfect** → background context

### Implement (priority order):
1. **Bit-sliced Boolean WHT** (INT32/INT64, bit-packed input, popcount-based)
   - Addresses core crypto use-case
   - No FP imprecision; exact results
   - Enables all Rothaus/Siegenthaler/Meier metric computations

2. **Bank-conflict-aware INT32 kernels** (Andrade insights)
   - Tune existing kernels for 16/32-bank GPUs
   - Auto-detect and select radix variant
   - Immediate perf gain with no API changes

3. **Tensor Core FP16/BF16 path** (HadaCore + Markovic)
   - For users with FP workloads (not primary crypto use-case but valuable)
   - Implement both Cooley–Tukey and constant geometry; benchmark and auto-select
   - Requires sm_80+ runtime gating

4. **Warp-shuffle small-N kernels** (HadaCore insight)
   - Pure register-based for N ≤ 1024
   - Complements existing shared-memory kernels

5. **Crypto metric utilities** (derive from all 4 crypto papers)
   - Nonlinearity, CI order, bent/balanced checks
   - GPU reductions for max|Walsh| and zero-pattern checks
   - Example code showing S-box analysis workflow

---

## Conclusion

**GPU papers (3):** All highly relevant; provide concrete algorithmic and architectural optimizations.

**Crypto papers (4):** Provide context but no implementation details. Should be moved to a "background" document to avoid cluttering the implementation-focused bibliography.

**What we can learn:**
- **Tensor Cores:** 1.1–3.5× speedup for FP16/BF16; need 16×16 tiling + transpose scaling
- **Bank conflicts:** Radix must match SMEM bank count; auto-detect and tune
- **Boolean WHT:** Bit-packing + popcount for exact ±1 transforms; primary crypto use-case
- **Metrics:** Provide post-WHT analysis functions for nonlinearity, CI, bent properties

**Next steps:**
1. Clean papers.bib (keep 3 GPU papers)
2. Create crypto-background.md (move 4 theory papers)
3. Prototype Boolean bit-sliced kernel
4. Add bank-conflict tuning to existing INT32 kernels
5. Implement Tensor Core path (FP16/BF16)
