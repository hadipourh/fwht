# WHT Time Complexity: Theory and Practice

## Time Complexity: Θ(n · 2^n) — Provably Optimal

The **Fast Walsh-Hadamard Transform (FWHT)** is already optimal in terms of time complexity.

### Why This is Optimal

**Lower bound argument:**
- The Walsh spectrum has **2^n entries** (output size)
- Each entry requires examining all 2^n input points (by definition of the Walsh transform)
- Therefore, Ω(n · 2^n) is a lower bound

**What FWHT achieves:**
- **O(n · 2^n)** operations via the butterfly structure
- This matches the lower bound → **optimal**

**Naïve approach would be:**
- Direct computation: O(2^n · 2^n) = O(4^n) — exponentially worse
- Compute each of 2^n spectrum entries independently, each requiring 2^n operations

---

## Algorithm Variants

All variants have **Θ(n · 2^n)** complexity but differ in constants and memory patterns.

### 1. Cooley-Tukey Decimation (Standard FWHT)

**Characteristics:**
- **Memory:** In-place (O(2^n))
- **Cache behavior:** Poor for large n (random access patterns)
- **Operations:** n stages, each doing 2^n butterfly ops
- **Status:** What we currently use in libfwht

**Butterfly structure:**
```
for stage s = 0 to n-1:
    stride = 2^s
    for each pair (i, i+stride):
        (a, b) = (data[i], data[i+stride])
        data[i] = a + b
        data[i+stride] = a - b
```

### 2. Stockham Autosort

**Characteristics:**
- **Memory:** Out-of-place (requires 2 × 2^n buffer)
- **Cache behavior:** Excellent (sequential access, write-combine friendly)
- **Operations:** Same n · 2^n, but better coalescing on GPU
- **Status:** Planned for large-N GPU optimization (see roadmap)

**Key advantage:**
- Reads are sequential within each stage
- Writes are sequential within each stage
- Better for GPU global memory coalescing

### 3. Constant Geometry (Markovic 2025)

**Characteristics:**
- **Memory:** In-place or hybrid
- **Cache behavior:** Better than Cooley-Tukey for certain sizes
- **Operations:** Same n · 2^n, rearranged tiling
- **Key insight:** Maintains data layout to improve cache reuse
- **Status:** From our references; reported faster than Cooley-Tukey on Tensor Cores

**Why it helps:**
- Reduces data movement between memory levels
- Better spatial/temporal locality
- Maps well to hierarchical GPU memory (registers → SMEM → global)

### 4. Recursive Divide-and-Conquer

**Characteristics:**
- **Memory:** O(2^n), recursive stack O(n)
- **Cache behavior:** Depends on recursion order
- **Operations:** Same n · 2^n asymptotically
- **Status:** Rarely used; iterative Cooley-Tukey is simpler

**Recursive structure:**
```
WHT(data, n):
    if n == 1:
        (a, b) = (data[0], data[1])
        data[0] = a + b
        data[1] = a - b
    else:
        WHT(data[0:2^(n-1)], n-1)    // first half
        WHT(data[2^(n-1):2^n], n-1)  // second half
        merge(data, n)               // combine results
```

---

## Key Insight from Our References

**None of the GPU papers change the asymptotic complexity.** They all do Θ(n · 2^n) operations.

### What They Optimize

1. **HadaCore (Agarwal et al., 2024):**
   - Memory access patterns (coalescing, alignment)
   - Hardware utilization (Tensor Cores via 16×16 MMA)
   - Register/SMEM/global memory hierarchy exploitation

2. **Markovic et al. (2025):**
   - Data layout (constant geometry vs Cooley-Tukey)
   - Tiling strategies for Tensor Cores
   - Algorithmic variant selection

3. **Andrade et al. (2014):**
   - Shared-memory bank conflicts
   - Radix scheduling (radix-16 vs radix-32)
   - Architecture-specific tuning (16-bank vs 32-bank GPUs)

### Where Speedups Come From

**Not from reducing Θ(n · 2^n) — that's already optimal.**

**Speedups come from optimizing constants:**

1. **Memory latency reduction:**
   - Eliminate bank conflicts (Andrade): 1.5–2× gain
   - Coalesced global memory access (Stockham): 1.2–1.5× gain
   - Cache-friendly layouts (constant geometry): 1.1–1.3× gain

2. **Hardware utilization:**
   - Tensor Cores (HadaCore): 1.1–3.5× gain for FP16/BF16
   - Warp shuffles vs SMEM (small-N): 2–3× gain
   - SIMD/vectorization (CPU): 4–8× gain (SSE/AVX)

3. **Parallelism:**
   - GPU vs CPU: 10–100× for large batches
   - Multi-core CPU: 4–16× (depends on core count)

---

## Practical Implications for libfwht

### What We Can't Improve
- **Asymptotic complexity:** Already Θ(n · 2^n), which is optimal
- **Number of arithmetic operations:** n · 2^n butterflies is fundamental

### What We Can Improve (and Are Improving)

1. **Memory system efficiency:**
   - Bank-conflict-aware kernels (Andrade insights)
   - Stockham for large-N coalescing
   - Constant geometry for cache reuse

2. **Hardware-specific optimization:**
   - Tensor Cores for FP16/BF16 (HadaCore approach)
   - Warp shuffles for small-N (register-only execution)
   - Bit-sliced for Boolean inputs (popcount-based correlation)

3. **Workload-specific paths:**
   - Exact integer for crypto (no rounding errors)
   - FP16/BF16 for ML/signal processing (speed over precision)
   - Batched for vectorial Boolean functions (S-box analysis)

---

## Complexity Summary Table

| Algorithm | Time | Space | Cache | GPU-Friendly | Status in libfwht |
|-----------|------|-------|-------|--------------|-------------------|
| Cooley-Tukey | Θ(n·2^n) | O(2^n) in-place | Poor | Moderate | ✅ Current default |
| Stockham | Θ(n·2^n) | O(2·2^n) out-of-place | Excellent | Very good | 🔄 Planned (roadmap) |
| Constant Geometry | Θ(n·2^n) | O(2^n) hybrid | Good | Very good | 🔄 Planned (from Markovic) |
| Recursive D&C | Θ(n·2^n) | O(2^n + n) | Varies | Poor | ❌ Not planned |
| Naïve (direct) | Θ(4^n) | O(2^n) | N/A | Poor | ❌ Never (too slow) |

---

## Bottom Line

### For Time Complexity
**FWHT at Θ(n · 2^n) is already optimal; no algorithm can do better asymptotically.**

### For Wall-Clock Time
**Constant factors matter enormously:**
- CPU: Cache-oblivious layouts, SIMD vectorization (4–16× over naïve)
- GPU: Coalesced access, bank-conflict elimination, Tensor Cores (10–100× over CPU)

### For libfwht
**We're implementing the theoretically optimal algorithm (FWHT). All our GPU work is about optimizing the constants — and that's where real 10–100× speedups come from.**

The optimization journey:
1. ✅ **Correct algorithm** (FWHT) → Θ(n·2^n) vs Θ(4^n) naïve
2. 🔄 **Memory patterns** (bank conflicts, coalescing) → 2–4× gain
3. 🔄 **Hardware features** (Tensor Cores, shuffles) → 2–10× gain
4. 🔄 **Workload specialization** (bit-sliced, batched) → 10–100× for specific cases

**Total potential:** ~1000× over naïve single-threaded implementation, while staying at the same Θ(n·2^n) complexity.
