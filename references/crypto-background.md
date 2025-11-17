# Background: WHT in Cryptography

These papers establish the theoretical foundation for why Walsh-Hadamard Transform is central to Boolean function analysis in cryptography. They define key properties (bent functions, correlation immunity, nonlinearity, perfect nonlinearity) that are **computed via WHT**, but they do not provide implementation algorithms.

**For implementation guidance, see the GPU optimization papers in `papers.bib`.**

---

## Foundational Papers

### Rothaus (1976) — Bent Functions

**Citation:**
```bibtex
@article{rothaus1976bent,
	title   = {On "Bent" Functions},
	author  = {Rothaus, Oscar S.},
	journal = {Journal of Combinatorial Theory, Series A},
	volume  = {20},
	number  = {3},
	pages   = {300--305},
	year    = {1976},
	doi     = {10.1016/0097-3165(76)90024-8}
}
```

**What it defines:**
- A Boolean function f: {0,1}^n → {0,1} is **bent** if its Walsh spectrum has constant absolute value 2^(n/2) for all α.
- Bent functions are maximally nonlinear: they are equidistant from all affine functions.
- Connection to Hadamard matrices: Walsh transform of (−1)^f(x) is related to Hadamard matrix properties.

**Why it matters for crypto:**
- Bent functions have perfect resistance to linear approximation attacks.
- Computing the Walsh spectrum is the **only practical way** to verify bent property.

**Not covered:** How to compute the Walsh transform efficiently.

---

### Siegenthaler (1984) — Correlation Immunity

**Citation:**
```bibtex
@article{siegenthaler1984correlationImmunity,
	title   = {Correlation-Immunity of Nonlinear Combining Functions for Cryptographic Applications},
	author  = {Siegenthaler, Thomas},
	journal = {IEEE Transactions on Information Theory},
	volume  = {30},
	number  = {5},
	pages   = {776--780},
	year    = {1984},
	doi     = {10.1109/TIT.1984.1056949}
}
```

**What it defines:**
- A Boolean function is **m-th order correlation immune** if every subset of ≤m input variables is statistically independent of the output.
- Characterized via Walsh spectrum: CI of order m ⟺ Walsh[α] = 0 for all α with Hamming weight ≤m.
- Trade-off: m + deg(f) ≤ n for n-variable function; balanced functions have m + deg(f) ≤ n−1.

**Why it matters for crypto:**
- Stream ciphers using correlation-immune combining functions resist correlation attacks.
- Walsh spectrum is the **only efficient tool** to determine CI order.

**Not covered:** How to compute or optimize the Walsh transform.

---

### Meier & Staffelbach (1990) — Nonlinearity Criteria

**Citation:**
```bibtex
@inproceedings{meier1990nonlinearity,
	title        = {Nonlinearity Criteria for Cryptographic Functions},
	author       = {Meier, Willi and Staffelbach, Othmar},
	booktitle    = {Advances in Cryptology -- EUROCRYPT'89},
	series       = {Lecture Notes in Computer Science},
	year         = {1990},
	doi          = {10.1007/3-540-46885-4_53}
}
```

**What it defines:**
- **Nonlinearity** of a Boolean function = minimum Hamming distance to all affine functions.
- Computed from Walsh spectrum: nl(f) = 2^(n−1) − 0.5 × max|Walsh[α]|.
- Two key criteria:
  1. Distance to affine functions (measured via Walsh spectrum)
  2. Distance to linear structures (measured via derivatives)
- Both criteria are invariant under affine transformations.

**Why it matters for crypto:**
- High nonlinearity is essential for resistance to linear cryptanalysis.
- Walsh spectrum provides the **direct measure** of cryptographic strength.

**Not covered:** Algorithmic details of computing the Walsh transform.

---

### Nyberg (1991) — Perfect Nonlinear S-boxes

**Citation:**
```bibtex
@inproceedings{nyberg1991perfect,
	title        = {Perfect Nonlinear S-boxes},
	author       = {Nyberg, Kaisa},
	booktitle    = {Advances in Cryptology -- EUROCRYPT'91},
	year         = {1991},
	doi          = {10.1007/3-540-46416-6_32}
}
```

**What it defines:**
- A **vectorial Boolean function** (S-box) F: {0,1}^n → {0,1}^m is **perfect nonlinear (PN)** if all its directional derivatives are evenly distributed.
- For PN property, need n ≥ 2m.
- Analysis via **component Boolean functions**: for each linear combination of output bits, compute Walsh spectrum and check properties.
- PN S-boxes have optimal resistance to differential cryptanalysis.

**Why it matters for crypto:**
- S-box design requires analyzing m component Boolean functions (each n→1).
- Each component needs its Walsh spectrum computed → **batched WHT** over all 2^m − 1 non-trivial components.

**Not covered:** How to implement fast batched Walsh transforms.

---

## Summary: What These Papers Tell Us

### They define **what** to compute:
1. **Bent property:** Check if all |Walsh[α]| = 2^(n/2)
2. **Correlation immunity:** Find largest m where Walsh[α] = 0 for all wt(α) ≤ m
3. **Nonlinearity:** Compute 2^(n−1) − 0.5 × max|Walsh[α]|
4. **Vectorial PN property:** Compute Walsh spectra of all component functions

### They do **not** tell us:
- How to compute the Walsh transform efficiently (→ see GPU papers in `papers.bib`)
- How to optimize for GPU architectures (→ Tensor Cores, bank conflicts, warp shuffles)
- How to handle different data types (int32, int64, fp16, bf16)
- How to batch transforms or process multiple functions in parallel

### What we need to implement (derived from these definitions):

**API for exact Boolean WHT:**
```c
// Input: truth table as packed bits; Output: exact integer Walsh spectrum
void fwht_boolean_i32(const uint32_t *truth_table, int n, int32_t *spectrum);
void fwht_boolean_i64(const uint64_t *truth_table, int n, int64_t *spectrum);
```

**API for crypto metrics:**
```c
typedef struct {
    int32_t max_walsh_abs;
    int nonlinearity;
    int correlation_immunity_order;
    bool is_bent;
    bool is_balanced;
} fwht_crypto_metrics;

fwht_crypto_metrics fwht_analyze_boolean(const int32_t *spectrum, int n);
```

**API for vectorial functions (S-boxes):**
```c
// Batch WHT across all 2^m - 1 component Boolean functions
void fwht_vectorial_analyze(const uint32_t *sbox_lut, int n, int m,
                             fwht_crypto_metrics *component_metrics);
```

---

## Where to Go for Implementation

For **how** to compute WHT efficiently on GPUs:
- See `papers.bib` for HadaCore (Tensor Cores), Markovic (algorithmic variants), Andrade (bank conflicts)
- See `docs/gpu_wht_optimization_roadmap.md` for our implementation plan
- See `references/REVIEW.md` for actionable insights derived from GPU papers

These theory papers provide **validation targets** for correctness testing, not implementation guidance.
