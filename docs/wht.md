# Walsh-Hadamard Background

This note gives a practical but precise explanation of the Walsh-Hadamard transform (WHT) and the implementation ideas used inside libfwht. Definitions for technical terms appear inline so newcomers can follow along without a separate glossary.

---

## 1. Intuition: Mixing With ±1 Patterns

The WHT takes an input vector of length `n = 2^k` (k is a non-negative integer) and correlates it with every possible pattern made solely of +1 and −1 values.

- **Walsh basis:** The complete set of +1/−1 patterns. “Basis” means any vector can be written as a weighted sum of these patterns.
- **Walsh spectrum:** The list of all correlation weights (one per basis pattern). Large magnitude values indicate the input strongly resembles that pattern. A sign flip simply means the pattern needs to be inverted.

Formally, if we view the input as a function `f : {0,1}^k → R`, the (unnormalized) Walsh coefficient for pattern `u ∈ {0,1}^k` is

```
W(u) = sum_{x ∈ {0,1}^k} f(x) · (-1)^{u·x}
```

where `u·x` is the bitwise dot product modulo 2 (XOR of pairwise ANDs). This is the version implemented by libfwht; other texts sometimes divide the right-hand side by `n = 2^k` (orthogonal normalization) or by `sqrt(n)` (orthonormal normalization).

When working with Boolean truth tables we map `0 → +1` and `1 → −1` before running the transform. This keeps the spectrum centered around zero, which makes cryptographic metrics such as nonlinearity easier to read.

---

## 2. Matrix View and a Tiny Example

The order-1 Hadamard matrix is:

```
H2 = [[ 1,  1],
	  [ 1, -1]]
```

Larger matrices follow the recursion

```
H_{2n} = [[H_n,  H_n],
		  [H_n, -H_n]]
```

For example, the order-4 matrix is:

```
H4 = [[ 1,  1,  1,  1],
	  [ 1, -1,  1, -1],
	  [ 1,  1, -1, -1],
	  [ 1, -1, -1,  1]]
```

Multiplying `H4` by an input `[a0, a1, a2, a3]` produces the four Walsh coefficients. Naïvely multiplying an `n × n` matrix by a vector costs `n^2` operations, which becomes `4^k` when `n = 2^k`—far too slow beyond tiny inputs.

---

## 3. Fast Walsh-Hadamard Transform (FWHT)

FWHT reorganizes the same math into `k` stages of **butterflies**. A butterfly is the two-point map `(x, y) → (x + y, x − y)`. Every stage doubles the distance between the partners:

```
Stage 0: stride 1  → combine neighbors
Stage 1: stride 2  → combine pairs two apart
...
Stage k-1: stride 2^(k-1) → combine the two halves
```

Key points:
- **Operation count:** Exactly `k · 2^k` additions/subtractions, which equals `n · log2(n)` with `n = 2^k`. This matches the lower bound because we must touch every element in each stage.
- **Recursive form:** One step of the FWHT can be written as `FWHT_{2n}(a, b) = (FWHT_n(a + b), FWHT_n(a − b))`, which is just the matrix recursion expressed over vectors `a` (first half) and `b` (second half).
- **Power-of-two requirement:** Each stage splits the data into two equal halves, so `n` must stay divisible by two all the way down. libfwht checks this via `fwht_is_power_of_2()` before launching a transform.
- **Normalization:** libfwht leaves the spectrum unscaled by default. Users can divide by `n` to obtain an orthogonal transform or by `sqrt(n)` to obtain an orthonormal transform `H_n / sqrt(n)`.

Because every stage performs `n` additions/subtractions, the total work satisfies the recurrence `T(n) = 2 · T(n/2) + n`, whose solution is `T(n) = n log2 n`.

---

## 4. Data Types and Numerical Intent

- **`int32` (fast default):** Exact arithmetic as long as intermediate sums stay in range. Suits Boolean and cryptographic workloads where determinism matters.
- **`int32` safe mode:** Internally widens each sum to 64 bits to detect overflow. Slightly slower but guarantees you will see an error instead of wrapping.
- **`float64`:** Doubles the dynamic range and keeps fractional terms, useful for probability distributions or filtering pipelines.
- **`float16` / `bfloat16`:** GPU-only mixed-precision modes that trade a bit of accuracy for large throughput gains, especially on Tensor Cores.

---

## 5. CPU Techniques Inside libfwht

1. **SIMD (Single Instruction, Multiple Data):** AVX2/AVX-512 on x86 and NEON/SVE on ARM allow one instruction to update multiple butterflies. We schedule data so each vector register always holds a contiguous group of elements.
2. **Cache-friendly batching:** `fwht_i32_batch` and `fwht_f64_batch` operate on many signals of the same length. Keeping the working set resident in the L1/L2 caches removes most memory stalls.
3. **Bit-packed Boolean kernels:** Functions such as `fwht_boolean_packed` and `fwht_boolean_batch` compress 32 or 64 Boolean values into a single machine word. We rely on the hardware `popcount` instruction (population count, i.e., number of ones) to accumulate correlation sums faster than unpacked arithmetic.

---

## 6. GPU Techniques Inside libfwht

### 6.1 Shared-memory tiling

CUDA shared memory is an explicitly managed on-chip scratchpad. We tile a large vector into chunks that fit inside shared memory, run multiple butterfly stages there, then write the chunk back to global memory. This minimizes expensive global loads/stores.

### 6.2 Bank-conflict-aware strides

Shared memory is divided into banks (hardware lanes). When two threads address the same bank, the hardware serializes the access. We pick radix sizes and strides so threads land on different banks, especially on GPUs with 32 banks where naive strides would conflict.

### 6.3 Warp shuffles

A warp is the smallest scheduling unit on NVIDIA GPUs (32 threads). **Warp shuffle instructions** let threads exchange register values without touching memory. For small transforms (`n ≤ 1024`) we perform entire stages using only shuffles, which is dramatically faster than shared-memory ping-pong.

### 6.4 Persistent GPU contexts

Creating CUDA buffers, streams, and plans inside every API call wastes time. libfwht exposes `fwht_gpu_context_*` helpers that allocate once and reuse the resources, which matters when the same application runs millions of transforms per second.

---

## 7. Tensor Core Path

Tensor Cores (available on NVIDIA Ampere, Ada, and Hopper architectures) execute fixed-size matrix multiply-accumulate (MMA) operations, typically 16×16. We map blocks of the Hadamard matrix to those MMA tiles:

- **Constant-geometry scheduling:** We keep the tile layout identical from stage to stage, so each kernel call reuses the same MMA instructions without reshuffling data.
- **Mixed precision:** Inputs reside in fp16 or bfloat16, while accumulators stay in fp32. This keeps numerical error predictable while unlocking the throughput gains of Tensor Cores.
- **Automatic fallback:** If Tensor Cores are unavailable, libfwht automatically switches to the shared-memory kernels without any API change.

---

## 8. Putting Theory Into Practice

1. **FWHT gives the optimal asymptotic cost** (`Θ(n log2 n)` with `n = 2^k`). No algorithm can beat the `n log n` bound while still producing all `n` coefficients.
2. **CPU optimizations** (SIMD, batching, bit-packing) shrink constant factors so that the theoretical optimum feels fast on general-purpose processors.
3. **GPU optimizations** (tiling, bank-aware strides, warp shuffles) harness thousands of threads efficiently.
4. **Tensor Cores** provide an additional multiplier for half-precision workloads without changing user code.

Because libfwht consolidates all of these techniques behind a stable API, beginners can focus on correct data preparation while experts still get access to high-end features such as GPU profiling, persistent contexts, and mixed-precision kernels.

---

## 9. Two Useful Properties to Remember

- **Self-inverse up to scaling:** Applying the WHT twice brings you back to the original signal up to a factor of `n`. In symbols, `H_n · H_n = n · I_n`, where `I_n` is the identity matrix. Practically, this means you can invert a transform by running FWHT again and then dividing by `n`.
- **Energy preservation with orthonormal scaling:** If you scale by `1/√n` to form `H_n / √n`, then the total energy (sum of squares) of the input equals the total energy of the spectrum. This is the Walsh analogue of Parseval’s identity for the Fourier transform.
