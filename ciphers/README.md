# Cipher Guideline

This folder is for small-block ciphers that use libfwht.

Supported tasks:

- Linear analysis
- Differential-linear analysis

Target ciphers are small-block designs such as SIMON32, SPECK32, and KATAN32.

## General Rules

- `r` rounds means `r` full rounds of the cipher.
- Keep encryption and decryption oracles separate.
- Keep the round count explicit in every API.
- Work with full masks or full differences over the whole block.
- Output vectors must use the natural integer order of masks.
- For many keys, report RMS strength, not the signed mean.
- In linear analysis, the fixed input mask or fixed output mask should be limited to Hamming weight 1 or 2.
- In differential-linear analysis, the fixed input difference or fixed output difference should be limited to Hamming weight 1 or 2.

## Cipher Interface

Each cipher module should provide:

- block size
- key size
- encryption oracle for `r` rounds
- decryption oracle for `r` rounds
- linear oracle interface
- differential-linear oracle interface
- helper to enumerate fixed Hamming weight 1 and Hamming weight 2 masks
- helper to enumerate fixed Hamming weight 1 and Hamming weight 2 differences

The round-reduced oracle must not guess hidden round conventions.

## Many Keys And RMS

- For one key, keep the signed correlation or signed bias.
- Across many keys, do not average the signed value.
- Use RMS over keys.

$$
\mathrm{RMS} = \sqrt{\frac{1}{K}\sum_{k=1}^{K} c_k^2}
$$

- The same rule is used for differential-linear bias.
- Keep one running `sum_squared[mask]` array.
- Do not keep a full `key_count x mask_count` table unless it is needed for debug output.

## Linear Analysis

Fixed-side rule:

- enumerate only fixed masks of Hamming weight 1 or 2
- do not iterate over all `2^n` fixed masks by default

For the forward side:

- fix one input mask
- for one query, stream over all outputs and call the decryption oracle
- if many queries use the same fixed key and memory is enough, build the inverse codebook once
- index the table by output value
- build the truth table `(-1)^(<input_mask, inverse_output>)`
- call libfwht once
- read all output-mask correlations from the FWHT output

For the backward side:

- fix one output mask
- for one query, stream over all inputs and call the encryption oracle
- if many queries use the same fixed key and memory is enough, build the forward codebook once
- index the table by input value
- build the truth table `(-1)^(<output_mask, forward_output>)`
- call libfwht once
- read all input-mask correlations from the FWHT output

Rule:

- if the free masks live on the output side, use a table indexed by outputs
- if the free masks live on the input side, use a table indexed by inputs

## Differential-Linear Analysis

Fixed-side rule:

- enumerate only fixed differences of Hamming weight 1 or 2
- do not iterate over all `2^n` fixed differences by default

For the forward side:

- fix one input difference
- for one query, stream over all inputs and compute the two encryptions on the fly
- if many queries use the same fixed key and memory is enough, build the forward codebook once
- compute `T[x] xor T[x xor diff]`
- build the histogram of derivative outputs
- call libfwht once
- read all output-mask correlations from the FWHT output

For the backward side:

- fix one output difference
- for one query, stream over all outputs and compute the two decryptions on the fly
- if many queries use the same fixed key and memory is enough, build the inverse codebook once
- compute `U[y] xor U[y xor diff]`
- build the histogram of inverse-derivative outputs
- call libfwht once
- read all input-mask correlations from the FWHT output

Rule:

- for fixed differences and all masks, transform the derivative histogram
- do not transform a raw list of states if a histogram is enough
- one FWHT gives all masks at once for one fixed difference

Exact formula:

$$
C(a, \beta) = 2^{-n}\sum_{\gamma} N_a(\gamma) (-1)^{\langle \beta, \gamma \rangle}
$$

where `N_a(gamma)` is the histogram of `T[x] xor T[x xor a]`.

Exact cost for block size `n`:

- build the histogram: `Theta(2^n * E_r)` where `E_r` is one `r`-round oracle call
- one FWHT: `Theta(n 2^n)`
- total exact time: `Theta(2^n * E_r + n 2^n)`
- memory without a stored codebook: `Theta(2^n)`
- memory with one stored codebook: still `Theta(2^n)`, but with a larger constant

## Codebook Or Streaming

- For one query, stream the oracle on demand.
- For many queries with the same fixed key and rounds, precompute one codebook only if the memory budget allows it.
- Do not precompute both forward and inverse tables unless both are really needed.
- For many keys, process one key at a time.
- Free the per-key codebook before the next key if memory is tight.
- At 32 bits, a stored codebook is usually a special mode, not the default mode.

## Performance Rules

- Reuse `fwht_create_context(...)` when many queries share the same size.
- Prefer contiguous buffers.
- Prefer `fwht_batch_i32_contiguous(...)` or `fwht_batch_f64_contiguous(...)` when many vectors share the same length.
- Avoid repeated allocation inside hot loops.
- Keep data on the best backend when repeated calls are needed.
- Use `fwht_boolean_packed_backend(...)` or `fwht_gpu_boolean_context_compute(...)` when the table is Boolean and repeated GPU calls are needed.
- Use CPU, OpenMP, or CUDA through libfwht instead of custom transform code.
- Parallelize over keys or over state-space chunks, but not with one huge table per thread.
- For exact 32-bit mode, use `fwht_f64(...)` rather than `fwht_i32(...)` because the unnormalized coefficients can reach `2^32`.

## Memory Notes

- A `2^n` codebook with `uint32_t` outputs costs `4 * 2^n` bytes.
- At `n = 32`, one forward table is 16 GiB.
- One inverse table is another 16 GiB.
- One `int32` FWHT vector of length `2^32` is 16 GiB but it cannot safely hold exact unnormalized coefficients at 32 bits.
- One `fp64` FWHT vector of length `2^32` is 32 GiB.
- So exact full-space work at 32 bits is a special high-memory mode.
- The default path for 32-bit ciphers should use streaming unless a memory check says a stored codebook is safe.

## Implementation Plan For A New Cipher

1. Define the round-reduced encryption and decryption oracles.
2. Fix the block format and bit order.
3. Implement the Hamming weight 1 and Hamming weight 2 mask enumerator and difference enumerator.
4. Implement the linear builder.
5. Implement the differential-linear builder.
6. Connect both builders to libfwht.
7. Add the RMS-over-keys reducer.
8. Add small known-answer tests.
9. Add one speed test for repeated queries.

## Validation

- Check small cases by brute force.
- Check that forward and backward conventions match.
- Check that round counting is exact.
- Check that mask indexing is stable.
- Check that repeated runs give the same result.
- Check that RMS is unchanged by random key sign flips.
- Check that the fixed-side enumerator returns only Hamming weight 1 and Hamming weight 2 masks or differences.