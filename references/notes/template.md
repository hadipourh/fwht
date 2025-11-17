---
slug: YEAR-firstAuthor-shortSlug
bibkey: keyInPapersBib
tags: [warp-shuffle, stockham, tensor-core, cp.async, persistent, autotune, bit-sliced]
gpu-arch: [Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper]
---

# Title: (Exact paper title)

## TL;DR
- One-paragraph summary of the core idea and when it helps.

## Contributions and context
- What’s new vs prior art? Which problem sizes/workloads are targeted?
- Code availability? Dependencies (CUTLASS, WMMA, cooperative groups)?

## Algorithm and GPU mapping
- Algorithmic variant (Stockham vs decimation, etc.)
- Kernel structure (per-stage vs fused; tiles; grid/block organization)
- Warp-level primitives used (e.g., `__shfl_xor_sync`), register blocking
- Shared memory usage and bank-conflict strategy
- Memory access patterns and coalescing across stages
- Async copies (`cp.async`) and double-buffering (if applicable)
- Tensor Core usage (WMMA tile sizes, data types, precision caveats)

## Performance and constraints
- Reported speedups and against what baseline
- GPU models, compiler flags, and key parameters
- Occupancy, register pressure, shared memory limits
- Edge cases: precision/overflow, misaligned sizes, large-N behavior

## Applicability to libfwht
- What to borrow: concrete kernel ideas
- Integration notes: API impact, build flags, optional paths
- Risks: portability, maintenance burden, robustness
- Quick prototype plan and expected gains on target GPUs

## Open questions
- Assumptions or missing details to validate in our context
