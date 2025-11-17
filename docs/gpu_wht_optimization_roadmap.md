# libfwht GPU WHT Optimization Roadmap

This living document outlines high‑impact improvements for the CUDA backend, prioritized by cost/benefit. It will be refined after we ingest the references in `references/` and write notes per paper.

## Current implementation (baseline)

- Per-transform block with shared-memory staging; simple butterfly loop per stage
  - Batch mode supported (multiple transforms) and persistent context to amortize allocations
  - Optional load/store device callbacks for fusing simple ops
- Strengths: simple and correct; decent for small–medium N
- Likely bottlenecks:
  - Not fully coalesced across all stages for large N
  - Heavy shared-memory usage with per-stage syncs
  - Launch overhead for very small N (many transforms)
  - Limited use of warp shuffles/register blocking

## Prioritized improvements

P0 — near-term, low–moderate effort, high ROI

1) Warp‑level/register FWHT for small N (≤ 1024)
- Use `__shfl_xor_sync` to implement butterflies in registers
- Minimal shared memory; fewer synchronizations; higher occupancy
- Map 1 transform per warp or per few warps; pack many per block for batching

2) Stockham autosort variant for large N
- Ensure coalesced reads/writes every stage via ping‑pong buffers
- Tile into shared memory; fuse multiple stages per kernel to reduce launch overhead
- Bank‑conflict‑free layouts for shared memory

3) Ampere+ async copies and double buffering
- Use `cp.async` to overlap global memory loads with compute inside tiles
- Two‑stage pipeline per tile (load next while computing current)

4) CUDA Graphs for repeated pipelines
- Record the transform sequence and replay to reduce launch overhead in batch scenarios

5) Auto‑tuned launch parameters
- Quick runtime sweep per device (block size, tile size, stages-per-kernel) cached in context

P1 — exploratory, potentially large gains, moderate risk

6) Tensor Core microkernels (WMMA/CUTLASS)
- Implement H16/H32 blocks via WMMA (±1 encoded as FP16/bfloat16) and compose via Kronecker product
- Use for batched small transforms (e.g., N ∈ {64,128,256}) or as building blocks inside larger kernels
- Validate numerical behavior (integer paths must remain exact; floats must be bounded error)

7) Persistent kernels with device‑side work queues
- Keep blocks resident, pop work items (transform descriptors) from global queue
- Improves throughput for many small, irregular batches

8) Boolean‑only bit‑sliced FWHT
- When inputs are ±1 from Boolean truth tables, pack bits and use `__popc/__popcll` to compute partial sums
- Specialized path can give large throughput gains for cryptanalysis workloads

P2 — longer‑term

9) Multi‑GPU distribution for massive batches
10) Mixed‑precision and normalization fusion (optional)

## API surface considerations

- Context flags to select kernel variant or enable auto‑tuning
- Optional “boolean mode” for bit‑sliced path (input constraints enforced)
- Graph/persistent modes exposed via context configuration
- Keep default behavior AUTO and safe; advanced features opt‑in

## Validation and benchmarking

- Extend `bench/fwht_bench.c` to measure:
  - Batch throughput vs latency (various N, batch sizes)
  - Small‑N regimes (N≤1024) and large‑N (≥ 4096)
  - Impact of Graphs/persistence; effect of cp.async
- Report GB/s and transforms/sec; include correctness checks vs CPU

## Next steps (blocked on references ingestion)

- Populate `references/papers.bib` and notes; annotate techniques and constraints
- Convert most promising techniques into small prototypes behind feature flags
- Measure on representative GPUs (Turing, Ampere, Ada)
