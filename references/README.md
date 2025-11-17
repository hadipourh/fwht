# References for Walsh–Hadamard Transform (WHT) and GPU Optimization

This folder collects the papers, tech reports, blog posts, and code artifacts relevant to fast WHT/FWHT on modern hardware (especially CUDA GPUs). The goal is to distill practical ideas to improve libfwht’s GPU backend with state‑of‑the‑art techniques.

## How to contribute references

- Place PDFs in this folder using a consistent name:
  - `YEAR-firstAuthor-shortSlug.pdf` (e.g., `2011-cheng-gpu-fwht.pdf`)
- If the reference is a webpage or code repo, add a `.url` or `.md` link file with the same slug.
- Add a BibTeX entry to `papers.bib` (same key as the slug, e.g., `cheng2011gpu_fwht`).
- Create a note file in `notes/` using the template: `notes/YEAR-firstAuthor-shortSlug.md` and fill it out while reading.

## What we’re looking for (coverage checklist)

- Algorithmic variants of FWHT on GPUs
  - Stockham autosort vs decimation in time/space
  - Tiled/shared-memory designs; register blocking
  - Warp-level butterflies with shuffle intrinsics
  - Multi-kernel vs fused kernels; grid-synchronization/cooperative groups
- Memory optimization
  - Coalesced access patterns across all stages
  - Bank-conflict-free shared memory layouts
  - Double buffering and asynchronous copies (`cp.async`) on Ampere+
- Hardware features
  - Tensor Cores / WMMA for small Hadamard blocks (H16/H32)
  - CUTLASS-based microkernels and integration considerations
  - POPC-based bit-sliced boolean FWHT specializations
- Execution models
  - Persistent kernels vs many launches; CUDA Graphs
  - Batching strategies for thousands of small transforms
  - Auto-tuning of tile sizes, block sizes, and kernel variant selection
- Evaluation methodology
  - Metrics: GB/s, transforms/sec, latency vs throughput regimes
  - Reproducibility details (problem sizes, GPU models, compiler flags)

## Reading workflow

1. Add the reference and a BibTeX entry in `papers.bib`.
2. Copy `notes/template.md` to `notes/<slug>.md` and summarize:
   - Key idea, where it wins, what it costs
   - How to map into libfwht (kernel structure, API impact)
   - Risks and edge cases (precision/overflow, occupancy, portability)
3. Tag notes with labels like: `warp-shuffle`, `stockham`, `tensor-core`, `cp.async`, `persistent`, `autotune`, `bit-sliced`.
4. Propose concrete changes and expected impact (e.g., “Replace small-N path with warp-shuffle kernel; expect 1.4–2.2× for N≤1024”).

## Seed topics (to guide collection)

- GPU FWHT/WHT implementations and surveys
- Stockham algorithms for FFT/WHT coalescing on GPUs
- Warp-level primitives for butterfly networks (`__shfl_xor_sync`)
- Tensor-Core Hadamard/Hadamard-like transforms (WMMA, CUTLASS)
- Asynchronous shared-memory copies (`cp.async`) and double-buffering
- Persistent kernels and CUDA Graphs in high-throughput pipelines
- Bit-sliced boolean-only WHT using `__popc`/`__popcll`

Once the references are added, we’ll review each and update the GPU optimization roadmap in `docs/gpu_wht_optimization_roadmap.md`.
