---
slug: 2024-agarwal-hadacore
bibkey: agarwal2024hadacore
tags: [tensor-core, warp-shuffle, shared-memory, transpose, wmma-ptx, bf16, fp16]
gpu-arch: [Ampere, Hopper]
---

# HadaCore: Tensor Core Accelerated Hadamard Transform Kernel (arXiv:2412.08832)

## TL;DR
- Proposes a Tensor Core–accelerated FWHT variant that applies 16×16 Hadamard blocks via MMA instructions, then uses transposes/rearrangements to scale to larger sizes. Reports speedups vs Dao-AILab’s CUDA FWHT of ~1.1–1.4× (A100) and ~1.0–1.3× (H100), with peak gains 3.5× and 3.6×, respectively.

## Contributions and context
- Hardware-aware decomposition of FWHT to match Tensor Core MMA (e.g., `mma.m16n8k16`) on NVIDIA GPUs.
- Scales beyond 256 by shared-memory staging + transposes; uses warp-level shuffles to coalesce or rearrange registers depending on size.
- Supports FP16 and BF16; BF16 accumulates in FP32 with conversion back to BF16.
- Benchmarks vs Dao-AILab’s fast-hadamard-transform on A100 and H100.

## Algorithm and GPU mapping
- Base compute unit: 16×16 Hadamard applied via two MMA operations per tile (maps to Tensor Core). Achieves up to size-256 by reshaping a 1×256 vector to 16×16, apply H16, transpose, apply again, transpose back.
- Larger than 256: process a full 1×n vector per threadblock. Each warp processes num_chunks of 256-element fragments; write to shared memory, synchronize, read transposed layout, then repeat chunked processing with 16×16 blocks.
- For sizes 512–2048: load coalesced and use warp shuffles to redistribute per-thread data to match Tensor Core layout. For 4096+, load all chunks first and then shuffle in registers to form MMA fragments.
- Non-power-of-16 sizes: factorize d = 2^m ⋅ 16^n; do n iterations of H16 blocks plus a final diagonal tiling for the 2^m component.

## Performance and constraints
- Reported speedups: A100 ~1.1–1.4× (peak 3.5×), H100 ~1.0–1.3× (peak 3.6×), versus Dao-AILab kernel; smaller gains on H100 for small sizes.
- Size 512 is the first that requires cross-256 synchronization and shows lower relative speedup due to data shuffling and shared-memory transposes aligned to MMA register layouts.
- Coalescing for ≥8K sizes requires more chunks per warp (8/16/32), trading off parallelism; configurations selected empirically.

## Applicability to libfwht
- Add an optional Tensor Core path for FP16/BF16 transforms using 16×16 microkernels composed to larger sizes via shared-memory transposes. Keep integer paths separate to preserve exactness.
- Use warp-level shuffle for coalescing and register re-mapping at 512–2048; shared-memory transpose for larger sizes.
- Consider “in-place rotation” optimization for cache residency (Appendix B).
- Provide AUTO selection gating by type (half/bfloat16), size (≥16), and device capability.

## Open questions
- Exact MMA tiling interfaces under WMMA vs inline PTX for best portability/perf.
- Occupancy/SMEM tradeoffs per GPU; need auto-tuning for warps-per-block and chunks.
