---
slug: dao-fast-hadamard-transform
bibkey: daoAILabFastHadamardTransform
tags: [cuda, warp-shuffle, shared-memory, batch, fp16, bf16, fp32]
gpu-arch: [Ampere]
---

# Dao-AILab: Fast Hadamard Transform in CUDA (GitHub)

## TL;DR
- CUDA FWHT with PyTorch interface supporting fp32/fp16/bf16 up to dim 32768. Benchmarked vs memcpy lower bound (README). Widely used baseline cited in HadaCore as state-of-the-art original algorithm implementation.

## Design (per arXiv 2412.08832 §2.4 summary)
- Right-Hadamard transform; parallelize across rows via grid; within row up to 256 threads per row.
- Each thread processes 8 contiguous elements per chunk; perform per-thread chunks first, then warp-level exchanges, then two block syncs with shared-memory transposes to cover many stages.
- Achieves 15 iterations with only 2 threadblock synchronizations (supports up to 2^15 size per the description).

## Performance (from README snippet)
- Speed table vs memcpy shows increasing speedup at larger sizes, with fp16/bf16 reaching up to ~1.8× at 32768; fp32 smaller gains.

## Applicability to libfwht
- Provides a strong warp/shared-memory template for non-Tensor-Core integer/float paths.
- Techniques to reduce sync count and improve locality are relevant for our non-WMMA kernels.
