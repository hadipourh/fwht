---
slug: 2024-pytorch-blog-hadacore
bibkey: pytorchBlog2024hadacore
tags: [tensor-core, mma, warp-shuffle, performance, fp8-context]
gpu-arch: [Ampere, Hopper]
---

# HadaCore: Tensor Core Accelerated Hadamard Transform Kernel (PyTorch Blog)

## TL;DR
- Blog announcement summarizing a Tensor Core–accelerated FWHT kernel with A100/H100 speedups vs Dao-AILab’s CUDA kernel. Provides high-level method (16×16 MMA tiles, warp-level parallelism), microbenchmarks, and MMLU end-to-end validation with FP8 attention.

## Key points
- Uses MMA PTX instructions (e.g., `mma.m16n8k16`) to apply 16×16 Hadamard blocks; processes 256-element fragments per warp and extends beyond by shuffling.
- Reported speedups: A100 ~1.1–1.4× (peak ~3.5×), H100 ~1.0–1.3× (peak ~3.6×) vs Dao-AILab’s kernel.
- MMLU for Llama 3.1-8B with FP8 attention shows similar accuracy preservation compared to the Dao kernel.
- Mentions future work for Hopper features (e.g., TMA/WGMMA), Triton version, and BF16 support.

## Applicability to libfwht
- Confirms viability of WMMA/PTX-based 16×16 building blocks for FP16/BF16 paths.
- Reinforces need for size- and device-aware kernel selection and for batch-heavy use cases.
