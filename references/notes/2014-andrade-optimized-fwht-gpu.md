---
slug: 2014-andrade-optimized-fwht-gpu
bibkey: andrade2014optimizedFWHTGPU
tags: [shared-memory, bank-conflicts, radix, gpu-architectures]
gpu-arch: [16-bank, 32-bank]
---

# Optimized Fast Walsh-Hadamard Transform on GPUs for non-binary LDPC decoding (Parallel Computing, 2014)

## TL;DR
- Short communication focusing on FWHT optimizations tailored to non-binary LDPC decoding (FT-SPA). The work tunes radix-n stages and shared-memory access to minimize bank conflicts and improve throughput on GPUs with 16 and 32 shared-memory bank architectures.

## Key points (from highlights/abstract)
- Analyzes FWHT role in FT-SPA for non-binary LDPC decoding.
- Quantifies the trade-off between bank conflicts and throughput.
- Employs radix-n approaches tuned to the number of shared-memory banks.
- Distinct tuning for architectures with 16 vs 32 SMEM banks.

## Applicability to libfwht
- For our non-Tensor-Core CUDA kernels, adopt bank-conflict-aware shared-memory layouts and choose radix scheduling that aligns with the target GPU bank count.
- Consider auto-detecting bank width and adjusting block-level radix/tiling accordingly.
