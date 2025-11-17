---
slug: 2025-markovic-walsh-hadamard-tensor-cores
bibkey: markovic2025walshHadamardTensorCores
tags: [tensor-core, cooley-tukey, constant-geometry, gpu]
gpu-arch: [Tensor Cores]
---

# Walsh-Hadamard spectra computation on GPU with tensor cores by Cooley-Tukey and constant geometry algorithms (Journal of Real-Time Image Processing, 2025)

## TL;DR
- Adapts both Cooley–Tukey and constant geometry WHT algorithms to NVIDIA Tensor Cores. Evaluates vs single/multi-core CPU and standard CUDA GPU approaches. Both tensor-core variants outperform conventional implementations; constant geometry achieves shorter execution time. Reported speedups grow with input variable count.

## Notes (from abstract and metadata on the DOI page)
- Focus: WHT spectra computation; tensor-core acceleration; algorithmic variants (Cooley–Tukey, constant geometry).
- Claim: Both TC-optimized algorithms outperform CPU and standard CUDA approaches; constant geometry faster among the two.
- Trend: Improvement increases with the number of input variables.
- Venue: Journal of Real-Time Image Processing, Vol 22, Issue 4 (2025). Published July 12, 2025. Publisher: Springer.

## Applicability to libfwht
- Reinforces the viability of Tensor Core microkernels beyond a single tiling strategy (constant-geometry variant may be preferable for certain sizes and layouts).
- Suggests adding both Cooley–Tukey and constant-geometry Tensor Core paths behind an AUTO selector and measuring across sizes.
- Compare against HadaCore-style 16×16 tiling: evaluate coalescing/synchronization trade-offs in our codebase.
