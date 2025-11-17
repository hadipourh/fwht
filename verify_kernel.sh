#!/bin/bash
# Quick verification script to check if simplified kernels are in place
echo "Checking fp32 kernel signature..."
grep -A 5 "hadamard_fp32_kernel" python/c_src/fwht_cuda.cu | head -15

echo ""
echo "Checking for EPT in fp32 kernel (should be ABSENT)..."
grep -c "constexpr int EPT" python/c_src/fwht_cuda.cu || echo "EPT not found (GOOD)"

echo ""
echo "Checking fp32 kernel uses simple smem pattern..."
grep "extern __shared__ float smem\[\]" python/c_src/fwht_cuda.cu && echo "FOUND simple smem" || echo "NOT FOUND"
