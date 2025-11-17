#!/bin/bash
# Test chunked kernel implementation

set -e

echo "=== Step 1: Rebuild C library ==="
make clean
make

echo ""
echo "=== Step 2: Rebuild Python bindings ==="
cd python
pip install -e . --force-reinstall --no-deps

echo ""
echo "=== Step 3: Run benchmarks on chunked range (1K-32K) ==="
cd ../tools
python compare_libs.py --sizes 1024 2048 4096 8192 16384 32768 \
  --batches 4 --dtype float16 \
  --meta-module faster_hadamard_transform \
  --meta-func hadamard_transform

echo ""
echo "=== Chunked kernel test complete! ==="
