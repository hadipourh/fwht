#!/bin/bash
# Setup script for RTX 4090 (SM 8.9, CUDA 12.4)
# Installs both libfwht and Meta's fast-hadamard-transform library

set -e  # Exit on error

echo "=== Installing dependencies ==="
pip install wheel pybind11 numpy packaging torch

echo ""
echo "=== Installing Meta's fast-hadamard-transform ==="
cd /workspace
if [ ! -d "fast-hadamard-transform" ]; then
    git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
fi
cd fast-hadamard-transform
pip install -e .
cd ..

echo ""
echo "=== Building libfwht C library ==="
cd /workspace
make clean && make

echo ""
echo "=== Installing pyfwht Python package ==="
cd python
pip install -e .

echo ""
echo "=== Testing installations ==="
cd /workspace
python << 'EOF'
import pyfwht
import fast_hadamard_transform
import torch

print("pyfwht:", pyfwht.__version__)
print("Meta library:", fast_hadamard_transform.__version__)
print("PyTorch CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
EOF

echo ""
echo "=== Setup complete! ==="
echo "Run comparison benchmark with:"
echo "  python tools/compare_libs.py --sizes 1024 2048 4096 --batches 1 100 1000 --dtype float16 --device gpu"
