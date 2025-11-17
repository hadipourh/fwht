#!/bin/bash
# Setup script for benchmark environment
# Installs both pyfwht and Meta's fast-hadamard-transform library
# 
# Requirements:
#   - CUDA-capable GPU (SM 8.0+, e.g., A100, RTX 3090/4090)
#   - CUDA 12.x toolkit
#   - Python 3.8+
#
# Usage:
#   bash tools/setup_benchmark_env.sh

set -e  # Exit on error

echo "========================================="
echo "Benchmark Environment Setup"
echo "========================================="

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo "CUDA Version:"
nvcc --version | grep "release"

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader

echo ""
echo "Step 1: Installing Python dependencies..."
# Install base dependencies first
pip install wheel pybind11 numpy packaging

# Install PyTorch with explicit CUDA version to ensure compatibility
echo "Installing PyTorch for CUDA 12.x..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "Step 2: Installing Meta's fast-hadamard-transform..."
# Reinstall Meta library to ensure it's compiled against the PyTorch we just installed
pip uninstall -y fast-hadamard-transform 2>/dev/null || true
pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git
echo "Meta library installed"

echo ""
echo "Step 3: Building libfwht C library (optional)..."
if [ -f "Makefile" ]; then
    make clean
    make
    echo "C library built"
else
    echo "Skipping C library (Makefile not found)"
fi

echo ""
echo "Step 4: Installing pyfwht Python package..."
cd python
pip install -e .
cd ..
echo "pyfwht installed"

echo ""
echo "Step 5: Verifying installations..."
python << 'EOF'
import sys

try:
    import pyfwht
    print(f"[OK] pyfwht version: {pyfwht.__version__}")
except ImportError as e:
    print(f"[FAIL] pyfwht not installed: {e}")
    sys.exit(1)

try:
    import fast_hadamard_transform
    print(f"[OK] Meta library version: {fast_hadamard_transform.__version__}")
except ImportError as e:
    print(f"[FAIL] Meta library not installed: {e}")
    sys.exit(1)

try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"[OK] PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[OK] Compute capability: {torch.cuda.get_device_capability(0)}")
except ImportError as e:
    print(f"[FAIL] PyTorch not installed: {e}")
    sys.exit(1)

print("\n[SUCCESS] All dependencies installed successfully!")
EOF

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "Run benchmarks with:"
echo "  python tools/compare_libs.py --sizes 1024 2048 4096 --batches 1 100 1000 --dtype float16 --device gpu"
