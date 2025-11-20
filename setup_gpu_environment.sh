#!/bin/bash
# Comprehensive GPU Environment Setup for libfwht
# 
# This script:
#   1. Installs Meta's fast-hadamard-transform library (with CUDA version patch)
#   2. Builds libfwht C library
#   3. Installs pyfwht Python package
#   4. Runs verification tests
#
# Requirements:
#   - CUDA-capable GPU (SM 8.0+, e.g., A100, RTX 3090/4090/5090)
#   - CUDA 12.x or 13.x toolkit
#   - Python 3.8+
#
# Usage:
#   bash setup_gpu_environment.sh

set -e  # Exit on error

echo "========================================="
echo "libfwht GPU Environment Setup"
echo "========================================="
echo ""

# Detect workspace directory (works from libfwht/ or any subdirectory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "=== System Information ==="
echo "CUDA Version:"
nvcc --version | grep "release"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
echo ""

# Step 1: Install Python dependencies
echo "=== Step 1/5: Installing Python Dependencies ==="
pip install --upgrade pip wheel setuptools
pip install pybind11 numpy packaging

# Install PyTorch with CUDA support
echo "Installing PyTorch for CUDA 12.x..."
pip install torch --index-url https://download.pytorch.org/whl/cu124
echo "✓ Python dependencies installed"
echo ""

# Step 2: Install Meta's fast-hadamard-transform library (with patch)
echo "=== Step 2/5: Installing Meta's fast-hadamard-transform ==="
pip uninstall -y fast-hadamard-transform 2>/dev/null || true

# Use the patching script
if [ -f "tools/install_meta_with_patch.sh" ]; then
    bash tools/install_meta_with_patch.sh || {
        echo "ERROR: Failed to install Meta library."
        echo "This is optional - pyfwht will still work without it."
        echo "Continue? (y/n)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    }
else
    echo "WARNING: tools/install_meta_with_patch.sh not found"
    echo "Trying direct install (may fail with CUDA version mismatch)..."
    export FORCE_CUDA=1
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
    pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git --no-build-isolation || {
        echo "WARNING: Meta library installation failed (this is optional)"
    }
fi
echo ""

# Step 3: Build libfwht C library
echo "=== Step 3/5: Building libfwht C Library ==="
if [ -f "Makefile" ]; then
    make clean
    make
    echo "✓ C library built successfully"
else
    echo "WARNING: Makefile not found, skipping C library build"
fi
echo ""

# Step 4: Install pyfwht Python package
echo "=== Step 4/5: Installing pyfwht Python Package ==="
cd python

# Force reinstall to pick up any C library changes
pip uninstall -y pyfwht 2>/dev/null || true
pip install -e . --force-reinstall --no-build-isolation

cd ..
echo "✓ pyfwht installed successfully"
echo ""

# Step 5: Run verification tests
echo "=== Step 5/5: Verifying Installation ==="
python << 'EOF'
import sys

print("Checking installations...\n")

# Check pyfwht
try:
    import pyfwht
    print(f"✓ pyfwht version: {pyfwht.__version__}")
    print(f"  - GPU available: {pyfwht.has_gpu()}")
    print(f"  - OpenMP available: {pyfwht.has_openmp()}")
except ImportError as e:
    print(f"✗ pyfwht not installed: {e}")
    sys.exit(1)

# Check Meta library (optional)
try:
    import fast_hadamard_transform as fht
    print(f"✓ Meta library version: {fht.__version__}")
except ImportError:
    print("⚠ Meta library not installed (optional, comparisons won't work)")

# Check PyTorch
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"✓ PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"  - Compute capability: {compute_cap[0]}.{compute_cap[1]}")
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("Running Quick Functionality Test")
print("="*50 + "\n")

# Quick GPU test
import numpy as np

# Test CPU
data_cpu = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
pyfwht.transform(data_cpu)
print(f"✓ CPU transform: {data_cpu}")

# Test GPU if available
if pyfwht.has_gpu() and torch.cuda.is_available():
    data_gpu = torch.randn(10, 1024, dtype=torch.float32, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_gpu)
    print(f"✓ GPU fp32 transform: shape={data_gpu.shape}, device={data_gpu.device}")
    
    # Test fp16 for speed
    data_fp16 = torch.randn(100, 4096, dtype=torch.float16, device='cuda')
    pyfwht.gpu.batch_transform_dlpack(data_fp16)
    print(f"✓ GPU fp16 transform: shape={data_fp16.shape}, device={data_fp16.device}")
else:
    print("⚠ GPU tests skipped (no CUDA)")

print("\n✓ All tests passed!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Setup Complete!"
    echo "========================================="
    echo ""
    echo "Available commands:"
    echo ""
    echo "  1. Run correctness + performance benchmark:"
    echo "     python python/tests/benchmark_all_precisions_fixed.py"
    echo ""
    echo "  2. Compare with Meta library:"
    echo "     python tools/compare_libs.py --sizes 1024 2048 4096 --batches 1 100 --dtype float16"
    echo ""
    echo "  3. Run test suite:"
    echo "     cd python && pytest tests/ -v"
    echo ""
    echo "  4. GPU multi-precision example:"
    echo "     python python/examples/gpu_multi_precision.py"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ Setup failed during verification"
    echo "========================================="
    exit 1
fi
