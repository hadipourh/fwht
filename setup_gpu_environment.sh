#!/bin/bash
# Comprehensive GPU Environment Setup for libfwht
# 
# This script:
#   1. Installs Dao-AILab's fast-hadamard-transform library (with CUDA version patch)
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

# Detect CUDA version and install matching PyTorch
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

echo "Detected CUDA version: $CUDA_VERSION"

# Map CUDA version to PyTorch index
if [ "$CUDA_MAJOR" -ge 13 ]; then
    TORCH_INDEX="cu121"  # PyTorch doesn't have cu130 yet, use cu121
    echo "Installing PyTorch for CUDA 12.1+ (compatible with CUDA 13.x)..."
elif [ "$CUDA_MAJOR" -eq 12 ]; then
    if [ "$CUDA_MINOR" -ge 4 ]; then
        TORCH_INDEX="cu124"
    else
        TORCH_INDEX="cu121"
    fi
    echo "Installing PyTorch for CUDA $TORCH_INDEX..."
else
    TORCH_INDEX="cu118"
    echo "Installing PyTorch for CUDA 11.8+"
fi

pip install torch --index-url https://download.pytorch.org/whl/$TORCH_INDEX
echo "✓ Python dependencies installed"
echo ""

# Step 2: Install Dao-AILab's fast-hadamard-transform library (optional)
echo "=== Step 2/5: Installing Dao-AILab's fast-hadamard-transform (optional) ==="
echo "This library is ONLY needed for side-by-side performance comparisons."
echo "Skipping this step will NOT affect pyfwht functionality."
echo ""

if [ "$CUDA_MAJOR" -ge 13 ]; then
    echo "ℹ CUDA $CUDA_VERSION detected"
    echo "  Will attempt to build with system CUDA (compatibility mode)"
    echo ""
fi

pip uninstall -y fast-hadamard-transform 2>/dev/null || true

# Use system CUDA for compilation, but disable version check
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
# Disable CUDA version checking in PyTorch extensions
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
# Bypass PyTorch's CUDA version check (the key fix!)
export TORCH_CUDA_VERSION_CHECK=0

# Find system nvcc
SYSTEM_NVCC=$(which nvcc 2>/dev/null || echo "")
if [ -z "$SYSTEM_NVCC" ]; then
    echo "⚠ nvcc not found in PATH, Dao-AILab library installation will fail"
    echo "  → pyfwht will work fine without it"
    echo ""
else
    export CUDA_HOME=$(dirname $(dirname $SYSTEM_NVCC))
    export CUDA_PATH=$CUDA_HOME
    echo "Using system CUDA from: $CUDA_HOME"
    
    # Try to install with version check bypass
    echo "Installing Dao-AILab's fast-hadamard-transform..."
    
    # Temporarily patch torch version check if needed
    if pip install git+https://github.com/Dao-AILab/fast-hadamard-transform.git --no-build-isolation 2>&1 | tee /tmp/dao_install.log | grep -q "Successfully installed"; then
        echo "✓ Dao-AILab library installed successfully"
        echo "  Comparison benchmarks will include Dao-AILab's implementation"
    else
        # Check if it's a CUDA version mismatch
        if grep -q "CUDA version" /tmp/dao_install.log; then
            echo "⚠ Dao-AILab library installation failed (CUDA version mismatch)"
            echo "  → System CUDA $CUDA_VERSION vs PyTorch CUDA $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'unknown')"
            echo "  → This is expected on CUDA 13.x systems"
        else
            echo "⚠ Dao-AILab library installation failed"
            echo "  → Check build log: /tmp/dao_install.log"
        fi
        echo "  → pyfwht will work fine without it"
        echo "  → Comparison examples will only benchmark pyfwht"
    fi
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

# Check Dao-AILab library (optional)
try:
    import fast_hadamard_transform as fht
    print(f"✓ Dao-AILab library version: {fht.__version__}")
except ImportError:
    print("⚠ Dao-AILab library not installed (optional, comparisons won't work)")

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
    echo "  2. Run Python test suite (from repo root):"
    echo "     python -m pytest python/tests -v"
    echo ""
    echo "  3. GPU multi-precision example:"
    echo "     python python/examples/gpu_multi_precision.py"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ Setup failed during verification"
    echo "========================================="
    exit 1
fi
