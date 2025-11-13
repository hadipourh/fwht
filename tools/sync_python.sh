#!/bin/bash
# Sync library sources to Python package
# Run this before building/releasing the Python package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."  # Go to project root

echo "Syncing library sources to Python package..."

# Create directories if they don't exist
mkdir -p python/include
mkdir -p python/c_src

# Copy header
echo "  ✓ Copying include/fwht.h → python/include/"
cp include/fwht.h python/include/

# Copy C sources
echo "  ✓ Copying src/*.c → python/c_src/"
cp src/fwht_backend.c python/c_src/
cp src/fwht_core.c python/c_src/

# Copy CUDA source
echo "  ✓ Copying src/fwht_cuda.cu → python/c_src/"
cp src/fwht_cuda.cu python/c_src/

# Copy internal header
echo "  ✓ Copying src/fwht_internal.h → python/c_src/"
cp src/fwht_internal.h python/c_src/

echo ""
echo "✓ Python sources synced successfully!"
echo ""
echo "Note: These files are in .gitignore and won't be committed."
echo "Run this script before building the Python package with:"
echo "  cd python && python -m build"
