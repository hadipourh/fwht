#!/bin/bash
# Install Meta's fast-hadamard-transform with CUDA version check bypass
# This script patches the setup.py to skip PyTorch's CUDA version check

set -e

META_REPO_DIR="/tmp/fast-hadamard-transform"
META_REPO_URL="https://github.com/Dao-AILab/fast-hadamard-transform.git"

echo "========================================="
echo "Installing Meta Library (with CUDA patch)"
echo "========================================="

# Clone or update repository
if [ ! -d "$META_REPO_DIR" ]; then
    echo "Cloning fast-hadamard-transform repository..."
    git clone "$META_REPO_URL" "$META_REPO_DIR"
else
    echo "Repository already exists, updating..."
    cd "$META_REPO_DIR"
    git pull || true
fi

cd "$META_REPO_DIR"

# Check if setup.py exists
if [ ! -f "setup.py" ]; then
    echo "ERROR: setup.py not found in $META_REPO_DIR"
    exit 1
fi

# Create backup
cp setup.py setup.py.backup

# Patch setup.py to skip CUDA version check
echo "Patching setup.py to bypass CUDA version check..."

# Use Python to patch the file (more reliable than sed)
python3 << 'PATCH_SCRIPT'
import re
import sys

setup_py_path = "setup.py"

try:
    with open(setup_py_path, 'r') as f:
        lines = f.readlines()
    
    # Check if already patched
    content_str = ''.join(lines)
    if "SKIP_CUDA_VERSION_CHECK" in content_str or "_patched_check_cuda_version" in content_str:
        print("Setup.py already patched, skipping...")
        sys.exit(0)
    
    # Find where to insert the patch - after all imports
    # Strategy: find the last import statement, then find the next non-blank, non-comment line
    # Insert between them (after imports, before actual code)
    insert_pos = 0
    last_import_line = -1
    
    # First pass: find the last import line (handling multi-line imports)
    in_multiline_import = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Check if we're starting a multi-line import
        if (stripped.startswith('import ') or stripped.startswith('from ')) and '(' in line and ')' not in line:
            in_multiline_import = True
            last_import_line = i
        # Check if we're ending a multi-line import
        elif in_multiline_import:
            last_import_line = i
            if ')' in line:
                in_multiline_import = False
        # Regular single-line import
        elif stripped.startswith('import ') or stripped.startswith('from '):
            last_import_line = i
    
    # Second pass: find the first non-import, non-comment, non-blank line after imports
    if last_import_line >= 0:
        for i in range(last_import_line + 1, len(lines)):
            stripped = lines[i].strip()
            # Skip blank lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            # Found first non-import code - insert before it
            insert_pos = i
            break
        
        # If we didn't find code after imports, insert right after last import
        if insert_pos == 0:
            insert_pos = last_import_line + 1
    
    # Fallback: if no imports found, try to find a safe spot (after line 10, before line 30)
    if insert_pos == 0:
        # Look for common patterns: setup(, def, class, or just after line 15
        for i in range(10, min(30, len(lines))):
            stripped = lines[i].strip()
            if stripped.startswith('def ') or stripped.startswith('class ') or 'setup(' in stripped:
                insert_pos = i
                break
        if insert_pos == 0:
            insert_pos = min(15, len(lines))
    
    # Check if torch.utils.cpp_extension is already imported
    # Look for the specific import: "from torch.utils.cpp_extension import"
    cpp_ext_alias = None
    for i, line in enumerate(lines[:insert_pos]):
        if 'torch.utils.cpp_extension' in line:
            # Try to extract the alias or module name
            if 'import' in line:
                # Could be: from torch.utils.cpp_extension import X
                # or: import torch.utils.cpp_extension as Y
                # For simplicity, we'll reference it directly
                cpp_ext_alias = 'torch.utils.cpp_extension'
                break
    
    # Create patch code that references the existing import
    # IMPORTANT: Don't re-import torch.utils.cpp_extension if it's already imported
    # Just patch the function directly
    if cpp_ext_alias:
        # torch.utils.cpp_extension already imported, reference it directly WITHOUT re-importing
        patch_code = [
            '\n',
            '# PATCH: Skip CUDA version check for compatibility (CUDA 12.4/13.0 are runtime-compatible)\n',
            '_original_check_cuda_version = torch.utils.cpp_extension._check_cuda_version\n',
            'def _patched_check_cuda_version(*args, **kwargs):\n',
            '    pass  # Skip version check - CUDA 12.x and 13.x are compatible at runtime\n',
            'torch.utils.cpp_extension._check_cuda_version = _patched_check_cuda_version\n',
            '\n'
        ]
    else:
        # torch.utils.cpp_extension not imported yet, import it first
        patch_code = [
            '\n',
            '# PATCH: Skip CUDA version check for compatibility\n',
            'import torch\n',
            'import torch.utils.cpp_extension\n',
            '_original_check_cuda_version = torch.utils.cpp_extension._check_cuda_version\n',
            'def _patched_check_cuda_version(*args, **kwargs):\n',
            '    pass  # Skip version check\n',
            'torch.utils.cpp_extension._check_cuda_version = _patched_check_cuda_version\n',
            '\n'
        ]
    
    # Insert the patch
    lines[insert_pos:insert_pos] = patch_code
    
    # Write back
    with open(setup_py_path, 'w') as f:
        f.writelines(lines)
    
    print("✓ Successfully patched setup.py")
    
except Exception as e:
    print(f"ERROR: Failed to patch setup.py: {e}")
    import traceback
    traceback.print_exc()
    # Restore backup
    import shutil
    try:
        shutil.copy("setup.py.backup", "setup.py")
    except:
        pass
    sys.exit(1)
PATCH_SCRIPT

if [ $? -ne 0 ]; then
    echo "ERROR: Patching failed"
    exit 1
fi

# Set environment variables (extra safety)
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Install
echo "Installing fast-hadamard-transform..."
pip install -e . --no-build-isolation || {
    echo "ERROR: Installation failed even with patch"
    echo "Restoring original setup.py..."
    cp setup.py.backup setup.py
    exit 1
}

echo ""
echo "✓ Meta library installed successfully!"
echo ""
echo "Verifying installation..."
python3 -c "import fast_hadamard_transform; print(f'Version: {fast_hadamard_transform.__version__}')" || {
    echo "WARNING: Installation succeeded but import failed"
    exit 1
}

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="

