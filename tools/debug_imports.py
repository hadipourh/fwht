#!/usr/bin/env python3
"""
Diagnostic script to debug why Meta library isn't being detected in benchmarks.
Run this with the same Python interpreter as compare_libs.py
"""

import sys
import os

print("=" * 70)
print("PYTHON ENVIRONMENT DIAGNOSTICS")
print("=" * 70)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path:")
for p in sys.path:
    print(f"  {p}")

print("\n" + "=" * 70)
print("CHECKING TORCH")
print("=" * 70)
try:
    import torch
    print(f"✓ torch imported successfully")
    print(f"  Version: {torch.__version__}")
    print(f"  Location: {torch.__file__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")
    print(f"  Exception type: {type(e).__name__}")

print("\n" + "=" * 70)
print("CHECKING FAST_HADAMARD_TRANSFORM")
print("=" * 70)
try:
    import fast_hadamard_transform
    print(f"✓ fast_hadamard_transform imported successfully")
    print(f"  Location: {fast_hadamard_transform.__file__}")
    print(f"  Module attributes:")
    for attr in dir(fast_hadamard_transform):
        if not attr.startswith('_'):
            print(f"    - {attr}")
    
    # Try to call the function
    print("\n  Testing hadamard_transform function:")
    if hasattr(fast_hadamard_transform, 'hadamard_transform'):
        fn = fast_hadamard_transform.hadamard_transform
        print(f"    ✓ hadamard_transform function exists")
        print(f"    Function: {fn}")
        
        # Try a simple call
        try:
            import torch
            x = torch.randn(1, 8, dtype=torch.float16, device='cuda')
            y = fn(x)
            print(f"    ✓ Test call succeeded: input shape {x.shape} -> output shape {y.shape}")
        except Exception as e:
            print(f"    ✗ Test call failed: {e}")
    else:
        print(f"    ✗ hadamard_transform function NOT found")
        
except Exception as e:
    print(f"✗ Failed to import fast_hadamard_transform: {e}")
    print(f"  Exception type: {type(e).__name__}")
    import traceback
    print(f"  Traceback:")
    traceback.print_exc()

print("\n" + "=" * 70)
print("CHECKING PYFWHT")
print("=" * 70)
try:
    import pyfwht
    print(f"✓ pyfwht imported successfully")
    print(f"  Version: {pyfwht.__version__}")
    print(f"  Location: {pyfwht.__file__}")
    print(f"  GPU available: {pyfwht.gpu.available}")
except Exception as e:
    print(f"✗ Failed to import pyfwht: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ENVIRONMENT VARIABLES")
print("=" * 70)
for key in ['PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'PATH']:
    val = os.environ.get(key, '(not set)')
    print(f"{key}: {val}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("If fast_hadamard_transform imports here but not in compare_libs.py,")
print("then the benchmark is using a different Python interpreter.")
print("Run: which python3")
print("Compare with sys.executable shown above.")
