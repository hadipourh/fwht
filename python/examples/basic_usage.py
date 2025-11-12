"""
Basic usage examples for pyfwht package.

This file demonstrates the main features of the pyfwht library.
"""
import numpy as np
import pyfwht as fwht

print("=" * 70)
print("pyfwht - Fast Walsh-Hadamard Transform for Python")
print("=" * 70)
print()

# ============================================================================
# Example 1: Basic in-place transform
# ============================================================================
print("Example 1: Basic in-place transform")
print("-" * 70)

data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
print(f"Original data: {data}")

fwht.transform(data)
print(f"After WHT:      {data}")

# Verify involution property: WHT(WHT(x)) = n*x
fwht.transform(data)
print(f"After 2nd WHT:  {data} (should be 8× original)")
print()

# ============================================================================
# Example 2: Out-of-place transform
# ============================================================================
print("Example 2: Out-of-place transform")
print("-" * 70)

original = np.array([1, -1, -1, 1], dtype=np.int32)
result = fwht.compute(original)

print(f"Original:  {original} (unchanged)")
print(f"Transform: {result}")
print()

# ============================================================================
# Example 3: Boolean function analysis
# ============================================================================
print("Example 3: Boolean function analysis")
print("-" * 70)

# XOR function: f(x,y) = x ⊕ y
# Truth table: f(00)=0, f(01)=1, f(10)=1, f(11)=0
xor_table = np.array([0, 1, 1, 0], dtype=np.uint8)
print(f"XOR truth table: {xor_table}")

# Compute WHT coefficients
wht_coeffs = fwht.from_bool(xor_table, signed=True)
print(f"WHT coefficients: {wht_coeffs}")

# Compute correlations
correlations = fwht.correlations(xor_table)
print(f"Correlations: {correlations}")
print(f"Max |correlation|: {np.max(np.abs(correlations)):.6f}")
print()

# ============================================================================
# Example 4: Explicit backend selection
# ============================================================================
print("Example 4: Backend selection")
print("-" * 70)

data = np.random.randn(256).astype(np.float64)

# CPU backend
fwht.transform(data.copy(), backend=fwht.Backend.CPU)
print(f"✓ CPU backend available")

# OpenMP backend
if fwht.has_openmp():
    fwht.transform(data.copy(), backend=fwht.Backend.OPENMP)
    print(f"✓ OpenMP backend available")
else:
    print(f"✗ OpenMP backend not available")

# GPU backend
if fwht.has_gpu():
    fwht.transform(data.copy(), backend=fwht.Backend.GPU)
    print(f"✓ GPU backend available")
else:
    print(f"✗ GPU backend not available")

# AUTO backend (recommended)
recommended = fwht.recommend_backend(len(data))
print(f"Recommended backend for n={len(data)}: {fwht.backend_name(recommended)}")
print()

# ============================================================================
# Example 5: Context API for repeated transforms
# ============================================================================
print("Example 5: Context API (efficient for repeated calls)")
print("-" * 70)

# Using context manager
with fwht.Context(backend=fwht.Backend.CPU) as ctx:
    for i in range(5):
        data = np.random.randn(128).astype(np.float64)
        ctx.transform(data)
    print(f"✓ Transformed 5 arrays using reusable context")
print()

# ============================================================================
# Example 6: Different data types
# ============================================================================
print("Example 6: Multiple data types")
print("-" * 70)

# int32 (most common for Boolean functions)
data_i32 = np.array([1, -1, -1, 1], dtype=np.int32)
fwht.transform(data_i32)
print(f"int32:   {data_i32}")

# float64 (for numerical applications)
data_f64 = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
fwht.transform(data_f64)
print(f"float64: {data_f64}")

# int8 (memory-efficient, but may overflow)
data_i8 = np.array([1, -1, -1, 1], dtype=np.int8)
fwht.transform(data_i8)
print(f"int8:    {data_i8}")
print()

# ============================================================================
# Example 7: Utility functions
# ============================================================================
print("Example 7: Utility functions")
print("-" * 70)

# Check if size is power of 2
sizes = [128, 256, 300, 1024]
for n in sizes:
    is_pow2 = fwht.is_power_of_2(n)
    if is_pow2:
        k = fwht.log2(n)
        print(f"n={n:4d}: power of 2 (k={k})")
    else:
        print(f"n={n:4d}: NOT power of 2")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print(f"pyfwht version: {fwht.__version__}")
print(f"C library version: {fwht.version()}")
print("=" * 70)
