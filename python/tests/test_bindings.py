#!/usr/bin/env python3
"""
Test script for pyfwht Python bindings.
Tests basic transforms, backends, context API, and GPU module (if available).
"""

import numpy as np
import pyfwht as fwht

def test_basic_transform():
    """Test basic in-place transform."""
    print("Testing basic transform...")
    data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
    original = data.copy()
    fwht.transform(data)
    print(f"  Input:  {original}")
    print(f"  Output: {data}")
    # Verify transform is reversible (applying twice gives scaled original)
    fwht.transform(data)
    assert np.allclose(data, original * 8), "Transform not reversible"
    print("  ✓ Basic transform works\n")

def test_compute():
    """Test out-of-place compute."""
    print("Testing out-of-place compute...")
    original = np.array([1, -1, -1, 1], dtype=np.int32)
    result = fwht.compute(original)
    print(f"  Input:  {original}")
    print(f"  Output: {result}")
    assert np.array_equal(original, [1, -1, -1, 1]), "Input was modified"
    print("  ✓ Out-of-place compute works\n")

def test_backends():
    """Test backend selection."""
    print("Testing backends...")
    data = np.random.randn(256)
    
    backends = [fwht.Backend.AUTO, fwht.Backend.CPU]
    if fwht.has_openmp():
        backends.append(fwht.Backend.OPENMP)
    if fwht.has_gpu():
        backends.append(fwht.Backend.GPU)
    
    for backend in backends:
        test_data = data.copy()
        fwht.transform(test_data, backend=backend)
        print(f"  ✓ {fwht.backend_name(backend)} backend works")
    print()

def test_context():
    """Test context API."""
    print("Testing context API...")
    
    # Test with context manager
    with fwht.Context(backend=fwht.Backend.CPU) as ctx:
        data1 = np.random.randn(128)
        ctx.transform(data1)
        
        data2 = np.random.randint(-100, 100, 128, dtype=np.int32)
        ctx.transform(data2)
        print("  ✓ Context manager works")
    
    # Test manual cleanup
    ctx = fwht.Context(backend=fwht.Backend.CPU)
    data = np.random.randn(64)
    ctx.transform(data)
    ctx.close()
    print("  ✓ Manual context cleanup works\n")

def test_boolean_api():
    """Test Boolean function API."""
    print("Testing Boolean API...")
    
    # XOR function: f(x,y) = x ⊕ y
    truth_table = np.array([0, 1, 1, 0], dtype=np.uint8)
    wht = fwht.from_bool(truth_table, signed=True)
    print(f"  Truth table: {truth_table}")
    print(f"  WHT coeffs:  {wht}")
    
    # Compute correlations
    corr = fwht.correlations(truth_table)
    print(f"  Correlations: {corr}")
    print("  ✓ Boolean API works\n")

def test_gpu_module():
    """Test GPU module (if available)."""
    print("Testing GPU module...")
    
    if not fwht.gpu.available:
        print("  ⚠ GPU not available (expected on macOS)")
        return
    
    print(f"  Device: {fwht.gpu.device_name()}")
    print(f"  Compute capability: {fwht.gpu.compute_capability()}")
    print(f"  SM count: {fwht.gpu.sm_count()}")
    print(f"  SMEM banks: {fwht.gpu.smem_banks()}")
    
    # Test profiling
    fwht.gpu.set_profiling(True)
    assert fwht.gpu.profiling_enabled(), "Profiling not enabled"
    
    data = np.random.randn(1024)
    fwht.transform(data, backend=fwht.Backend.GPU)
    
    metrics = fwht.gpu.get_last_metrics()
    print(f"  Metrics: H2D={metrics.h2d_ms:.3f}ms, "
          f"Kernel={metrics.kernel_ms:.3f}ms, "
          f"D2H={metrics.d2h_ms:.3f}ms")
    
    fwht.gpu.set_profiling(False)
    
    # Test batch operations
    batch = np.random.randint(-100, 100, (16, 256), dtype=np.int32)
    fwht.gpu.batch_transform_i32(batch)
    print("  ✓ Batch transform works")
    
    # Test GPU context
    with fwht.gpu.Context(max_n=512, batch_size=8) as ctx:
        data = np.random.randn(512)
        ctx.transform_f64(data)
        print("  ✓ GPU context works")
    
    print("  ✓ GPU module works\n")

def test_utilities():
    """Test utility functions."""
    print("Testing utilities...")
    
    assert fwht.is_power_of_2(256), "is_power_of_2 failed"
    assert not fwht.is_power_of_2(100), "is_power_of_2 failed"
    assert fwht.log2(256) == 8, "log2 failed"
    
    backend = fwht.recommend_backend(1024)
    print(f"  Recommended backend for N=1024: {fwht.backend_name(backend)}")
    
    print("  ✓ Utilities work\n")

def main():
    """Run all tests."""
    print("=" * 60)
    print("pyfwht Python Bindings Test Suite")
    print("=" * 60)
    print(f"Version: {fwht.version()}")
    print(f"OpenMP support: {fwht.has_openmp()}")
    print(f"GPU support: {fwht.has_gpu()}")
    print("=" * 60)
    print()
    
    test_basic_transform()
    test_compute()
    test_backends()
    test_context()
    test_boolean_api()
    test_utilities()
    test_gpu_module()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

if __name__ == "__main__":
    main()
