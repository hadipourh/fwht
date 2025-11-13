"""
GPU-specific tests for pyfwht.

Run these tests on a machine with CUDA support:
    pytest tests/test_gpu.py -v
"""
import pytest
import numpy as np
import pyfwht as fwht


class TestGPUAvailability:
    """Test GPU detection and availability."""
    
    @pytest.mark.skipif(not fwht.has_gpu(), reason="GPU not available")
    def test_gpu_detected(self):
        """Verify GPU is available on this system."""
        assert fwht.has_gpu(), "GPU/CUDA not detected. Ensure CUDA is installed and USE_CUDA=1 during build."
    
    def test_gpu_backend_enum(self):
        """Verify GPU backend enum exists."""
        assert hasattr(fwht.Backend, 'GPU')
        assert fwht.Backend.GPU.value == 3


class TestGPUTransforms:
    """Test GPU transform correctness."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self):
        """Skip all tests in this class if GPU not available."""
        if not fwht.has_gpu():
            pytest.skip("GPU not available")
    
    def test_gpu_i32_basic(self):
        """Test basic int32 GPU transform."""
        data = np.array([1, -1, -1, 1], dtype=np.int32)
        fwht.transform(data, backend=fwht.Backend.GPU)
        expected = np.array([0, 0, 0, 4], dtype=np.int32)
        np.testing.assert_array_equal(data, expected)
    
    def test_gpu_f64_basic(self):
        """Test basic float64 GPU transform."""
        data = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        fwht.transform(data, backend=fwht.Backend.GPU)
        expected = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float64)
        np.testing.assert_allclose(data, expected, rtol=1e-10)
    
    def test_gpu_i8_basic(self):
        """Test basic int8 GPU transform."""
        data = np.array([1, -1, -1, 1], dtype=np.int8)
        fwht.transform(data, backend=fwht.Backend.GPU)
        expected = np.array([0, 0, 0, 4], dtype=np.int8)
        np.testing.assert_array_equal(data, expected)
    
    def test_gpu_involution(self):
        """Test WHT involution property on GPU: WHT(WHT(x)) = n*x."""
        n = 256
        data = np.random.randint(-100, 100, n, dtype=np.int32)
        original = data.copy()
        
        fwht.transform(data, backend=fwht.Backend.GPU)
        fwht.transform(data, backend=fwht.Backend.GPU)
        
        np.testing.assert_array_equal(data, n * original)
    
    def test_gpu_linearity(self):
        """Test WHT linearity on GPU: WHT(a*x + b*y) = a*WHT(x) + b*WHT(y)."""
        n = 128
        x = np.random.randn(n).astype(np.float64)
        y = np.random.randn(n).astype(np.float64)
        a, b = 2.5, -1.3
        
        # Compute WHT(a*x + b*y)
        combined = a * x + b * y
        fwht.transform(combined, backend=fwht.Backend.GPU)
        
        # Compute a*WHT(x) + b*WHT(y)
        fwht.transform(x, backend=fwht.Backend.GPU)
        fwht.transform(y, backend=fwht.Backend.GPU)
        expected = a * x + b * y
        
        np.testing.assert_allclose(combined, expected, rtol=1e-10)
    
    def test_gpu_large_array(self):
        """Test GPU with large array (2^20 = 1M elements)."""
        n = 2**20
        data = np.random.randint(-10, 10, n, dtype=np.int32)
        original = data.copy()
        
        fwht.transform(data, backend=fwht.Backend.GPU)
        fwht.transform(data, backend=fwht.Backend.GPU)
        
        np.testing.assert_array_equal(data, n * original)
    
    def test_gpu_compute_i32(self):
        """Test GPU out-of-place transform for int32."""
        original = np.array([1, -1, -1, 1], dtype=np.int32)
        result = fwht.compute(original, backend=fwht.Backend.GPU)
        
        # Original unchanged
        np.testing.assert_array_equal(original, [1, -1, -1, 1])
        # Result correct
        np.testing.assert_array_equal(result, [0, 0, 0, 4])
    
    def test_gpu_compute_f64(self):
        """Test GPU out-of-place transform for float64."""
        original = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        result = fwht.compute(original, backend=fwht.Backend.GPU)
        
        np.testing.assert_array_equal(original, [1.0, -1.0, -1.0, 1.0])
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 4.0], rtol=1e-10)


class TestGPUvsCPU:
    """Test GPU results match CPU results."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self):
        """Skip if GPU not available."""
        if not fwht.has_gpu():
            pytest.skip("GPU not available")
    
    @pytest.mark.parametrize("size", [16, 256, 4096, 65536])
    def test_gpu_cpu_consistency_i32(self, size):
        """Test GPU and CPU give identical results for int32."""
        data = np.random.randint(-1000, 1000, size, dtype=np.int32)
        
        data_cpu = data.copy()
        data_gpu = data.copy()
        
        fwht.transform(data_cpu, backend=fwht.Backend.CPU)
        fwht.transform(data_gpu, backend=fwht.Backend.GPU)
        
        np.testing.assert_array_equal(data_cpu, data_gpu)
    
    @pytest.mark.parametrize("size", [16, 256, 4096, 65536])
    def test_gpu_cpu_consistency_f64(self, size):
        """Test GPU and CPU give identical results for float64."""
        data = np.random.randn(size).astype(np.float64)
        
        data_cpu = data.copy()
        data_gpu = data.copy()
        
        fwht.transform(data_cpu, backend=fwht.Backend.CPU)
        fwht.transform(data_gpu, backend=fwht.Backend.GPU)
        
        np.testing.assert_allclose(data_cpu, data_gpu, rtol=1e-10)
    
    @pytest.mark.parametrize("size", [16, 256, 4096])
    def test_gpu_openmp_consistency(self, size):
        """Test GPU and OpenMP give identical results."""
        if not fwht.has_openmp():
            pytest.skip("OpenMP not available")
        
        data = np.random.randint(-1000, 1000, size, dtype=np.int32)
        
        data_openmp = data.copy()
        data_gpu = data.copy()
        
        fwht.transform(data_openmp, backend=fwht.Backend.OPENMP)
        fwht.transform(data_gpu, backend=fwht.Backend.GPU)
        
        np.testing.assert_array_equal(data_openmp, data_gpu)


class TestGPUContext:
    """Test GPU context API."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self):
        """Skip if GPU not available."""
        if not fwht.has_gpu():
            pytest.skip("GPU not available")
    
    def test_gpu_context_basic(self):
        """Test creating and using GPU context."""
        with fwht.Context(backend=fwht.Backend.GPU) as ctx:
            data = np.array([1, -1, -1, 1], dtype=np.int32)
            ctx.transform(data)
            np.testing.assert_array_equal(data, [0, 0, 0, 4])
    
    def test_gpu_context_multiple_transforms(self):
        """Test GPU context with multiple transforms."""
        with fwht.Context(backend=fwht.Backend.GPU) as ctx:
            for _ in range(100):
                data = np.random.randint(-10, 10, 256, dtype=np.int32)
                original = data.copy()
                
                ctx.transform(data)
                ctx.transform(data)
                
                np.testing.assert_array_equal(data, 256 * original)
    
    def test_gpu_context_different_sizes(self):
        """Test GPU context with different array sizes."""
        with fwht.Context(backend=fwht.Backend.GPU) as ctx:
            for k in range(4, 12):
                n = 2**k
                data = np.random.randint(-100, 100, n, dtype=np.int32)
                original = data.copy()
                
                ctx.transform(data)
                ctx.transform(data)
                
                np.testing.assert_array_equal(data, n * original)


class TestGPUPerformance:
    """Performance-oriented tests for GPU."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self):
        """Skip if GPU not available."""
        if not fwht.has_gpu():
            pytest.skip("GPU not available")
    
    def test_gpu_very_large_array(self):
        """Test GPU with very large array (2^24 = 16M elements)."""
        n = 2**24
        print(f"\nTesting GPU with n={n:,} elements...")
        
        data = np.random.randint(-10, 10, n, dtype=np.int32)
        original = data.copy()
        
        import time
        start = time.time()
        fwht.transform(data, backend=fwht.Backend.GPU)
        gpu_time = time.time() - start
        print(f"GPU transform time: {gpu_time:.4f}s")
        
        # Verify correctness with involution
        fwht.transform(data, backend=fwht.Backend.GPU)
        np.testing.assert_array_equal(data, n * original)
    
    def test_gpu_batch_processing(self):
        """Test GPU batch processing performance."""
        n_arrays = 1000
        size = 1024
        print(f"\nProcessing {n_arrays} arrays of size {size} on GPU...")
        
        import time
        
        # With context (should be faster)
        start = time.time()
        with fwht.Context(backend=fwht.Backend.GPU) as ctx:
            for _ in range(n_arrays):
                data = np.random.randint(-10, 10, size, dtype=np.int32)
                ctx.transform(data)
        context_time = time.time() - start
        print(f"With context: {context_time:.4f}s ({n_arrays/context_time:.1f} transforms/s)")
        
        # Without context (for comparison)
        start = time.time()
        for _ in range(n_arrays):
            data = np.random.randint(-10, 10, size, dtype=np.int32)
            fwht.transform(data, backend=fwht.Backend.GPU)
        no_context_time = time.time() - start
        print(f"Without context: {no_context_time:.4f}s ({n_arrays/no_context_time:.1f} transforms/s)")
        print(f"Speedup: {no_context_time/context_time:.2f}x")


class TestGPUBooleanFunctions:
    """Test GPU with boolean function operations."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self):
        """Skip if GPU not available."""
        if not fwht.has_gpu():
            pytest.skip("GPU not available")
    
    def test_gpu_from_bool(self):
        """Test from_bool works with GPU backend."""
        # XOR function
        xor_table = np.array([0, 1, 1, 0], dtype=np.uint8)
        wht = fwht.from_bool(xor_table, signed=True)
        
        # XOR has maximum correlation 1.0 with itself
        expected = np.array([0, 0, 0, 4], dtype=np.int32)
        np.testing.assert_array_equal(wht, expected)
    
    def test_gpu_correlations(self):
        """Test correlations computation."""
        # XOR function: should have max |correlation| = 1.0
        xor_table = np.array([0, 1, 1, 0], dtype=np.uint8)
        corr = fwht.correlations(xor_table)
        
        max_corr = np.max(np.abs(corr))
        assert np.isclose(max_corr, 1.0, rtol=1e-10)
        
        # Should have exactly one coefficient with |corr| = 1.0
        assert np.sum(np.abs(corr) > 0.99) == 1


if __name__ == "__main__":
    # Quick standalone test
    print("Testing GPU availability...")
    print(f"GPU available: {fwht.has_gpu()}")
    
    if fwht.has_gpu():
        print("\nRunning basic GPU test...")
        data = np.array([1, -1, -1, 1], dtype=np.int32)
        print(f"Input: {data}")
        fwht.transform(data, backend=fwht.Backend.GPU)
        print(f"Output: {data}")
        print("✓ GPU test passed!")
    else:
        print("GPU not available. Build with USE_CUDA=1 to enable.")
