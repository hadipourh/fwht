"""
Basic tests for pyfwht package.

Tests core transform functionality with different dtypes and backends.
"""
import numpy as np
import pytest

try:
    import pyfwht as fwht
except ImportError:
    pytest.skip("pyfwht not installed", allow_module_level=True)


class TestBasicTransforms:
    """Test basic in-place transforms."""
    
    def test_transform_i32(self):
        """Test int32 transform."""
        data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        fwht.transform(data)
        
        # Check involution property: WHT(WHT(x)) = n*x
        original = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        fwht.transform(data)
        assert np.array_equal(data, original * 8)
    
    def test_transform_f64(self):
        """Test float64 transform."""
        data = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        fwht.transform(data)
        
        # Check involution
        original = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        fwht.transform(data)
        np.testing.assert_array_almost_equal(data, original * 4)
    
    def test_transform_i8(self):
        """Test int8 transform (small arrays only to avoid overflow)."""
        data = np.array([1, -1, -1, 1], dtype=np.int8)
        fwht.transform(data)
        
        # Check result
        assert data.dtype == np.int8
    
    def test_invalid_dtype(self):
        """Test that invalid dtypes raise error."""
        data = np.array([1, 2, 3, 4], dtype=np.int16)
        with pytest.raises(TypeError, match="Unsupported dtype"):
            fwht.transform(data)
    
    def test_invalid_shape(self):
        """Test that non-1D arrays raise error."""
        data = np.array([[1, 2], [3, 4]], dtype=np.int32)
        with pytest.raises(ValueError, match="1-dimensional"):
            fwht.transform(data)
    
    def test_invalid_size(self):
        """Test that non-power-of-2 sizes raise error."""
        data = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises((ValueError, RuntimeError)):
            fwht.transform(data)


class TestOutOfPlace:
    """Test out-of-place compute functions."""
    
    def test_compute_i32(self):
        """Test int32 out-of-place transform."""
        original = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        result = fwht.compute(original)
        
        # Original should be unchanged
        assert np.array_equal(original, [1, -1, -1, 1, -1, 1, 1, -1])
        
        # Result should match in-place
        expected = original.copy()
        fwht.transform(expected)
        assert np.array_equal(result, expected)
    
    def test_compute_f64(self):
        """Test float64 out-of-place transform."""
        original = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        result = fwht.compute(original)
        
        # Original unchanged
        np.testing.assert_array_equal(original, [1.0, -1.0, -1.0, 1.0])
        
        # Result matches in-place
        expected = original.copy()
        fwht.transform(expected)
        np.testing.assert_array_almost_equal(result, expected)


class TestBooleanFunctions:
    """Test Boolean function convenience API."""
    
    def test_from_bool_signed(self):
        """Test Boolean to WHT conversion with signed representation."""
        # XOR function: f(x,y) = x ⊕ y
        truth_table = np.array([0, 1, 1, 0], dtype=np.uint8)
        wht = fwht.from_bool(truth_table, signed=True)
        
        assert wht.dtype == np.int32
        assert len(wht) == 4
        # XOR has perfect correlation with one linear function
        assert any(abs(wht) == 4)
    
    def test_from_bool_unsigned(self):
        """Test Boolean to WHT conversion without signed representation."""
        truth_table = np.array([0, 1, 1, 0], dtype=np.uint8)
        wht = fwht.from_bool(truth_table, signed=False)
        
        assert wht.dtype == np.int32
        assert len(wht) == 4
    
    def test_correlations(self):
        """Test correlation computation."""
        truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        corr = fwht.correlations(truth_table)
        
        assert corr.dtype == np.float64
        assert len(corr) == 8
        # Correlations should be in [-1, 1]
        assert np.all(np.abs(corr) <= 1.0)
        # Sum of squared correlations should equal 1 (Parseval)
        assert abs(np.sum(corr**2) - 1.0) < 1e-10


class TestBackends:
    """Test explicit backend selection."""
    
    def test_cpu_backend(self):
        """Test CPU backend."""
        data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        fwht.transform(data, backend=fwht.Backend.CPU)
        # Just verify it runs without error
    
    @pytest.mark.skipif(not fwht.has_openmp(), reason="OpenMP not available")
    def test_openmp_backend(self):
        """Test OpenMP backend if available."""
        data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        fwht.transform(data, backend=fwht.Backend.OPENMP)
    
    @pytest.mark.skipif(not fwht.has_gpu(), reason="GPU not available")
    def test_gpu_backend(self):
        """Test GPU backend if available."""
        data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        fwht.transform(data, backend=fwht.Backend.GPU)
    
    def test_auto_backend(self):
        """Test AUTO backend selection."""
        data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        fwht.transform(data, backend=fwht.Backend.AUTO)


class TestContext:
    """Test context API for repeated transforms."""
    
    def test_context_basic(self):
        """Test basic context usage."""
        ctx = fwht.Context(backend=fwht.Backend.CPU)
        data = np.array([1, -1, -1, 1], dtype=np.int32)
        ctx.transform(data)
        ctx.close()
    
    def test_context_manager(self):
        """Test context manager protocol."""
        with fwht.Context(backend=fwht.Backend.CPU) as ctx:
            data1 = np.array([1, -1, -1, 1], dtype=np.int32)
            ctx.transform(data1)
            
            data2 = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            ctx.transform(data2)
    
    def test_context_closed_error(self):
        """Test that using closed context raises error."""
        ctx = fwht.Context()
        ctx.close()
        
        data = np.array([1, -1, -1, 1], dtype=np.int32)
        with pytest.raises(RuntimeError, match="closed"):
            ctx.transform(data)


class TestUtilities:
    """Test utility functions."""
    
    def test_is_power_of_2(self):
        """Test power-of-2 check."""
        assert fwht.is_power_of_2(1)
        assert fwht.is_power_of_2(2)
        assert fwht.is_power_of_2(256)
        assert fwht.is_power_of_2(1024)
        
        assert not fwht.is_power_of_2(0)
        assert not fwht.is_power_of_2(3)
        assert not fwht.is_power_of_2(100)
    
    def test_log2(self):
        """Test log2 computation."""
        assert fwht.log2(1) == 0
        assert fwht.log2(2) == 1
        assert fwht.log2(256) == 8
        assert fwht.log2(1024) == 10
        
        # Non-power-of-2 should return -1
        assert fwht.log2(3) == -1
    
    def test_recommend_backend(self):
        """Test backend recommendation."""
        backend = fwht.recommend_backend(256)
        assert isinstance(backend, fwht.Backend)
    
    def test_version(self):
        """Test version string."""
        ver = fwht.version()
        assert isinstance(ver, str)
        assert len(ver) > 0


class TestConsistency:
    """Test consistency between different backends."""
    
    @pytest.mark.skipif(not fwht.has_openmp(), reason="OpenMP not available")
    def test_cpu_vs_openmp(self):
        """Test CPU vs OpenMP consistency."""
        data_cpu = np.random.randn(256).astype(np.float64)
        data_omp = data_cpu.copy()
        
        fwht.transform(data_cpu, backend=fwht.Backend.CPU)
        fwht.transform(data_omp, backend=fwht.Backend.OPENMP)
        
        np.testing.assert_array_almost_equal(data_cpu, data_omp)
    
    @pytest.mark.skipif(not fwht.has_gpu(), reason="GPU not available")
    def test_cpu_vs_gpu(self):
        """Test CPU vs GPU consistency."""
        data_cpu = np.random.randn(1024).astype(np.float64)
        data_gpu = data_cpu.copy()
        
        fwht.transform(data_cpu, backend=fwht.Backend.CPU)
        fwht.transform(data_gpu, backend=fwht.Backend.GPU)
        
        # GPU may have small numerical errors
        np.testing.assert_array_almost_equal(data_cpu, data_gpu, decimal=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
