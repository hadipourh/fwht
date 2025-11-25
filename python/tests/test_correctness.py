"""
Comprehensive correctness tests for pyfwht Python bindings.

Tests all CPU/correctness functionality including:
- Basic transforms (int32, float64, int8)
- Overflow detection
- Batch processing
- Boolean function API
- Out-of-place transforms
- Backend consistency

Run with: pytest tests/test_correctness.py -v
"""

import numpy as np
import pytest

try:
    import pyfwht
except ImportError:
    pytest.skip("pyfwht not installed", allow_module_level=True)


class TestBasicTransforms:
    """Test basic transform functionality."""
    
    def test_fwht_i32_basic(self):
        """Test int32 transform with fwht()."""
        data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        result = pyfwht.fwht(data)
        
        # Verify result is modified
        assert not np.array_equal(result, [1, -1, -1, 1, -1, 1, 1, -1])
        
        # Check involution: fwht(fwht(x)) = n*x
        result2 = pyfwht.fwht(result)
        expected = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32) * 8
        np.testing.assert_array_equal(result2, expected)
    
    def test_fwht_f64_basic(self):
        """Test float64 transform."""
        data = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        result = pyfwht.fwht(data)
        result2 = pyfwht.fwht(result)
        expected = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64) * 4
        np.testing.assert_allclose(result2, expected, rtol=1e-10)
    
    def test_fwht_i8_basic(self):
        """Test int8 transform (small arrays to avoid overflow) - CPU only."""
        data = np.array([1, -1, -1, 1], dtype=np.int8)
        # int8 only supports CPU/AUTO backends
        result = pyfwht.fwht(data, backend='cpu')
        assert result.dtype == np.int8
    
    def test_fwht_constant_zero(self):
        """Test constant function f(x)=0 (represented as all +1)."""
        data = np.ones(8, dtype=np.int32)
        result = pyfwht.fwht(data)
        # WHT[0] = n, all others = 0
        assert result[0] == 8
        assert np.all(result[1:] == 0)
    
    def test_fwht_constant_one(self):
        """Test constant function f(x)=1 (represented as all -1)."""
        data = -np.ones(8, dtype=np.int32)
        result = pyfwht.fwht(data)
        # WHT[0] = -n, all others = 0
        assert result[0] == -8
        assert np.all(result[1:] == 0)
    
    def test_fwht_linear_function(self):
        """Test linear function (single bit)."""
        # f(x) = x_0 (first bit) → alternating pattern
        data = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int32)
        result = pyfwht.fwht(data)
        # Perfect correlation with u=1
        assert result[1] == 8
        assert result[0] == 0
        assert np.all(result[2:] == 0)


class TestOverflowDetection:
    """Test overflow detection with transform_safe()."""
    
    def test_overflow_addition(self):
        """Test detection of addition overflow."""
        data = np.array([2147483647, 2, 0, 0], dtype=np.int32)  # INT32_MAX + positive
        with pytest.raises(RuntimeError, match="[Oo]verflow"):
            pyfwht.transform_safe(data)
    
    def test_overflow_subtraction(self):
        """Test detection of subtraction overflow."""
        data = np.array([-2147483648, -2, 0, 0], dtype=np.int32)  # INT32_MIN - positive
        with pytest.raises(RuntimeError, match="[Oo]verflow"):
            pyfwht.transform_safe(data)
    
    def test_no_overflow_safe_values(self):
        """Test that safe values don't trigger overflow."""
        data = np.array([1000, -500, 250, -125], dtype=np.int32)
        pyfwht.transform_safe(data)  # Should not raise


class TestBatchProcessing:
    """Test vectorized batch processing."""
    
    def test_vectorized_batch_i32(self):
        """Test int32 batch processing."""
        batch_size = 10
        n = 256
        
        # Create batch of random arrays
        batch = [np.random.randint(-100, 100, n, dtype=np.int32) for _ in range(batch_size)]
        originals = [arr.copy() for arr in batch]
        
        # Process batch - new API requires n parameter
        pyfwht.vectorized_batch_i32(batch, n)
        
        # Verify each matches individual transform
        for i, (result, original) in enumerate(zip(batch, originals)):
            expected = pyfwht.fwht(original.copy())
            np.testing.assert_array_equal(result, expected, 
                                         err_msg=f"Mismatch at batch index {i}")
    
    def test_vectorized_batch_f64(self):
        """Test float64 batch processing."""
        n = 128
        batch = [np.random.randn(n).astype(np.float64) for _ in range(5)]
        originals = [arr.copy() for arr in batch]
        
        # Process batch - new API requires n parameter
        pyfwht.vectorized_batch_f64(batch, n)
        
        for result, original in zip(batch, originals):
            expected = pyfwht.fwht(original.copy())
            np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_batch_various_sizes(self):
        """Test batch with various power-of-2 sizes."""
        for size in [2, 4, 8, 16, 32, 64, 128, 256]:
            batch = [np.random.randint(-10, 10, size, dtype=np.int32) for _ in range(3)]
            originals = [arr.copy() for arr in batch]
            
            # Process batch - new API requires size parameter
            pyfwht.vectorized_batch_i32(batch, size)
            
            for result, original in zip(batch, originals):
                expected = pyfwht.fwht(original.copy())
                np.testing.assert_array_equal(result, expected)


class TestBooleanFunctions:
    """Test Boolean function API."""
    
    def test_from_bool_xor(self):
        """Test XOR function: f(x,y) = x ⊕ y."""
        # XOR truth table for 2 variables
        truth_table = np.array([0, 1, 1, 0], dtype=np.uint8)
        wht = pyfwht.from_bool(truth_table, signed=True)
        
        assert wht.dtype == np.int32
        assert len(wht) == 4
        # XOR has perfect correlation (WHT coefficient = ±n)
        assert any(abs(wht) == 4)
    
    def test_from_bool_constant(self):
        """Test constant Boolean functions."""
        # Constant 0
        truth_zero = np.zeros(8, dtype=np.uint8)
        wht_zero = pyfwht.from_bool(truth_zero, signed=True)
        assert wht_zero[0] == 8  # All positive
        assert np.all(wht_zero[1:] == 0)
        
        # Constant 1
        truth_one = np.ones(8, dtype=np.uint8)
        wht_one = pyfwht.from_bool(truth_one, signed=True)
        assert wht_one[0] == -8  # All negative
        assert np.all(wht_one[1:] == 0)
    
    def test_correlations(self):
        """Test correlation computation."""
        # XOR function
        truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        corr = pyfwht.correlations(truth_table)
        
        assert corr.dtype == np.float64
        assert len(corr) == 8
        # Correlations in [-1, 1]
        assert np.all(np.abs(corr) <= 1.0)
        # Parseval: sum of squared correlations = 1
        assert abs(np.sum(corr**2) - 1.0) < 1e-10
    
    def test_correlations_perfect(self):
        """Test perfect correlation detection."""
        # Single-bit function (linear)
        truth_table = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        corr = pyfwht.correlations(truth_table)
        
        # Should have max |correlation| = 1.0
        max_corr = np.max(np.abs(corr))
        assert np.isclose(max_corr, 1.0, rtol=1e-10)
    
    def test_boolean_batch(self):
        """Test batch processing of packed Boolean functions."""
        # Create multiple packed functions
        packed_list = [
            np.array([0x96], dtype=np.uint64),  # XOR-like
            np.array([0xFF], dtype=np.uint64),  # All ones
            np.array([0x00], dtype=np.uint64),  # All zeros
        ]
        
        results = pyfwht.boolean_batch(packed_list, n=8)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert len(result) == 8
            assert result.dtype == np.int32


class TestOutOfPlace:
    """Test out-of-place compute functions."""
    
    def test_compute_i32(self):
        """Test int32 out-of-place transform."""
        original = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        result = pyfwht.compute(original)
        
        # Original unchanged
        assert np.array_equal(original, [1, -1, -1, 1, -1, 1, 1, -1])
        
        # Result matches in-place
        expected = pyfwht.fwht(original.copy())
        assert np.array_equal(result, expected)
    
    def test_compute_f64(self):
        """Test float64 out-of-place transform."""
        original = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        result = pyfwht.compute(original)
        
        # Original unchanged
        np.testing.assert_array_equal(original, [1.0, -1.0, -1.0, 1.0])
        
        # Result matches
        expected = pyfwht.fwht(original.copy())
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestBackends:
    """Test backend selection and consistency."""
    
    def test_cpu_backend(self):
        """Test explicit CPU backend."""
        data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        result = pyfwht.fwht(data, backend='cpu')
        # Verify it ran (non-trivial output)
        assert not np.all(result == data)
    
    @pytest.mark.skipif(not pyfwht.has_openmp(), reason="OpenMP not available")
    def test_cpu_vs_openmp_consistency(self):
        """Test CPU and OpenMP give same results."""
        data = np.random.randn(256).astype(np.float64)
        
        result_cpu = pyfwht.fwht(data.copy(), backend='cpu')
        result_omp = pyfwht.fwht(data.copy(), backend='openmp')
        
        np.testing.assert_allclose(result_cpu, result_omp, rtol=1e-10)
    
    def test_auto_backend(self):
        """Test AUTO backend selection."""
        data = np.array([1, -1, -1, 1], dtype=np.int32)
        result = pyfwht.fwht(data, backend='auto')
        # Should produce valid output
        assert result is not None


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_size_1(self):
        """Test WHT of size 1 (identity)."""
        data = np.array([42], dtype=np.int32)
        result = pyfwht.fwht(data)
        assert result[0] == 42
    
    def test_size_2(self):
        """Test smallest non-trivial size."""
        data = np.array([1, -1], dtype=np.int32)
        result = pyfwht.fwht(data)
        expected = np.array([0, 2], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_large_size(self):
        """Test large array (2^16 = 65536)."""
        n = 65536
        data = np.random.randint(-10, 10, n, dtype=np.int32)
        original = data.copy()
        
        result = pyfwht.fwht(data)
        result2 = pyfwht.fwht(result)
        
        # Involution property
        np.testing.assert_array_equal(result2, original * n)
    
    def test_invalid_size_not_power_of_2(self):
        """Test that non-power-of-2 size raises error."""
        data = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises((ValueError, RuntimeError)):
            pyfwht.fwht(data)
    
    def test_invalid_dtype(self):
        """Test that unsupported dtype raises error."""
        data = np.array([1, 2, 3, 4], dtype=np.int16)
        with pytest.raises(TypeError):
            pyfwht.fwht(data)
    
    def test_multidimensional_array(self):
        """Test that 2D arrays are handled correctly."""
        # 2D arrays should be treated as batch
        data = np.array([[1, -1, -1, 1],
                        [1, 1, -1, -1]], dtype=np.int32)
        result = pyfwht.fwht(data)
        
        # Each row should be transformed
        assert result.shape == (2, 4)


class TestUtilities:
    """Test utility functions."""
    
    def test_is_power_of_2(self):
        """Test power-of-2 check."""
        assert pyfwht.is_power_of_2(1)
        assert pyfwht.is_power_of_2(2)
        assert pyfwht.is_power_of_2(256)
        assert pyfwht.is_power_of_2(1024)
        
        assert not pyfwht.is_power_of_2(0)
        assert not pyfwht.is_power_of_2(3)
        assert not pyfwht.is_power_of_2(100)
    
    def test_log2(self):
        """Test log2 computation."""
        assert pyfwht.log2(1) == 0
        assert pyfwht.log2(2) == 1
        assert pyfwht.log2(256) == 8
        assert pyfwht.log2(1024) == 10
        
        # Non-power-of-2 returns -1
        assert pyfwht.log2(3) == -1
    
    def test_version(self):
        """Test version string."""
        ver = pyfwht.version()
        assert isinstance(ver, str)
        assert len(ver) > 0
        # Should match semver pattern
        assert '.' in ver


class TestMathematicalProperties:
    """Test mathematical properties of WHT."""
    
    def test_involution(self):
        """Test WHT(WHT(x)) = n*x."""
        n = 16
        data = np.random.randint(-100, 100, n, dtype=np.int32)
        original = data.copy()
        
        result = pyfwht.fwht(data)
        result2 = pyfwht.fwht(result)
        
        np.testing.assert_array_equal(result2, n * original)
    
    def test_linearity(self):
        """Test WHT(a*x + b*y) = a*WHT(x) + b*WHT(y)."""
        n = 8
        x = np.random.randn(n).astype(np.float64)
        y = np.random.randn(n).astype(np.float64)
        a, b = 2.5, -1.3
        
        # Compute WHT(a*x + b*y)
        combined = a * x + b * y
        wht_combined = pyfwht.fwht(combined)
        
        # Compute a*WHT(x) + b*WHT(y)
        wht_x = pyfwht.fwht(x.copy())
        wht_y = pyfwht.fwht(y.copy())
        expected = a * wht_x + b * wht_y
        
        np.testing.assert_allclose(wht_combined, expected, rtol=1e-10)
    
    def test_orthogonality(self):
        """Test orthogonality: WHT matrix is orthogonal (up to scaling)."""
        n = 16
        
        # Create identity-like input (basis vector)
        for i in range(min(n, 4)):  # Test first 4 basis vectors
            data = np.zeros(n, dtype=np.float64)
            data[i] = 1.0
            
            result = pyfwht.fwht(data)
            
            # Energy preservation (up to scaling by n)
            energy_in = np.sum(data**2)
            energy_out = np.sum(result**2)
            assert np.isclose(energy_out, energy_in * n, rtol=1e-10)


class TestConsistency:
    """Test consistency between different dtypes."""
    
    def test_i32_vs_f64_consistency(self):
        """Test that int32 and float64 give equivalent results."""
        data_i32 = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
        data_f64 = data_i32.astype(np.float64)
        
        result_i32 = pyfwht.fwht(data_i32.copy())
        result_f64 = pyfwht.fwht(data_f64.copy())
        
        # Results should match (convert to float for comparison)
        np.testing.assert_allclose(result_i32.astype(np.float64), result_f64, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
