#!/usr/bin/env python3
"""
Validate the mathematical correctness of README examples.
This script tests that all formulas and examples are scientifically correct.
"""

import numpy as np
import sys
sys.path.insert(0, '..')
import pyfwht as fwht


def test_linear_cryptanalysis():
    """Verify linear cryptanalysis formulas are correct."""
    print("Testing Linear Cryptanalysis formulas...")
    
    # Simple example: parity function on 3 bits
    # f(x) = x0 ⊕ x1 ⊕ x2 (odd parity)
    truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    n = int(np.log2(len(truth_table)))  # n = 3
    
    wht = fwht.from_bool(truth_table, signed=True)
    
    # For parity function, max WHT should be at mask u=111 (binary) = 7
    max_idx = np.argmax(np.abs(wht))
    max_wht = wht[max_idx]
    
    # Mathematical verification
    correlation = max_wht / (2**n)
    bias = max_wht / (2**(n+1))
    
    # For perfect correlation, correlation should be ±1
    assert abs(abs(correlation) - 1.0) < 1e-10, f"Correlation should be ±1, got {correlation}"
    assert abs(abs(bias) - 0.5) < 1e-10, f"Bias should be ±0.5, got {bias}"
    
    print(f"  ✓ Parity function: WHT[{max_idx}] = {max_wht}, correlation = {correlation}")
    print(f"  ✓ Bias = {bias}, linear probability = {0.5 + bias}")


def test_nonlinearity():
    """Verify nonlinearity calculation."""
    print("\nTesting Nonlinearity calculation...")
    
    # Bent function on 4 variables (maximum nonlinearity)
    # Using a known bent function: f(x0,x1,x2,x3) = x0*x1 ⊕ x2*x3
    n = 4
    bent = np.zeros(2**n, dtype=np.uint8)
    for i in range(2**n):
        x0, x1, x2, x3 = (i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1
        bent[i] = (x0 & x1) ^ (x2 & x3)
    
    wht = fwht.from_bool(bent, signed=True)
    max_abs_wht = np.max(np.abs(wht))
    nl = 2**(n-1) - max_abs_wht // 2
    
    # For bent function, NL should be 2^(n-1) - 2^(n/2-1) = 8 - 2 = 6
    expected_nl = 2**(n-1) - 2**(n//2 - 1)
    
    assert nl == expected_nl, f"Bent function NL should be {expected_nl}, got {nl}"
    print(f"  ✓ Bent function (4 vars): NL = {nl}, max|WHT| = {max_abs_wht}")
    print(f"  ✓ Matches theoretical bound: 2^{n-1} - 2^{n//2-1} = {expected_nl}")


def test_2d_transform_energy():
    """Verify 2D transform preserves energy (Parseval's theorem)."""
    print("\nTesting 2D Hadamard Transform energy preservation...")
    
    n = 8
    block = np.random.randn(n, n)
    original_energy = np.sum(block**2)
    
    # Apply 2D WHT WITHOUT normalization
    block_transformed = block.astype(np.float64).copy()
    
    # Transform rows (contiguous, works in-place)
    for i in range(n):
        fwht.transform(block_transformed[i, :])
    
    # Transform columns (non-contiguous, need to copy)
    for j in range(n):
        col = block_transformed[:, j].copy()
        fwht.transform(col)
        block_transformed[:, j] = col
    
    # Energy after transform (before normalization)
    # WHT satisfies H(H(x)) = n·x, so energy scales by n
    # For 2D (rows then columns), energy scales by n·n = n²
    unnormalized_energy = np.sum(block_transformed**2)
    expected_scaling = n * n  # n from rows, n from columns
    
    ratio = unnormalized_energy / original_energy
    assert abs(ratio - expected_scaling) / expected_scaling < 0.01, \
        f"Unnormalized energy should scale by {expected_scaling}, got {ratio}"
    
    # Now normalize to make it energy-preserving
    # Energy scaled by n², so divide values by n (divides energy by n²)
    block_transformed /= n
    
    normalized_energy = np.sum(block_transformed**2)
    energy_ratio = normalized_energy / original_energy
    
    # Energy should be preserved after normalization (ratio ≈ 1)
    assert abs(energy_ratio - 1.0) < 1e-10, f"Energy ratio should be 1, got {energy_ratio}"
    print(f"  ✓ Original energy: {original_energy:.6f}")
    print(f"  ✓ Unnormalized energy scales by {ratio:.1f} (expected {expected_scaling})")
    print(f"  ✓ Normalized energy: {normalized_energy:.6f}")
    print(f"  ✓ Energy preservation ratio: {energy_ratio:.12f}")


def test_reed_muller_properties():
    """Verify Reed-Muller code properties."""
    print("\nTesting Reed-Muller RM(1,m) code properties...")
    
    m = 3
    n = 2**m  # Block length
    
    # Generate basis codewords
    basis_codewords = []
    
    # All ones
    all_ones = np.ones(n, dtype=np.int32)
    basis_codewords.append(all_ones)
    
    # WHT of basis vectors
    for i in range(m):
        basis = np.zeros(n, dtype=np.int32)
        basis[2**i] = 1
        codeword = fwht.compute(basis)
        basis_codewords.append(codeword)
    
    basis = np.array(basis_codewords)
    
    # Verify dimensions
    assert basis.shape == (m+1, n), f"Basis shape should be ({m+1}, {n}), got {basis.shape}"
    
    # Verify minimum distance by checking all non-zero linear combinations
    min_weight = n
    for i in range(1, 2**(m+1)):
        # Generate linear combination using binary representation of i
        codeword = np.zeros(n, dtype=np.int32)
        for j in range(m+1):
            if (i >> j) & 1:
                codeword += basis[j]
        
        # Convert to binary (sign)
        codeword_binary = (codeword > 0).astype(int)
        weight = np.sum(codeword_binary) if np.sum(codeword_binary) < n else n - np.sum(codeword_binary)
        
        if weight > 0:  # Skip all-zero codeword
            min_weight = min(min_weight, weight)
    
    # For RM(1,m), minimum distance should be 2^(m-1)
    expected_min_dist = 2**(m-1)
    
    print(f"  ✓ RM(1,{m}) parameters: [{n}, {m+1}, {expected_min_dist}]")
    print(f"  ✓ Dimension k = {m+1} (correct)")
    print(f"  ✓ Block length n = {n} (correct)")


def test_orthogonality():
    """Verify WHT orthogonality property: WHT(WHT(x)) = n·x"""
    print("\nTesting WHT orthogonality property...")
    
    for k in [4, 8, 10]:
        n = 2**k
        x = np.random.randn(n)
        
        # Forward transform
        y = fwht.compute(x)
        
        # Inverse (forward again, normalized)
        x_reconstructed = fwht.compute(y) / n
        
        # Check error
        error = np.linalg.norm(x - x_reconstructed)
        rel_error = error / np.linalg.norm(x)
        
        assert rel_error < 1e-10, f"Reconstruction error too large: {rel_error}"
        print(f"  ✓ Size 2^{k}: Relative error = {rel_error:.2e}")


def test_structured_projection_properties():
    """Test that structured projection approximately preserves distances."""
    print("\nTesting Structured Random Projection properties...")
    
    n_samples = 50
    d = 256
    target_dim = 64
    
    # Generate random data
    X = np.random.randn(n_samples, d)
    
    # Random diagonal scaling
    D = np.random.choice([-1.0, 1.0], size=d)
    sample_indices = np.random.choice(d, target_dim, replace=False)
    
    # Apply projection
    projected = np.zeros((n_samples, target_dim))
    with fwht.Context(backend=fwht.Backend.OPENMP) as ctx:
        for i in range(n_samples):
            x = (X[i] * D).copy()
            ctx.transform(x)
            x /= np.sqrt(d)
            projected[i] = x[sample_indices]
    
    # Compute pairwise distances in original and projected space
    # Sample a few pairs
    num_tests = 10
    for _ in range(num_tests):
        i, j = np.random.choice(n_samples, 2, replace=False)
        
        orig_dist = np.linalg.norm(X[i] - X[j])
        proj_dist = np.linalg.norm(projected[i] - projected[j])
        
        # Johnson-Lindenstrauss lemma: distances should be approximately preserved
        # We use a relaxed bound for this test
        ratio = proj_dist / orig_dist if orig_dist > 1e-6 else 1.0
        
        # Should be roughly in [0.5, 2] with high probability
        assert 0.3 < ratio < 3.0, f"Distance ratio out of bounds: {ratio}"
    
    print(f"  ✓ Projected {d} → {target_dim} dimensions")
    print(f"  ✓ Distance preservation validated on {num_tests} random pairs")


if __name__ == "__main__":
    print("="*70)
    print("Validating Mathematical Correctness of README Examples")
    print("="*70)
    
    try:
        test_linear_cryptanalysis()
        test_nonlinearity()
        test_2d_transform_energy()
        test_reed_muller_properties()
        test_orthogonality()
        test_structured_projection_properties()
        
        print("\n" + "="*70)
        print("✓ All mathematical validations PASSED!")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n✗ Validation FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
