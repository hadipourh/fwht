"""
S-box Cryptanalysis Example using pyfwht

Demonstrates:
- Linear Approximation Table (LAT) computation for S-boxes
- Boolean component Walsh spectrum analysis
- Extracting cryptographic properties (nonlinearity, max bias)
- GPU acceleration for large S-boxes
"""

import numpy as np
import pyfwht as fwht

# AES S-box (256-entry lookup table)
AES_SBOX = np.array([
    99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118,
    202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192,
    183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21,
    4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117,
    9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132,
    83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207,
    208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168,
    81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210,
    205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115,
    96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219,
    224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121,
    231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8,
    186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138,
    112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158,
    225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223,
    140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22
], dtype=np.uint32)


def example_basic_sbox_analysis():
    """Basic S-box analysis: component spectra and LAT metrics."""
    print("=" * 70)
    print("Example 1: Basic S-box Analysis (AES S-box)")
    print("=" * 70)
    print()
    
    # Analyze S-box (8x8: 8 input bits, 8 output bits)
    result = fwht.analyze_sbox(
        AES_SBOX,
        compute_lat=True,      # Compute LAT metrics
        profile_timings=True   # Show performance breakdown
    )
    
    # Component analysis results
    print("Boolean Component Analysis:")
    print(f"  Input bits (m):        {result.components.m_bits}")
    print(f"  Output bits (n):       {result.components.n_bits}")
    print(f"  Table size:            {result.components.size}")
    print(f"  Max |Walsh|:           {result.components.max_walsh}")
    print(f"  Min nonlinearity:      {result.components.min_nonlinearity}")
    print(f"  FWHT time:             {result.components.fwht_ms:.3f} ms")
    print()
    
    # LAT analysis results
    if result.lat:
        print("Linear Approximation Table (LAT) Analysis:")
        print(f"  Max |LAT| entry:       {result.lat.lat_max} (excluding trivial LAT[0,0])")
        print(f"  Max bias:              {result.lat.lat_max_bias:.6f}")
        print(f"  Max correlation:       {2 * result.lat.lat_max_bias:.6f}  (correlation = 2 × bias)")
        print(f"  Column synthesis:      {result.lat.column_ms:.3f} ms")
        print(f"  Column FWHT:           {result.lat.fwht_ms:.3f} ms")
        print()
    
    # Cryptographic interpretation
    print("Cryptographic Properties:")
    print(f"  Nonlinearity:          {result.components.min_nonlinearity} (higher = better)")
    print(f"  Linear bias:           {result.lat.lat_max_bias:.6f} (lower = better)")
    print(f"  Linear correlation:    {2 * result.lat.lat_max_bias:.6f}")
    print(f"  Balanced:              {result.components.max_walsh == 0 or 'No' if result.components.max_walsh != 0 else 'Yes'}")
    print()


def example_with_spectra():
    """Retrieve full Walsh spectra for each Boolean component."""
    print("=" * 70)
    print("Example 2: Detailed Component Walsh Spectra")
    print("=" * 70)
    print()
    
    # Analyze with full spectra
    result = fwht.analyze_sbox(
        AES_SBOX,
        compute_lat=False,      # Skip LAT for this example
        return_spectra=True     # Get full Walsh spectra
    )
    
    if result.components.spectra is not None:
        print(f"Spectra shape: {result.components.spectra.shape}")
        print(f"  (n_components={result.components.spectra.shape[0]}, ")
        print(f"   spectrum_size={result.components.spectra.shape[1]})")
        print()
        
        # Analyze each Boolean component (output bit)
        for i in range(result.components.spectra.shape[0]):
            spectrum = result.components.spectra[i]
            max_walsh = np.max(np.abs(spectrum))
            nonlinearity = (result.components.size - max_walsh) / 2
            
            print(f"Component {i} (output bit {i}):")
            print(f"  Max |Walsh|:       {max_walsh}")
            print(f"  Nonlinearity:      {nonlinearity}")
            print(f"  Sample coeffs:     {spectrum[:8].tolist()}")
            print()


def example_with_full_lat():
    """Retrieve and analyze the complete LAT matrix."""
    print("=" * 70)
    print("Example 3: Full LAT Matrix Analysis")
    print("=" * 70)
    print()
    
    # Analyze with full LAT matrix
    result = fwht.analyze_sbox(
        AES_SBOX,
        compute_lat=True,
        return_lat=True         # Get full LAT matrix
    )
    
    if result.lat and result.lat.lat is not None:
        lat = result.lat.lat
        print(f"LAT matrix shape: {lat.shape}")
        print(f"  (size={lat.shape[0]}, output_masks={lat.shape[1]})")
        print()
        
        # Find strongest linear approximations
        abs_lat = np.abs(lat)
        max_positions = np.argwhere(abs_lat == result.lat.lat_max)
        
        print(f"Found {len(max_positions)} entries with max |LAT| = {result.lat.lat_max}:")
        for idx, (input_mask, output_mask) in enumerate(max_positions[:5]):
            lat_value = lat[input_mask, output_mask]
            print(f"  LAT[0x{input_mask:02x}, 0x{output_mask:02x}] = {lat_value:4d} "
                  f"(bias = {lat_value/256:.6f})")
        if len(max_positions) > 5:
            print(f"  ... and {len(max_positions) - 5} more")
        print()
        
        # Distribution of LAT entries
        unique, counts = np.unique(abs_lat, return_counts=True)
        print("LAT value distribution (top 5):")
        sorted_indices = np.argsort(counts)[::-1]
        for i in sorted_indices[:5]:
            print(f"  |LAT| = {unique[i]:3d}: {counts[i]:5d} entries")
        print()


def example_gpu_accelerated():
    """Use GPU acceleration for S-box analysis (if available)."""
    print("=" * 70)
    print("Example 4: GPU-Accelerated S-box Analysis")
    print("=" * 70)
    print()
    
    if not fwht.has_gpu():
        print("GPU not available. Skipping GPU example.")
        print("(Build with CUDA support to enable GPU acceleration)")
        return
    
    print(f"GPU: {fwht.gpu.device_name()}")
    print()
    
    # Analyze using GPU backend
    result = fwht.analyze_sbox(
        AES_SBOX,
        backend='gpu',          # Force GPU backend
        compute_lat=True,
        profile_timings=True
    )
    
    print("GPU Performance:")
    print(f"  Component FWHT:    {result.components.fwht_ms:.3f} ms")
    if result.lat:
        print(f"  LAT columns:       {result.lat.column_ms:.3f} ms")
        print(f"  LAT FWHT:          {result.lat.fwht_ms:.3f} ms")
        total_ms = result.lat.column_ms + result.lat.fwht_ms
        print(f"  Total LAT time:    {total_ms:.3f} ms")
    print()


def example_small_sbox():
    """Analyze a smaller 4-bit S-box for educational purposes."""
    print("=" * 70)
    print("Example 5: Small 4-bit S-box Analysis")
    print("=" * 70)
    print()
    
    # Example 4-bit S-box (PRESENT cipher S-box)
    present_sbox = np.array([
        0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
        0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2
    ], dtype=np.uint32)
    
    print("PRESENT S-box (4x4):")
    print(f"  Lookup table: {present_sbox.tolist()}")
    print()
    
    # Full analysis with spectra and LAT
    result = fwht.analyze_sbox(
        present_sbox,
        compute_lat=True,
        return_spectra=True,
        return_lat=True
    )
    
    print("Analysis Results:")
    print(f"  Size:              {result.components.size} (4 input bits)")
    print(f"  Max |Walsh|:       {result.components.max_walsh}")
    print(f"  Nonlinearity:      {result.components.min_nonlinearity}")
    if result.lat:
        print(f"  Max |LAT|:         {result.lat.lat_max} (excluding LAT[0,0])")
        print(f"  Max bias:          {result.lat.lat_max_bias:.6f}")
        print(f"  Max correlation:   {2 * result.lat.lat_max_bias:.6f}")
    print()
    
    # Show full LAT for small S-box (it's manageable to display)
    if result.lat and result.lat.lat is not None:
        print("Full LAT matrix:")
        print("     ", end="")
        for j in range(16):
            print(f"{j:4d}", end="")
        print()
        print("    " + "-" * 64)
        
        for i in range(16):
            print(f"{i:2d} | ", end="")
            for j in range(16):
                print(f"{result.lat.lat[i, j]:4d}", end="")
            print()
        print()


def example_large_sbox_gpu():
    """Analyze a large 16-bit random S-box using GPU acceleration."""
    print("=" * 70)
    print("Example 6: Large S-box LAT Computation on GPU (16-bit)")
    print("=" * 70)
    print()
    
    if not fwht.has_gpu():
        print("GPU not available. Skipping large S-box example.")
        print("(This example requires CUDA-enabled GPU)")
        return
    
    print(f"GPU: {fwht.gpu.device_name()}")
    print()
    
    # Generate random 16-bit S-box (65536 entries)
    print("Generating random 16-bit S-box (65536 entries)...")
    np.random.seed(42)
    large_sbox = np.random.permutation(65536).astype(np.uint32)
    print(f"  S-box size: {len(large_sbox)} entries")
    print(f"  Memory: {large_sbox.nbytes / 1024:.1f} KB")
    print()
    
    # Analyze with full LAT on GPU
    print("Computing full LAT on GPU...")
    result = fwht.analyze_sbox(
        large_sbox,
        backend='gpu',
        compute_lat=True,
        return_lat=True,        # Get full LAT matrix
        profile_timings=True
    )
    
    print()
    print("Results:")
    print(f"  Input/output bits:     {result.components.m_bits}×{result.components.n_bits}")
    print(f"  Max |Walsh|:           {result.components.max_walsh}")
    print(f"  Min nonlinearity:      {result.components.min_nonlinearity}")
    print()
    
    if result.lat:
        print("LAT Analysis:")
        print(f"  LAT shape:             {result.lat.lat.shape if result.lat.lat is not None else 'N/A'}")
        print(f"  LAT memory:            {result.lat.lat.nbytes / (1024**2):.1f} MB" if result.lat.lat is not None else "")
        print(f"  Max |LAT| entry:       {result.lat.lat_max} (excluding LAT[0,0])")
        print(f"  Max bias:              {result.lat.lat_max_bias:.6f}")
        print(f"  Max correlation:       {2 * result.lat.lat_max_bias:.6f}")
        print()
        
        print("Performance:")
        print(f"  Component FWHT:        {result.components.fwht_ms:.3f} ms")
        print(f"  LAT column synthesis:  {result.lat.column_ms:.3f} ms")
        print(f"  LAT FWHT:              {result.lat.fwht_ms:.3f} ms")
        total_ms = result.lat.column_ms + result.lat.fwht_ms
        print(f"  Total LAT time:        {total_ms:.3f} ms")
        print()
        
        # Show throughput
        n_components = result.components.n_bits
        n_lut_entries = result.components.size
        total_transforms = n_components + n_lut_entries  # Components + LAT columns
        if total_ms > 0:
            throughput = total_transforms / (total_ms / 1000.0)
            print(f"  Throughput:            {throughput:.1f} transforms/sec")
            print(f"  ({n_components} component + {n_lut_entries} LAT column transforms)")
        print()
        
        # Find strongest linear approximations
        if result.lat.lat is not None:
            abs_lat = np.abs(result.lat.lat)
            max_positions = np.argwhere(abs_lat == result.lat.lat_max)
            
            print(f"Strongest linear approximations ({len(max_positions)} found):")
            for idx, (input_mask, output_mask) in enumerate(max_positions[:3]):
                lat_value = result.lat.lat[input_mask, output_mask]
                print(f"  LAT[0x{input_mask:04x}, 0x{output_mask:04x}] = {lat_value:6d} "
                      f"(bias = {lat_value/65536:.8f})")
            if len(max_positions) > 3:
                print(f"  ... and {len(max_positions) - 3} more")
    print()


def compare_backends():
    """Compare CPU vs GPU performance for S-box analysis."""
    print("=" * 70)
    print("Example 7: Backend Performance Comparison")
    print("=" * 70)
    print()
    
    backends = [('auto', 'AUTO'), ('cpu', 'CPU')]
    if fwht.has_openmp():
        backends.append(('openmp', 'OpenMP'))
    if fwht.has_gpu():
        backends.append(('gpu', 'GPU'))
    
    print(f"Testing {len(backends)} backends with AES S-box (256 entries)...")
    print()
    
    for backend_name, label in backends:
        result = fwht.analyze_sbox(
            AES_SBOX,
            backend=backend_name,
            compute_lat=True,
            profile_timings=True
        )
        
        total = result.components.fwht_ms
        if result.lat:
            total += result.lat.column_ms + result.lat.fwht_ms
        
        print(f"{label:8s}: {total:7.3f} ms total")
        print(f"          Component FWHT: {result.components.fwht_ms:.3f} ms")
        if result.lat:
            print(f"          LAT synthesis:  {result.lat.column_ms:.3f} ms")
            print(f"          LAT FWHT:       {result.lat.fwht_ms:.3f} ms")
        print()


def main():
    """Run all S-box analysis examples."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "pyfwht S-box Cryptanalysis Examples" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    example_basic_sbox_analysis()
    print("\n" + "─" * 70 + "\n")
    
    example_with_spectra()
    print("\n" + "─" * 70 + "\n")
    
    example_with_full_lat()
    print("\n" + "─" * 70 + "\n")
    
    example_gpu_accelerated()
    print("\n" + "─" * 70 + "\n")
    
    example_small_sbox()
    print("\n" + "─" * 70 + "\n")
    
    example_large_sbox_gpu()
    print("\n" + "─" * 70 + "\n")
    
    compare_backends()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("The analyze_sbox() function provides comprehensive cryptanalysis:")
    print()
    print("1. Component Analysis:")
    print("   - Walsh spectrum for each Boolean component (output bit)")
    print("   - Maximum Walsh coefficient (lower = better)")
    print("   - Nonlinearity (higher = better, measures resistance to linear attacks)")
    print()
    print("2. LAT Analysis:")
    print("   - Linear Approximation Table for all input/output mask pairs")
    print("   - Maximum |LAT| entry (excluding trivial LAT[0,0])")
    print("   - Maximum bias: LAT_max / (2^m) where m = input bits")
    print("   - Maximum correlation: 2 × bias (since correlation = 2×Pr[linear] - 1)")
    print("   - Lower bias/correlation = better resistance to linear cryptanalysis")
    print("   - Full LAT matrix (useful for finding best linear approximations)")
    print()
    print("3. Performance:")
    print("   - CPU: Single-threaded SIMD-optimized")
    print("   - OpenMP: Multi-threaded CPU parallelism")
    print("   - GPU: Massive parallelism for large S-boxes (requires CUDA)")
    print()
    print("For more information, see:")
    print("  - pyfwht documentation: https://github.com/hadipourh/fwht")
    print("  - fwht.analyze_sbox() API reference")
    print("=" * 70)


if __name__ == "__main__":
    main()
