"""
pyfwht - Python Bindings for Fast Walsh-Hadamard Transform

High-performance Walsh-Hadamard Transform library with NumPy integration
and support for CPU (SIMD), OpenMP, and CUDA backends.

Basic Usage:
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> data = np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int32)
    >>> fwht.transform(data)  # In-place transform
    >>> print(data)

Copyright (C) 2025 Hosein Hadipour
License: GPL-3.0-or-later
"""

from ._version import __version__
from . import _pyfwht
from ._pyfwht import Backend, Config

import numpy as np
import warnings
import os
from typing import Optional, Union, Any

# Global flag for fp16 precision warning (shown only once)
_fp16_warning_shown = False

# Re-export low-level C bindings for advanced users
from ._pyfwht import (
    fwht_i32 as _fwht_i32,
    fwht_i32_safe as _fwht_i32_safe,
    fwht_f64 as _fwht_f64,
    fwht_i8 as _fwht_i8,
    fwht_i32_backend as _fwht_i32_backend,
    fwht_f64_backend as _fwht_f64_backend,
    fwht_i32_batch as _fwht_i32_batch,
    fwht_f64_batch as _fwht_f64_batch,
    fwht_compute_i32 as _fwht_compute_i32,
    fwht_compute_f64 as _fwht_compute_f64,
    fwht_compute_i32_backend as _fwht_compute_i32_backend,
    fwht_compute_f64_backend as _fwht_compute_f64_backend,
    fwht_from_bool as _fwht_from_bool,
    fwht_correlations as _fwht_correlations,
    fwht_boolean_packed as _fwht_boolean_packed,
    fwht_boolean_packed_backend as _fwht_boolean_packed_backend,
    fwht_boolean_batch as _fwht_boolean_batch,
    Context as _Context,
    is_power_of_2,
    log2,
    recommend_backend,
    has_openmp,
    has_gpu,
    backend_name,
    version,
    default_config,
)

# GPU module (if available) – adapt to current bindings layout with submodule _pyfwht.gpu
_GPU_AVAILABLE = False
try:
    # Import the pybind11 core module
    from . import _pyfwht as _pb
    # Check if the GPU submodule exists (built with USE_CUDA)
    if hasattr(_pb, "gpu"):
        _GPU_AVAILABLE = True
except ImportError:
    _pb = None  # type: ignore
    _GPU_AVAILABLE = False

__all__ = [
    '__version__',
    'Backend',
    'Config',
    'Context',
    'fwht',
    'transform',
    'transform_safe',
    'compute',
    'from_bool',
    'correlations',
    'boolean_packed',
    'boolean_batch',
    'vectorized_batch_i32',
    'vectorized_batch_f64',
    'is_power_of_2',
    'log2',
    'recommend_backend',
    'has_openmp',
    'has_gpu',
    'backend_name',
    'version',
    'gpu',  # GPU module
    'gpu_get_compute_capability',
]


def _warn_fp16_precision():
    """
    Show one-time warning about fp16 precision tradeoffs.
    Can be suppressed with FWHT_SILENCE_FP16_WARNING=1 environment variable.
    """
    global _fp16_warning_shown
    
    if _fp16_warning_shown:
        return
    
    # Check if warning should be suppressed
    if os.environ.get('FWHT_SILENCE_FP16_WARNING') == '1':
        _fp16_warning_shown = True
        return
    
    warnings.warn(
        "\n"
        "╔═══════════════════════════════════════════════════════════════════════════╗\n"
        "║ FP16 Tensor Core Precision Notice                                         ║\n"
        "╠═══════════════════════════════════════════════════════════════════════════╣\n"
        "║ Using float16 Tensor Cores provides 25-36× speedup.                       ║\n"
        "║                                                                           ║\n"
        "║ Observed behavior (RTX 4090, CUDA 12.6):                                  ║\n"
        "║   • Boolean {-1,+1} inputs: bit-exact vs CPU (max|error| = 0)             ║\n"
        "║   • Random fp32/fp64 data: max|error| ≈ 1.3e-1, mean ≈ 2.5e-2             ║\n"
        "║   • Relative error: < 6e-4 for coefficients around ±4000                  ║\n"
        "║                                                                           ║\n"
        "║ Recommended use cases:                                                    ║\n"
        "║   ✓ Machine learning / signal processing (use PyTorch DLPack)             ║\n"
        "║   ✓ Boolean cryptanalysis (truth tables stay exact)                       ║\n"
        "║   ✗ High-precision floating workloads (prefer fp32/fp64)                  ║\n"
        "║                                                                           ║\n"
        "║ To suppress this warning: set FWHT_SILENCE_FP16_WARNING=1                 ║\n"
        "╚═══════════════════════════════════════════════════════════════════════════╝",
        UserWarning,
        stacklevel=3
    )
    
    _fp16_warning_shown = True


def fwht(
    data: np.ndarray,
    backend: Optional[Union[Backend, str]] = None
) -> np.ndarray:
    """
    Walsh-Hadamard Transform supporting fp16/fp32/fp64 and batches.
    
    Parameters
    ----------
    data : np.ndarray
        1-D or 2-D NumPy array.
        Supported dtypes: float16, float32, float64, int8, int32.
        For 2-D: shape (batch_size, n) for batch processing.
    backend : Backend or str, optional
        Backend selection ('auto', 'cpu', 'openmp', 'cuda').
        If None, uses 'cuda' for GPU-resident data, 'auto' otherwise.
    
    Returns
    -------
    np.ndarray
        Transformed array (new copy).
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht
    >>> data = np.random.randn(4096).astype(np.float16)
    >>> result = pyfwht.fwht(data, backend='cuda')  # Uses Tensor Cores!
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    # Copy data to avoid modifying input
    result = data.copy()
    
    # Handle string backend
    if isinstance(backend, str):
        name = backend.strip().lower()
        mapping = {'auto': Backend.AUTO, 'cpu': Backend.CPU, 'openmp': Backend.OPENMP, 'gpu': Backend.GPU, 'cuda': Backend.GPU}
        backend = mapping.get(name, Backend.AUTO)
    elif backend is None:
        backend = Backend.GPU if has_gpu() else Backend.AUTO
    
    # Batch processing (2-D array)
    if result.ndim == 2:
        batch_size, n = result.shape
        if result.dtype in (np.float16, np.float32, np.float64) and backend == Backend.GPU and has_gpu():
            # Use GPU batch processing with native precision (Tensor Cores for fp16/fp32!)
            if result.dtype == np.float32:
                gpu.batch_f32(result, n, batch_size)
            elif result.dtype == np.float16:
                # NumPy float16 view as uint16 for C++ binding
                _warn_fp16_precision()
                result_u16 = result.view(np.uint16)
                gpu.batch_f16(result_u16, n, batch_size)
            else:
                gpu.batch_f64(result, n, batch_size)
        elif result.dtype == np.int32 and backend == Backend.GPU and has_gpu():
            gpu.batch_i32(result, n, batch_size)
        else:
            # CPU batch: process each row
            for i in range(batch_size):
                transform(result[i], backend)
        return result
    
    # Single transform (1-D array)
    if result.ndim != 1:
        raise ValueError("Input must be 1-D or 2-D array")
    
    # Handle different dtypes
    if result.dtype == np.float16 and backend == Backend.GPU and has_gpu():
        # fp16: use native GPU kernel with Tensor Cores!
        _warn_fp16_precision()
        n = len(result)
        result_u16 = result.view(np.uint16)
        gpu.batch_f16(result_u16, n, 1)
        return result
    elif result.dtype == np.float32 and backend == Backend.GPU and has_gpu():
        # fp32: use native GPU kernel with Tensor Cores!
        n = len(result)
        gpu.batch_f32(result, n, 1)
        return result
    elif result.dtype in (np.float16, np.float32):
        # CPU fallback: convert to fp64
        result_f64 = result.astype(np.float64)
        transform(result_f64, backend)
        return result_f64.astype(result.dtype)
    else:
        # Native supported types
        transform(result, backend)
        return result


def transform(
    data: np.ndarray,
    backend: Optional[Backend] = None
) -> None:
    """
    In-place Walsh-Hadamard Transform with automatic dtype dispatch.
    
    Parameters
    ----------
    data : np.ndarray
        1-D NumPy array of int8, int32, or float64.
        Must have power-of-2 length.
        Modified in-place.
    backend : Backend, optional
        Backend selection (AUTO, CPU, OPENMP, GPU).
        If None, uses AUTO backend.
    
    Raises
    ------
    ValueError
        If array is not 1-D or length is not power of 2.
    RuntimeError
        If backend is unavailable.
    TypeError
        If dtype is not int8, int32, or float64.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> data = np.array([1, -1, -1, 1], dtype=np.int32)
    >>> fwht.transform(data)
    >>> print(data)
    [ 0  4  0  0]
    
    >>> # Explicit backend
    >>> data = np.random.randn(256)
    >>> fwht.transform(data, backend=fwht.Backend.CPU)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    if data.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    
    # Accept legacy string backend specifiers (e.g., 'cpu','gpu') for backwards compatibility.
    # pybind11 enum types are not subscriptable, so use getattr mapping.
    if isinstance(backend, str):
        name = backend.strip().lower()
        mapping = {
            'auto': Backend.AUTO,
            'cpu': Backend.CPU,
            'openmp': Backend.OPENMP,
            'gpu': Backend.GPU,
            'cuda': Backend.GPU,
        }
        if name not in mapping:
            raise ValueError(
                f"Unknown backend string '{backend}'. Expected one of: auto,cpu,openmp,gpu,cuda"
            )
        backend = mapping[name]

    if backend is None:
        backend = Backend.AUTO
    
    # Dispatch based on dtype
    if data.dtype == np.int32:
        _fwht_i32_backend(data, backend)
    elif data.dtype == np.float64:
        _fwht_f64_backend(data, backend)
    elif data.dtype == np.int8:
        if backend != Backend.AUTO and backend != Backend.CPU:
            raise ValueError("int8 transforms only support AUTO and CPU backends")
        _fwht_i8(data)
    else:
        raise TypeError(
            f"Unsupported dtype: {data.dtype}. "
            "Supported types: int8, int32, float64"
        )


def transform_safe(data: np.ndarray) -> None:
    """
    In-place Walsh-Hadamard Transform for int32 with overflow detection.
    
    This variant detects integer overflow during computation and raises an
    exception if overflow occurs. It's 5-10% slower than regular transform
    but guarantees correctness or fails safely.
    
    Parameters
    ----------
    data : np.ndarray
        1-D NumPy array of int32.
        Must have power-of-2 length.
        Modified in-place.
    
    Raises
    ------
    RuntimeError
        If integer overflow is detected during computation.
    ValueError
        If array is not 1-D, not int32, or length is not power of 2.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> data = np.array([1, -1, -1, 1], dtype=np.int32)
    >>> fwht.transform_safe(data)  # Safe from overflow
    >>> print(data)
    [ 0  4  0  0]
    
    >>> # Large values that might overflow
    >>> data = np.array([2**30, 2**30, 0, 0], dtype=np.int32)
    >>> try:
    ...     fwht.transform_safe(data)
    ... except RuntimeError as e:
    ...     print("Overflow detected:", e)
    
    Notes
    -----
    Use this function when:
    - Input magnitudes are large or unknown
    - Safety is more important than performance  
    - You need to validate that n * max(|input|) < 2^31
    
    For maximum performance without overflow checks, use transform() instead.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    if data.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    
    if data.dtype != np.int32:
        raise TypeError("transform_safe only supports int32 arrays")
    
    _fwht_i32_safe(data)


def vectorized_batch_i32(data_list: list, n: int) -> None:
    """
    SIMD-optimized batch WHT for multiple int32 arrays (in-place).
    
    This function processes multiple independent transforms simultaneously
    using SIMD vectorization. It's 3-5× faster than processing arrays
    sequentially for small to medium sizes (n ≤ 256).
    
    Parameters
    ----------
    data_list : list of np.ndarray
        List of 1-D int32 arrays, each of length n.
        All arrays are modified in-place.
    n : int
        Size of each array (must be power of 2, same for all arrays).
    
    Raises
    ------
    ValueError
        If arrays have different sizes or n is not power of 2.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> # Process 8 S-boxes in parallel (3-5× faster than loop!)
    >>> sboxes = [np.random.randint(-10, 10, 256, dtype=np.int32) 
    ...           for _ in range(8)]
    >>> fwht.vectorized_batch_i32(sboxes, 256)
    >>> # All sboxes now contain their WHT coefficients
    
    Notes
    -----
    Performance characteristics:
    - n ≤ 256: 3-5× faster than sequential (optimal SIMD usage)
    - n > 256: 1.2-1.5× faster (memory-bound, less SIMD benefit)
    
    This is different from GPU batch processing:
    - CPU vectorized batch: Processes arrays in parallel using SIMD (AVX2/NEON)
    - GPU batch: Processes arrays on GPU using CUDA
    
    For GPU-resident data, use gpu.batch_transform_i32() instead.
    """
    if not isinstance(data_list, list):
        raise TypeError("data_list must be a list of NumPy arrays")
    
    _fwht_i32_batch(data_list, n)


def vectorized_batch_f64(data_list: list, n: int) -> None:
    """
    SIMD-optimized batch WHT for multiple float64 arrays (in-place).
    
    Same as vectorized_batch_i32() but for float64 arrays.
    Provides 3-5× speedup over sequential processing for n ≤ 256.
    
    Parameters
    ----------
    data_list : list of np.ndarray
        List of 1-D float64 arrays, each of length n.
        All arrays are modified in-place.
    n : int
        Size of each array (must be power of 2, same for all arrays).
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> # Process multiple transforms in parallel
    >>> data = [np.random.randn(512) for _ in range(16)]
    >>> fwht.vectorized_batch_f64(data, 512)
    
    See Also
    --------
    vectorized_batch_i32 : Integer version with detailed documentation
    gpu.batch_transform_f64 : GPU batch processing for large batches
    """
    if not isinstance(data_list, list):
        raise TypeError("data_list must be a list of NumPy arrays")
    
    _fwht_f64_batch(data_list, n)


def compute(
    data: np.ndarray,
    backend: Optional[Backend] = None
) -> np.ndarray:
    """
    Out-of-place Walsh-Hadamard Transform.
    
    Parameters
    ----------
    data : np.ndarray
        1-D NumPy array of int32 or float64.
        Input is not modified.
    backend : Backend, optional
        Backend selection (AUTO, CPU, OPENMP, GPU).
        If None, uses AUTO backend.
    
    Returns
    -------
    np.ndarray
        New array containing the WHT of input.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> original = np.array([1, -1, -1, 1], dtype=np.int32)
    >>> result = fwht.compute(original)
    >>> print(original)  # Unchanged
    [ 1 -1 -1  1]
    >>> print(result)
    [ 0  4  0  0]
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    if data.ndim != 1:
        raise ValueError("Input must be 1-dimensional")
    
    if backend is None:
        backend = Backend.AUTO
    
    # Dispatch based on dtype
    if data.dtype == np.int32:
        return _fwht_compute_i32_backend(data, backend)
    elif data.dtype == np.float64:
        return _fwht_compute_f64_backend(data, backend)
    else:
        raise TypeError(
            f"Unsupported dtype: {data.dtype}. "
            "Supported types: int32, float64"
        )


def from_bool(
    truth_table: np.ndarray,
    signed: bool = True
) -> np.ndarray:
    """
    Compute WHT coefficients from Boolean function truth table.
    
    Parameters
    ----------
    truth_table : np.ndarray
        1-D array of 0s and 1s representing Boolean function.
        Length must be power of 2.
    signed : bool, default=True
        If True, converts 0→+1, 1→-1 before transform (cryptographic convention).
        If False, uses values as-is.
    
    Returns
    -------
    np.ndarray
        WHT coefficients as int32 array.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> # XOR function: f(x,y) = x ⊕ y
    >>> truth_table = np.array([0, 1, 1, 0], dtype=np.uint8)
    >>> wht = fwht.from_bool(truth_table, signed=True)
    >>> print(wht)
    [0 0 0 4]
    """
    if not isinstance(truth_table, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    # Convert to uint8 if needed
    if truth_table.dtype != np.uint8:
        truth_table = truth_table.astype(np.uint8)
    
    return _fwht_from_bool(truth_table, signed)


def correlations(truth_table: np.ndarray) -> np.ndarray:
    """
    Compute correlations between Boolean function and all linear functions.
    
    Parameters
    ----------
    truth_table : np.ndarray
        1-D array of 0s and 1s representing Boolean function.
        Length must be power of 2.
    
    Returns
    -------
    np.ndarray
        Correlation values in range [-1.0, +1.0] as float64 array.
        correlations[u] = Cor(f, ℓ_u) where ℓ_u(x) = popcount(u & x) mod 2
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> truth_table = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    >>> corr = fwht.correlations(truth_table)
    >>> max_corr = np.max(np.abs(corr))
    >>> print(f"Max correlation: {max_corr}")
    """
    if not isinstance(truth_table, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    # Convert to uint8 if needed
    if truth_table.dtype != np.uint8:
        truth_table = truth_table.astype(np.uint8)
    
    return _fwht_correlations(truth_table)


def boolean_packed(
    packed_bits: np.ndarray,
    n: int,
    backend: Optional[Backend] = None
) -> np.ndarray:
    """
    Compute WHT from bit-packed Boolean function (memory-efficient).
    
    This function is optimized for cryptanalysis applications where you
    need to analyze many Boolean functions. By packing 64 Boolean values
    into each uint64, you save 32× memory compared to unpacked representation.
    
    Parameters
    ----------
    packed_bits : np.ndarray
        1-D array of uint64 containing bit-packed truth table.
        Bit i of word j represents bool_func[j*64 + i].
        Array should have length ceil(n/64).
    n : int
        Transform size (number of Boolean function inputs).
        Must be power of 2, n ≤ 65536.
    backend : Backend, optional
        Backend selection (AUTO, CPU, OPENMP, GPU).
        If None, uses AUTO backend.
    
    Returns
    -------
    np.ndarray
        WHT coefficients as int32 array of length n.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> 
    >>> # Pack truth table [0,1,1,0,1,0,0,1] into single uint64
    >>> # Bits 1,2,4,7 are set → 0b10010110 = 0x96
    >>> packed = np.array([0x96], dtype=np.uint64)
    >>> wht = fwht.boolean_packed(packed, n=8)
    >>> print(wht)
    
    >>> # For larger functions, pack into multiple uint64s
    >>> truth_table = np.random.randint(0, 2, 256, dtype=np.uint8)
    >>> n_words = (256 + 63) // 64  # Need 4 words
    >>> packed = np.zeros(n_words, dtype=np.uint64)
    >>> for i in range(256):
    ...     if truth_table[i]:
    ...         word_idx = i // 64
    ...         bit_idx = i % 64
    ...         packed[word_idx] |= (1 << bit_idx)
    >>> wht = fwht.boolean_packed(packed, n=256)
    
    Notes
    -----
    Memory savings: For n=65536, packed representation uses 1KB vs 256KB
    for unpacked uint8 array (256× savings).
    
    The current implementation unpacks internally to use SIMD butterfly,
    so performance is similar to unpacked version. Future GPU kernels
    may provide significant speedup for truly bit-sliced operations.
    """
    if not isinstance(packed_bits, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    
    if packed_bits.dtype != np.uint64:
        raise TypeError("Packed bits must have dtype uint64")
    
    if packed_bits.ndim != 1:
        raise ValueError("Packed bits must be 1-dimensional")
    
    if backend is None:
        return _fwht_boolean_packed(packed_bits, n)
    else:
        return _fwht_boolean_packed_backend(packed_bits, n, backend)


def boolean_batch(packed_list: list, n: int) -> list:
    """
    Batch WHT for multiple bit-packed Boolean functions (S-box cryptanalysis).
    
    This function efficiently computes WHT for multiple Boolean functions
    simultaneously, ideal for analyzing all component functions of an S-box.
    Provides 50-100× speedup over sequential unpacked transforms.
    
    Parameters
    ----------
    packed_list : list of np.ndarray
        List of bit-packed Boolean functions (uint64 arrays).
        Each array should have length ceil(n/64).
    n : int
        Transform size (must be power of 2, same for all functions).
    
    Returns
    -------
    list of np.ndarray
        List of WHT spectra (int32 arrays of length n).
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> 
    >>> # Analyze 8-bit S-box with 8 component functions
    >>> # Each component is a Boolean function with 256 inputs
    >>> n = 256
    >>> n_components = 8
    >>> 
    >>> # Pack each component function
    >>> packed_sbox = []
    >>> for comp in range(n_components):
    ...     # Generate random Boolean function for demo
    ...     truth_table = np.random.randint(0, 2, n, dtype=np.uint8)
    ...     
    ...     # Pack into uint64 array
    ...     n_words = (n + 63) // 64
    ...     packed = np.zeros(n_words, dtype=np.uint64)
    ...     for i in range(n):
    ...         if truth_table[i]:
    ...             packed[i // 64] |= (1 << (i % 64))
    ...     
    ...     packed_sbox.append(packed)
    >>> 
    >>> # Compute WHT for all components in one call
    >>> wht_spectra = fwht.boolean_batch(packed_sbox, n)
    >>> 
    >>> # Find maximum Walsh coefficient for each component
    >>> for i, spectrum in enumerate(wht_spectra):
    ...     max_walsh = np.max(np.abs(spectrum))
    ...     print(f"Component {i}: max|W| = {max_walsh}")
    
    Notes
    -----
    Performance: 50-100× faster than:
    1. Unpacking each Boolean function
    2. Computing WHT sequentially
    3. Processing in Python loop
    
    Memory efficient: Uses bit-packed representation (32× less memory).
    
    Ideal for:
    - S-box linear cryptanalysis
    - Computing nonlinearity of vectorial Boolean functions
    - Batch analysis of Boolean function properties
    """
    if not isinstance(packed_list, list):
        raise TypeError("packed_list must be a list of NumPy uint64 arrays")
    
    return _fwht_boolean_batch(packed_list, n)


class Context:
    """
    FWHT computation context for efficient repeated transforms.
    
    Creating a context amortizes setup costs (thread pools, GPU memory, etc.)
    for applications that compute many WHTs.
    
    Parameters
    ----------
    backend : Backend, optional
        Backend selection. Default: AUTO
    num_threads : int, optional
        Number of OpenMP threads (0 = auto-detect). Default: 0
    gpu_device : int, optional
        GPU device ID for CUDA backend. Default: 0
    normalize : bool, optional
        Divide by sqrt(n) after transform. Default: False
    
    Examples
    --------
    >>> import numpy as np
    >>> import pyfwht as fwht
    >>> 
    >>> # Context manager (automatic cleanup)
    >>> with fwht.Context(backend=fwht.Backend.CPU) as ctx:
    ...     data1 = np.random.randn(256)
    ...     ctx.transform(data1)
    ...     data2 = np.random.randn(256)
    ...     ctx.transform(data2)
    >>> 
    >>> # Manual management
    >>> ctx = fwht.Context(backend=fwht.Backend.OPENMP, num_threads=4)
    >>> ctx.transform(data)
    >>> ctx.close()
    """
    
    def __init__(
        self,
        backend: Backend = Backend.AUTO,
        num_threads: int = 0,
        gpu_device: int = 0,
        normalize: bool = False
    ):
        config = Config()
        config.backend = backend
        config.num_threads = num_threads
        config.gpu_device = gpu_device
        config.normalize = normalize
        
        self._ctx = _Context(config)
        self._closed = False
    
    def transform(self, data: np.ndarray) -> None:
        """
        In-place transform using this context.
        
        Parameters
        ----------
        data : np.ndarray
            1-D array of int32 or float64.
        """
        if self._closed:
            raise RuntimeError("Context is closed")
        
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        
        if data.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        
        # Dispatch based on dtype
        if data.dtype == np.int32:
            self._ctx.transform_i32(data)
        elif data.dtype == np.float64:
            self._ctx.transform_f64(data)
        else:
            raise TypeError(
                f"Unsupported dtype: {data.dtype}. "
                "Supported types: int32, float64"
            )
    
    def close(self) -> None:
        """Release resources associated with this context."""
        if not self._closed:
            self._ctx.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        self.close()


# ============================================================================
# GPU Module
# ============================================================================

class GPUModule:
    """
    GPU-specific functionality for CUDA backend.
    
    Provides access to device information, profiling, batch operations,
    and GPU contexts with persistent allocations.
    
    Attributes
    ----------
    available : bool
        Whether GPU support is available.
    
    Examples
    --------
    >>> import pyfwht as fwht
    >>> if fwht.gpu.available:
    ...     print(f"Device: {fwht.gpu.device_name()}")
    ...     print(f"Compute: {fwht.gpu.compute_capability()}")
    """
    
    def __init__(self):
           # Check GPU availability at runtime, not import time
           # This handles cases where the module was built with CUDA but imported before GPU init
           self.available = has_gpu()
    
    def device_name(self) -> str:
        """Get GPU device name (e.g., 'NVIDIA H100')."""
        if not self.available:
            raise RuntimeError("GPU support not available")
        # bindings: _pyfwht.gpu.Info.get_device_name()
        return _pb.gpu.Info.get_device_name()
    
    def compute_capability(self) -> tuple:
        """
        Get CUDA compute capability as (major, minor) tuple.
        
        Returns
        -------
        tuple
            (major, minor) version, e.g., (9, 0) for SM 9.0
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        cc = _pb.gpu.Info.get_compute_capability()
        # bindings return e.g. 90 for SM 9.0; convert to (major, minor)
        major = cc // 10
        minor = cc % 10
        return (major, minor)
    
    def sm_count(self) -> int:
        """Get number of streaming multiprocessors."""
        if not self.available:
            raise RuntimeError("GPU support not available")
        return _pb.gpu.Info.get_sm_count()
    
    def smem_banks(self) -> int:
        """
        Get shared memory bank count (16 or 32).
        
        Used for bank-aware padding optimization.
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        return _pb.gpu.Info.get_smem_banks()
    
    def set_profiling(self, enabled: bool) -> None:
        """
        Enable or disable detailed GPU profiling.
        
        When enabled, tracks H2D, kernel, and D2H timings separately.
        
        Parameters
        ----------
        enabled : bool
            Whether to enable profiling.
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        _pb.gpu.Profiling.set_profiling(enabled)
    
    def profiling_enabled(self) -> bool:
        """Check if GPU profiling is currently enabled."""
        if not self.available:
            raise RuntimeError("GPU support not available")
        return _pb.gpu.Profiling.profiling_enabled()
    
    def get_last_metrics(self) -> Any:
        """
        Get profiling metrics from the last GPU operation.
        
        Returns
        -------
        GPUMetrics
            Object with h2d_ms, kernel_ms, d2h_ms, n, batch_size,
            bytes_transferred, samples, valid fields.
        
        Examples
        --------
        >>> import pyfwht as fwht
        >>> fwht.gpu.set_profiling(True)
        >>> data = np.random.randn(1024)
        >>> fwht.transform(data, backend=fwht.Backend.GPU)
        >>> metrics = fwht.gpu.get_last_metrics()
        >>> print(f"Kernel: {metrics.kernel_ms:.3f} ms")
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        return _pb.gpu.Profiling.get_last_metrics()
    
    def batch_transform_i32(self, batch: np.ndarray) -> None:
        """
        In-place batch transform for int32 arrays.
        
        Parameters
        ----------
        batch : np.ndarray
            2-D array of shape (batch_size, n) with dtype int32.
            Each row is transformed independently.
        
        Examples
        --------
        >>> import numpy as np
        >>> import pyfwht as fwht
        >>> batch = np.random.randint(-100, 100, (16, 256), dtype=np.int32)
        >>> fwht.gpu.batch_transform_i32(batch)
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        if not isinstance(batch, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if batch.ndim != 2:
            raise ValueError("Input must be 2-dimensional (batch_size, n)")
        if batch.dtype != np.int32:
            raise TypeError("Input must have dtype int32")
        # bindings: function requires data, n, batch_size
        n = batch.shape[1]
        bsz = batch.shape[0]
        _pb.gpu.batch_i32(batch, int(n), int(bsz))
    
    def batch_transform_f64(self, batch: np.ndarray) -> None:
        """
        In-place batch transform for float64 arrays.
        
        Parameters
        ----------
        batch : np.ndarray
            2-D array of shape (batch_size, n) with dtype float64.
            Each row is transformed independently.
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        if not isinstance(batch, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if batch.ndim != 2:
            raise ValueError("Input must be 2-dimensional (batch_size, n)")
        if batch.dtype != np.float64:
            raise TypeError("Input must have dtype float64")
        n = batch.shape[1]
        bsz = batch.shape[0]
        _pb.gpu.batch_f64(batch, int(n), int(bsz))

    # Low-level batch operations accepting (data, n, batch_size)
    # These mirror the pybind11 exports for direct use in fwht()
    def batch_f64(self, data: np.ndarray, n: int, batch_size: int) -> None:
        if not self.available:
            raise RuntimeError("GPU support not available")
        _pb.gpu.batch_f64(data, int(n), int(batch_size))

    def batch_f32(self, data: np.ndarray, n: int, batch_size: int) -> None:
        if not self.available:
            raise RuntimeError("GPU support not available")
        _pb.gpu.batch_f32(data, int(n), int(batch_size))

    def batch_f16(self, data_u16: np.ndarray, n: int, batch_size: int) -> None:
        if not self.available:
            raise RuntimeError("GPU support not available")
        # Expects uint16 view of float16 array
        if data_u16.dtype != np.uint16:
            raise TypeError("data_u16 must be a uint16 view of float16 array")
        _pb.gpu.batch_f16(data_u16, int(n), int(batch_size))

    def batch_i32(self, data: np.ndarray, n: int, batch_size: int) -> None:
        if not self.available:
            raise RuntimeError("GPU support not available")
        _pb.gpu.batch_i32(data, int(n), int(batch_size))
    
    def batch_transform_dlpack(self, tensor, n: Optional[int] = None, batch_size: Optional[int] = None) -> None:
        """
        Zero-copy in-place batch transform for GPU tensors (PyTorch, CuPy, JAX).
        
        This function uses DLPack for zero-copy interoperability, eliminating
        host-to-device and device-to-host memory transfers. The tensor must
        already be on the GPU.
        
        Parameters
        ----------
        tensor : GPU tensor object
            A tensor supporting DLPack protocol (__dlpack__, __dlpack_device__).
            Supported frameworks: PyTorch, CuPy, JAX, TensorFlow, etc.
            Must be 2-D with shape (batch_size, n) and on CUDA device.
            Supported dtypes: float64, float32, float16, int32.
        n : int, optional
            Size of each transform (must be power of 2).
            If None, inferred from tensor shape[1].
        batch_size : int, optional
            Number of transforms in batch.
            If None, inferred from tensor shape[0].
        
        Examples
        --------
        PyTorch float64 (cryptographic precision):
        
        >>> import torch
        >>> import pyfwht
        >>> # High precision for cryptographic applications
        >>> data = torch.randn(1000, 4096, dtype=torch.float64, device='cuda')
        >>> pyfwht.gpu.batch_transform_dlpack(data)
        >>> # Result is in-place in the same tensor
        
        PyTorch float16 (maximum speed, Meta-inspired):
        
        >>> import torch
        >>> import pyfwht
        >>> # 11× faster than float64, suitable for ML/AI
        >>> data = torch.randn(1000, 4096, dtype=torch.float16, device='cuda')
        >>> pyfwht.gpu.batch_transform_dlpack(data)
        
        CuPy example:
        
        >>> import cupy as cp
        >>> import pyfwht
        >>> data = cp.random.randn(1000, 4096, dtype=cp.float32)  # 2× faster than fp64
        >>> pyfwht.gpu.batch_transform_dlpack(data)
        
        Notes
        -----
        - **80% faster** than batch_transform_f64() for large batches
          (eliminates H2D/D2H memory transfers)
        - Tensor must be contiguous in memory
        - Transform is done in-place
        - For NumPy arrays, use batch_transform_f64() instead
        
        Performance (RTX 4090, batch=1000, n=4096):
        - float64: ~74 GOps/s (best precision, cryptographic use)
        - float32: ~400 GOps/s (balanced speed/precision)
        - float16: ~800 GOps/s (maximum speed, ML/AI use, matches Meta)
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        
        # Check if tensor supports DLPack
        if not hasattr(tensor, '__dlpack__'):
            raise TypeError(
                "Tensor must support DLPack protocol (__dlpack__ method). "
                "Supported: PyTorch, CuPy, JAX. For NumPy, use batch_transform_f64()."
            )
        
        # Infer dimensions from tensor if not provided
        if hasattr(tensor, 'shape'):
            tensor_shape = tensor.shape
            if len(tensor_shape) != 2:
                raise ValueError(f"Tensor must be 2-D, got shape {tensor_shape}")
            
            if batch_size is None:
                batch_size = tensor_shape[0]
            if n is None:
                n = tensor_shape[1]
            
            # Validate
            if tensor_shape[0] != batch_size or tensor_shape[1] != n:
                raise ValueError(
                    f"Tensor shape {tensor_shape} doesn't match (batch_size={batch_size}, n={n})"
                )
        else:
            if n is None or batch_size is None:
                raise ValueError("Must provide n and batch_size if tensor has no shape attribute")
        
        # Get DLPack capsule
        dlpack_tensor = tensor.__dlpack__()
        
        # Determine dtype and call appropriate function
        if hasattr(tensor, 'dtype'):
            dtype_str = str(tensor.dtype)
            if 'float64' in dtype_str or 'double' in dtype_str:
                _pb.gpu.batch_f64_dlpack(dlpack_tensor, int(n), int(batch_size))
            elif 'float32' in dtype_str or dtype_str == 'float':
                # Meta-inspired fp32 kernel (2× faster than fp64)
                _pb.gpu.batch_f32_dlpack(dlpack_tensor, int(n), int(batch_size))
            elif 'float16' in dtype_str or 'half' in dtype_str:
                # Meta-inspired fp16 kernel (11× faster than fp64, lower precision)
                _warn_fp16_precision()
                _pb.gpu.batch_f16_dlpack(dlpack_tensor, int(n), int(batch_size))
            elif 'int32' in dtype_str:
                _pb.gpu.batch_i32_dlpack(dlpack_tensor, int(n), int(batch_size))
            else:
                raise TypeError(
                    f"Unsupported dtype: {tensor.dtype}. Supported: float64, float32, float16, int32"
                )
        else:
            # Default to float64
            _pb.gpu.batch_f64_dlpack(dlpack_tensor, int(n), int(batch_size))
    
    def set_multi_shuffle(self, enabled: bool) -> None:
        """
        Enable or disable multi-shuffle optimization for N in (32, 512].
        
        Multi-shuffle uses warp-level primitives for medium sizes,
        potentially faster than shared memory on some GPUs.
        Default: disabled (uses shared memory instead).
        
        Parameters
        ----------
        enabled : bool
            Whether to enable multi-shuffle.
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        _pb.gpu.Toggles.set_multi_shuffle(enabled)
    
    def multi_shuffle_enabled(self) -> bool:
        """Check if multi-shuffle optimization is enabled."""
        if not self.available:
            raise RuntimeError("GPU support not available")
        return _pb.gpu.Toggles.multi_shuffle_enabled()
    
    def set_block_size(self, block_size: int) -> None:
        """
        Configure CUDA block size for kernel execution.
        
        Provides manual control over thread block size for performance tuning.
        Usually automatic selection is optimal, but this allows experimentation.
        
        Parameters
        ----------
        block_size : int
            Block size to use (must be power-of-2 in [1, 1024]).
            Pass 0 to revert to automatic selection.
        
        Examples
        --------
        >>> import pyfwht
        >>> if pyfwht.has_gpu():
        ...     # Try 512 threads per block
        ...     pyfwht.gpu.set_block_size(512)
        ...     
        ...     # Revert to auto
        ...     pyfwht.gpu.set_block_size(0)
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        _pb.gpu.Toggles.set_block_size(block_size)
    
    def get_block_size(self) -> int:
        """
        Get current CUDA block size configuration.
        
        Returns
        -------
        int
            Current block size, or 0 if using automatic selection.
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        return _pb.gpu.Toggles.get_block_size()
    
    def Context(
        self,
        max_n: int = 1024,
        batch_size: int = 1
    ):
        """
        Create a GPU context with persistent device allocations.
        
        Avoids repeated cudaMalloc/cudaFree overhead for
        applications computing many transforms of fixed size.
        
        Parameters
        ----------
        max_n : int, default=1024
            Maximum transform size to pre-allocate for.
        batch_size : int, default=1
            Batch size to pre-allocate for.
        
        Returns
        -------
        GPUContext
            Context object with transform methods.
        
        Examples
        --------
        >>> import numpy as np
        >>> import pyfwht as fwht
        >>> with fwht.gpu.Context(max_n=512, batch_size=8) as ctx:
        ...     for _ in range(100):
        ...         data = np.random.randn(512)
        ...         ctx.transform_f64(data)
        """
        if not self.available:
            raise RuntimeError("GPU support not available")
        return GPUContext(max_n, batch_size)


class GPUContext:
    """
    GPU computation context with persistent device memory.
    
    Do not instantiate directly; use `fwht.gpu.Context()` instead.
    """
    
    def __init__(self, max_n: int, batch_size: int):
        # bindings: GPU Context class under submodule
        self._ctx = _pb.gpu.Context(int(max_n), int(batch_size))
        self._closed = False
    
    def transform_i32(self, data: np.ndarray) -> None:
        """
        In-place int32 transform using pre-allocated GPU memory.
        
        Parameters
        ----------
        data : np.ndarray
            1-D array of int32, length <= max_n.
        """
        if self._closed:
            raise RuntimeError("Context is closed")
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if data.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        if data.dtype != np.int32:
            raise TypeError("Input must have dtype int32")
        n = data.shape[0]
        self._ctx.compute_i32(data, int(n), 1)
    
    def transform_f64(self, data: np.ndarray) -> None:
        """
        In-place float64 transform using pre-allocated GPU memory.
        
        Parameters
        ----------
        data : np.ndarray
            1-D array of float64, length <= max_n.
        """
        if self._closed:
            raise RuntimeError("Context is closed")
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        if data.ndim != 1:
            raise ValueError("Input must be 1-dimensional")
        if data.dtype != np.float64:
            raise TypeError("Input must have dtype float64")
        n = data.shape[0]
        self._ctx.compute_f64(data, int(n), 1)
    
    def close(self) -> None:
        """Free GPU resources."""
        if not self._closed:
            # pybind class exposes close()
            self._ctx.close()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        self.close()


# Create singleton GPU module instance
gpu = GPUModule()

# Convenience functions at module level
def gpu_get_compute_capability():
    """Get GPU compute capability (e.g., 89 for SM 8.9)."""
    if not has_gpu():
        return 0
    return _pb.gpu.Info.get_compute_capability() if _GPU_AVAILABLE else 0
