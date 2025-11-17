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
from typing import Optional, Union, Any

# Re-export low-level C bindings for advanced users
from ._pyfwht import (
    fwht_i32 as _fwht_i32,
    fwht_f64 as _fwht_f64,
    fwht_i8 as _fwht_i8,
    fwht_i32_backend as _fwht_i32_backend,
    fwht_f64_backend as _fwht_f64_backend,
    fwht_compute_i32 as _fwht_compute_i32,
    fwht_compute_f64 as _fwht_compute_f64,
    fwht_compute_i32_backend as _fwht_compute_i32_backend,
    fwht_compute_f64_backend as _fwht_compute_f64_backend,
    fwht_from_bool as _fwht_from_bool,
    fwht_correlations as _fwht_correlations,
    fwht_boolean_packed as _fwht_boolean_packed,
    fwht_boolean_packed_backend as _fwht_boolean_packed_backend,
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
    'transform',
    'compute',
    'from_bool',
    'correlations',
    'boolean_packed',
    'is_power_of_2',
    'log2',
    'recommend_backend',
    'has_openmp',
    'has_gpu',
    'backend_name',
    'version',
    'gpu',  # GPU module
]


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
        }
        if name not in mapping:
            raise ValueError(
                f"Unknown backend string '{backend}'. Expected one of: auto,cpu,openmp,gpu"
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
