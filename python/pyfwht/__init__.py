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
from typing import Optional, Union

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

__all__ = [
    '__version__',
    'Backend',
    'Config',
    'Context',
    'transform',
    'compute',
    'from_bool',
    'correlations',
    'is_power_of_2',
    'log2',
    'recommend_backend',
    'has_openmp',
    'has_gpu',
    'backend_name',
    'version',
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
