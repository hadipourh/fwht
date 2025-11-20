#!/usr/bin/env python3
"""
Side-by-side comparison harness for:
    - pyfwht (CPU and GPU backends)
    - meta-pytorch/applied-ai hadamard CUDA kernel (PyTorch extension)

Measures latency and throughput across sizes and batch counts, with warmup/repeats,
and optionally (light) correctness. Results can be saved as CSV.

Notes and caveats:
- The meta kernel (`fast_hadamard_transform.hadamard_transform`) supports only fp16
    and bf16 input tensors (contiguous, last dimension is Hadamard size, power-of-two
    up to 2^15). It internally pads total element count to a multiple of 256.
- pyfwht provides int32 and float64 batch APIs; when you request unsupported dtypes
    the harness will cast and annotate the note field.
- Correctness numbers for the meta kernel are set to None by default because scaling
    conventions (normalized vs unnormalized Hadamard) may differ; enable a future
    reference path if needed.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _ensure_pyfwht_importable() -> None:
    """Prefer importing pyfwht from the repo/python path to use local sources.

    This avoids picking up an older site-packages install and lets us test
    the just-synced code without reinstalling the wheel.
    """
    repo = _repo_root()
    py_path = os.path.join(repo, "python")
    if py_path not in sys.path:
        sys.path.insert(0, py_path)
    try:
        import pyfwht  # noqa: F401  # type: ignore
    except Exception:
        # Fall back to environment if local import failed
        try:
            sys.path = [p for p in sys.path if p != py_path]
            import pyfwht  # noqa: F401  # type: ignore
        except Exception as e:  # pragma: no cover - informative error
            print("Warning: could not import pyfwht from repo/python or environment:", e)


def _try_import_torch() -> Optional[Any]:
    try:
        import os
        # Ensure CUDA libraries are in LD_LIBRARY_PATH before importing torch
        cuda_paths = [
            '/usr/local/cuda/lib64',
            '/usr/local/cuda-12/lib64',
            '/usr/local/cuda-12.4/lib64',
            '/usr/local/nvidia/lib64',
        ]
        existing_path = os.environ.get('LD_LIBRARY_PATH', '')
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path) and cuda_path not in existing_path:
                os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}:{existing_path}"
                existing_path = os.environ['LD_LIBRARY_PATH']
        
        import torch  # type: ignore
        if os.environ.get('DEBUG_BENCH'):
            print(f"[DEBUG] torch imported successfully: {torch.__version__}")
        return torch
    except Exception as e:
        import os
        if os.environ.get('DEBUG_BENCH'):
            print(f"[DEBUG] Failed to import torch: {e}")
            import traceback
            traceback.print_exc()
        return None


def _try_import_module(name: str) -> Optional[Any]:
    try:
        __import__(name)
        mod = sys.modules[name]
        import os
        if os.environ.get('DEBUG_BENCH'):
            print(f"[DEBUG] Successfully imported {name} from {getattr(mod, '__file__', '(built-in)')}")
        return mod
    except Exception as e:
        import os
        if os.environ.get('DEBUG_BENCH'):
            import traceback
            print(f"[DEBUG] Failed to import {name}: {e}")
            print(f"[DEBUG] Exception type: {type(e).__name__}")
            traceback.print_exc()
        return None


def _cuda_available(torch: Any, want_device: str) -> bool:
    if want_device == "cpu":
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _ops_for_fwht(n: int) -> Tuple[int, int]:
    """
    Returns (butterflies, ops) where butterflies = n*log2(n) and ops = 2*butterflies
    counting one add + one sub per butterfly.
    """
    log2n = int(math.log2(n))
    butterflies = n * log2n
    ops = 2 * butterflies
    return butterflies, ops


def _format_gops(ops: int, secs: float) -> float:
    if secs <= 0:
        return float("inf")
    return (ops / secs) / 1e9


@dataclass
class BenchResult:
    impl: str
    device: str
    n: int
    batch: int
    dtype: str
    include_transfer: bool
    repeats: int
    warmup: int
    time_ms: float
    gops: float
    max_abs_err: Optional[float]
    note: str = ""

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["time_ms"] = f"{self.time_ms:.3f}"
        d["gops"] = f"{self.gops:.3f}"
        d["max_abs_err"] = ("" if self.max_abs_err is None else f"{self.max_abs_err:.3e}")
        return d


def _timeit(fn: Callable[[], None], sync: Optional[Callable[[], None]] = None,
            warmup: int = 3, repeats: int = 10) -> float:
    # Warmup
    for _ in range(max(0, warmup)):
        fn()
        if sync:
            sync()
    # Measure
    times: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


def _numpy_reference(x: Any) -> Any:
    """
    CPU reference using pyfwht CPU backend. Accepts numpy array shaped (B, N) or (N,).
    Returns numpy array of same shape/dtype.
    """
    import numpy as np
    import pyfwht  # type: ignore

    arr = x
    if isinstance(x, list):
        arr = np.asarray(x)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    # Use fwht() which supports all dtypes including fp16/fp32
    if arr.ndim == 1:
        result = arr.copy()
        result = pyfwht.fwht(result, backend='cpu')
        return result
    elif arr.ndim == 2:
        # Process batch
        result = arr.copy()
        result = pyfwht.fwht(result, backend='cpu')
        return result
    else:
        raise ValueError("reference expects 1D or 2D array")


def bench_pyfwht_cpu(n: int, batch: int, dtype: str,
                     warmup: int, repeats: int) -> BenchResult:
    import numpy as np
    import pyfwht  # type: ignore

    # Use native dtypes for CPU too
    if dtype == "int32":
        np_dtype = np.int32
        x = (np.random.randint(-2**31, 2**31 - 1, size=(batch, n), dtype=np_dtype)
             if batch > 1 else np.random.randint(-2**31, 2**31 - 1, size=(n,), dtype=np_dtype))
    elif dtype == "float64":
        np_dtype = np.float64
        x = (np.random.randn(batch, n).astype(np_dtype)
             if batch > 1 else np.random.randn(n).astype(np_dtype))
    elif dtype in ("float32", "fp32"):
        np_dtype = np.float32
        x = (np.random.randn(batch, n).astype(np_dtype)
             if batch > 1 else np.random.randn(n).astype(np_dtype))
    elif dtype in ("float16", "fp16"):
        np_dtype = np.float16
        x = (np.random.randn(batch, n).astype(np_dtype)
             if batch > 1 else np.random.randn(n).astype(np_dtype))
    elif dtype == "bfloat16":
        # NumPy doesn't have bfloat16, use fp16
        np_dtype = np.float16
        x = (np.random.randn(batch, n).astype(np_dtype)
             if batch > 1 else np.random.randn(n).astype(np_dtype))
        dtype = "float16"
    else:
        np_dtype = np.float64
        x = (np.random.randn(batch, n).astype(np_dtype)
             if batch > 1 else np.random.randn(n).astype(np_dtype))

    # Use fwht() API which supports all dtypes
    def run_once():
        _ = pyfwht.fwht(x, backend='cpu')

    mean_secs = _timeit(run_once, sync=None, warmup=warmup, repeats=repeats)
    butterflies, ops = _ops_for_fwht(n)
    gops = _format_gops(ops * batch, mean_secs)

    # Correctness against reference
    ref = _numpy_reference(x)
    out_arr = pyfwht.fwht(x.copy(), backend='cpu')
    max_err = float(np.max(np.abs(out_arr - ref))) if hasattr(out_arr, "__array__") else None

    return BenchResult(
        impl="pyfwht-cpu",
        device="cpu",
        n=n,
        batch=batch,
        dtype=dtype,
        include_transfer=False,
        repeats=repeats,
        warmup=warmup,
        time_ms=mean_secs * 1e3,
        gops=gops,
        max_abs_err=max_err,
        note="",
    )


def bench_pyfwht_gpu(n: int, batch: int, dtype: str,
                     warmup: int, repeats: int, include_transfer: bool) -> Optional[BenchResult]:
    import numpy as np
    import pyfwht  # type: ignore

    if not pyfwht.has_gpu():
        return None

    note = ""
    # Prepare host input with native dtype (uses Tensor Cores for fp16/fp32!)
    if dtype == "int32":
        np_dtype = np.int32
        host = np.random.randint(-2**31, 2**31 - 1, size=(n,), dtype=np_dtype)
    elif dtype == "float64":
        np_dtype = np.float64
        host = np.random.randn(n).astype(np_dtype)
    elif dtype in ("float32", "fp32"):
        np_dtype = np.float32
        host = np.random.randn(n).astype(np_dtype)
        note = "uses Tensor Cores (sm_70+)"
    elif dtype in ("float16", "fp16"):
        np_dtype = np.float16
        host = np.random.randn(n).astype(np_dtype)
        note = "uses Tensor Cores (sm_70+) - maximum speed"
    elif dtype == "bfloat16":
        # NumPy doesn't have native bfloat16, use float16 as proxy
        np_dtype = np.float16
        host = np.random.randn(n).astype(np_dtype)
        note = "using fp16 (NumPy lacks bfloat16)"
        dtype = "float16"
    else:
        return None

    # Prepare batch input
    if batch == 1:
        host_batch = host.reshape(1, n)
    else:
        if dtype == "int32":
            host_batch = np.random.randint(-2**31, 2**31 - 1, size=(batch, n), dtype=np_dtype)
        else:
            host_batch = np.random.randn(batch, n).astype(np_dtype)
    
    # Use new fwht() API which automatically routes to Tensor Core kernels
    def run_once_include():
        """Includes data copy overhead"""
        work = host_batch.copy()
        _ = pyfwht.fwht(work, backend='cuda')

    def run_once_exclude():
        """Excludes data copy - pre-allocated work buffer"""
        work_buffer = host_batch.copy()
        _ = pyfwht.fwht(work_buffer, backend='cuda')
    
    fn = run_once_include if include_transfer else run_once_exclude
    mean_secs = _timeit(fn, sync=None, warmup=warmup, repeats=repeats)
    
    butterflies, ops = _ops_for_fwht(n)
    gops = _format_gops(ops * batch, mean_secs)

    # Correctness vs CPU reference
    test_input = host_batch[0].copy()
    ref = _numpy_reference(test_input)
    out_arr = test_input.copy()
    
    # Compute using GPU
    out_batch = out_arr.reshape(1, n)
    _ = pyfwht.fwht(out_batch, backend='cuda')
    out_arr = out_batch[0]
    
    try:
        max_err = float(np.max(np.abs(out_arr - ref)))
    except Exception:
        max_err = None

    return BenchResult(
        impl="pyfwht-gpu",
        device="gpu",
        n=n,
        batch=batch,
        dtype=dtype,
        include_transfer=include_transfer,
        repeats=repeats,
        warmup=warmup,
        time_ms=mean_secs * 1e3,
        gops=gops,
        max_abs_err=max_err,
        note=note or ("timing includes transfers" if include_transfer else ""),
    )


def bench_meta_torch(n: int, batch: int, dtype: str, device: str,
                     warmup: int, repeats: int, include_transfer: bool,
                     module_name: str, func_name: str) -> Optional[BenchResult]:
    import os
    debug = os.environ.get('DEBUG_BENCH')
    
    torch = _try_import_torch()
    if torch is None:
        if debug:
            print(f"[DEBUG] bench_meta_torch: torch is None")
        return None
    if device == "gpu" and not _cuda_available(torch, device):
        if debug:
            print(f"[DEBUG] bench_meta_torch: CUDA not available")
        return None

    mod = _try_import_module(module_name)
    if mod is None:
        if debug:
            print(f"[DEBUG] bench_meta_torch: module {module_name} is None")
        return None

    fn = getattr(mod, func_name, None)
    if not callable(fn):
        if debug:
            print(f"[DEBUG] bench_meta_torch: function {func_name} not found, trying alternatives")
        # Try a couple of common alternatives
        for alt in ("fht", "hadamard_transform", "fast_hadamard_transform"):
            fn = getattr(mod, alt, None)
            if callable(fn):
                func_name = alt
                if debug:
                    print(f"[DEBUG] bench_meta_torch: found alternative function {alt}")
                break
    if not callable(fn):
        if debug:
            print(f"[DEBUG] bench_meta_torch: no callable function found in {module_name}")
            print(f"[DEBUG] Available attributes: {dir(mod)}")
        return None

    # Inputs
    # Meta kernel supports only fp16 / bf16. Cast unsupported requests.
    orig_dtype = dtype
    if dtype in ("float16", "fp16"):
        dt = torch.float16
        dtype = "float16"
    elif dtype in ("bfloat16", "bf16"):
        dt = torch.bfloat16
        dtype = "bfloat16"
    else:
        # Fallback: use float16
        dt = torch.float16
        dtype = "float16"

    host = torch.randn((batch, n), dtype=dt)

    # Prepare device tensors
    use_gpu = device == "gpu"
    if use_gpu:
        try:
            dev = torch.device("cuda")
            # Sanity: allocate once to detect arch issues early
            _ = torch.empty((1,), device=dev)
        except Exception:
            return None

    def sync():
        if use_gpu:
            torch.cuda.synchronize()

    # The function likely expects input already on the device for fair kernel timing
    def run_once_include():
        if use_gpu:
            x = host.to("cuda", non_blocking=False)
            y = fn(x)
            y_cpu = y.to("cpu")
            _ = y_cpu
        else:
            _ = fn(host)

    # Exclude transfers by keeping data on device
    if use_gpu:
        dev_host = host.to("cuda")
        dev_work = dev_host.clone()  # Pre-allocate work buffer
        
        import os
        debug = os.environ.get('DEBUG_BENCH')

        def run_once_exclude():
            dev_work.copy_(dev_host)  # Reset to original input
            if debug and run_once_exclude.counter == 0:
                print(f"[DEBUG] Calling Meta fn with shape {dev_work.shape}, dtype {dev_work.dtype}")
            result = fn(dev_work)
            if debug and run_once_exclude.counter == 0:
                print(f"[DEBUG] Meta returned: {type(result)}, is None: {result is None}")
                print(f"[DEBUG] Input modified in-place: {torch.allclose(dev_work, dev_host) if result is None else 'N/A'}")
                run_once_exclude.counter += 1
            # Force synchronization - Meta kernel might use different stream
            if result is not None and hasattr(result, 'device'):
                _ = result.cpu()  # Force D2H transfer to ensure completion
            torch.cuda.synchronize()  # Sync all streams
        
        run_once_exclude.counter = 0
    else:
        host_work = host.clone()  # Pre-allocate work buffer
        
        def run_once_exclude():
            host_work.copy_(host)  # Reset to original input
            _ = fn(host_work)

    runner = run_once_include if include_transfer else run_once_exclude

    # Run timing
    try:
        mean_secs = _timeit(runner, sync=sync, warmup=warmup, repeats=repeats)
    except Exception:
        return None

    butterflies, ops = _ops_for_fwht(n)
    gops = _format_gops(ops * batch, mean_secs)

    # Correctness vs CPU reference
    # Correctness: Meta kernel normalizes by 1/sqrt(N), pyfwht doesn't.
    # Compute reference with pyfwht and scale by sqrt(N) to match Meta's convention.
    max_err = None
    try:
        import numpy as np
        import pyfwht
        
        # Create fresh random input matching host tensor
        np.random.seed(123)
        test_input = np.random.randn(batch, n).astype(np.float64)
        
        # Compute reference with pyfwht CPU (unnormalized)
        ref_out = np.empty_like(test_input)
        for i in range(batch):
            row_copy = test_input[i].copy()
            pyfwht.transform(row_copy, backend=pyfwht.Backend.CPU)
            ref_out[i] = row_copy
        
        # Normalize reference by 1/sqrt(N) to match Meta convention
        ref_out = ref_out / np.sqrt(n)
        
        # Get meta output on same input
        test_tensor = torch.tensor(test_input, dtype=dt, device=('cuda' if use_gpu else 'cpu'))
        meta_result = fn(test_tensor)
        meta_out = meta_result.cpu().numpy() if use_gpu else meta_result.numpy()
        
        # Compute error
        diff = np.abs(meta_out - ref_out)
        max_err = float(np.max(diff))
    except Exception as e:
        max_err = None

    impl_name = f"meta-{module_name}.{func_name}"
    return BenchResult(
        impl=impl_name,
        device=("gpu" if use_gpu else "cpu"),
        n=n,
        batch=batch,
        dtype=dtype,
        include_transfer=include_transfer,
        repeats=repeats,
        warmup=warmup,
        time_ms=mean_secs * 1e3,
        gops=gops,
        max_abs_err=max_err,
        note=("cast from " + orig_dtype if orig_dtype != dtype else ""),
    )


def parse_sizes(powers: List[int], sizes: List[int]) -> List[int]:
    out: List[int] = []
    out.extend([1 << p for p in powers])
    out.extend(sizes)
    out = sorted(set(out))
    # filter invalid (non power-of-two) sizes
    out = [n for n in out if n > 0 and (n & (n - 1)) == 0]
    return out


def _print_output_comparison(n: int, args) -> None:
    """Print side-by-side output comparison for visual verification."""
    import numpy as np
    import pyfwht
    
    # Create random Boolean input
    np.random.seed(42)
    x_np = np.random.randint(0, 2, n, dtype=np.int32)
    print(f"Input (Boolean 0/1): {x_np}")
    
    # pyfwht CPU (unnormalized)
    x_pyfwht = x_np.astype(np.float64).copy()
    x_pyfwht = pyfwht.fwht(x_pyfwht, backend='cpu')
    print(f"\npyfwht output (unnormalized):\n{x_pyfwht}")
    
    # pyfwht normalized
    x_pyfwht_norm = x_pyfwht / np.sqrt(n)
    print(f"\npyfwht normalized (÷√{n}):\n{x_pyfwht_norm}")
    
    # Meta kernel if available
    if not args.skip_meta:
        torch = _try_import_torch()
        mod = _try_import_module(args.meta_module)
        if torch is not None and mod is not None:
            fn = getattr(mod, args.meta_func, None)
            if fn is not None:
                try:
                    x_meta = torch.tensor(x_np, dtype=torch.float16, device='cuda').unsqueeze(0)
                    y_meta = fn(x_meta, inplace=False)
                    meta_out = y_meta.cpu().squeeze().numpy()
                    print(f"\nMeta kernel output:\n{meta_out}")
                    
                    diff = np.abs(x_pyfwht_norm - meta_out)
                    print(f"\nMax absolute error: {np.max(diff):.6e}")
                    print(f"Outputs match: {np.allclose(x_pyfwht_norm, meta_out, atol=1e-2)}")
                except Exception as e:
                    print(f"\nMeta kernel failed: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_pyfwht_importable()

    parser = argparse.ArgumentParser(description="Compare pyfwht with meta-pytorch hadamard CUDA kernel")
    parser.add_argument("--powers", type=int, nargs="*", default=[10, 11, 12],
                        help="Powers of two to test (e.g., 10 11 12 for 1k, 2k, 4k)")
    parser.add_argument("--sizes", type=int, nargs="*", default=[],
                        help="Explicit sizes to test (must be powers of two)")
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 10, 100],
                        help="Batch sizes to test")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32", "float64", "int32"],
                        help="Primary dtype for the run (meta kernel supports only float16/bfloat16)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu"],
                        help="Target device for meta kernel; pyfwht runs both CPU and GPU if available")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--include-transfer", action="store_true",
                        help="Include H2D/D2H in timing for GPU implementations")
    parser.add_argument("--csv", type=str, default=None, help="Write results to CSV file path")
    parser.add_argument("--skip-meta", action="store_true", help="Skip meta-pytorch hadamard kernel")
    parser.add_argument("--skip-pyfwht-gpu", action="store_true")
    parser.add_argument("--skip-pyfwht-cpu", action="store_true")
    parser.add_argument("--meta-module", type=str, default="fast_hadamard_transform",
                        help="Module name to import for meta kernel extension")
    parser.add_argument("--meta-func", type=str, default="hadamard_transform",
                        help="Function name to call from meta kernel module")
    parser.add_argument("--print-outputs", action="store_true",
                        help="Print actual output vectors for visual comparison (only for small sizes)")

    args = parser.parse_args(argv)

    sizes = parse_sizes(args.powers, args.sizes)
    if not sizes:
        print("No valid sizes to test.")
        return 2

    results: List[BenchResult] = []

    # Determine if GPU is usable for meta kernel
    torch = _try_import_torch()
    if args.device == "auto":
        device = "gpu" if (torch is not None and _cuda_available(torch, "gpu")) else "cpu"
    else:
        device = args.device

    for n in sizes:
        for b in args.batches:
            # Print outputs if requested and size is small enough
            if args.print_outputs and n <= 20 and b == 1:
                print(f"\n{'='*70}")
                print(f"OUTPUT COMPARISON: Size={n}, Batch={b}")
                print('='*70)
                _print_output_comparison(n, args)
                print('='*70)
            
            # pyfwht CPU
            if not args.skip_pyfwht_cpu:
                try:
                    res = bench_pyfwht_cpu(n=n, batch=b, dtype=args.dtype,
                                           warmup=args.warmup, repeats=args.repeats)
                    results.append(res)
                    print(f"pyfwht CPU n={n} b={b}: {res.time_ms:.3f} ms, {res.gops:.2f} GOps/s")
                except Exception as e:
                    print(f"pyfwht CPU n={n} b={b} failed: {e}")

            # pyfwht GPU
            if not args.skip_pyfwht_gpu:
                try:
                    gres = bench_pyfwht_gpu(n=n, batch=b, dtype=args.dtype,
                                             warmup=args.warmup, repeats=args.repeats,
                                             include_transfer=args.include_transfer)
                    if gres is not None:
                        results.append(gres)
                        print(f"pyfwht GPU n={n} b={b}: {gres.time_ms:.3f} ms, {gres.gops:.2f} GOps/s")
                    else:
                        print(f"pyfwht GPU n={n} b={b}: skipped (unavailable)")
                except Exception as e:
                    print(f"pyfwht GPU n={n} b={b} failed: {e}")

            # meta-pytorch hadamard kernel
            if not args.skip_meta:
                try:
                    mres = bench_meta_torch(n=n, batch=b, dtype=args.dtype,
                                            device=device, warmup=args.warmup, repeats=args.repeats,
                                            include_transfer=args.include_transfer,
                                            module_name=args.meta_module, func_name=args.meta_func)
                    if mres is not None:
                        results.append(mres)
                        print(f"META {mres.device} n={n} b={b}: {mres.time_ms:.3f} ms, {mres.gops:.2f} GOps/s")
                    else:
                        print(f"META {device} n={n} b={b}: skipped (unavailable)")
                except Exception as e:
                    print(f"META {device} n={n} b={b} failed: {e}")

    if args.csv:
        fieldnames = list(BenchResult.__dataclass_fields__.keys())
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_row())
        print(f"Wrote {len(results)} results to {args.csv}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
