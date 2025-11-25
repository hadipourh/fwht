"""
Compare Dao-AILab/fast-hadamard-transform (PyTorch CUDA) vs libfwht (this repo).

This script benchmarks:
- Dao-AILab's GPU kernel via `fast_hadamard_transform.hadamard_transform` (fp16/fp32)
- libfwht GPU (if available) using pyfwht.gpu batch transforms (int32 and float64)

Notes:
- Dao-AILab library is GPU-only (CUDA). It supports fp16, bfloat16, and fp32, up to dim=32768.
- libfwht supports int32, fp64, fp32, and fp16 (with Tensor Cores) on GPU.
- FP16 Tensor Core support: n=256, 512, 1024, 2048, 4096, 8192, 16384, 32768.
- Results are memory-bandwidth bound; dtype differences mainly affect bytes moved and math mode.

Prereqs:
    pip install torch --index-url https://download.pytorch.org/whl/cu121  # or your CUDA-matched wheel
    pip install fast-hadamard-transform
    pip install numpy
    pip install -e .   # from this repo's python/ folder

Run:
    python python/examples/compare_fast_hadamard.py --sizes 4096,8192,16384,32768 --repeats 20 --warmup 5
"""

import argparse
import time
import math
import numpy as np

try:
    import torch
    import fast_hadamard_transform as dao_fht
except Exception as e:
    torch = None
    dao_fht = None

import pyfwht as fwht


def time_cuda(fn, repeats=10, warmup=5):
    # Warmup
    for _ in range(warmup):
        fn()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
    # Timed
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.min(timings)), float(np.mean(timings)), float(np.std(timings))


def hadamard_ops(n: int) -> int:
    # Number of butterfly operations ~ n * log2(n)
    return n * int(math.log2(n))


def bench_dao_gpu(sizes, repeats, warmup, dtype):
    results = []
    if torch is None or not torch.cuda.is_available() or dao_fht is None:
        return results
    for n in sizes:
        # Use a reasonable batch to amortize launch overhead
        batch = 1024 if n <= 8192 else 256
        x = torch.randn(batch, n, device='cuda', dtype=dtype)
        def run():
            y = dao_fht.hadamard_transform(x, scale=1.0)
            return y
        tmin, tmean, tstd = time_cuda(run, repeats=repeats, warmup=warmup)
        ops = hadamard_ops(n) * batch
        results.append({
            'lib': 'dao_fht',
            'dtype': str(dtype).replace('torch.', ''),
            'n': n,
            'batch': batch,
            't_min_ms': tmin,
            't_mean_ms': tmean,
            't_std_ms': tstd,
            'GOps/s': ops / (tmin / 1000.0) / 1e9,
        })
    return results


def bench_pyfwht_gpu(sizes, repeats, warmup):
    results = []
    if not fwht.has_gpu():
        return results
    # int32 path
    for n in sizes:
        batch = 1024 if n <= 8192 else 256
        x = np.random.randint(-1, 2, size=(batch, n), dtype=np.int32)
        def run():
            fwht.gpu.batch_transform_i32(x)
            return x
        tmin, tmean, tstd = time_cuda(run, repeats=repeats, warmup=warmup)
        ops = hadamard_ops(n) * batch
        results.append({
            'lib': 'pyfwht',
            'dtype': 'int32',
            'n': n,
            'batch': batch,
            't_min_ms': tmin,
            't_mean_ms': tmean,
            't_std_ms': tstd,
            'GOps/s': ops / (tmin / 1000.0) / 1e9,
        })
    # float64 path
    for n in sizes:
        batch = 1024 if n <= 8192 else 256
        x = np.random.randn(batch, n).astype(np.float64)
        def run():
            fwht.gpu.batch_transform_f64(x)
            return x
        tmin, tmean, tstd = time_cuda(run, repeats=repeats, warmup=warmup)
        ops = hadamard_ops(n) * batch
        results.append({
            'lib': 'pyfwht',
            'dtype': 'float64',
            'n': n,
            'batch': batch,
            't_min_ms': tmin,
            't_mean_ms': tmean,
            't_std_ms': tstd,
            'GOps/s': ops / (tmin / 1000.0) / 1e9,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=str, default='4096,8192,16384,32768')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--dao-dtype', type=str, default='float32', choices=['float16','float32','bfloat16'])
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]

    print('=== Environment ===')
    print('GPU available (pyfwht):', fwht.has_gpu())
    if torch is not None:
        print('Torch CUDA available:', torch.cuda.is_available())
        print('Torch version:', torch.__version__)
        print('CUDA version (torch):', torch.version.cuda)
        if torch.cuda.is_available():
            print('Device:', torch.cuda.get_device_name())
    print()

    all_results = []

    # Dao-AILab GPU
    torch_dtype = {
        'float16': torch.float16 if torch is not None else None,
        'float32': torch.float32 if torch is not None else None,
        'bfloat16': torch.bfloat16 if torch is not None else None,
    }[args.dao_dtype]
    dao_results = bench_dao_gpu(sizes, args.repeats, args.warmup, torch_dtype)
    all_results.extend(dao_results)

    # pyfwht GPU (int32 and float64)
    fwht_results = bench_pyfwht_gpu(sizes, args.repeats, args.warmup)
    all_results.extend(fwht_results)

    # Pretty print
    if not all_results:
        print('No results (missing GPU or packages).')
        return

    print('\n=== Results (min over repeats) ===')
    rows = sorted(all_results, key=lambda r: (r['n'], r['lib'], r['dtype']))
    for r in rows:
        print(f"{r['lib']:6s} {r['dtype']:7s}  n={r['n']:6d}  batch={r['batch']:5d}  "
              f"min={r['t_min_ms']:7.3f} ms  mean={r['t_mean_ms']:7.3f}±{r['t_std_ms']:5.3f} ms  "
              f"[{r['GOps/s']:5.2f} GOps/s]")


if __name__ == '__main__':
    main()
