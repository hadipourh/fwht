"""Benchmark pyfwht against SymPy's FWHT implementation on CPU.

This script compares correctness and single-thread CPU performance of the
pyfwht library versus SymPy's reference Walsh-Hadamard transform for
sizes 2^6 through 2^19.
"""
from __future__ import annotations

import sys
import time
from typing import Callable, List, Sequence

import numpy as np
import pyfwht as fwht

try:
    from sympy.discrete.transforms import fwht as sympy_fwht
except ImportError:  # pragma: no cover - optional dependency for the example
    print("SymPy is required for this benchmark. Install it via `pip install sympy`.", file=sys.stderr)
    sys.exit(1)


def benchmark(fn: Callable[[], None], repeat: int = 3) -> float:
    """Return average runtime (seconds) of *repeat* executions of fn."""
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    end = time.perf_counter()
    return (end - start) / repeat


def sympy_transform(data: Sequence[float]) -> np.ndarray:
    """Convenience wrapper returning a NumPy array from SymPy's fwht."""
    return np.array(sympy_fwht(list(data)), dtype=np.float64)


def run_case(n: int, rng: np.random.Generator) -> dict:
    """Benchmark a single problem size and validate correctness."""
    values = rng.standard_normal(n).astype(np.float64)
    values_list: List[float] = values.tolist()

    sympy_out = sympy_transform(values_list)

    lib_out = values.copy()
    fwht.transform(lib_out, backend=fwht.Backend.CPU)

    max_err = float(np.max(np.abs(lib_out - sympy_out)))
    if max_err > 1e-9:
        raise AssertionError(f"Mismatch detected for n={n}: max |diff| = {max_err}")

    # SymPy is slow for larger n, so reduce repeats beyond 2^12
    sympy_repeat = 3 if n <= 4096 else 1
    lib_repeat = 10 if n <= 4096 else 5

    sympy_time = benchmark(lambda: sympy_fwht(values_list), repeat=sympy_repeat)
    lib_time = benchmark(lambda: fwht.transform(values.copy(), backend=fwht.Backend.CPU), repeat=lib_repeat)

    speedup = sympy_time / lib_time if lib_time > 0 else float("inf")
    return {
        "n": n,
        "sympy_time": sympy_time,
        "lib_time": lib_time,
        "speedup": speedup,
        "max_err": max_err,
    }


if __name__ == "__main__":
    print("pyfwht vs SymPy FWHT (CPU)")
    print("-" * 50)
    rng = np.random.default_rng(seed=0)

    results = []
    for power in range(6, 20):
        n = 1 << power
        print(f"Running n={n} ...", end=" ")
        res = run_case(n, rng)
        res["power"] = power
        print("ok")
        results.append(res)

    header = f"{'n (2^k)':>10} | {'SymPy (s)':>10} | {'pyfwht (s)':>11} | {'speedup':>8} | {'max |diff|':>11}"
    print("\n" + header)
    print("-" * len(header))
    for res in results:
        print(
            f"2^{res['power']:>2d}={res['n']:5d} | "
            f"{res['sympy_time']:10.5f} | "
            f"{res['lib_time']:11.5f} | "
            f"{res['speedup']:8.2f} | "
            f"{res['max_err']:11.3e}"
        )

    best_speedup = max(r["speedup"] for r in results)
    print("\nLargest observed speedup: {:.2f}×".format(best_speedup))
