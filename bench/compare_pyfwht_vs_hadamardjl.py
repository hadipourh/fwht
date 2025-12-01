"""Compare pyfwht against Hadamard.jl using shared random inputs.

The script spawns Julia to run the vendored Hadamard.jl package so both
implementations operate on identical data. Results include timing for
both libraries, correctness checks, and aggregate speedups. Example:

    python3 bench/compare_pyfwht_vs_hadamardjl.py \
        --julia-bin julia \
        --julia-project otherlibs/Hadamard.jl

The Julia command must support the `--project` flag (Julia >= 1.0).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyfwht as fwht

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JULIA_PROJECT = ROOT / "otherlibs" / "Hadamard.jl"
RUNNER_SCRIPT = ROOT / "bench" / "compare_hadamard_runner.jl"
TMP_DEFAULT = ROOT / "bench" / "tmp"
BACKEND_MAP = {}
for candidate in ("AUTO", "CPU", "OPENMP", "GPU"):
    if hasattr(fwht.Backend, candidate):
        BACKEND_MAP[candidate.lower()] = getattr(fwht.Backend, candidate)

CORRECTNESS_TOL = 1e-9
CHECK_PASS = "✔"
CHECK_FAIL = "✘"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-power", type=int, default=6, help="Smallest power of two (inclusive)")
    parser.add_argument("--max-power", type=int, default=30, help="Largest power of two (inclusive)")
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of repetitions per size for both implementations",
    )
    parser.add_argument("--julia-bin", default="julia", help="Path to Julia executable")
    parser.add_argument(
        "--julia-project",
        default=str(DEFAULT_JULIA_PROJECT),
        help="Julia project to use (should contain Hadamard.jl)",
    )
    parser.add_argument(
        "--julia-script",
        default=str(RUNNER_SCRIPT),
        help="Helper Julia script (compare_hadamard_runner.jl)",
    )
    parser.add_argument("--tmp-dir", default=str(TMP_DEFAULT), help="Directory for shared buffers")
    parser.add_argument(
        "--julia-threads",
        type=int,
        default=None,
        help="Set JULIA_NUM_THREADS (defaults to detected CPU cores)",
    )
    parser.add_argument(
        "--py-backend",
        default="auto",
        help="pyfwht backend to use (one of: {}), defaults to AUTO".format(
            ", ".join(sorted(BACKEND_MAP))
        ),
    )
    parser.add_argument(
        "--py-threads",
        type=int,
        default=None,
        help="Number of CPU threads for pyfwht OpenMP backend (0 or omitted = auto)",
    )
    parser.add_argument(
        "--no-scale",
        dest="scale_output",
        action="store_false",
        help="Disable scaling Hadamard.jl output by n (default keeps outputs comparable)",
    )
    parser.set_defaults(scale_output=True)
    return parser.parse_args()


def ensure_tmp_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_py_backend(name: str) -> fwht.Backend:
    key = name.lower()
    if key not in BACKEND_MAP:
        raise SystemExit(f"Unknown pyfwht backend '{name}'. Choices: {', '.join(sorted(BACKEND_MAP))}.")
    return BACKEND_MAP[key]


def run_pyfwht(
    data: np.ndarray,
    repeat: int,
    backend: fwht.Backend,
    ctx: fwht.Context | None,
) -> Dict[str, float]:
    buf = data.copy()
    start = time.perf_counter()
    for _ in range(repeat):
        buf[:] = data
        if ctx is not None:
            ctx.transform(buf)
        else:
            fwht.transform(buf, backend=backend)
    elapsed = (time.perf_counter() - start) / repeat
    return {"output": buf.copy(), "time": elapsed}


def run_hadamard(
    julia_bin: str,
    project: str,
    script: str,
    input_path: Path,
    output_path: Path,
    n: int,
    repeat: int,
    scale_output: bool,
    julia_threads: int | None,
) -> Dict[str, float]:
    cmd = [
        julia_bin,
        f"--project={project}",
        script,
        str(input_path),
        str(output_path),
        str(n),
        str(repeat),
    ]
    if scale_output:
        cmd.append("--scale-output")
    env = os.environ.copy()
    if julia_threads is not None:
        env["JULIA_NUM_THREADS"] = str(julia_threads)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Julia benchmark failed")
    last_line = proc.stdout.strip().splitlines()[-1]
    stats = json.loads(last_line)
    result = np.fromfile(output_path, dtype=np.float64, count=n)
    return {"time": float(stats["time"]), "output": result}


def format_table_row(
    power: int,
    n: int,
    py_time: float,
    jl_time: float,
    max_err: float,
    passed: bool,
) -> str:
    if py_time <= jl_time:
        ratio = jl_time / py_time if py_time > 0 else float("inf")
        verdict = f"pyfwht ×{ratio:7.2f}"
    else:
        ratio = py_time / jl_time if jl_time > 0 else float("inf")
        verdict = f"Hadamard ×{ratio:7.2f}"
    correctness = CHECK_PASS if passed else CHECK_FAIL
    return (
        f"2^{power:>2d}={n:10d} | "
        f"{py_time:12.6f} | "
        f"{jl_time:15.6f} | "
        f"{verdict:>16} | "
        f"{max_err:11.3e} | "
        f"{correctness:^9}"
    )


def main() -> None:
    args = parse_args()
    if args.min_power > args.max_power:
        raise SystemExit("min-power must be <= max-power")
    if args.repeat <= 0:
        raise SystemExit("repeat must be positive")
    if args.py_threads is not None and args.py_threads < 0:
        raise SystemExit("py-threads must be >= 0")

    py_backend = resolve_py_backend(args.py_backend)

    if py_backend == fwht.Backend.OPENMP and not fwht.has_openmp():
        raise SystemExit("pyfwht was built without OpenMP support")
    if py_backend == fwht.Backend.GPU and not fwht.has_gpu():
        raise SystemExit("pyfwht GPU backend requested but CUDA is unavailable")

    if args.julia_threads is None:
        detected = os.cpu_count() or 1
        args.julia_threads = detected

    tmp_dir = ensure_tmp_dir(Path(args.tmp_dir))
    rng = np.random.default_rng(seed=0)

    results: List[Dict[str, float]] = []

    py_threads = args.py_threads if args.py_threads is not None else 0
    ctx_kwargs = {"backend": py_backend, "num_threads": py_threads}

    print("pyfwht vs Hadamard.jl (CPU)")
    print("-" * 72)
    print(f"Julia project : {args.julia_project}")
    print(f"Julia threads : {args.julia_threads}")
    print(f"pyfwht backend: {py_backend.name}")
    if py_backend == fwht.Backend.OPENMP:
        thread_source = args.py_threads
        thread_label = "auto" if thread_source in (None, 0) else str(thread_source)
        print(f"pyfwht threads: {thread_label}")
    print(f"Repeat count  : {args.repeat}")

    with fwht.Context(**ctx_kwargs) as py_ctx:
        for power in range(args.min_power, args.max_power + 1):
            n = 1 << power
            repeat = args.repeat
            print(f"Running n={n} (repeat={repeat}) ...")
            data = rng.standard_normal(n).astype(np.float64)
            input_path = tmp_dir / f"input_{n}.bin"
            output_path = tmp_dir / f"hadamard_{n}.bin"
            data.tofile(input_path)

            py_stats = run_pyfwht(data, repeat, py_backend, ctx=py_ctx)
            jl_stats = run_hadamard(
                args.julia_bin,
                args.julia_project,
                args.julia_script,
                input_path,
                output_path,
                n,
                repeat,
                args.scale_output,
                args.julia_threads,
            )

            max_err = float(np.max(np.abs(py_stats["output"] - jl_stats["output"])))

            results.append(
                {
                    "power": power,
                    "n": n,
                    "py_time": py_stats["time"],
                    "jl_time": jl_stats["time"],
                    "max_err": max_err,
                    "passed": max_err <= CORRECTNESS_TOL,
                }
            )

    header = (
        f"{'n (2^k)':>12} | {'pyfwht (s)':>12} | {'Hadamard.jl (s)':>15} | "
        f"{'Faster (×)':>16} | {'max |diff|':>11} | {'Correct?':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for res in results:
        print(
            format_table_row(
                res["power"],
                res["n"],
                res["py_time"],
                res["jl_time"],
                res["max_err"],
                res["passed"],
            )
        )

    worst_err = max(r["max_err"] for r in results)
    total_cases = len(results)
    py_wins = [r for r in results if r["py_time"] <= r["jl_time"]]
    jl_wins = [r for r in results if r["py_time"] > r["jl_time"]]

    print("\nSummary")
    print("-" * 72)
    if py_wins:
        py_speedups = [r["jl_time"] / r["py_time"] for r in py_wins if r["py_time"] > 0]
        avg_py_speedup = sum(py_speedups) / len(py_speedups)
        print(
            f"pyfwht faster on {len(py_wins)}/{total_cases} sizes "
            f"(avg speedup {avg_py_speedup:.2f}×)."
        )
    else:
        print("pyfwht was never faster in this sweep.")

    if jl_wins:
        jl_speedups = [r["py_time"] / r["jl_time"] for r in jl_wins if r["jl_time"] > 0]
        avg_jl_speedup = sum(jl_speedups) / len(jl_speedups)
        print(
            f"Hadamard.jl faster on {len(jl_wins)}/{total_cases} sizes "
            f"(avg speedup {avg_jl_speedup:.2f}×)."
        )
    else:
        print("Hadamard.jl was never faster in this sweep.")

    print(f"Max |diff|: {worst_err:.3e}")


if __name__ == "__main__":
    main()
