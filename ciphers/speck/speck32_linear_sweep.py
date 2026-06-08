#!/usr/bin/env python3

import argparse
import csv
import errno
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


KeyMode = Dict[str, Optional[Union[int, str]]]


TOP_RESULT_RE = re.compile(
    r"^\s*(\d+)\.\s+mask=(0x[0-9a-fA-F]+)\s+"
    r"(correlation|rms)=(\S+)(?:\s+abs=(\S+))?\s*$"
)
DEFAULT_KEY = 0x1918111009080100


def parse_int(value: str) -> int:
    return int(value, 0)


def enumerate_hw_one_two() -> List[int]:
    values: List[int] = []

    for bit in range(32):
        values.append(1 << bit)

    for first_bit in range(32):
        for second_bit in range(first_bit + 1, 32):
            values.append((1 << first_bit) | (1 << second_bit))

    return values


def hamming_weight(value: int) -> int:
    weight = 0

    while value:
        value &= value - 1
        weight += 1

    return weight


def parse_top_results(output: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    for line in output.splitlines():
        match = TOP_RESULT_RE.match(line)
        if not match:
            continue
        metric_name = match.group(3)
        metric_pow2 = match.group(4)
        abs_metric_pow2 = match.group(5) if match.group(5) is not None else metric_pow2
        results.append(
            {
                "rank": match.group(1),
                "free_mask_hex": match.group(2).lower(),
                "metric_name": metric_name,
                "metric_pow2": metric_pow2,
                "abs_metric_pow2": abs_metric_pow2,
            }
        )

    return results


def resolve_key_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> KeyMode:
    if args.key is not None and args.num_keys is not None:
        parser.error("choose either --key or --num-keys")
    if args.seed is not None and args.num_keys is None:
        parser.error("--seed can only be used with --num-keys")

    if args.key is not None:
        if args.key < 0 or args.key > 0xFFFFFFFFFFFFFFFF:
            parser.error("key must fit in 64 bits")
        return {
            "mode": "single_key",
            "key": args.key,
            "num_keys": None,
            "seed": None,
        }

    if args.num_keys is not None:
        if args.num_keys <= 0:
            parser.error("num-keys must be positive")
        seed = args.seed if args.seed is not None else secrets.randbits(64)
        if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
            parser.error("seed must fit in 64 bits")
        return {
            "mode": "random_keys",
            "key": None,
            "num_keys": args.num_keys,
            "seed": seed,
        }

    return {
        "mode": "single_key",
        "key": DEFAULT_KEY,
        "num_keys": None,
        "seed": None,
    }


def build_key_arguments(key_mode: KeyMode) -> List[str]:
    if key_mode["mode"] == "single_key":
        return ["--key", f"0x{int(key_mode['key']):016x}"]

    return [
        "--num-keys",
        str(int(key_mode["num_keys"])),
        "--seed",
        f"0x{int(key_mode['seed']):016x}",
    ]


def build_environment(omp_threads: Optional[int]) -> Dict[str, str]:
    env = os.environ.copy()
    if omp_threads is not None:
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["OMP_DYNAMIC"] = "false"
        env.setdefault("OMP_PROC_BIND", "spread")
        env.setdefault("OMP_PLACES", "cores")
    return env


def run_query(
    binary_path: Path,
    fwht_root: Path,
    rounds: int,
    key_mode: KeyMode,
    fixed_side: str,
    fixed_mask: int,
    top_k: int,
    use_codebook: bool,
    unsafe_memory: bool,
    omp_threads: Optional[int],
) -> List[Dict[str, str]]:
    command = [
        str(binary_path),
        "--rounds",
        str(rounds),
        *build_key_arguments(key_mode),
        f"--{fixed_side}-mask",
        f"0x{fixed_mask:08x}",
        "--top",
        str(top_k),
        "--force",
    ]

    if use_codebook:
        command.append("--codebook")
    if unsafe_memory:
        command.append("--unsafe-memory")

    env = build_environment(omp_threads)

    try:
        completed = subprocess.run(
            command,
            cwd=fwht_root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        if exc.errno == errno.ENOEXEC:
            raise RuntimeError(
                f"cannot execute {binary_path}\n"
                "The existing Speck binary is not runnable on this host. "
                "This usually means a stale binary copied from another OS or CPU architecture.\n"
                "Rebuild it on the current machine with: make speck32-linear NO_CUDA=1"
            ) from exc
        raise RuntimeError(f"failed to launch {binary_path}: {exc}") from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed for {fixed_side} mask 0x{fixed_mask:08x}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    results = parse_top_results(completed.stdout)
    if not results:
        raise RuntimeError(
            f"no top results found for {fixed_side} mask 0x{fixed_mask:08x}\n"
            f"stdout:\n{completed.stdout}"
        )
    return results


def run_backend_sweep(
    binary_path: Path,
    fwht_root: Path,
    rounds: int,
    key_mode: KeyMode,
    fixed_side: str,
    output_path: Path,
    top_k: int,
    use_codebook: bool,
    unsafe_memory: bool,
    omp_threads: Optional[int],
    chunk_size: Optional[int],
) -> None:
    command = [
        str(binary_path),
        "--rounds",
        str(rounds),
        *build_key_arguments(key_mode),
        "--fixed-side",
        fixed_side,
        "--csv-output",
        str(output_path),
        "--top",
        str(top_k),
        "--force",
    ]

    if use_codebook:
        command.append("--codebook")
    if unsafe_memory:
        command.append("--unsafe-memory")
    if chunk_size is not None:
        command.extend(["--chunk-size", str(chunk_size)])

    try:
        completed = subprocess.run(
            command,
            cwd=fwht_root,
            env=build_environment(omp_threads),
            check=False,
        )
    except OSError as exc:
        if exc.errno == errno.ENOEXEC:
            raise RuntimeError(
                f"cannot execute {binary_path}\n"
                "The shared-process Speck binary is not runnable on this host. "
                "This usually means a stale binary copied from another OS or CPU architecture.\n"
                "Rebuild it on the current machine with: make speck32-linear NO_CUDA=1"
            ) from exc
        raise RuntimeError(f"failed to launch {binary_path}: {exc}") from exc

    if completed.returncode != 0:
        raise RuntimeError(
            f"shared-process sweep command failed for fixed {fixed_side} masks "
            f"with exit code {completed.returncode}"
        )


def output_path_for(
    output_dir: Path,
    rounds: int,
    key_mode: KeyMode,
    suffix: str,
    top_k: int,
) -> Path:
    if key_mode["mode"] == "single_key":
        return output_dir / (
            f"speck32_{rounds}r_key_{int(key_mode['key']):016x}_{suffix}_top{top_k}.csv"
        )

    return output_dir / (
        f"speck32_{rounds}r_{int(key_mode['num_keys'])}keys_seed_{int(key_mode['seed']):016x}_{suffix}_top{top_k}.csv"
    )


def write_sweep(
    binary_path: Path,
    fwht_root: Path,
    rounds: int,
    key_mode: KeyMode,
    fixed_side: str,
    output_path: Path,
    top_k: int,
    use_codebook: bool,
    unsafe_memory: bool,
    omp_threads: Optional[int],
) -> None:
    fixed_masks = enumerate_hw_one_two()
    key_hex = "" if key_mode["key"] is None else f"0x{int(key_mode['key']):016x}"
    seed_hex = "" if key_mode["seed"] is None else f"0x{int(key_mode['seed']):016x}"
    num_keys_text = "" if key_mode["num_keys"] is None else str(int(key_mode["num_keys"]))

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rounds",
                "key_mode",
                "key_hex",
                "num_keys",
                "seed_hex",
                "fixed_side",
                "fixed_mask_hex",
                "fixed_mask_weight",
                "rank",
                "free_mask_hex",
                "metric_name",
                "metric_pow2",
                "abs_metric_pow2",
            ]
        )

        total = len(fixed_masks)
        for index, fixed_mask in enumerate(fixed_masks, start=1):
            print(
                f"[{fixed_side} {index}/{total}] 0x{fixed_mask:08x}",
                file=sys.stderr,
                flush=True,
            )
            results = run_query(
                binary_path=binary_path,
                fwht_root=fwht_root,
                rounds=rounds,
                key_mode=key_mode,
                fixed_side=fixed_side,
                fixed_mask=fixed_mask,
                top_k=top_k,
                use_codebook=use_codebook,
                unsafe_memory=unsafe_memory,
                omp_threads=omp_threads,
            )
            for result in results:
                writer.writerow(
                    [
                        rounds,
                        key_mode["mode"],
                        key_hex,
                        num_keys_text,
                        seed_hex,
                        fixed_side,
                        f"0x{fixed_mask:08x}",
                        hamming_weight(fixed_mask),
                        result["rank"],
                        result["free_mask_hex"],
                        result["metric_name"],
                        result["metric_pow2"],
                        result["abs_metric_pow2"],
                    ]
                )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Speck32 exact linear masks of Hamming weight 1 and 2.",
    )
    parser.add_argument("--rounds", required=True, type=int, help="number of rounds")
    parser.add_argument("--key", type=parse_int, default=None, help="64-bit key in hex or decimal")
    parser.add_argument("--num-keys", type=int, default=None, help="number of uniformly random master keys")
    parser.add_argument("--seed", type=parse_int, default=None, help="seed for the random multi-key experiment")
    parser.add_argument("--top-k", type=int, default=5, help="top results per fixed mask")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="directory for the csv files",
    )
    parser.add_argument("--codebook", action="store_true", help="use the codebook mode")
    parser.add_argument(
        "--unsafe-memory",
        action="store_true",
        help="pass --unsafe-memory to the Speck tool",
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        help="set OMP_NUM_THREADS for each run",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="shared-process backend only: number of fixed masks per chunk",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.rounds < 1 or args.rounds > 22:
        parser.error("rounds must be in [1, 22]")
    if args.top_k <= 0:
        parser.error("top-k must be positive")
    if args.chunk_size is not None and args.chunk_size <= 0:
        parser.error("chunk-size must be positive")

    key_mode = resolve_key_mode(parser, args)
    if key_mode["mode"] == "random_keys":
        print(
            f"using {int(key_mode['num_keys'])} random master keys with seed 0x{int(key_mode['seed']):016x}",
            file=sys.stderr,
        )

    fwht_root = Path(__file__).resolve().parents[2]
    sweep_binary_path = fwht_root / "build" / "speck32_linear_sweep"
    binary_path = fwht_root / "build" / "speck32_linear"
    use_shared_backend = sweep_binary_path.is_file()
    if not use_shared_backend and not binary_path.is_file():
        parser.error("missing build/speck32_linear or build/speck32_linear_sweep, run: make speck32-linear NO_CUDA=1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = output_path_for(
        output_dir,
        args.rounds,
        key_mode,
        "linear_fixed_input_masks",
        args.top_k,
    )
    output_path = output_path_for(
        output_dir,
        args.rounds,
        key_mode,
        "linear_fixed_output_masks",
        args.top_k,
    )

    if use_shared_backend:
        print(
            f"using shared-process sweep backend: {sweep_binary_path}",
            file=sys.stderr,
        )
        run_backend_sweep(
            binary_path=sweep_binary_path,
            fwht_root=fwht_root,
            rounds=args.rounds,
            key_mode=key_mode,
            fixed_side="input",
            output_path=input_path,
            top_k=args.top_k,
            use_codebook=args.codebook,
            unsafe_memory=args.unsafe_memory,
            omp_threads=args.omp_threads,
            chunk_size=args.chunk_size,
        )
        run_backend_sweep(
            binary_path=sweep_binary_path,
            fwht_root=fwht_root,
            rounds=args.rounds,
            key_mode=key_mode,
            fixed_side="output",
            output_path=output_path,
            top_k=args.top_k,
            use_codebook=args.codebook,
            unsafe_memory=args.unsafe_memory,
            omp_threads=args.omp_threads,
            chunk_size=args.chunk_size,
        )
    else:
        write_sweep(
            binary_path=binary_path,
            fwht_root=fwht_root,
            rounds=args.rounds,
            key_mode=key_mode,
            fixed_side="input",
            output_path=input_path,
            top_k=args.top_k,
            use_codebook=args.codebook,
            unsafe_memory=args.unsafe_memory,
            omp_threads=args.omp_threads,
        )
        write_sweep(
            binary_path=binary_path,
            fwht_root=fwht_root,
            rounds=args.rounds,
            key_mode=key_mode,
            fixed_side="output",
            output_path=output_path,
            top_k=args.top_k,
            use_codebook=args.codebook,
            unsafe_memory=args.unsafe_memory,
            omp_threads=args.omp_threads,
        )

    print(f"wrote {input_path}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())