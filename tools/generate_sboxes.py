#!/usr/bin/env python3
"""Utility to generate random n-bit S-box tables for benchmarking.

By default the script produces `count` random bijective S-boxes of size `2^n_bits`
(using Python's `random.Random` seeded RNG for reproducibility) and saves them as
comma-separated text files under the requested output directory. A manifest file
(`manifest.json`) captures the generation parameters and the list of emitted
artifacts so benchmark harnesses can pick them up automatically.

Example:

    python3 tools/generate_sboxes.py \
        --n-bits 8 \
        --count 64 \
        --seed 1337 \
        --out build/gpu_sboxes

This creates files like `build/gpu_sboxes/sbox_0000.txt` containing a single
comma-separated line with 256 integers ranging from 0 to 255. You can feed any
of those files to the CLI on a GPU machine via:

    ./build/fwht_cli --backend gpu --sbox --input build/gpu_sboxes/sbox_0000.txt \
        --sbox-lat build/results/sbox_0000.lat --sbox-components build/results/sbox_0000.comp

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List


def generate_sbox(n_bits: int, rng: random.Random) -> List[int]:
    size = 1 << n_bits
    values = list(range(size))
    rng.shuffle(values)
    return values


def write_sbox(path: Path, values: List[int]) -> None:
    path.write_text(
        ",".join(str(v) for v in values) + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random S-box tables for benchmarking.")
    parser.add_argument("--n-bits", type=int, default=None,
                        help="Generate only this size (overrides min/max range).")
    parser.add_argument("--min-bits", type=int, default=8,
                        help="Minimum S-box width to generate (inclusive, default: 8).")
    parser.add_argument("--max-bits", type=int, default=17,
                        help="Maximum S-box width to generate (inclusive, default: 17).")
    parser.add_argument("--count", type=int, default=32,
                        help="Number of distinct S-box tables per size (default: 32).")
    parser.add_argument("--seed", type=int, default=0xC0FFEE,
                        help="Seed for the pseudo-random generator (default: 0xC0FFEE).")
    parser.add_argument("--out", type=Path, default=Path("build/gpu_sboxes"),
                        help="Directory to write the generated tables (default: build/gpu_sboxes).")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("count must be positive.")

    if args.n_bits is not None:
        min_bits = max_bits = args.n_bits
    else:
        min_bits = args.min_bits
        max_bits = args.max_bits

    if min_bits is None or max_bits is None:
        parser.error("Must specify either --n-bits or both --min-bits/--max-bits.")
    if min_bits <= 0 or max_bits <= 0:
        parser.error("Bit widths must be positive.")
    if min_bits > max_bits:
        parser.error("min-bits cannot exceed max-bits.")
    if max_bits > 17:
        parser.error("max-bits > 17 would create 262k+ entries; please cap at 17.")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest: Dict[str, object] = {
        "min_bits": min_bits,
        "max_bits": max_bits,
        "count_per_size": args.count,
        "seed": args.seed,
        "files": [],
    }

    for n_bits in range(min_bits, max_bits + 1):
        size = 1 << n_bits
        size_dir = out_dir / f"n{n_bits:02d}"
        size_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(args.count):
            values = generate_sbox(n_bits, rng)
            file_path = size_dir / f"sbox_{n_bits:02d}_{idx:04d}.txt"
            write_sbox(file_path, values)
            manifest["files"].append({
                "n_bits": n_bits,
                "size": size,
                "path": str(file_path),
            })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(f"[generate_sboxes] wrote {args.count} tables per size for n ∈ [{min_bits}, {max_bits}]")
    print(f"[generate_sboxes] output root: {out_dir}")
    print(f"[generate_sboxes] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
