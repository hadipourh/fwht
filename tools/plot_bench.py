#!/usr/bin/env python3
"""Plot or summarize libfwht vs SboxU benchmark results.

Expected input is the CSV emitted by tests/compare_sboxu_fwht.cpp.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from collections import defaultdict


def read_rows(csv_path: pathlib.Path):
    rows = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "sboxu_us_per_iter" not in row or "speedup" not in row:
                raise ValueError("CSV missing expected columns (rerun compare_sboxu_fwht)")
            rows.append(
                {
                    "n": int(row["n"]),
                    "threads": int(row["threads"]),
                    "iters": int(row["iters"]),
                    "sboxu_us": float(row["sboxu_us_per_iter"]),
                    "libfwht_us": float(row["libfwht_us_per_iter"]),
                    "speedup": float(row["speedup"]),
                }
            )
    return sorted(rows, key=lambda r: (r["threads"], r["n"]))


def print_table(rows):
    if not rows:
        print("No rows available.")
        return
    header = f"{'n':>10} {'threads':>8} {'SboxU us/iter':>15} {'libfwht us/iter':>17} {'speedup':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['n']:>10} {row['threads']:>8} {row['sboxu_us']:>15.2f} "
            f"{row['libfwht_us']:>17.3f} {row['speedup']:>10.2f}"
        )


def _group_by_threads(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["threads"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["n"])
    return dict(sorted(groups.items()))


def plot(rows, output: pathlib.Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - dependency hint
        print("matplotlib is not installed; install it to enable plotting.")
        return

    groups = _group_by_threads(rows)
    if not groups:
        print("No rows available for plotting.")
        return

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.15},
    )

    total_top_curves = len(groups) * 2 or 1
    base_colors = list(plt.get_cmap("Set1").colors)
    base_colors.extend(color for color in plt.get_cmap("tab20").colors if color not in base_colors)
    if len(base_colors) < total_top_curves:
        repeats = math.ceil(total_top_curves / len(base_colors))
        base_colors = (base_colors * repeats)[:total_top_curves]
    curve_colors = {}
    color_idx = 0
    for threads in groups:
        curve_colors[("sboxu", threads)] = base_colors[color_idx % len(base_colors)]
        color_idx += 1
        curve_colors[("libfwht", threads)] = base_colors[color_idx % len(base_colors)]
        color_idx += 1

    speedup_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    speedup_colors = {}
    for idx, threads in enumerate(groups):
        speedup_colors[threads] = speedup_cycle[idx % len(speedup_cycle)]

    xticks = sorted({row["n"] for row in rows})

    for threads, entries in groups.items():
        xs = [row["n"] for row in entries]
        sboxu = [row["sboxu_us"] for row in entries]
        libfwht = [row["libfwht_us"] for row in entries]
        speedup = [row["speedup"] for row in entries]

        label_suffix = f"T={threads}"
        ax_top.semilogy(
            xs,
            sboxu,
            marker="o",
            color=curve_colors[("sboxu", threads)],
            label=f"SboxU ({label_suffix})",
        )
        ax_top.semilogy(
            xs,
            libfwht,
            marker="s",
            linestyle="--",
            color=curve_colors[("libfwht", threads)],
            label=f"libfwht ({label_suffix})",
        )
        ax_bottom.plot(
            xs,
            speedup,
            marker="o",
            linestyle="-",
            color=speedup_colors[threads],
            label=f"Speedup ({label_suffix})",
        )

    ax_top.set_ylabel("Per-iteration time (microseconds)")
    ax_top.set_title("Per-iteration runtime (log y-scale) – lower is better")
    ax_top.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_top.legend(fontsize="small", ncol=2)

    ax_bottom.set_xlabel("Transform size (n)")
    ax_bottom.set_ylabel("Speedup (SboxU / libfwht)")
    ax_bottom.set_yscale("log")
    ax_bottom.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax_bottom.legend(fontsize="small", ncol=2)

    ax_bottom.set_xticks(xticks)
    tick_labels = []
    for n in xticks:
        log_val = math.log2(n)
        if abs(round(log_val) - log_val) < 1e-9:
            tick_labels.append(f"2^{int(round(log_val))}")
        else:
            tick_labels.append(str(n))
    ax_bottom.set_xticklabels(tick_labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    print(f"Wrote plot to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("build/compare_sboxu_fwht.csv"),
        help="Input CSV path (default: build/compare_sboxu_fwht.csv)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("build/compare_sboxu_fwht.pdf"),
        help="Path to write the plot (default: build/compare_sboxu_fwht.pdf).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation even if an output path is provided.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV file {args.csv} does not exist. Run the benchmark harness first.")
        return

    rows = read_rows(args.csv)
    print_table(rows)
    if not args.no_plot and args.output:
        plot(rows, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
