#!/usr/bin/env python3
"""Render a markdown table from qtaim-embed-bench JSON results.

Usage:
  python profiling/summarize_bench.py profiling/bench_results
  python profiling/summarize_bench.py profiling/bench_results --filter encoder_fn=schnet \
      --cols name batch_size samples_per_s peak_mem_gb
"""

import argparse
import json
from pathlib import Path

DEFAULT_COLS = [
    "name", "task", "mode", "encoder_fn", "batch_size", "precision", "hidden_size",
    "it_per_s", "samples_per_s", "atoms_per_s", "data_wait_frac", "gpu_util_mean",
    "peak_mem_gb", "kernels_per_step", "sync_per_step", "gather_scatter_ms",
    "matmul_ms", "encoder_ms", "neighbor_radius_ms", "collate_ms_per_batch",
]


def flatten(result: dict) -> dict:
    row = {"name": result["name"], "task": result["task"], "mode": result["mode"],
           "git_sha": result.get("git_sha")}
    row.update(result.get("summary", {}))
    row.update(result.get("metrics", {}))
    return row


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--cols", nargs="+", default=DEFAULT_COLS)
    p.add_argument("--filter", action="append", default=[], help="key=value, repeatable")
    p.add_argument("--sort", default="name")
    args = p.parse_args(argv)

    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "name" in data and "metrics" in data:
            rows.append(flatten(data))

    for flt in args.filter:
        k, _, v = flt.partition("=")
        rows = [r for r in rows if str(r.get(k)) == v]

    rows.sort(key=lambda r: (r.get(args.sort) is None, r.get(args.sort)))
    cols = [c for c in args.cols if any(r.get(c) is not None for r in rows)]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
