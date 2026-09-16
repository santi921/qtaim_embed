#!/usr/bin/env python3
"""
Analyze PyG benchmark results and compare to DGL baseline.

DGL baseline (from dgl_baseline/optimization_results.txt):
  workers=0: 1.87 it/s
  workers=2: 2.48 it/s (+32.6%)
  workers=4: 2.25 it/s (+20.3%)
  workers=8: 1.67 it/s (-10.7%)
  compiled:  2.02 it/s (+8.0%)
"""

import re
import json
from pathlib import Path

PROFILING_DIR = Path(__file__).parent

# DGL baseline numbers for direct comparison
DGL_BASELINE = {
    "pyg_link_workers0": {"dgl_its": 1.87, "dgl_label": "workers=0"},
    "pyg_link_workers2": {"dgl_its": 2.48, "dgl_label": "workers=2"},
    "pyg_link_workers4": {"dgl_its": 2.25, "dgl_label": "workers=4"},
    "pyg_link_compiled":  {"dgl_its": 2.02, "dgl_label": "compiled"},
}


def parse_log(log_path: Path) -> dict:
    if not log_path.exists():
        return {"error": f"not found: {log_path}"}

    text = log_path.read_text()

    # Extract it/s from training epoch lines only (excludes fast validation steps).
    # Matches: "Epoch N: 100%|...| M/M [mm:ss<mm:ss, X.XXit/s, ...]"
    epoch_its = re.findall(
        r'Epoch \d+: 100%\|[^|]+\| \d+/\d+ \[.*?,\s*([\d.]+)it/s', text
    )
    # Fall back to s/it format
    epoch_sit = re.findall(
        r'Epoch \d+: 100%\|[^|]+\| \d+/\d+ \[.*?,\s*([\d.]+)s/it', text
    )

    its_values = [float(v) for v in epoch_its]
    its_values += [1.0 / float(v) for v in epoch_sit if float(v) > 0]

    # Use median of steady-state epochs (skip epoch 0 warm-up if multiple epochs)
    steady = its_values[1:] if len(its_values) > 1 else its_values
    avg_its = sum(steady) / len(steady) if steady else None

    # Extract epoch count
    epochs = re.findall(r'Epoch (\d+):', text)
    max_epoch = max(int(e) for e in epochs) if epochs else 0

    return {
        "its_values": its_values,
        "avg_its": avg_its,
        "max_epoch": max_epoch,
    }


def main():
    experiments = [
        # Set 1: workers scaling
        "pyg_link_workers0",
        "pyg_link_workers2",
        "pyg_link_workers4",
        "pyg_link_compiled",
        # Set 2: batch size
        "pyg_link_batch32",
        "pyg_link_batch128",
        # Set 3: heavy model
        "pyg_heavy_workers0",
        "pyg_heavy_workers4",
    ]

    results = {}
    for name in experiments:
        log = PROFILING_DIR / f"{name}.log"
        results[name] = parse_log(log)

    # ------------------------------------------------------------------
    print("=" * 70)
    print("PyG BENCHMARK RESULTS")
    print("=" * 70)
    print()

    # Set 1: workers scaling with DGL comparison
    print("Set 1: Link prediction - workers scaling")
    print(f"{'Experiment':<28} {'PyG it/s':>10} {'DGL it/s':>10} {'vs DGL':>10} {'vs PyG w0':>12}")
    print("-" * 72)

    pyg_w0 = results.get("pyg_link_workers0", {}).get("avg_its")
    for name in ["pyg_link_workers0", "pyg_link_workers2", "pyg_link_workers4", "pyg_link_compiled"]:
        r = results[name]
        if "error" in r or r["avg_its"] is None:
            print(f"  {name:<26} {'N/A':>10}")
            continue
        pyg_its = r["avg_its"]
        dgl_info = DGL_BASELINE.get(name, {})
        dgl_its = dgl_info.get("dgl_its")
        vs_dgl = f"{(pyg_its / dgl_its - 1) * 100:+.1f}%" if dgl_its else "N/A"
        vs_w0 = f"{(pyg_its / pyg_w0 - 1) * 100:+.1f}%" if pyg_w0 and name != "pyg_link_workers0" else "--"
        print(f"  {name:<26} {pyg_its:>10.2f} {dgl_its or 0:>10.2f} {vs_dgl:>10} {vs_w0:>12}")
    print()

    # Set 2: batch size
    print("Set 2: Link prediction - batch size scaling")
    print(f"{'Experiment':<28} {'PyG it/s':>10}")
    print("-" * 40)
    for name in ["pyg_link_batch32", "pyg_link_batch128"]:
        r = results[name]
        if "error" in r or r["avg_its"] is None:
            print(f"  {name:<26} {'N/A':>10}")
        else:
            print(f"  {name:<26} {r['avg_its']:>10.2f}")
    print()

    # Set 3: heavy model
    print("Set 3: Heavy model (GAT+MLP)")
    print(f"{'Experiment':<28} {'PyG it/s':>10}")
    print("-" * 40)
    for name in ["pyg_heavy_workers0", "pyg_heavy_workers4"]:
        r = results[name]
        if "error" in r or r["avg_its"] is None:
            print(f"  {name:<26} {'N/A':>10}")
        else:
            print(f"  {name:<26} {r['avg_its']:>10.2f}")
    print()

    # ------------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY vs DGL BASELINE")
    print("=" * 70)
    print()
    print("DGL baseline (small dataset, batch=128, ResidualBlock):")
    print("  workers=0: 1.87 it/s")
    print("  workers=2: 2.48 it/s (best DGL)")
    print("  workers=4: 2.25 it/s")
    print()

    w0_pyg = results.get("pyg_link_workers0", {}).get("avg_its")
    w2_pyg = results.get("pyg_link_workers2", {}).get("avg_its")
    if w0_pyg and w2_pyg:
        print(f"PyG workers=0: {w0_pyg:.2f} it/s  ({(w0_pyg/1.87-1)*100:+.1f}% vs DGL w0)")
        print(f"PyG workers=2: {w2_pyg:.2f} it/s  ({(w2_pyg/2.48-1)*100:+.1f}% vs DGL w2 best)")
        if w0_pyg >= 1.87 * 0.9:
            print("\nResult: PyG meets >=90% of DGL baseline throughput. PASS.")
        else:
            print("\nResult: PyG is >10% slower than DGL. INVESTIGATE.")
    print()

    # Save raw results
    out_json = PROFILING_DIR / "pyg_benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Raw results saved: {out_json}")


if __name__ == "__main__":
    main()
