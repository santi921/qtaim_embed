#!/usr/bin/env python3
"""
Analyze PyG graph-level profiling results.

Reads JSON result files from profiling/graph_results/ and produces
comparative tables for throughput, GPU utilization, memory, and
bottleneck analysis across model sizes, worker counts, batch sizes,
and torch.compile.

Usage:
    python profiling/analyze_pyg_graph_results.py
"""

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "graph_results"


def load_results():
    """Load all result JSON files."""
    results = {}
    if not RESULTS_DIR.exists():
        print(f"No results directory found: {RESULTS_DIR}")
        return results

    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            results[data["experiment"]] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: skipping {f.name}: {e}")

    return results


def print_model_scaling_table(results):
    """Compare throughput across model sizes (Set 1: workers=0)."""
    print()
    print("=" * 90)
    print("SET 1: MODEL SIZE SCALING (workers=0, batch=64)")
    print("=" * 90)

    header = f"{'Experiment':<30} {'Conv':<15} {'Params':>10} {'it/s':>8} {'samp/s':>10} {'GPU Mem':>8} {'CUDA%':>7}"
    print(header)
    print("-" * 90)

    baseline_its = None
    names = [
        "graph_baseline_w0",
        "graph_resid_medium_w0",
        "graph_gat_small_w0",
        "graph_gat_medium_w0",
        "graph_gat_large_w0",
    ]
    for name in names:
        if name not in results:
            print(f"  {name:<28} {'(not run)':>10}")
            continue
        r = results[name]
        its = r["throughput"]["avg_it_per_s"]
        sps = r["throughput"]["avg_samples_per_s"]
        params = r["model"]["total_params"]
        mem = r["throughput"]["gpu_peak_memory_gb"]
        cuda_pct = r.get("profiler", {}).get("cuda_pct", -1)
        conv = r["config"]["conv_fn"]

        if baseline_its is None:
            baseline_its = its
            vs = "--"
        else:
            vs = f"{(its / baseline_its - 1) * 100:+.0f}%"

        cuda_str = f"{cuda_pct:.0f}%" if cuda_pct >= 0 else "N/A"
        print(f"  {name:<28} {conv:<15} {params:>10,} {its:>8.2f} {sps:>10.1f} {mem:>7.2f}G {cuda_str:>7} {vs:>8}")


def print_worker_scaling_table(results):
    """Compare throughput across worker counts (Set 2)."""
    print()
    print("=" * 90)
    print("SET 2: WORKER SCALING")
    print("=" * 90)

    models = ["resid_medium", "gat_small", "gat_medium", "gat_large"]
    for model in models:
        print(f"\n  Model: {model}")
        header = f"    {'Workers':<10} {'it/s':>8} {'samp/s':>10} {'GPU Mem':>8} {'vs w=0':>10}"
        print(header)
        print("    " + "-" * 50)

        w0_its = None
        for nw in [0, 2, 4, 8]:
            name = f"graph_{model}_w{nw}"
            if name not in results:
                print(f"    {nw:<10} {'(not run)':>8}")
                continue
            r = results[name]
            its = r["throughput"]["avg_it_per_s"]
            sps = r["throughput"]["avg_samples_per_s"]
            mem = r["throughput"]["gpu_peak_memory_gb"]

            if nw == 0:
                w0_its = its
                vs = "--"
            else:
                vs = f"{(its / w0_its - 1) * 100:+.1f}%" if w0_its else "N/A"

            print(f"    {nw:<10} {its:>8.2f} {sps:>10.1f} {mem:>7.2f}G {vs:>10}")


def print_batch_scaling_table(results):
    """Compare throughput across batch sizes (Set 3)."""
    print()
    print("=" * 90)
    print("SET 3: BATCH SIZE SCALING (gat_large, workers=4)")
    print("=" * 90)

    header = f"  {'Batch':<10} {'it/s':>8} {'samp/s':>10} {'GPU Mem':>8} {'vs b=64':>10}"
    print(header)
    print("  " + "-" * 50)

    b64_sps = None
    for bs in [32, 64, 128, 256]:
        if bs == 64:
            # Batch scaling experiments use workers=4; the w0 baseline has
            # different worker count so should not be used as a batch-size
            # comparison point.
            name = "graph_gat_large_w4"
            if name not in results:
                name = "graph_gat_large_w0"
        else:
            name = f"graph_gat_large_b{bs}"

        if name not in results:
            print(f"  {bs:<10} {'(not run)':>8}")
            continue
        r = results[name]
        its = r["throughput"]["avg_it_per_s"]
        sps = r["throughput"]["avg_samples_per_s"]
        mem = r["throughput"]["gpu_peak_memory_gb"]

        if bs == 64 or b64_sps is None:
            b64_sps = sps
            vs = "--"
        else:
            vs = f"{(sps / b64_sps - 1) * 100:+.1f}%" if b64_sps else "N/A"

        print(f"  {bs:<10} {its:>8.2f} {sps:>10.1f} {mem:>7.2f}G {vs:>10}")


def print_compile_table(results):
    """Compare torch.compile benefit (Set 4)."""
    print()
    print("=" * 90)
    print("SET 4: torch.compile BENEFIT")
    print("=" * 90)

    models = ["resid_medium", "gat_small", "gat_medium", "gat_large"]
    header = f"  {'Model':<25} {'Normal it/s':>12} {'Compiled it/s':>14} {'Speedup':>10}"
    print(header)
    print("  " + "-" * 65)

    for model in models:
        # Normal uses w4 for fair comparison
        name_norm = f"graph_{model}_w4"
        # Fall back to w0 if w4 not available
        if name_norm not in results:
            name_norm = f"graph_{model}_w0"
        name_comp = f"graph_{model}_compiled"

        if name_norm not in results or name_comp not in results:
            print(f"  {model:<25} {'(incomplete)':>12}")
            continue

        its_norm = results[name_norm]["throughput"]["avg_it_per_s"]
        its_comp = results[name_comp]["throughput"]["avg_it_per_s"]
        speedup = (its_comp / its_norm - 1) * 100 if its_norm > 0 else 0

        print(f"  {model:<25} {its_norm:>12.2f} {its_comp:>14.2f} {speedup:>+9.1f}%")


def print_production_table(results):
    """Show production model results (Set 5)."""
    print()
    print("=" * 90)
    print("SET 5: PRODUCTION MODEL (gat_production)")
    print("=" * 90)

    names = [
        "graph_gat_production_w0",
        "graph_gat_production_w4",
        "graph_gat_production_compiled",
    ]
    header = f"  {'Experiment':<35} {'Params':>10} {'it/s':>8} {'samp/s':>10} {'GPU Mem':>8} {'CUDA%':>7}"
    print(header)
    print("  " + "-" * 80)

    for name in names:
        if name not in results:
            print(f"  {name:<35} {'(not run)':>10}")
            continue
        r = results[name]
        its = r["throughput"]["avg_it_per_s"]
        sps = r["throughput"]["avg_samples_per_s"]
        params = r["model"]["total_params"]
        mem = r["throughput"]["gpu_peak_memory_gb"]
        cuda_pct = r.get("profiler", {}).get("cuda_pct", -1)
        cuda_str = f"{cuda_pct:.0f}%" if cuda_pct >= 0 else "N/A"

        print(f"  {name:<35} {params:>10,} {its:>8.2f} {sps:>10.1f} {mem:>7.2f}G {cuda_str:>7}")


def print_bottleneck_analysis(results):
    """Analyze where time is spent for each model size."""
    print()
    print("=" * 90)
    print("BOTTLENECK ANALYSIS (CPU time breakdown by category)")
    print("=" * 90)

    names_with_profiler = [
        name for name, r in results.items()
        if "profiler" in r and "categories" in r.get("profiler", {})
    ]

    if not names_with_profiler:
        print("  No profiler data available. Run without --skip_profiler.")
        return

    # Sort by param count
    names_with_profiler.sort(
        key=lambda n: results[n]["model"]["total_params"]
    )

    categories = ["conv/linear", "optimizer", "activation", "norm", "loss", "data_transfer", "other"]

    header = f"  {'Experiment':<30} {'Params':>8}"
    for cat in categories:
        header += f" {cat[:8]:>8}"
    print(header)
    print("  " + "-" * (38 + 8 * len(categories)))

    for name in names_with_profiler:
        r = results[name]
        cats = r["profiler"]["categories"]
        total = sum(cats.values())
        params = r["model"]["total_params"]
        row = f"  {name:<30} {params:>8,}"
        for cat in categories:
            pct = cats.get(cat, 0) / total * 100 if total > 0 else 0
            row += f" {pct:>7.1f}%"
        print(row)


def print_avg_throughput_table(results):
    """Print average throughput across all experiments."""
    print()
    print("=" * 90)
    print("AVERAGE THROUGHPUT (all experiments)")
    print("=" * 90)

    header = f"  {'Experiment':<35} {'Params':>10} {'it/s':>8} {'samp/s':>10} {'GPU Mem':>8}"
    print(header)
    print("  " + "-" * 75)

    # Sort by param count
    sorted_names = sorted(results.keys(), key=lambda n: (
        results[n]["model"]["total_params"],
        results[n]["config"].get("num_workers", 0),
    ))

    for name in sorted_names:
        r = results[name]
        its = r["throughput"]["avg_it_per_s"]
        sps = r["throughput"]["avg_samples_per_s"]
        params = r["model"]["total_params"]
        mem = r["throughput"]["gpu_peak_memory_gb"]
        print(f"  {name:<35} {params:>10,} {its:>8.2f} {sps:>10.1f} {mem:>7.2f}G")

    # Grand averages grouped by model
    print()
    print("  Per-model averages:")
    print(f"  {'Model':<25} {'Avg it/s':>10} {'Avg samp/s':>12} {'Experiments':>12}")
    print("  " + "-" * 60)

    from collections import defaultdict
    model_groups = defaultdict(list)
    for name, r in results.items():
        # Extract model identifier from experiment name
        parts = name.replace("graph_", "").split("_")
        # Find model name (everything before w0/w2/w4/w8/b32/b128/b256/compiled)
        model_id = []
        for p in parts:
            if p.startswith("w") and p[1:].isdigit():
                break
            if p.startswith("b") and p[1:].isdigit():
                break
            if p == "compiled":
                break
            model_id.append(p)
        model_key = "_".join(model_id)
        model_groups[model_key].append(r)

    for model_key in sorted(model_groups.keys(), key=lambda k: model_groups[k][0]["model"]["total_params"]):
        runs = model_groups[model_key]
        avg_its = sum(r["throughput"]["avg_it_per_s"] for r in runs) / len(runs)
        avg_sps = sum(r["throughput"]["avg_samples_per_s"] for r in runs) / len(runs)
        print(f"  {model_key:<25} {avg_its:>10.2f} {avg_sps:>12.1f} {len(runs):>12}")


def print_overall_summary(results):
    """Print high-level findings."""
    print()
    print("=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    if not results:
        print("  No results to summarize.")
        return

    # Find best throughput
    best_name = max(results, key=lambda n: results[n]["throughput"]["avg_samples_per_s"])
    best = results[best_name]
    print(f"  Best throughput:    {best_name} ({best['throughput']['avg_samples_per_s']:.1f} samples/s)")

    # Largest model
    largest_name = max(results, key=lambda n: results[n]["model"]["total_params"])
    largest = results[largest_name]
    print(f"  Largest model:      {largest_name} ({largest['model']['total_params']:,} params)")
    print(f"    Throughput:       {largest['throughput']['avg_it_per_s']:.2f} it/s | {largest['throughput']['avg_samples_per_s']:.1f} samples/s")
    print(f"    GPU peak mem:    {largest['throughput']['gpu_peak_memory_gb']:.2f} GB")

    # GPU-boundedness progression
    profiled = {
        name: r for name, r in results.items()
        if "profiler" in r and r["profiler"].get("cuda_pct", -1) >= 0
    }
    if profiled:
        print()
        print("  GPU-boundedness (CUDA time %):")
        for name in sorted(profiled, key=lambda n: profiled[n]["model"]["total_params"]):
            r = profiled[name]
            print(f"    {name:<35} {r['model']['total_params']:>10,} params -> {r['profiler']['cuda_pct']:.1f}% CUDA")


def main():
    results = load_results()

    if not results:
        print("No results found. Run profiling first:")
        print("  bash profiling/run_pyg_graph_benchmark.sh")
        return

    print(f"Loaded {len(results)} result files from {RESULTS_DIR}")

    print_avg_throughput_table(results)
    print_model_scaling_table(results)
    print_worker_scaling_table(results)
    print_batch_scaling_table(results)
    print_compile_table(results)
    print_bottleneck_analysis(results)
    print_overall_summary(results)


if __name__ == "__main__":
    main()
