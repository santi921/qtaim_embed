#!/usr/bin/env python3
"""
Analyze profiling experiment results and extract key metrics.
Parses log files to extract timing information and generate comparison report.
"""

import re
import json
from pathlib import Path
from typing import Dict, Optional


def parse_log_file(log_path: Path) -> Dict[str, any]:
    """Extract timing and performance metrics from a log file."""

    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}"}

    content = log_path.read_text()

    metrics = {
        "config": log_path.stem,
        "total_time": None,
        "epochs": 0,
        "steps_per_epoch": None,
        "time_per_step": None,
        "samples_per_sec": None,
    }

    # Extract total training time
    # Look for patterns like "Epoch 1: 100%|██████████| 2/2 [00:03<00:00,  1.50s/it]"
    time_pattern = r'\[(\d+:\d+)<.*?,\s+([\d.]+)([ms]?)s/it\]'
    matches = re.findall(time_pattern, content)
    if matches:
        # Get last match (final timing)
        elapsed, per_step, unit = matches[-1]
        metrics["time_per_step"] = float(per_step) * (1000 if unit == 's' else 1)  # Convert to ms

    # Extract epoch information
    epoch_pattern = r'Epoch (\d+):'
    epochs = re.findall(epoch_pattern, content)
    if epochs:
        metrics["epochs"] = max(int(e) for e in epochs)

    # Extract steps per epoch
    step_pattern = r'(\d+)/(\d+) \['
    steps = re.findall(step_pattern, content)
    if steps:
        metrics["steps_per_epoch"] = int(steps[0][1])  # Total steps

    # Calculate throughput if we have the data
    if metrics["time_per_step"] and metrics["steps_per_epoch"]:
        # Assuming batch_size=128 from config
        batch_size = 128
        metrics["samples_per_sec"] = (batch_size * 1000) / metrics["time_per_step"]

    return metrics


def main():
    """Analyze all experiment logs and generate comparison report."""

    profiling_dir = Path(__file__).parent

    experiments = [
        "exp_workers_0",
        "exp_workers_2",
        "exp_workers_4",
        "exp_workers_8",
        "exp_compiled",
    ]

    results = []

    print("=" * 80)
    print("PROFILING EXPERIMENT RESULTS")
    print("=" * 80)
    print()

    for exp_name in experiments:
        log_file = profiling_dir / f"{exp_name}.log"
        metrics = parse_log_file(log_file)
        results.append(metrics)

        if "error" in metrics:
            print(f"{exp_name}: {metrics['error']}")
            continue

        print(f"Experiment: {exp_name}")
        print(f"  Time per step: {metrics['time_per_step']:.1f} ms")
        if metrics["samples_per_sec"]:
            print(f"  Throughput: {metrics['samples_per_sec']:.1f} samples/sec")
        print()

    # Find baseline (workers_0) for comparison
    baseline = next((r for r in results if r["config"] == "exp_workers_0"), None)

    if baseline and baseline.get("samples_per_sec"):
        print("=" * 80)
        print("COMPARISON TO BASELINE (num_workers=0)")
        print("=" * 80)
        print()

        for result in results:
            if "error" in result or result["config"] == "exp_workers_0":
                continue

            if result.get("samples_per_sec"):
                speedup = (result["samples_per_sec"] / baseline["samples_per_sec"] - 1) * 100
                print(f"{result['config']}:")
                print(f"  Throughput: {result['samples_per_sec']:.1f} samples/sec")
                print(f"  Speedup: {speedup:+.1f}%")
                print()

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("Based on the results above:")
    print("1. Best num_workers value: [to be determined from data]")
    print("2. torch.compile impact: [to be determined from data]")
    print("3. Recommended default config: [to be determined from data]")
    print()

    # Save raw results to JSON
    output_file = profiling_dir / "experiment_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Raw results saved to: {output_file}")


if __name__ == "__main__":
    main()
