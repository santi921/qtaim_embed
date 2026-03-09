#!/usr/bin/env python3
"""
Create complete experiment configs based on the baseline.
"""

import json
from pathlib import Path

# Load the baseline config
baseline_path = Path(__file__).parent / "baseline_config_full.json"
with open(baseline_path) as f:
    baseline = json.load(f)

# Update to use LMDB paths and batch_size=128
baseline["optim"]["train_batch_size"] = 128

# Remove test_lmdb to skip testing phase (which crashes)
if "test_lmdb" in baseline["dataset"]:
    del baseline["dataset"]["test_lmdb"]

experiments = [
    {
        "name": "exp_workers_0",
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,  # Can't be true with num_workers=0
        "compiled": False,
    },
    {
        "name": "exp_workers_2",
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "compiled": False,
    },
    {
        "name": "exp_workers_4",
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "compiled": False,
    },
    {
        "name": "exp_workers_8",
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "compiled": False,
    },
    {
        "name": "exp_compiled",
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "compiled": True,
    },
]

profiling_dir = Path(__file__).parent

for exp in experiments:
    config = json.loads(json.dumps(baseline))  # Deep copy

    # Use large dataset (20x bigger) for realistic profiling
    config["dataset"]["train_lmdb"] = "tests/data/lmdb_link_large/train/"
    config["dataset"]["val_lmdb"] = "tests/data/lmdb_link_large/val/"

    # Update optim settings
    config["optim"]["num_workers"] = exp["num_workers"]
    config["optim"]["pin_memory"] = exp["pin_memory"]
    config["optim"]["persistent_workers"] = exp["persistent_workers"]

    # Update model settings
    config["model"]["compiled"] = exp["compiled"]
    config["model"]["max_epochs"] = 2

    # Write config file
    output_path = profiling_dir / f"{exp['name']}.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created: {output_path}")

print("\nAll experiment configs created!")
