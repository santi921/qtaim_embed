#!/usr/bin/env python3
"""
Generate Config C profiling experiments: GAT + MLP predictor (3-5M params)
This should be GPU-bound with 50-80% GPU utilization.
"""

import json
from pathlib import Path

# Load the baseline config
baseline_path = Path(__file__).parent / "baseline_config_full.json"
with open(baseline_path) as f:
    baseline = json.load(f)

# Update to use LMDB paths and batch_size=128
baseline["optim"]["train_batch_size"] = 128

# Remove test_lmdb to skip testing phase
if "test_lmdb" in baseline["dataset"]:
    del baseline["dataset"]["test_lmdb"]

# Config C: GAT + MLP (very high complexity, ~3-5M params)
model_config = {
    "conv_fn": "GATConv",
    "hidden_size": 256,
    "embedding_size": 256,
    "n_conv_layers": 8,
    "num_heads_gat": 4,
    "dropout_feat_gat": 0.1,
    "dropout_attn_gat": 0.1,
    "residual_gat": True,
    "predictor": "MLP",
    "predictor_param_dict": {
        "fc_layer_size": [1024, 1024],
        "fc_dropout": 0.2,
        "batch_norm": True,
        "activation": "ReLU",
    },
}

profiling_dir = Path(__file__).parent

# Create 3 experiments: workers=0 (baseline), workers=4 (optimal), workers=4+compiled (best)
experiments = [
    {"workers": 0, "compiled": False, "name": "heavy_gat_mlp_workers0"},
    {"workers": 4, "compiled": False, "name": "heavy_gat_mlp_workers4"},
    {"workers": 4, "compiled": True, "name": "heavy_gat_mlp_workers4_compiled"},
]

for exp in experiments:
    config = json.loads(json.dumps(baseline))  # Deep copy

    # Use large dataset (20x bigger) for realistic profiling
    config["dataset"]["train_lmdb"] = "tests/data/lmdb_link_large/train/"
    config["dataset"]["val_lmdb"] = "tests/data/lmdb_link_large/val/"

    # Update model settings
    for key, val in model_config.items():
        config["model"][key] = val

    # Update optim settings
    config["optim"]["num_workers"] = exp["workers"]
    config["optim"]["pin_memory"] = True
    config["optim"]["persistent_workers"] = exp["workers"] > 0
    config["model"]["compiled"] = exp["compiled"]
    config["model"]["max_epochs"] = 2

    # Write config file
    output_path = profiling_dir / f"{exp['name']}.json"
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created: {output_path}")

print("\n" + "="*60)
print("Config C (GAT + MLP) experiments created!")
print("="*60)
print("\nExpected characteristics:")
print("  - Parameters: ~3-5M (100-170x larger than baseline)")
print("  - GPU utilization: 50-80% (GPU-bound!)")
print("  - Forward pass: 50-60% of time (vs 3-18% in baseline)")
print("  - Optimizer: 10-15% of time (vs 40% in baseline)")
print("\nExperiments:")
print("  1. heavy_gat_mlp_workers0.json - Baseline (no parallelism)")
print("  2. heavy_gat_mlp_workers4.json - Optimized (parallel data loading)")
print("  3. heavy_gat_mlp_workers4_compiled.json - Best (+ torch.compile)")
print("\nTo run: bash profiling/run_config_c.sh")
print("Estimated time: ~45-60 minutes for all 3 experiments")
