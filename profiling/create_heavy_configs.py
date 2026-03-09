#!/usr/bin/env python3
"""
Generate profiling configs for heavy models (GAT, Attention, MLP predictors).
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

# Define heavy model configurations
configs = {
    # Config A: GAT + Dot (moderate complexity, ~200-300K params)
    "heavy_gat_dot": {
        "conv_fn": "GATConv",
        "hidden_size": 128,
        "embedding_size": 128,
        "n_conv_layers": 8,
        "num_heads_gat": 4,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "residual_gat": True,
        "predictor": "Dot",
        "predictor_param_dict": {},
    },
    # Config B: GAT + Attention (high complexity, ~300-400K params)
    "heavy_gat_attention": {
        "conv_fn": "GATConv",
        "hidden_size": 128,
        "embedding_size": 128,
        "n_conv_layers": 8,
        "num_heads_gat": 4,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "residual_gat": True,
        "predictor": "Attention",
        "predictor_param_dict": {},
    },
    # Config C: GAT + MLP (very high complexity, ~3-5M params)
    "heavy_gat_mlp": {
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
    },
}

profiling_dir = Path(__file__).parent
created_count = 0

# For each base config, create workers and compile variants
for base_name, model_config in configs.items():
    for workers in [0, 4]:  # Test baseline and optimal
        for compiled in [False, True]:
            # Skip redundant configs: only test compiled with workers=4
            if compiled and workers != 4:
                continue

            config = json.loads(json.dumps(baseline))  # Deep copy

            # Use large dataset (20x bigger) for realistic profiling
            config["dataset"]["train_lmdb"] = "tests/data/lmdb_link_large/train/"
            config["dataset"]["val_lmdb"] = "tests/data/lmdb_link_large/val/"

            # Update model settings
            for key, val in model_config.items():
                config["model"][key] = val

            # Update optim settings
            config["optim"]["num_workers"] = workers
            config["optim"]["pin_memory"] = True
            config["optim"]["persistent_workers"] = workers > 0
            config["model"]["compiled"] = compiled
            config["model"]["max_epochs"] = 2

            # Generate filename
            suffix = f"_workers{workers}"
            if compiled:
                suffix += "_compiled"
            filename = f"{base_name}{suffix}.json"

            # Write config file
            output_path = profiling_dir / filename
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Created: {output_path}")
            created_count += 1

print(f"\nCreated {created_count} heavy model experiment configs!")
print("\nQuick start configs (Config A - moderate):")
print("  - heavy_gat_dot_workers0.json (baseline)")
print("  - heavy_gat_dot_workers4.json (optimized)")
print("  - heavy_gat_dot_workers4_compiled.json (best)")
print("\nTo run all experiments: bash profiling/run_heavy_experiments.sh")
