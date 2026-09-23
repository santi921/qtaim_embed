#!/usr/bin/env python3
"""
Generate PyG graph-level profiling configs for heavy benchmarking.

Tests graph-level regression (GCNGraphPred) on the QM9 10K LMDB dataset
across multiple model sizes, worker counts, batch sizes, and torch.compile.

Dataset: data/lmdb_graph_qm9_10000 (8010 train, ~1000 val, ~1000 test)
Graph structure: HeteroData with atom(18), bond(13), global(3) features
"""

import json
from pathlib import Path
from copy import deepcopy

PROFILING_DIR = Path(__file__).parent
DATA_ROOT = "data/lmdb_graph_qm9_10000"

# ---------------------------------------------------------------------------
# Base graph-level config (matches get_default_graph_level_config structure)
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    "dataset": {
        "verbose": False,
        "element_set": [],
        "allowed_ring_size": [3, 4, 5, 6, 7],
        "allowed_charges": None,
        "allowed_spins": None,
        "self_loop": True,
        "extra_keys": {
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["extra_feat_bond_esp_total", "bond_length"],
            "global": ["extra_feat_global_E1_CAM"],
        },
        "target_list": ["extra_feat_global_E1_CAM"],
        "target_dict": {"global": ["extra_feat_global_E1_CAM"]},
        "extra_dataset_info": {},
        "debug": False,
        "bond_key": "bonds",
        "map_key": "extra_feat_bond_indices_qtaim",
        "log_scale_features": False,
        "log_scale_targets": False,
        "standard_scale_features": True,
        "standard_scale_targets": True,
        "val_prop": 0.15,
        "test_prop": 0.1,
        "seed": 42,
        "train_lmdb": f"{DATA_ROOT}/train/",
        "val_lmdb": f"{DATA_ROOT}/val/",
        "test_lmdb": f"{DATA_ROOT}/test/",
        "train_dataset_loc": "tests/data/labelled_data.pkl",
        "num_workers": 0,
        "impute": False,
        "edge_dropout": 0.0,
        "log_save_dir": "./profiling/logs/",
    },
    "model": {
        "classifier": False,
        "compiled": False,
        "n_conv_layers": 8,
        "resid_n_graph_convs": 2,
        "target_dict": {"global": ["extra_feat_global_E1_CAM"]},
        "conv_fn": "ResidualBlock",
        "global_pooling_fn": "SumPoolingThenCat",
        "dropout": 0.2,
        "batch_norm": False,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "aggregate": "sum",
        "lr": 0.01,
        "scheduler_name": "reduce_on_plateau",
        "weight_decay": 1e-5,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.6,
        "loss_fn": "mse",
        "embedding_size": 20,
        "shape_fc": "cone",
        "fc_hidden_size_1": 256,
        "fc_num_layers": 3,
        "fc_dropout": 0.2,
        "fc_batch_norm": True,
        "lstm_iters": 3,
        "lstm_layers": 2,
        "num_heads_gat": 2,
        "dropout_feat_gat": 0.2,
        "dropout_attn_gat": 0.2,
        "hidden_size": 64,
        "residual_gat": True,
        "restore": False,
        "max_epochs": 3,
        "initializer": "kaiming",
        "extra_stop_patience": 100,
    },
    "optim": {
        "num_devices": 1,
        "num_nodes": 1,
        "num_workers": 0,
        "gradient_clip_val": 5.0,
        "strategy": "auto",
        "precision": "bf16",
        "accumulate_grad_batches": 1,
        "pin_memory": True,
        "persistent_workers": False,
        "train_batch_size": 64,
    },
}


# ---------------------------------------------------------------------------
# Model architecture configs (small to production-sized)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    # Config A: Baseline small (ResidualBlock, ~30K params)
    "baseline": {
        "conv_fn": "ResidualBlock",
        "hidden_size": 64,
        "embedding_size": 20,
        "n_conv_layers": 8,
        "global_pooling_fn": "SumPoolingThenCat",
        "fc_hidden_size_1": 256,
        "fc_num_layers": 3,
    },

    # Config B: Medium ResidualBlock (~150K params)
    "resid_medium": {
        "conv_fn": "ResidualBlock",
        "hidden_size": 128,
        "embedding_size": 128,
        "n_conv_layers": 8,
        "global_pooling_fn": "SumPoolingThenCat",
        "fc_hidden_size_1": 512,
        "fc_num_layers": 3,
    },

    # Config C: GAT small (~500K-1M params)
    # GAT output = hidden_size * num_heads, so keep hidden small
    "gat_small": {
        "conv_fn": "GATConv",
        "hidden_size": 64,
        "embedding_size": 64,
        "n_conv_layers": 8,
        "num_heads_gat": 4,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "residual_gat": True,
        "global_pooling_fn": "SumPoolingThenCat",
        "fc_hidden_size_1": 256,
        "fc_num_layers": 3,
    },

    # Config D: GAT medium (~2-5M params)
    "gat_medium": {
        "conv_fn": "GATConv",
        "hidden_size": 96,
        "embedding_size": 96,
        "n_conv_layers": 8,
        "num_heads_gat": 4,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "residual_gat": True,
        "global_pooling_fn": "SumPoolingThenCat",
        "fc_hidden_size_1": 384,
        "fc_num_layers": 3,
    },

    # Config E: GAT large (~10-20M params)
    "gat_large": {
        "conv_fn": "GATConv",
        "hidden_size": 128,
        "embedding_size": 128,
        "n_conv_layers": 10,
        "num_heads_gat": 4,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "residual_gat": True,
        "global_pooling_fn": "SumPoolingThenCat",
        "fc_hidden_size_1": 512,
        "fc_num_layers": 3,
    },
}

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
EXPERIMENTS = []

# Set 1: Model scaling (all models, workers=0, batch=64, no compile)
for model_name in ["baseline", "resid_medium", "gat_small", "gat_medium", "gat_large"]:
    EXPERIMENTS.append({
        "name": f"graph_{model_name}_w0",
        "model_config": model_name,
        "num_workers": 0,
        "batch_size": 64,
        "compiled": False,
    })

# Set 2: Worker scaling on medium and large models
for model_name in ["resid_medium", "gat_small", "gat_medium", "gat_large"]:
    for nw in [2, 4, 8]:
        EXPERIMENTS.append({
            "name": f"graph_{model_name}_w{nw}",
            "model_config": model_name,
            "num_workers": nw,
            "batch_size": 64,
            "compiled": False,
        })

# Set 3: Batch size scaling on large model
for bs in [32, 128, 256]:
    EXPERIMENTS.append({
        "name": f"graph_gat_large_b{bs}",
        "model_config": "gat_large",
        "num_workers": 4,
        "batch_size": bs,
        "compiled": False,
    })

# Set 4: torch.compile on medium and large models
for model_name in ["resid_medium", "gat_medium", "gat_large"]:
    EXPERIMENTS.append({
        "name": f"graph_{model_name}_compiled",
        "model_config": model_name,
        "num_workers": 4,
        "batch_size": 64,
        "compiled": True,
    })

# ---------------------------------------------------------------------------
# Generate config files
# ---------------------------------------------------------------------------
def main():
    output_dir = PROFILING_DIR / "graph_configs"
    output_dir.mkdir(exist_ok=True)

    for exp in EXPERIMENTS:
        config = deepcopy(BASE_CONFIG)
        model_overrides = MODEL_CONFIGS[exp["model_config"]]

        # Apply model overrides
        for k, v in model_overrides.items():
            config["model"][k] = v

        # Apply experiment settings
        nw = exp["num_workers"]
        config["optim"]["num_workers"] = nw
        config["optim"]["pin_memory"] = True
        config["optim"]["persistent_workers"] = nw > 0
        config["optim"]["train_batch_size"] = exp["batch_size"]
        config["model"]["compiled"] = exp["compiled"]
        config["model"]["max_epochs"] = 3

        # Save
        filename = f"{exp['name']}.json"
        out = output_dir / filename
        out.write_text(json.dumps(config, indent=2))

    print(f"Generated {len(EXPERIMENTS)} config files in {output_dir}/")
    print()
    print("Experiment sets:")
    print(f"  Set 1 (model scaling):   {sum(1 for e in EXPERIMENTS if '_w0' in e['name'])} configs")
    print(f"  Set 2 (worker scaling):  {sum(1 for e in EXPERIMENTS if any(f'_w{n}' in e['name'] for n in [2,4,8]))} configs")
    print(f"  Set 3 (batch scaling):   {sum(1 for e in EXPERIMENTS if '_b' in e['name'])} configs")
    print(f"  Set 4 (torch.compile):   {sum(1 for e in EXPERIMENTS if 'compiled' in e['name'])} configs")
    print(f"  Total:                   {len(EXPERIMENTS)} configs")
    print()

    # Print experiment names for runner script
    print("All experiments:")
    for exp in EXPERIMENTS:
        print(f"  {exp['name']}")


if __name__ == "__main__":
    main()
