#!/usr/bin/env python3
"""
Generate PyG benchmark experiment configs.

Mirrors the DGL baseline experiments in dgl_baseline/ so results are
directly comparable. Tests:
  - Link prediction: workers scaling (0, 2, 4) + batch sizes (32, 128)
  - Link prediction: heavy model (GAT+MLP, Config C equivalent)
  - Graph regression: workers scaling (0, 2, 4)

DGL baseline reference (dgl_baseline/optimization_results.txt):
  workers=0: 1.87 it/s
  workers=2: 2.48 it/s (+32.6%)
  workers=4: 2.25 it/s (+20.3%)
  compiled=True: 2.02 it/s (+8.0%)
"""

import json
from copy import deepcopy
from pathlib import Path

PROFILING_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Base link prediction config (mirrors dgl_baseline/baseline_config_full.json)
# ---------------------------------------------------------------------------
BASE_LINK_CONFIG = {
    "dataset": {
        "verbose": True,
        "element_set": None,
        "allowed_ring_size": [3, 4, 5, 6, 7],
        "allowed_charges": None,
        "allowed_spins": None,
        "self_loop": True,
        "extra_keys": {
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["bond_length"],
        },
        "target_dict": {
            "atom": ["extra_feat_atom_esp_total"],
            "bond": [],
        },
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
        "train_dataset_loc": "tests/data/labelled_data.pkl",
        "train_lmdb": "tests/data/lmdb_link/train/",
        "val_lmdb": "tests/data/lmdb_link/val/",
        "pin_memory": False,
        "persistent_workers": False,
    },
    "model": {
        "classifier": False,
        "compiled": False,
        "n_conv_layers": 8,
        "resid_n_graph_convs": 2,
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
        "max_epochs": 2,
        "predictor": "Dot",
        "predictor_param_dict": {},
        "aggregator_type": "none",
        "initializer": "kaiming",
        "extra_stop_patience": 25,
    },
    "optim": {
        "num_devices": 1,
        "num_nodes": 1,
        "num_workers": 0,
        "gradient_clip_val": 5.0,
        "strategy": "auto",
        "precision": "bf16",
        "accumulate_grad_batches": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "train_batch_size": 128,
    },
}

# ---------------------------------------------------------------------------
# Experiment sets
# ---------------------------------------------------------------------------

# Set 1: Link prediction - workers scaling (exact DGL mirror)
link_workers_experiments = [
    {"name": "pyg_link_workers0", "num_workers": 0, "compiled": False},
    {"name": "pyg_link_workers2", "num_workers": 2, "compiled": False},
    {"name": "pyg_link_workers4", "num_workers": 4, "compiled": False},
    {"name": "pyg_link_compiled",  "num_workers": 4, "compiled": True},
]

for exp in link_workers_experiments:
    config = deepcopy(BASE_LINK_CONFIG)
    nw = exp["num_workers"]
    config["optim"]["num_workers"] = nw
    config["optim"]["pin_memory"] = True
    config["optim"]["persistent_workers"] = nw > 0
    config["model"]["compiled"] = exp["compiled"]
    config["model"]["max_epochs"] = 2
    out = PROFILING_DIR / f"{exp['name']}.json"
    out.write_text(json.dumps(config, indent=2))
    print(f"Created: {out}")

# Set 2: Link prediction - batch size scaling
batch_experiments = [
    {"name": "pyg_link_batch32",  "batch_size": 32,  "num_workers": 4},
    {"name": "pyg_link_batch128", "batch_size": 128, "num_workers": 4},
]

for exp in batch_experiments:
    config = deepcopy(BASE_LINK_CONFIG)
    config["optim"]["train_batch_size"] = exp["batch_size"]
    config["optim"]["num_workers"] = exp["num_workers"]
    config["optim"]["pin_memory"] = True
    config["optim"]["persistent_workers"] = True
    config["model"]["max_epochs"] = 2
    out = PROFILING_DIR / f"{exp['name']}.json"
    out.write_text(json.dumps(config, indent=2))
    print(f"Created: {out}")

# Set 3: Heavy model - GAT + MLP (Config C equivalent)
heavy_model_cfg = {
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

heavy_experiments = [
    {"name": "pyg_heavy_workers0", "num_workers": 0, "compiled": False},
    {"name": "pyg_heavy_workers4", "num_workers": 4, "compiled": False},
]

for exp in heavy_experiments:
    config = deepcopy(BASE_LINK_CONFIG)
    for k, v in heavy_model_cfg.items():
        config["model"][k] = v
    nw = exp["num_workers"]
    config["optim"]["num_workers"] = nw
    config["optim"]["pin_memory"] = True
    config["optim"]["persistent_workers"] = nw > 0
    config["optim"]["train_batch_size"] = 128
    config["model"]["compiled"] = exp["compiled"]
    config["model"]["max_epochs"] = 2
    out = PROFILING_DIR / f"{exp['name']}.json"
    out.write_text(json.dumps(config, indent=2))
    print(f"Created: {out}")

print("\n" + "=" * 60)
print("PyG benchmark configs created!")
print("=" * 60)
print("\nExperiment sets:")
print("  Set 1 (link workers): pyg_link_workers{0,2,4}, pyg_link_compiled")
print("  Set 2 (batch size):   pyg_link_batch{32,128}")
print("  Set 3 (heavy model):  pyg_heavy_workers{0,4}")
print("\nRun: bash profiling/run_pyg_benchmark.sh")
