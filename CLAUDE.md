# CLAUDE.md - QTAIM-Embed Development Guide

## Project Overview

QTAIM-Embed is a Graph Neural Network (GNN) package for molecular property prediction using heterogeneous graphs. The project implements machine learning models that work with Quantum Theory of Atoms in Molecules (QTAIM) features to predict molecular and reaction properties. It handles complex molecular representations including spin states, charged species, and sophisticated atom/bond features.

**Citation**: Digital Discovery (2024) by Vargas, Gee, and Alexandrova.

## Quick Reference

```bash
# Setup environment
conda env create -f env.yml
conda activate qtaim_embed
pip install -e .

# Run tests
pytest tests/

# Train a model (graph-level regression)
qtaim-embed-train-graph -dataset_loc path/to/data.pkl -project_name my_project

# Bayesian optimization
qtaim-embed-bayes-opt-graph -dataset_loc path/to/data.pkl
```

## Repository Structure

```
qtaim_embed/
├── qtaim_embed/           # Main source package
│   ├── core/              # Core dataset and data module classes
│   │   ├── dataset.py     # HeteroGraphNodeLabelDataset - main dataset class
│   │   ├── datamodule.py  # PyTorch Lightning data modules
│   │   └── molwrapper.py  # MoleculeWrapper class
│   ├── data/              # Data processing and featurization
│   │   ├── featurizer.py  # Molecular featurizers (atom, bond, global)
│   │   ├── processing.py  # Scalers (HeteroGraphStandardScaler, etc.)
│   │   ├── dataloader.py  # Custom DataLoaders for different tasks
│   │   ├── lmdb.py        # LMDB database management
│   │   ├── grapher.py     # Graph construction from molecules
│   │   ├── transforms.py  # Graph transformations (edge dropout, etc.)
│   │   └── xai.py         # Explainability tools
│   ├── models/            # Neural network architectures
│   │   ├── layers.py      # Custom GNN layers (UnifySize, ResidualBlock, pooling)
│   │   ├── layers_homo.py # Homogeneous graph layers
│   │   ├── graph_level/   # Graph-level models (regression, classification)
│   │   ├── node_level/    # Node-level prediction models
│   │   ├── link_pred/     # Link prediction models
│   │   ├── utils.py       # Model utilities and checkpoint loading
│   │   └── initializers.py# Weight initialization strategies
│   ├── scripts/           # Training and utility scripts
│   │   ├── train/         # Training scripts and configs
│   │   ├── helpers/       # Data conversion utilities (mol2lmdb)
│   │   ├── vis/           # Visualization tools
│   │   └── translate/     # Format translation utilities
│   └── utils/             # Common utilities
│       ├── data.py        # Config defaults, dataset splitting
│       ├── models.py      # Model loading, hyperparameter handling
│       ├── descriptors.py # Molecular descriptors and encodings
│       └── translation.py # Format conversions
├── tests/                 # Test suite (pytest)
├── data/                  # Sample datasets and plots
├── experiments/           # Experimental notebooks
├── pyproject.toml         # Project configuration
├── env.yml                # Conda environment specification
└── README.md              # User documentation
```

## Key Architecture Concepts

### Heterogeneous Graphs

Molecules are represented as heterogeneous graphs with three node types:
- **atom**: Atomic features (element, hybridization, charge, etc.)
- **bond**: Bond features (bond type, QTAIM properties)
- **global**: Global molecular features

### Task Types

1. **Graph-level regression**: Predict molecular properties (e.g., energy)
2. **Graph-level classification**: Classify molecules
3. **Node-level prediction**: Predict per-atom/bond properties
4. **Link prediction**: Predict edges/bonds

### Model Components

- **Message-passing functions**: `GraphConvDropoutBatch`, `ResidualBlock`, `GATConv`
- **Global pooling**: `SumPoolingThenCat`, `MeanPoolingThenCat`, `WeightAndSumThenCat`, `GlobalAttentionPoolingThenCat`, `Set2SetThenCat`
- **Scalers**: `HeteroGraphStandardScaler`, `HeteroGraphLogMagnitudeScaler`

## Configuration System

The project uses hierarchical configuration dictionaries:

```python
config = {
    "dataset": {
        "train_dataset_loc": "path/to/data.pkl",
        "allowed_ring_size": [3, 4, 5, 6, 7],
        "allowed_charges": None,  # None = all allowed
        "allowed_spins": None,
        "standard_scale_features": True,
        "log_scale_targets": False,
        "val_prop": 0.15,
        "test_prop": 0.1,
        "extra_keys": {"atom": [], "bond": [], "global": []},
    },
    "model": {
        "n_conv_layers": 8,
        "conv_fn": "ResidualBlock",  # or "GraphConvDropoutBatch", "GATConv"
        "global_pooling_fn": "SumPoolingThenCat",
        "hidden_size": 128,
        "embedding_size": 128,
        "dropout": 0.2,
        "batch_norm": True,
        "activation": "ReLU",
        "lr": 1e-3,
        "loss_fn": "mse",  # or "mae"
    },
    "optim": {
        "precision": 16,  # or "bf16", 32
        "max_epochs": 100,
        "gradient_clip_val": 1.0,
    }
}
```

Default configs are available via:
```python
from qtaim_embed.utils.data import get_default_graph_level_config
config = get_default_graph_level_config()
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest tests/ --cov=qtaim_embed
```

### Key Test Files
- `test_models.py`: Model training and checkpointing
- `test_scalers.py`: Feature scaling (extensive)
- `test_featurizers.py`: Molecular featurization (~260 cases)
- `test_layers.py`: Custom GNN layers
- `test_core.py`: Dataset functionality

### Adding New Features

1. **New model layer**: Add to `qtaim_embed/models/layers.py`
2. **New pooling function**: Add to `layers.py`, register in `models/utils.py`
3. **New featurizer**: Add to `qtaim_embed/data/featurizer.py`
4. **New training script**: Add to `qtaim_embed/scripts/train/`

### Code Style

- Use type hints for function signatures
- Follow existing patterns for PyTorch Lightning modules
- Docstrings for public classes and methods
- Configuration via dictionaries, not command-line-heavy interfaces

## Common Tasks

### Loading and Training a Model

```python
import pytorch_lightning as pl
from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule
from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.utils.data import get_default_graph_level_config

# Setup config
config = get_default_graph_level_config()
config["dataset"]["train_dataset_loc"] = "path/to/data.pkl"
config["model"]["target_dict"]["global"] = ["target_property"]

# Create data module and model
dm = QTAIMGraphTaskDataModule(config=config)
model = load_graph_level_model_from_config(config["model"])

# Train
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(model, dm)
```

### Converting Data to LMDB

```bash
qtaim-embed-mol2lmdb -input_file data.pkl -output_dir ./lmdb_data/
```

### Hyperparameter Optimization

```bash
qtaim-embed-bayes-opt-graph \
    -dataset_loc data.pkl \
    -project_name hp_search \
    -sweep_config sweep_config.json
```

## Dependencies

**Core:**
- Python 3.11
- PyTorch 2.4.1 (CUDA 12.4)
- DGL (Deep Graph Library)
- PyTorch Lightning

**Chemistry:**
- RDKit
- PyMatgen
- ASE

**ML/Data:**
- scikit-learn
- torchmetrics
- LMDB
- e3nn

## CLI Entry Points

| Command | Description |
|---------|-------------|
| `qtaim-embed-train-graph` | Train graph-level regression |
| `qtaim-embed-train-graph-classifier` | Train graph-level classification |
| `qtaim-embed-train-node` | Train node-level prediction |
| `qtaim-embed-bayes-opt-graph` | Bayesian optimization for graph models |
| `qtaim-embed-bayes-opt-node` | Bayesian optimization for node models |
| `qtaim-embed-bayes-opt-graph-classifier` | Bayesian optimization for classifiers |
| `qtaim-embed-mol2lmdb` | Convert molecule data to LMDB |
| `qtaim-embed-mol2lmdb-node` | Convert node-labeled data to LMDB |
| `qtaim-embed-data-summary` | Summarize dataset statistics |

## Important Files to Know

| File | Purpose |
|------|---------|
| `core/dataset.py` | Main dataset class (~950 lines) |
| `core/datamodule.py` | Lightning data modules (~950 lines) |
| `models/layers.py` | All custom GNN layers (~738 lines) |
| `models/graph_level/base_gcn.py` | Graph-level regression model |
| `scripts/train/train_qtaim_graph.py` | Main training script |
| `utils/data.py` | Default configs and utilities |
| `utils/models.py` | Model loading utilities |

## Data Flow

1. **Input**: Molecular structures (RDKit molecules in pickle files)
2. **Wrapping**: Convert to `MoleculeWrapper` objects with metadata
3. **Featurization**: Generate atom, bond, and global features
4. **Graph Construction**: Build heterogeneous DGL graphs
5. **Scaling**: Normalize features (standard/log scales)
6. **Batching**: Collate multiple graphs into batched DGL graphs
7. **Model Prediction**: Pass through GNN layers with message passing
8. **Pooling**: Aggregate node features to graph-level predictions
9. **Loss & Optimization**: Compute loss, backpropagate, update weights

## Debugging Tips

- Use `debug=True` in training scripts for smaller dataset subsets
- Check `config["dataset"]["extra_keys"]` for feature configuration issues
- Verify scaler serialization with `test_scalers.py` patterns
- For LMDB issues, ensure proper closing of database connections
- Use `torch.set_float32_matmul_precision("high")` for performance

## W&B Integration

The project uses Weights & Biases for experiment tracking:
```python
from pytorch_lightning.loggers import WandbLogger
logger = WandbLogger(project="project_name", entity="username")
trainer = pl.Trainer(logger=logger)
```

Sweep configs are in `scripts/train/sweep_config*.json`.
