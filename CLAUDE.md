# CLAUDE.md - QTAIM-Embed Development Guide

## Project Overview

QTAIM-Embed is a Graph Neural Network (GNN) package for molecular property prediction using heterogeneous graphs. The project implements machine learning models that work with Quantum Theory of Atoms in Molecules (QTAIM) features to predict molecular and reaction properties. It handles complex molecular representations including spin states, charged species, and sophisticated atom/bond features.

**Citation**: Digital Discovery (2024) by Vargas, Gee, and Alexandrova.

## Shared Skills

For available skills and tool guidance, read the relevant files in `/home/santiagovargas/dev/claude-skills/` as needed:
- **Scientific** (PyG, PyTorch Lightning, RDKit, pymatgen, matplotlib, scikit-learn): `scientific/`
- **Code review & planning** (multi-agent reviews, brainstorm/plan/work workflows): `compound/`
- **Document processing** (PDF, XLSX): `documents/`

Read the specific skill file when you need detailed API patterns or usage guidance for a task.

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
│   │   ├── encoders/      # 3D geometric atom encoders (SchNet/DimeNet++/MACE-style) + neighbor module
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
5. **Bond classification (T3)**: `GCNBondPred` in `models/link_pred/bond_model.py` scores geometric candidate pairs (`candidate_pairs` at 2x covalent radii) for QTAIM bond-path existence. Atom embeddings come from `encoder_fn` over `atom.pos`/`atom.z` (plus `atom.feat` only if `use_atom_feat` is set, which leaks labels when atom features derive from the bond list); labels are read from `a2b` inside the step; no message passing over bond nodes (they are the labels). Always report next to the distance-rule baseline (`models/link_pred/baselines.py`, `qtaim-embed-eval-bond-baselines`). Plan: `docs/plans/2026-09-08-feat-t3-bond-classifier-plan.md`.

### Model Components

- **Message-passing functions**: `GraphConvDropoutBatch`, `ResidualBlock`, `GATConv`, and `ResidualBlockDense` (same math as `ResidualBlock` on a padded per-molecule layout, `models/layers_dense.py`: a2b/b2a are `bmm` against an incidence matrix, global edges are masked sums/broadcasts, no gather/scatter; pair with `dataset.bucketing: true` so shapes are static (the sampler stamps each batch's class shape as `graph.dense_shape`, train drops ragged last batches) and `compiled: true` captures the conv stack as CUDA graphs, one per shape class; eval always runs the eager dense blocks. 1.4-2.2x over `ResidualBlock` at bf16 on 60-350 atom molecules in the conv stack alone, 1.3-1.5x in the full training step (5,038 to 7,755 samples/s at hidden 128, batch 1024, where the data path then caps it). `convert_model_to_dense` in `models/utils.py` maps trained `ResidualBlock` weights onto it.)
- **3D geometric encoders** (`encoder_fn`): `SchNetEncoder`, `DimeNetPPEncoder`, `EquivariantEncoder` (see below)
- **Global pooling**: `SumPoolingThenCat`, `MeanPoolingThenCat`, `WeightAndSumThenCat`, `WeightAndMeanThenCat`, `GlobalAttentionPoolingThenCat`, `Set2SetThenCat`
- **Scalers**: `HeteroGraphStandardScaler`, `HeteroGraphLogMagnitudeScaler`

### 3D Geometric Encoders

Optional per-atom encoders in `models/encoders/` that consume only `atom.pos`
(float32 [N,3], Angstrom) and `atom.z` (int64 [N]) from the heterograph and
return per-atom embeddings of width `encoder_hidden`. The output is
concatenated with `atom.feat` before `UnifySize`; the hetero conv stack, heads,
and existing feature pathway are unchanged. Graphs built by grapher commit
`9084345` (2026-07-27) or later carry `pos`/`z`; older LMDBs do not and will
fail with a schema error.

- `encoder_fn: "schnet"` - invariant continuous-filter convolution, adapter
  around PyG's reference blocks (weight-copy parity tested against
  `torch_geometric.nn.models.SchNet`).
- `encoder_fn: "dimenetpp"` - directional/angular message passing via a
  torch-only triplet builder (no torch-sparse); memory scales with
  sum(deg^2), capped by `encoder_max_neighbors`.
- `encoder_fn: "equivariant"` - MACE-style e3nn message passing with
  l = 0..`encoder_lmax` irreps and invariant scalar (l=0) readout.
  `encoder_tp: "channelwise"` (default) uses depthwise `uvu` tensor products
  plus an `o3.Linear` mix (256 weights per edge at lmax 1, hidden 64);
  `"fully_connected"` is the original per-edge FullyConnectedTensorProduct
  (16,384 weights per edge, OOMs at batch 128) kept only for old checkpoints.
- `encoder_fn: "none"` (default) - current behaviour, no encoder.

All `encoder_*` knobs and their defaults live in `ENCODER_DEFAULTS`
(`models/encoders/__init__.py`); loaders use `encoder_kwargs_from_config`, default
configs splat the dict, models call `attach_encoder` / `check_encoder_hparams`.
A new knob goes there, into `build_encoder`, and as one keyword line in each of
the four model constructors; `tests/test_encoder_defaults.py` enforces that.

Supported by `GCNNodePred`, `GCNGraphPred`, and `GCNGraphPredClassifier` (not
the link model). Neighbor lists are built inside the encoder forward
(`models/encoders/neighbors.py` - torch_cluster and torch_sparse are
deliberately NOT dependencies): batched inputs use one per-molecule dense
distance block over `to_dense_batch` positions (one kernel, one sync; 65x
faster than the old batch-wide chunked cdist at batch 512), single molecules
and oversized blocks fall back to the row-chunked cdist. The build is
data-dependent, so it is incompatible with `torch.compile`: `compiled: true`
with an encoder raises at construction. `encoder_max_neighbors` defaults to
16 (dimenetpp triplet memory is sum(deg^2); cutoff 5 / cap 32 needs 9.4 GB at
batch 128 on 60-atom molecules).

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
        "bucketing": False,   # True: BucketBatchSampler groups graphs by padded (atoms, bonds) shape (LMDB path; DDP-safe, shards by rank)
        "bucket_grid": 16,
        "bucketing_eval": True,  # False: val/test loaders in dataset order (bucketed order is class-major; see sampler.indices())
        "bucket_drop_last": True,  # train only: drop each class's ragged last batch so the batch dimension is static
    },
    "model": {
        "n_conv_layers": 8,
        "conv_fn": "ResidualBlock",  # or "ResidualBlockDense" (padded, needs dataset.bucketing), "GraphConvDropoutBatch", "GATConv"
        "dense_grid": 16,            # ResidualBlockDense: pad atoms/bonds to multiples of this
        "global_pooling_fn": "SumPoolingThenCat",
        "hidden_size": 128,
        "embedding_size": 128,
        "dropout": 0.2,
        "batch_norm": True,   # REQUIRED for ResidualBlock on 40+ atom molecules (see Important Notes)
        "bn_before_activation": True,   # default via configs: conv -> BN -> activation -> dropout, final layer linear (fixes the eval-mode divergence); model constructors default to False so pre-2026-09-09 checkpoints keep their layer order
        "global_aggr": "sum",           # "mean": average (not sum) atoms/bonds into the global node (a2g/b2g)
        "activation": "ReLU",
        "lr": 1e-3,
        "loss_fn": "mse",  # or "mae"
        # optional 3D encoder (needs atom.pos/atom.z on the graphs)
        "encoder_fn": "none",  # or "schnet", "dimenetpp", "equivariant"
        "encoder_hidden": 64,          # output width, concatenated onto atom.feat
        "encoder_cutoff": 5.0,         # radius-graph cutoff, Angstrom
        "encoder_n_interactions": 3,
        "encoder_num_gaussians": 50,   # schnet RBF size
        "encoder_num_radial": 6,       # dimenetpp/equivariant radial basis size
        "encoder_lmax": 1,             # equivariant only
        "encoder_max_neighbors": 16,   # dimenetpp only, caps triplet blowup
        "encoder_tp": "channelwise",   # equivariant only
        "encoder_max_z": 119,          # embedding rows for atomic numbers (all encoders)
    },
    "optim": {
        "precision": "bf16-mixed",  # default; "16-mixed" or 32 also work, never bare 16
        "max_epochs": 100,
        "gradient_clip_val": 1.0,
        "train_batch_size": 128,    # node default is 1024 with lr 8e-3 and warmup_epochs 1 (E1 gate on 119K graphs); use 128 / 1e-3 on small datasets
        "num_workers": 8,           # LMDB path needs >= 8 at batch >= 512 (data-bound otherwise)
        "pin_memory": True,
        "warmup_epochs": 0,         # > 0 adds LinearWarmup (linear LR ramp, then ReduceLROnPlateau)
    }
}
```

Default configs are available via:
```python
from qtaim_embed.utils.data import get_default_graph_level_config
config = get_default_graph_level_config()
```

## Development Workflow

### Conda Environment Activation (Claude Code)

When running commands that require conda environments, use this pattern:

```bash
# Correct pattern for activating conda and running commands
source /home/santiagovargas/miniconda3/etc/profile.d/conda.sh && conda activate generator && <command>

# Example: running tests
source /home/santiagovargas/miniconda3/etc/profile.d/conda.sh && conda activate generator && pytest tests/
```

**Important**: The `generator` environment is the primary development environment for this project.

### Running Tests

```bash
# Run all tests (with proper conda activation)
source /home/santiagovargas/miniconda3/etc/profile.d/conda.sh && conda activate generator && pytest tests/

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
- `test_neighbors.py`: Radius/candidate/triplet construction vs brute force
- `test_encoder_parity.py`: 3D encoders vs PyG reference blocks (weight-copy parity)
- `test_equivariance.py`: Rotation invariance/equivariance of the 3D encoders (both `encoder_tp` modes)
- `test_collate.py`: direct LMDB collate vs `Batch.from_data_list`
- `test_warmup.py`: `LinearWarmup` callback
- `test_layers_dense.py`, `test_models_dense.py`: `ResidualBlockDense` and dense models vs their `ResidualBlock` twins (exact parity, masked batch norm)
- `test_bucketing.py`: `BucketBatchSampler` and the `LMDBDataModule` bucketing hook
- `test_bond_pairs.py`, `test_bond_model.py`: bond-pair labels, candidate recall, pair head symmetry, `GCNBondPred` vs distance rule

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
- DONT USE EMOJIS or emdashes
- run code reviews after any large changes 
- plan mode automatically for any changes that seem to edit more than one file or any file that is critical/highly called upon

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
- PyTorch Geometric (PyG)
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
| `qtaim-embed-train-bond` | Train the T3 bond classifier (`GCNBondPred`, LMDB only) |
| `qtaim-embed-eval-bond-baselines` | Candidate recall and distance-rule reference table for T3 |
| `qtaim-embed-bayes-opt-graph` | Bayesian optimization for graph models |
| `qtaim-embed-bayes-opt-node` | Bayesian optimization for node models |
| `qtaim-embed-bayes-opt-graph-classifier` | Bayesian optimization for classifiers |
| `qtaim-embed-mol2lmdb` | Convert molecule data to LMDB |
| `qtaim-embed-mol2lmdb-node` | Convert node-labeled data to LMDB |
| `qtaim-embed-data-summary` | Summarize dataset statistics |
| `qtaim-embed-bench` | Training-throughput harness (raw or Lightning mode, GPU util, kernels/step, data wait); configs in `profiling/bench_configs/`, results in `profiling/bench_results/` |

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
4. **Graph Construction**: Build heterogeneous PyG graphs
5. **Scaling**: Normalize features (standard/log scales)
6. **Batching**: Collate multiple graphs into batched PyG graphs (the LMDB loader uses `collate_hetero_direct`, 4x faster than `Batch.from_data_list`; output is a `HeteroData` with `batch`/`ptr`/`num_graphs`, not a `Batch`, so `to_data_list` is unavailable)
7. **Model Prediction**: Pass through GNN layers with message passing
8. **Pooling**: Aggregate node features to graph-level predictions
9. **Loss & Optimization**: Compute loss, backpropagate, update weights

## Debugging Tips

- Use `debug=True` in training scripts for smaller dataset subsets
- Check `config["dataset"]["extra_keys"]` for feature configuration issues
- Verify scaler serialization with `test_scalers.py` patterns
- For LMDB issues, ensure proper closing of database connections
- Use `torch.set_float32_matmul_precision("high")` for performance
- Throughput questions: run `qtaim-embed-bench` first; measurements and gate decisions live in `docs/research/2026-09-track-a-measurements.md`

## W&B Integration

The project uses Weights & Biases for experiment tracking:
```python
from pytorch_lightning.loggers import WandbLogger
logger = WandbLogger(project="project_name", entity="username")
trainer = pl.Trainer(logger=logger)
```

Sweep configs are in `scripts/train/sweep_config*.json`.


## Important Notes
- `batch_norm` must stay True for the hetero conv stack: PyG's `GraphConv` has no degree normalization (the `norm` key is inert since the DGL migration), so on 40+ atom molecules activations grow 60-100x per block and a `batch_norm: False` model collapses to predicting the label mean. Measured 2026-09-09, `docs/research/2026-09-track-a-measurements.md`.
- OPEN: even with batch norm on, the tm_react reference run (batch 128, lr 1e-3) diverges in eval mode from epoch 2. Diagnosed (`docs/research/2026-09-tm-react-eval-divergence.md`): BN sits after ReLU in `GraphConvDropoutBatch`, channels dead for a whole batch drive `running_var` to the float32 floor, and in eval mode they amplify rare inputs 100-260x; the unnormalized a2g/b2g sums feed the amplifier. Weights are fine (batch-stat val MSE keeps improving). Fixed by `model.bn_before_activation: true` (conv -> BN -> activation -> dropout, final prediction layer linear; sparse and dense twins, parity tested): 40 epochs from scratch with no excursion, val R2 0.87 vs the reference's 0.72 before it diverged. `model.global_aggr: "mean"` adds nothing on top and still diverges on its own. Default True in every default config and loader since 2026-09-09; the model constructors keep False so old checkpoints load with their original order. Do not train 40+ atom models with it off.
- Research the codebase before editing. Never change code you haven't read. Also don't make changes to code without asking first.
- Don't use emojis and emdashes anywhere
- User instructions always override this file.
- Do not re-read files already read unless file may have changed.
- Be concise in output but thorough in reasoning.
- No inline prose. Use comments sparingly - only where logic is unclear.
- 