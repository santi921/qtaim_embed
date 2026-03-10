# DGL to PyG Migration Guide

This guide covers the migration of QTAIM-Embed from DGL (Deep Graph Library) to PyTorch Geometric (PyG).

## Overview

All graph operations now use PyTorch Geometric instead of DGL. This change was motivated by:

- PyG is more actively maintained and has a larger ecosystem
- PyG is available on PyPI (no custom index URLs needed)
- Better integration with the broader PyTorch ecosystem

**This is a breaking change.** Models trained with the DGL version are not compatible with the PyG version. Retraining is required.

## Installation

Remove DGL and install PyG:

```bash
# Remove DGL
pip uninstall dgl

# Install PyG (CPU)
pip install torch_geometric

# Install PyG (CUDA 12.4)
pip install torch_geometric torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Or with conda:

```bash
conda install -c pyg pytorch-geometric
```

Then install QTAIM-Embed:

```bash
pip install -e .
```

## API Changes

### Graph Format

DGL `HeteroGraph` has been replaced by PyG `HeteroData`.

```python
# Before (DGL)
import dgl
g = dgl.heterograph({...})
g.ndata["feat"]  # access node features

# After (PyG)
from torch_geometric.data import HeteroData
data = HeteroData()
data["atom"].feat  # access node features by type
```

### Batching

```python
# Before (DGL)
batched = dgl.batch(graph_list)

# After (PyG)
from torch_geometric.data import Batch
batched = Batch.from_data_list(graph_list)
```

### Node Features

```python
# Before (DGL)
feats = g.ndata["feat"]  # returns dict {ntype: tensor}

# After (PyG)
feats = {nt: data[nt].feat for nt in data.node_types}
```

### Edge Indices

```python
# Before (DGL)
src, dst = g.edges(etype=("atom", "a2b", "bond"))

# After (PyG)
edge_index = data["atom", "a2b", "bond"].edge_index  # shape [2, E]
src, dst = edge_index[0], edge_index[1]
```

## Model Changes

DGL and PyG implement `GraphConv` differently:

- **DGL**: Shared weight matrix with degree normalization
- **PyG**: Separate root and neighbor weight matrices (`W_root`, `W_rel`) with additive aggregation

These are both valid GNN formulations but are **not numerically equivalent**. Models trained with the DGL backend will produce different outputs with the PyG backend even with the same weights. Retraining is required.

## LMDB Compatibility

Old DGL-serialized LMDB databases are **not compatible** with the PyG version. The serialization functions have changed:

- `serialize_dgl_graph` / `load_dgl_graph_from_serialized` have been removed
- `serialize_graph` / `load_graph_from_serialized` now handle PyG `HeteroData` objects

Regenerate LMDB databases from source pickle files:

```bash
qtaim-embed-mol2lmdb -input_file data.pkl -output_dir ./lmdb_data/
```

## Checkpoint Compatibility

Old model checkpoints saved with the DGL version will **not load** correctly due to changed layer parameter names and shapes. Retrain models from scratch.

## Scaler Compatibility

Scaler files (standard/log scalers) saved with `torch.save` are compatible across both versions since they store plain tensors, not graph objects.

## Performance Notes

- Batch size has the largest impact on throughput (up to 5.7x improvement)
- `num_workers=4` provides consistent 1.8-2.3x speedup for data loading
- `fused=True` optimizer provides 20-40% speedup on GPU
- `torch.compile` provides no benefit for small datasets (<1000 steps/epoch)
