---
status: done
priority: p1
issue_id: "004"
tags: [code-review, dgl-migration, xai]
dependencies: []
---

# 004 - data/xai.py: DGL API called on PyG Data objects

## Problem Statement

`qtaim_embed/data/xai.py` calls `.ndata`, `.edata`, and `.to_networkx()` on an object
returned by `construct_homograph_blank()` - which now returns a PyG `Data` object.
These are DGL-only APIs and will raise `AttributeError`.

## Findings

**`qtaim_embed/data/xai.py`** lines 40-43:
```python
homo_graph_empty.ndata["feats"] = edge_mask_z_processed["atom"]   # DGL only
homo_graph_empty.edata["feats"] = edge_mask_z_processed["bond"]   # DGL only
nxg = homo_graph_empty.to_networkx(node_attrs=["feats"], edge_attrs=["feats"])  # DGL only
```

`construct_homograph_blank` in `utils/grapher.py:141` returns a PyG `Data` object.
PyG `Data` uses attribute access (`graph.x`, `graph.edge_attr`) and has its own
`to_networkx` via `torch_geometric.utils.to_networkx`.

Also: `tests/test_transforms.py:2` has unused `import dgl` left over from migration.
This will cause `ImportError` in environments without DGL installed.

## Proposed Solutions

### Option 1 (Recommended): Rewrite for PyG Data API
```python
from torch_geometric.utils import to_networkx

# Instead of homo_graph_empty.ndata["feats"] = ...
homo_graph_empty.x = edge_mask_z_processed["atom"]
homo_graph_empty.edge_attr = edge_mask_z_processed["bond"]

# Instead of homo_graph_empty.to_networkx(...)
nxg = to_networkx(homo_graph_empty, node_attrs=["x"], edge_attrs=["edge_attr"])
```

**Effort:** Small (30 min for xai.py, 5 min for test_transforms.py)
**Risk:** Low

### Option 2: Remove or deprecate xai.py
If XAI is not actively used, mark the module as deprecated pending full PyG rewrite.

## Acceptance Criteria

- [x] `xai.py` uses PyG `Data` attribute access (`graph.x`, `graph.edge_attr`)
- [x] `xai.py` uses `torch_geometric.utils.to_networkx` instead of `.to_networkx()`
- [x] `tests/test_transforms.py` has `import dgl` removed
- [x] `xai.py` does not import or use `dgl`

## Work Log

- 2026-03-06: Identified by pattern-recognition-specialist review agent
