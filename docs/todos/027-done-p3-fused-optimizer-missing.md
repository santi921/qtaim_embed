---
status: done
priority: p3
issue_id: "027"
tags: [code-review, performance, optimizer]
dependencies: []
---

# 012 - Fused optimizer pattern missing from all four model classes

## Problem Statement

MEMORY.md documents the `fused=True` optimizer pattern as a key performance improvement
(20-40% speedup) implemented across all models. However, none of the four model
`configure_optimizers` methods actually use it. Only the profiling test scripts show
the fused pattern.

## Findings

All four models use plain `torch.optim.Adam`:
- `models/graph_level/base_gcn.py` - `configure_optimizers`
- `models/graph_level/base_gcn_classifier.py` - `configure_optimizers`
- `models/node_level/base_gcn.py` - `configure_optimizers`
- `models/link_pred/link_model.py:549` - `torch.optim.Adam(...)`

None use the documented try/except fused pattern from MEMORY.md:
```python
try:
    optimizer = torch.optim.Adam(..., fused=True)
except RuntimeError:
    optimizer = torch.optim.Adam(...)
```

## Proposed Solutions

### Option 1: Add fused=True try/except to all configure_optimizers
```python
def configure_optimizers(self):
    try:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
            fused=True,
        )
    except RuntimeError:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
    ...
```

**Effort:** Small (4 files, ~5 lines each)
**Risk:** Low - `fused=True` only activates on CUDA; CPU fallback is explicit

## Acceptance Criteria

- [x] All four model `configure_optimizers` use fused try/except pattern
- [x] Training tests still pass
- [x] GPU training shows measurable speedup vs baseline

## Work Log

- 2026-03-06: Identified by pattern-recognition-specialist review agent
