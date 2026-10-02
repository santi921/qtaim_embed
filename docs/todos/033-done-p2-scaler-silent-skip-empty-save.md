---
status: done
priority: p2
issue_id: 033
tags: [code-review, data-integrity, scaler]
dependencies: []
---

# Scaler update() batch-level skip and silent empty-scaler save

## Problem Statement

`HeteroGraphStandardScalerIterative.update()` (qtaim_embed/data/processing.py)
derives `node_types` from `graphs[0]` only. Two failure modes:

1. A batch whose first graph lacks the track (e.g. labels) skips every graph
   in the batch, including ones that have labels - scaler stats silently fit
   on a subset. Before PR #13 this crashed loudly (IndexError); now it is
   silent.
2. A scaler fed only track-less graphs finalizes and saves as an empty
   scaler with no error. Applying it later to featurized graphs crashes with
   an unrelated-looking TypeError; applying to unfeaturized graphs is fully
   silent (dataset written unscaled).

## Findings

- Current pipelines call update() per single graph with uniform label
  presence per dataset, so scenario 1 is theoretical today (drops to P3 if
  that invariant is guaranteed upstream).
- Scenario 2 is reachable by miswiring (e.g. fitting the feature scaler on
  raw build_graph output, which now looks more graph-like with pos/z).

## Proposed Solutions

1. **Per-graph filtering + finalize guard** (recommended): in update(), skip
   individual graphs lacking the track instead of the whole batch; in
   finalize(), warn or raise when `features_tf=True` and `dict_node_sizes`
   is empty ("feature scaler finalized with zero observations"). Effort:
   Small. Risk: Low.
2. **Assert batch homogeneity**: raise if graphs in one update() call have
   differing node_types for the track. Effort: Small. Risk: could break
   legitimate heterogeneous callers if any exist.

## Recommended Action

Option 1, with the loud failure placed at apply time rather than finalize: the
generator's `Converter.finalize` runs on every shard, and a shard that produced
no graphs must not crash there. `finalize()` warns; `__call__` raises.

## Acceptance Criteria

- [x] update() on a mixed batch fits every graph that has the track
- [x] finalize()/save of a zero-observation feature scaler is loud

## Work Log

- 2026-07-28: Finding from PR #13 review (data-integrity-guardian +
  kieran-python-reviewer converged).

## Resources

- PR: https://github.com/santi921/qtaim_embed/pull/13 (merged)
- 2026-09-09: fixed in `qtaim_embed/data/processing.py`. `update()` and `__call__`
  collect the track per graph via `_get_ndata` instead of reading `graphs[0]`;
  `finalize()` logs a WARNING when `features_tf` and zero observations;
  `__call__` raises ValueError when a graph carries a node type the scaler has
  no statistics for (previously silent pass-through when `_mean` was empty,
  KeyError when partially populated). Tests:
  `tests/test_scaler_merge_fallback.py` (mixed batch, zero-observation warn +
  raise). `inverse()` still reads `graphs[0]`; left as is.
