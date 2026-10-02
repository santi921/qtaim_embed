---
status: done
priority: p3
issue_id: 034
tags: [code-review, quality, featurizer]
dependencies: []
---

# Misc P3 hardening from PR #13 review

## Problem Statement

Collected nice-to-haves from the multi-agent review of PR #13, none blocking.

## Findings

1. **z dtype decision before the 4M build**: `atom.z` is int64; int32 halves
   its storage (~0.9 GB at 4M graphs) and `nn.Embedding` accepts IntTensor
   since torch 1.10. PyG convention is int64. Changing later means a corpus
   rebuild - decide before the Livermore run, then leave it alone. Total
   pos+z disk cost measured: ~1.66 KB/graph, ~6.6 GB at 4M (matches +2.2%).
2. **Width-test coverage**: `TestZeroBondFallbackWidth` exercises ring + boo
   + rbf but not `bond_length` or plain scalar keys - the old bug lived in
   the "trivial" term. Add `"bond_length"` and one scalar key to the config.
3. **boo_/rbf_ substring matching**: a descriptor key merely containing
   "boo_"/"rbf_" is misclassified (`int(key.split("_")[1])` can raise).
   `startswith` is safer. Multiple boo keys: last wins silently; multiple
   rbf keys: first wins silently - a hard error would prevent silent
   feature loss.
4. **selected_keys=None**: still crashes in the boo-detection loop despite a
   (now removed) dead `!= None` guard downstream. Either normalize with
   `selected_keys or []` in `__init__` or document list-only.
5. **update() homogeneity comment**: note in processing.py that the
   `graphs[0]` check assumes homogeneous batches (superseded if todo 002
   lands).
6. **Optional simplification**: derive the fallback width as
   `len(self._feature_name)` by moving the name block above the fallback
   (requires an unconditional `self._feature_name = []` reset, which also
   fixes name accumulation across calls when `allowed_ring_size == []`).
   Partially superseded by the merged scalar_keys refactor.

## Proposed Solutions

Batch items 2-5 as one small cleanup PR; decide item 1 as a config decision
before the full-corpus rebuild; take item 6 only if touching the function
again.

## Recommended Action

Items 2-5 done 2026-09-09 (see work log). Item 1 decided 2026-09-09: keep int64. Item 6 taken only as far as the unconditional
`_feature_name` reset.

## Acceptance Criteria

- [x] z dtype decision recorded before 4M build
- [x] width test covers all five width terms

## Work Log

- 2026-07-28: Findings from PR #13 review (performance-oracle,
  kieran-python-reviewer, code-simplicity-reviewer, data-integrity-guardian).

## Resources

- PR: https://github.com/santi921/qtaim_embed/pull/13 (merged)
- 2026-09-09: items 2-5 done in `qtaim_embed/data/featurizer.py` and
  `tests/test_featurizers.py::TestZeroBondFallbackWidth`. boo_/rbf_ keys are
  matched with `startswith`; a second boo or rbf key, or a malformed rbf key
  (`rbf_cutoff`, unknown basis), raises ValueError instead of silently winning
  or producing a width mismatch. `selected_keys=None` normalizes to `[]` in all
  three featurizers. `_feature_name` is reset unconditionally (item 6 partial),
  so names no longer accumulate across calls when `allowed_ring_size == []`.
  Item 5 is superseded by todo 033 (update() no longer reads graphs[0]).
  Width test now covers ring block, bond_length, boo, rbf and two scalars, one
  named `odd_boo_name` to pin the prefix rule. Item 1 (z dtype) still open.
- 2026-09-09: item 1 decided, `atom.z` stays int64. Measured on
  `tests/data/lmdb_link/train` (75 graphs, 16.1 atoms/graph): z costs ~364
  B/graph serialized, ~270 B of which is torch.save per-tensor framing, not
  payload. int32 saves 64 B/graph (0.6%), uint8 saves 95 B/graph (1.0%);
  at 4M graphs that is under 0.4 GB of ~39.7 GB. Both narrower dtypes also
  need a `.long()` cast in TransformMol and break `F.one_hot` (int32) or
  `nn.Embedding`/indexing (uint8) wherever a reader skips that cast. Not
  worth a rebuild coupling. If disk ever matters the lever is the per-graph
  torch.save format, not the integer width. Todo closed.
