---
status: done
priority: p2
issue_id: "035"
tags: [code-review, encoders, correctness]
dependencies: []
---

# 035 - DimeNet++ encoder fails for Z >= 95

## Problem Statement

`DimeNetPPEncoder` wraps PyG's `EmbeddingBlock`, which hard-codes
`nn.Embedding(95, hidden)`. Any atom with Z >= 95 (Am, Cm, Bk, Cf, ... in
`settings_classifier_actinides.json`) raises `IndexError: index out of range in self`
at the first batch. `schnet` and `equivariant` use `max_z=119` and work, so the
failure is encoder-specific and only surfaces at runtime.

## Proposed Solution

Replace `self.emb.emb` (the `nn.Embedding(95, H)`) with `nn.Embedding(119, H)` after
constructing the PyG block, or build the embedding block locally. About 10 lines.
Add a forward test with `z = torch.tensor([96, 1, 8])` to `tests/test_encoder_parity.py`.

## Acceptance Criteria

- [x] `DimeNetPPEncoder` forward succeeds for Z up to 118
- [x] Parity test against PyG blocks still passes for Z < 95

## Work Log

- 2026-09-08: found by code review (angles A and C); deferred because the file is being edited in the performance window.
- 2026-09-09: done. `DimeNetPPEncoder(max_z=119)` replaces `self.emb.emb`
  after constructing PyG's EmbeddingBlock and re-applies its
  uniform(-sqrt(3), sqrt(3)) init. Test
  `tests/test_encoder_parity.py::TestDimeNetPP::test_heavy_elements_embed`
  (Z = 96, 118 forward + gradient on row 96). `max_z` is not yet exposed in
  `build_encoder` (encoders/__init__.py is being edited by the perf window);
  the default covers the periodic table. DimeNet++ checkpoints saved before
  this change carry a (95, H) embedding and need a non-strict load.
