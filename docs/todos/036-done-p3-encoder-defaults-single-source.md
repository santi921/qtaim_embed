---
status: done
priority: p3
issue_id: "036"
tags: [code-review, simplification, encoders, config]
dependencies: []
---

# 036 - Encoder knobs are spelled out in ~12 places

## Problem Statement

The eight `encoder_*` settings (`encoder_fn`, `encoder_hidden`, `encoder_cutoff`,
`encoder_n_interactions`, `encoder_num_gaussians`, `encoder_num_radial`, `encoder_lmax`,
`encoder_max_neighbors`) appear as literal lists in: three loaders in `models/utils.py`
(115-122, 162-169, 255-262) plus the key list in `load_bond_model_from_config`, the
constructor signatures and params dicts of `GCNGraphPred`, `GCNGraphPredClassifier`,
`GCNNodePred`, `GCNBondPred`, and four default configs in `utils/data.py`. Knobs the
encoders already accept (`num_spherical`, `int_emb_size`, `num_filters`, `max_z`) are
unreachable from config because each costs ~12 edits.

## Proposed Solution

One `ENCODER_DEFAULTS` dict in `models/encoders/__init__.py` next to `ENCODER_FNS`.
Loaders do `{k: config.get(k, v) for k, v in ENCODER_DEFAULTS.items()}`; default
configs do `**ENCODER_DEFAULTS`; `build_encoder` reads from it. Also lift the
duplicated encoder wiring in the three hetero models (assert, params entries,
`build_encoder`, atom width += encoder_hidden) into one `attach_encoder` helper; the
classifier currently lacks the compiled-xor-encoder assert the other two carry.

## Acceptance Criteria

- [x] Adding a ninth encoder knob touches one file plus the encoder that consumes it
- [x] All four default configs and four models read defaults from the same object
- [x] Classifier: not applicable, `GCNGraphPredClassifier` has no `compiled` flag; the guard lives in `check_encoder_hparams` and the classifier calls it without `compiled`

## Work Log

- 2026-09-08: found by code review (angles D, E, G); deferred, file owned by the performance window.
- 2026-09-09: done. `ENCODER_DEFAULTS` (ten knobs, incl. the new `encoder_max_z`)
  and `encoder_kwargs_from_config` in `models/encoders/__init__.py`; the three
  loaders in `models/utils.py` splat `encoder_kwargs_from_config(config)` and
  `load_bond_model_from_config` takes its key list from `ENCODER_DEFAULTS`; the
  four default configs in `utils/data.py` splat `ENCODER_DEFAULTS` (bond
  overrides `encoder_fn` to schnet). `build_encoder` reads knobs through
  `_knob`, so hparams saved before a knob existed fall back to the default.
  `attach_encoder` / `check_encoder_hparams` replace the copied assert and
  width logic in `GCNGraphPred`, `GCNGraphPredClassifier`, `GCNNodePred`,
  `GCNBondPred`. Model constructors keep explicit encoder_* keywords (Lightning
  hparams), so a new knob is: ENCODER_DEFAULTS + build_encoder + one keyword
  line in each of the four constructors; `tests/test_encoder_defaults.py`
  fails if any of the four models or four configs misses one.
