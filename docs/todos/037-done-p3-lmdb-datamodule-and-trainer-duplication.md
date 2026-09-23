---
status: done
priority: p3
issue_id: "037"
tags: [code-review, simplification, datamodule, training-scripts]
dependencies: []
---

# 037 - Three LMDB datamodules, ten Trainer constructions

## Problem Statement

`LMDBDataModule`, `LMDBLinkDataModule` and (until 2026-09-09) `LMDBBondDataModule` copied
the same `__init__`/`setup` block; `LMDBBondDataModule` now subclasses `LMDBDataModule`
but `LMDBLinkDataModule` still does not. Every worker-stability, dtype, or schema fix has
to be applied per copy. Separately, `pl.Trainer(...)` is constructed by hand in ten
scripts; `num_sanity_val_steps` reached three of them and is ignored by the other seven
(`bayes_opt_*`, `train_lmdb_classifier`, `train_qtaim_graph_classifier`).

## Proposed Solution

1. `LMDBLinkDataModule(LMDBDataModule)` overriding only the loader methods.
2. One `build_trainer(config, loggers, callbacks)` in `qtaim_embed/utils/training.py`
   used by every train and bayes-opt script; `train_qtaim_bond.build_trainer` is the
   starting point.
3. Fold `DataLoaderBondLMDB` into `DataLoaderLMDB` with a `with_labels` flag.

Do this as one cleanup PR after the bond-classifier and performance branches land, so
it does not collide with either.

## Acceptance Criteria

- [x] One LMDB datamodule base, two thin subclasses
- [x] `num_sanity_val_steps` honoured by every entry point
- [x] Full test suite green

## Work Log

- 2026-09-08: found by code review (angles D, E, G).
- 2026-09-09: done: `LMDBLinkDataModule` and `LMDBBondDataModule` now subclass
  `LMDBDataModule` (one setup path); `qtaim_embed/utils/training.py:build_trainer`
  written and adopted by `train_qtaim_bond.py`. Deferred: adopting `build_trainer`
  in the other nine scripts (perf window is editing them for LinearWarmup) and folding
  `DataLoaderBondLMDB` into `DataLoaderLMDB` (perf window rewrote its collate to
  `collate_hetero_direct`). Do both after that branch lands.
- 2026-09-09 (later): `build_trainer` adopted in the six scripts the perf window
  has not touched: `train_lmdb_classifier.py`, `train_qtaim_link.py`,
  `bayes_opt_graph.py`, `bayes_opt_graph_classifier.py`, `bayes_opt_link.py`,
  `bayes_opt_node.py`. Accelerator and default_root_dir are passed explicitly so
  behaviour is unchanged apart from `num_sanity_val_steps` and `warmup_epochs`
  now being honoured. Verified: `tests/test_cli_entry_points.py` (19) and an
  end-to-end `qtaim-embed-train-link --use_lmdb` run on the lmdb_link fixture
  (3 epochs, test metrics reported). Still open: `train_qtaim_graph.py`,
  `train_qtaim_graph_classifier.py`, `train_qtaim_node.py` (perf window edits
  them for LinearWarmup) and the `DataLoaderBondLMDB` fold-in.
- 2026-09-09 (final): performance branch finished, so the last three scripts
  (`train_qtaim_graph.py`, `train_qtaim_graph_classifier.py`,
  `train_qtaim_node.py`) now call `build_trainer`, which gained
  `use_distributed_sampler=not dataset.bucketing` (BucketBatchSampler shards
  by rank itself) and owns the LinearWarmup wiring. `DataLoaderBondLMDB` is
  gone: `DataLoaderLMDB(with_labels=False)` returns the batched HeteroData
  from `collate_hetero_direct`; `LMDBBondDataModule._loader` uses it. All ten
  Trainer constructions go through one function.
