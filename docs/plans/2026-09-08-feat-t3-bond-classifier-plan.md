---
title: "feat: T3 bond classifier (geometry-aware QTAIM bond-path prediction)"
type: feat
date: 2026-09-08
status: implemented 2026-09-08, review fixes applied 2026-09-09 (uncommitted on feat/3d-encoders); plumbing validated on mo_hydrides structural bonds; QTAIM numbers pending regenerated qtaim.lmdb
source_plan: qtaim_generator/docs/plans/2026-07-27-neurips-r1-05-t3-bond-classification-plan.md
related:
  - docs/roadmap_v4.md (Section 3.5 Bond Prediction Validation Program)
  - qtaim_generator/docs/plans/2026-07-27-neurips-r1-00-dependency-map.md (lock 5)
  - qtaim_generator/docs/plans/2026-09-08-neurips-r2-10-qtaim-regeneration-plan.md (label gate)
  - qtaim_generator/docs/plans/2026-07-27-3d-encoders-and-t3-candidates-plan.md (measurement provenance)
  - docs/plans/2026-09-08-feat-performance-engineering-plan.md (encoder-path throughput)
---

# T3 bond classifier

This is the qtaim_embed implementation plan for the T3 task defined in
qtaim_generator's doc 05. Doc 05 owns the task definition, the measured
reference numbers, and the scoping decision (model vs reference table). This
document owns the code: which files change, in what order, with what tests.
Both documents link to each other; edit the task definition there and the
implementation here.

## Summary

- **Task**: for every candidate atom pair (i, j) with
  `d_ij <= 2.0 * (rcov_i + rcov_j)`, predict whether a QTAIM bond critical
  point exists. One logit per pair. Near-balanced at 2.0x (positive fraction
  0.427 on tm_react), so no class weighting.
- **Reference to beat**: tuned distance rule `d <= 1.1 * (rcov_i + rcov_j)`,
  macro-F1 0.971 on 800 tm_react structures (doc 05 section 3). Always report
  it next to any learned number.
- **Design**: a new `GCNBondPred` LightningModule. Atom embeddings come from
  the shared `encoder_fn` 3D encoder over geometry only; a symmetric pair head
  scores candidates with an RBF-expanded distance. Candidates and labels are
  generated inside the training step from `atom.pos`, `atom.z`, and the
  graph's `a2b` connectivity. No stored candidates, no stored labels, no
  `hetero_to_homo`, no message passing over ground-truth bond nodes.
- **Label gate**: trustworthy QTAIM labels need graphs built from the
  regenerated `qtaim.lmdb` (generator doc 10). Code and tests proceed on the
  existing fixtures now; reported numbers wait for regenerated graphs.

## Why a new module rather than fixing `GCNLinkPred`

Doc 05 section 4 found two bugs in the existing predictors
(`qtaim_embed/models/layers_homo.py`): `MLPPredictor` ends in `nn.Sigmoid()`
while the loss is BCE-with-logits (double sigmoid, and `FullPredictor`
sigmoids a third time), and `AttentionPredictor` softmaxes over a size-1
dimension so the destination node is ignored. Beyond those, `GCNLinkPred` has
three structural problems for T3:

1. It runs on `hetero_to_homo` output, which drops `pos` and `z`
   (`qtaim_embed/data/transforms.py:92-124`). The model is 3D-blind by
   construction.
2. Negatives come from uniform `negative_sampling` over all non-edges
   (`qtaim_embed/data/dataloader.py:161-189`), which are mostly far-apart
   pairs. The task becomes trivially separable and the metric meaningless.
3. Its message passing runs over the positive-edge graph, i.e. over the
   labels. At inference from geometry there is no such graph.

Point 3 also rules out reusing `GCNNodePred` with a classification branch:
that model's hetero conv stack passes messages along `a2b`/`b2a` edges, which
for a QTAIM-bonded graph are the answer. `GCNBondPred` therefore uses only:
encoder(pos, z) and, optionally, `atom.feat` columns that are derivable from
geometry (element one-hot). `GCNLinkPred` stays as-is for its existing users;
nothing in it is modified by this plan.

## Label leakage rules (read before configuring a run)

- Never enable `use_atom_feat` on graphs built with `bonding_scheme: "qtaim"`
  unless the atom feature set is element one-hot only. `total_degree`,
  `is_in_ring`, and `ring_size_*` are computed from the graph's bond list and
  leak the labels. The default is `use_atom_feat: false`.
- The bond node features (`bond_length`, RBF expansions, `boo_*`) are never
  read by this model.
- Message passing happens only inside the encoder over its radius graph.

## Architecture

```
atom.pos (N,3), atom.z (N,), atom.batch (N,)
        |
        v
 encoder_fn in {none, schnet, dimenetpp, equivariant}    -> h (N, H)
   none: nn.Embedding(119, H) on z                       (learned distance rule)
        |
 candidate_pairs(pos, z, batch, pool_multiplier=2.0)     -> i, j, d_ij   (P,)
        |
 pair head (symmetric):
   x = [h_i + h_j, |h_i - h_j|, rbf(d_ij), d_ij / (rcov_i + rcov_j)]
   MLP(x) -> 1 logit per pair
        |
 labels: pair (i,j) in bond set from a2b edge_index      -> y (P,) in {0,1}
 loss:   BCE-with-logits
 metrics: AUROC, AP, precision, recall, F1(pos), macro-F1 at a calibrated
          threshold; threshold chosen on val to maximise macro-F1
```

`rbf` is `sinusoidal_bessel_rbf` from `qtaim_embed/utils/descriptors.py`
(Bessel with p=5 polynomial envelope, 50 basis, cutoff defaulting to the
largest possible candidate distance, pool_multiplier * 2 * max(rcov) = 10.4 A) or `gaussian_rbf`, selected by `pair_rbf`.

The `d / (rcov_i + rcov_j)` scalar is included so the head can express the
reference rule exactly; with `encoder_fn: none` the model is a learned
element-aware distance rule and is the correct "no geometry beyond distance"
ablation.

## Files

### New

| File | Purpose |
|------|---------|
| `qtaim_embed/data/bonds.py` | `bond_pairs_from_heterograph(graph)` -> (B,2) sorted atom pairs from `a2b`; `candidate_labels(i, j, bond_pairs, num_nodes)` -> (P,) via integer pair keys and `torch.isin`. Pure torch, batch-safe. |
| `qtaim_embed/models/link_pred/pair_head.py` | `SymmetricPairHead(nn.Module)`: builds the feature vector above, MLP to one logit. Invariant to swapping i and j by construction. |
| `qtaim_embed/models/link_pred/bond_model.py` | `GCNBondPred(pl.LightningModule)`: encoder, pair head, candidate generation, labels, loss, metrics, threshold calibration, `predict_bonds(graph)` returning (i, j, prob). |
| `qtaim_embed/models/link_pred/baselines.py` | `distance_rule(d, z_i, z_j, k)`, `fit_distance_rule(d, z_i, z_j, y, k_grid)`, `fit_pairwise_distance_rule(...)` (per element-pair k), `macro_f1`, `candidate_recall(bond_pairs, i, j)`. |
| `qtaim_embed/scripts/eval/eval_bond_baselines.py` | CLI: given an LMDB (file or shard dir) report candidate recall at 2.0x, the k-sweep table, and the per-element-pair rule; optional `--holdout_manifest` to stratify by suite. Writes CSV + markdown. |
| `qtaim_embed/scripts/train/train_qtaim_bond.py` | Training entry point mirroring `train_qtaim_link.py`; LMDB only. |
| `tests/test_bond_pairs.py` | Pair extraction and label assignment vs brute force; candidate recall on the fixture (>= 0.99 at 2.0x, 1.0 at 3.0x); pair head symmetry. |
| `tests/test_bond_model.py` | Head symmetry; forward shapes for every `encoder_fn`; overfit test on `tests/data/lmdb_link` that asserts val macro-F1 above the fitted distance rule on the same fixture; threshold calibration; checkpoint round trip. |

### Modified

| File | Change |
|------|--------|
| `qtaim_embed/data/dataloader.py` | `DataLoaderBondLMDB`: `Batch.from_data_list` only, returns the batched `HeteroData`. |
| `qtaim_embed/core/datamodule.py` | `LMDBBondDataModule`, same shape as `LMDBLinkDataModule` minus the homograph conversion. |
| `qtaim_embed/utils/data.py` | `get_default_bond_level_config()`. |
| `qtaim_embed/models/utils.py` | `load_bond_model_from_config()`. |
| `pyproject.toml` | entry point `qtaim-embed-train-bond`. |
| `CLAUDE.md` | one paragraph under Task Types and CLI table. |
| `docs/roadmap_v4.md` | Section 3.5 gets the label gate, the two predictor bugs, and a pointer here. |
| `qtaim_generator/docs/plans/2026-07-27-neurips-r1-05-t3-bond-classification-plan.md` | pointer to this document at the top. |

### Not touched

`GCNLinkPred`, `layers_homo.py`, `hetero_to_homo`, `FullPredictor`. Wiring
`GCNBondPred` into `FullPredictor` in place of the link model is a follow-up
(roadmap v4 Phase 2), once the classifier has a reported number.

## Config surface

```python
"model": {
    "encoder_fn": "schnet",           # none | schnet | dimenetpp | equivariant
    "encoder_hidden": 64,
    "encoder_cutoff": 5.0,
    "encoder_n_interactions": 3,
    "encoder_num_gaussians": 50,
    "encoder_num_radial": 6,
    "encoder_lmax": 1,
    "encoder_max_neighbors": 32,
    "use_atom_feat": False,           # see leakage rules
    "atom_input_size": 0,             # required only when use_atom_feat
    "pool_multiplier": 2.0,           # candidate radius, doc 05 section 2
    "pair_rbf": "bessel",             # bessel | gaussian
    "pair_rbf_n": 50,
    "pair_rbf_cutoff": None,          # None -> pool_multiplier * 2 * max(rcov), about 10.4 A
    "pair_hidden": [256, 128],
    "pair_dropout": 0.1,
    "lr": 1e-3, "weight_decay": 1e-5, "lr_plateau_patience": 10, "lr_scale_factor": 0.5,
    "threshold": None,                # calibrated on val when None
}
```

## Steps

| # | Step | Depends on | Test |
|---|------|------------|------|
| 1 | `data/bonds.py`: pair extraction and labels | nothing | brute-force parity on fixtures, batched graphs, graphs with zero bonds |
| 2 | `baselines.py` + `eval_bond_baselines.py` | 1 | reproduces k=1.1 as argmax on the fixture; recall = 1.0 at 2.0x on fixtures |
| 3 | `pair_head.py` | nothing | swap invariance to 1e-6; output shape (P,) |
| 4 | `bond_model.py` forward + loss + metrics | 1, 3 | forward for each encoder_fn on one fixture batch; loss finite; bf16 autocast |
| 5 | Threshold calibration and `predict_bonds` | 4 | chosen threshold maximises macro-F1 on a synthetic logit set |
| 6 | Dataloader, datamodule, default config, loader, training script, entry point | 4 | `LMDBBondDataModule` yields batches; 3-epoch smoke run on `tests/data/lmdb_link` |
| 7 | Overfit test with accuracy floor | 2, 6 | macro-F1 on fixture val > fitted distance rule on the same split after a fixed budget |
| 8 | Docs and cross-links | all | none |

Steps 1 to 3 are independent and small; 4 to 7 are sequential.

**Status 2026-09-08**: steps 1-8 done. 16 new tests pass (`tests/test_bond_pairs.py`,
`tests/test_bond_model.py`); the existing link, CLI, node and encoder tests still pass.
End-to-end GPU smoke run via `qtaim-embed-train-bond` with `encoder_fn: schnet`,
`bf16-mixed`, 3 epochs on the fixture: val macro-F1 0.868, test macro-F1 0.882 at the
calibrated threshold 0.45. Two implementation notes:

- Fused Adam is disabled when the trainer clips gradients (Lightning refuses the
  combination under mixed precision); same rule as the classifier since commit 0f82e55.
- `nn.ModuleDict` cannot use the key `"train"` (clashes with `nn.Module.train`), so the
  per-mode metric dicts are keyed `m_train` / `m_val` / `m_test`.

Review fixes 2026-09-09 (from the eight-angle code review): placeholder bond nodes
tolerated in `bond_pairs_from_heterograph` with a sync-free fast path; default
`pair_rbf_cutoff` derived from the candidate pool; `LMDBBondDataModule` now
subclasses `LMDBDataModule`; the eval script reuses `_resolve_lmdb_path` and accepts
`-config`; metric math shared between `BinaryScores` and `BinnedBinaryStats` via
`scores_from_counts`; RBF constants cached as buffers in the pair head; one
`models/optim.build_adam` used by all five LightningModules (fused Adam off when the
trainer clips gradients); `finalize()` clamps negative variance and `merge_scalers`
accepts scalers built from stored mean/std. Deferred: a single ENCODER_DEFAULTS
source (touches `encoders/__init__.py`, owned by the perf window right now).

The overfit test on the all-covalent fixture gives model 1.000 vs distance rule 1.000, so
it is a regression floor, not evidence of headroom. Headroom can only be shown where the
rule fails: the hydrogen-bond bond paths found in the recall analysis below, and the
H1/H6 suites.

## Evaluation protocol (what gets reported)

On the main composition split and on each of H1, H3, H6, H7, H8, for
`encoder_fn` in {none, schnet, dimenetpp}:

- candidate recall at 2.0x (upper bound for every model)
- distance rule k=1.1, per-element-pair rule, `GCNBondPred`: precision,
  recall, F1(pos), macro-F1, AUROC, AP
- per-molecule exact-topology rate
- calibration error at the chosen threshold, per suite

Until generator doc 10 lands, any number computed on the May `qtaim.lmdb`
graphs is a development number and is labelled as such.

## Run 2026-09-09: 30 epochs on mo_hydrides (structural-bond labels)

First run beyond the fixture, to check that the loss decreases at real scale.
Data: `qtaim_generator/data/graphs_posz_local/splits/mo_hydrides`
(T1 corpus, `bonding_scheme: structural`, so labels are covalent-radius bonds,
not QTAIM bond paths). Config: `encoder_fn: schnet`, batch 64, bf16-mixed,
lr 1e-3, GPU 0 shared with the perf window's benchmarks.

| epoch | train_loss | val_loss | val macro-F1 |
|-------|-----------|----------|--------------|
| 0 | 0.365 | 0.1160 | 0.958 |
| 3 | 0.0268 | 0.0196 | 0.994 |
| 9 | 0.0113 | 0.0103 | 0.996 |
| 18 | 0.00627 | 0.00698 | 0.997 |
| 29 | 0.00519 | 0.00752 | 0.997 |

Test split (312 graphs, 21,003 atoms, 24,548 bonds, 58,117 candidates at 2.0x):
macro-F1 0.9971, precision 0.9978, recall 0.9956, AUROC 0.99998, AP 0.99997,
calibrated threshold 0.41. Loss decreases monotonically; the plumbing works.

Baseline on the same test split (`qtaim-embed-eval-bond-baselines -config ... --split test`):
candidate recall 1.000 at 2.0x, distance rule best k = 1.30 with macro-F1
**1.0000**, per-element-pair rule also 1.0000. That is expected: the structural
bonding scheme that built these graphs is the rule `d <= 1.3 * (rcov_i + rcov_j)`,
so the label is a deterministic function of the model's own `d / r_ref` input.

Two conclusions:

- The T1 structural-bond corpus is a plumbing test only. It cannot show T3
  headroom, and a model that scores below 1.0 on it is losing to a one-line rule.
  Real T3 numbers wait for graphs built on the regenerated `qtaim.lmdb` (doc 10).
- The 0.3 % residual is a capacity/optimisation gap on a hard step function at
  ratio 1.30, not a data limit: 50 Bessel functions over the 10.4 A default cutoff
  give ~0.2 A resolution, and 30 epochs at a flat 1e-3 learning rate leave the
  boundary soft. Before the QTAIM run, try `pair_rbf_n: 100`, a cosine or plateau
  schedule that actually fires (val_loss still improved at epoch 29), and 100+
  epochs; the ratio scalar alone should let the head represent the rule exactly.

## Finding 2026-09-08: candidate recall at 2.0x is not 1.0

Measured on the 75-molecule organic fixture `tests/data/lmdb_link/train`
(1207 atoms, 1230 QTAIM bond paths):

| pool_multiplier | candidates/atom | positive fraction | recall | missed bond paths |
|-----------------|-----------------|-------------------|--------|-------------------|
| 2.0 | 1.76 | 0.579 | 0.9935 | 8 |
| 2.5 | 2.91 | 0.351 | 0.9959 | 5 |
| 3.0 | 4.03 | 0.253 | 1.0000 | 0 |

All eight misses are O-H (seven) or C-H (one) bond paths at 1.95-2.91 A, i.e.
2.0-2.8x the covalent-radius sum: hydrogen bonds that QTAIM records as bond
paths. Covalent bonds sit at ratio 0.79-1.05 (99th percentile 1.05). So on
organics the 2.0x pool is a 99.3 % recall ceiling and the missed class is
chemically specific. Consequences:

- `pool_multiplier` stays a config knob; the eval script reports recall per
  multiplier so the ceiling is always visible next to the model number.
- The distance rule cannot recover these pairs at any k without destroying
  precision (k >= 2.0 admits every non-bonded near contact), which is where a
  learned scorer has real headroom: an O-H pair at 2.5 A is a bond path or
  not depending on the environment, not the distance.
- Expect H1 (metal-ligand) and H6 (lanthanide) suites to show a lower ceiling
  at 2.0x; measure before training on them (roadmap v4 Section 7).

## Open items carried from doc 05

- Reconcile the 1.4 vs 2.0 `pool_multiplier` constant in
  `qtaim_generator/qtaim_gen/source/analysis/bond_agreement.py` and the paper.
  Not a qtaim_embed change; the model default here is 2.0.
- Whether T3 ships as a model or as the reference table: decided by the
  step 7 number on regenerated graphs.
- Candidate pool blow-up on `pdb_pockets_*`: `candidate_pairs` is chunked so
  memory is bounded, but P per batch should be logged; add
  `max_candidates_per_atom` if needed.
