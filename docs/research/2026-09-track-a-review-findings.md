---
title: "Track A code review findings (2026-09-09, not yet applied)"
status: items 1-10 and 12 applied 2026-09-09 (tests green); 11 and the nice-to-haves open
scope: ResidualBlockDense, bucketing, direct collate, LinearWarmup, neighbors, equivariant TP, qtaim-embed-bench
---

# Track A code review findings

Two review passes (correctness/performance, Python quality) over the uncommitted
Track A changes. Suite state at review time: 323 passed, 1 skipped. Nothing
below has been changed in code.

## Must fix (correctness)

1. APPLIED. The masked path computes mean/var in fp32 and returns the input
   dtype, matching `nn.BatchNorm1d` under autocast (which also emits bf16 with
   fp32 running stats; the reviewer's "reference runs in fp32" was only true
   of the statistics). Test: bf16 masked and index paths vs BatchNorm1d.
2. APPLIED. `BucketBatchSampler.batch_shape` maps any batch to its (merged)
   class shape; `DataLoaderLMDB(dense_shape_of=...)` stamps it as
   `graph.dense_shape`; the models pass it to `to_dense_hetero`. Train samplers
   use `drop_last=True` (`dataset.bucket_drop_last`) so G is static too. Eval
   now always runs the eager dense blocks (exact cuDNN batch norm, no eval
   recompiles), so the compiled shape set equals the train class count. The
   `or True` test is replaced by shapes-subset-of-classes and static-G checks.
3. APPLIED. `BucketBatchSampler(rank, world_size)` shards the epoch's batches
   (equal count per rank; shuffled tail dropped, unshuffled tail repeated),
   `LMDBDataModule` reads rank/world size from `torch.distributed`, the three
   train scripts pass `use_distributed_sampler=not bucketing`.
4. APPLIED. `LinearWarmup` persists base LRs and total steps through the
   callback `state_dict`, reads `initial_lr` when present, and raises on an
   unsized loader. Test: resume after epoch 1 of a 2-epoch warmup continues
   the ramp to the original base LR.
5. APPLIED. `_cast_graph_floats` casts labels to
   `promote_types(dtype, float32)`; features still follow `dataset.dtype`.

## Must fix (bench harness numbers)

6. APPLIED (exact leaf-op keys; `_BATCH_NORM`, `_OPTIMIZER` tuples). Rows in
   `profiling/bench_results/` written before this fix keep the inflated
   per-category columns; throughput columns are unaffected. Original finding:
   profiler categories double count parent and
   child ops: `matmul_ms` includes `aten::matmul` plus its `mm`/`bmm` children,
   `optimizer_ms` counts `Optimizer.step` plus `_fused_adam`, `batch_norm_ms`
   misses `cudnn_batch_norm` and backward, `aten::index.Tensor` never matches.
   samples/s, step ms, memory, GPU util, data wait are unaffected; the
   per-category ms columns in the measurements doc are inflated.
7. APPLIED. `bench_neighbors.py` calls `_radius_neighbors_chunked` directly
   for the chunked arm.

## Should fix

8. APPLIED (`with_valid=False` on the compiled path).
9. APPLIED with item 3: val/test stay bucketed by default (static eval shapes),
   `sampler.indices()` gives the yielded order, `dataset.bucketing_eval: false`
   restores dataset order.
10. APPLIED. The dense block now always has a trainable rel-bias, like the
    sparse reference (GraphConv always builds `lin_rel.bias`).
11. NOT FIXED. `load_from_checkpoint` re-runs `__init__`, whose default fills
    `encoder_tp="channelwise"`, so an hparams-based fallback cannot detect an
    old checkpoint; detection would have to inspect the state_dict for the
    missing `encoder.linears.*` keys in `on_load_checkpoint`. No such
    checkpoints are known to exist (the fully connected encoder OOMed at every
    useful batch), so this is left as documented: pass
    `encoder_tp: "fully_connected"` explicitly when loading one.
12. APPLIED. Docstrings now describe the shipped stamping path and the
    measured 1.4-2.2x; `BucketBatchSampler` has a class docstring and
    `shape_class` / `graph_sizes` / `collate_hetero_direct` have type hints;
    the indentation in `models/utils.py` is fixed; every
    `encoder_max_neighbors` constructor and loader fallback is 16.

## Nice to have

Bench: `_profile_window` unused args, `assert` for config validation, module
import side effects, `data_wait` includes H2D copy when not pinned, GB vs GiB
mix, `rng.choice` on small datasets. Data: size cache keyed by count only,
warn-once on collate fallback, `to_dense_batch` recomputes two syncs the caller
already paid. Datamodule: link/bond `_loader` overrides shadow the base
signature. Profiling scripts: `bench_collate.py` duplicates the shipped
collate, `bench_bucketing.py` lacks the zero-bond floor, `report_track_a.py`
divide by zero on a 0 ms row.

## Verified correct by the reviewers

Dense math vs PyG GraphConv and HeteroConv sum, masking of padded rows,
incidence orientation, MaskedBatchNorm index path vs nn.BatchNorm1d, compiled
reduce-overhead path (fwd/bwd/running stats/eval parity, no output aliasing
across replays), no dynamic shapes in the compiled signature, sampler covers
every index once per epoch, direct collate ptr/batch/edge offsets, checkpoint
loading defaults, non-dense path behaviour unchanged, dense neighbor edge
order identical to the chunked builder.
