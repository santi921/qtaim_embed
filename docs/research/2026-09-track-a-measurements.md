---
title: "Track A measurements: training throughput on >40-atom systems"
date: 2026-09-09
status: living document (tables regenerate from profiling/bench_results with profiling/report_track_a.py)
plan: docs/plans/2026-09-08-feat-performance-engineering-plan.md (A0-A5)
---

# Track A measurements and gate decisions

Every number here comes from `qtaim-embed-bench` or one of the `profiling/bench_*.py`
scripts on one RTX A5000 (24 GB), torch 2.11.0+cu130, torch-geometric 2.7.0,
bf16-mixed unless stated. JSON results live in `profiling/bench_results/`; the
tables in the appendix are rendered from them by `profiling/report_track_a.py`.

Datasets: tm_react shard split (119K train graphs, mean 59 atoms, 75 % over 40
atoms, pos/z on graphs, six charge targets, node level), H7 holdout (18K
graphs, 251-350 atoms), TMQM graph level (48K, no pos/z, the old 25 it/s
baseline). Model unless stated: 8 hetero conv layers as 4 ResidualBlocks,
hidden 128, dropout 0.2; batch norm off in the E1-E6 rows (the pre-2026-09-09
default), on in every row marked `__bn__` and in the A3 tables.

## Summary of decisions

| item | evidence | decision |
|---|---|---|
| A1 batch size | E1: 3,006 to 4,983 samples/s from batch 128 to 256, then flat with 4 workers because data wait rises to 40-53 %; 7,163 at batch 1024 with 8 workers. Accuracy gate (40 epochs, bn_before_activation on): batch 1024 / lr 8e-3 / 1 warmup epoch reaches best val loss 1.466 and test MAE 0.1925 against 1.463 and 0.1938 for batch 128 / lr 1e-3 (0.2 % worse val loss, 0.7 % better test MAE, within the 2 % gate); batch 1024 at the unscaled lr 1e-3 is 3 % worse (1.505, MAE 0.1939) | node default config: batch 1024, lr 8e-3, warmup_epochs 1, 8 workers; graph-level defaults unchanged (no graph-level accuracy run) |
| A1 precision | bf16-mixed within 7 % of fp32 at every batch; the residual stream stays fp32 so gathers move fp32 data | bf16-mixed default (halves activation memory; needed for the dense path where it matters) |
| A2 fused sparse conv | E3: 0.83-0.99x of the reference above batch 128, more memory | dropped |
| A3 dense padded conv | E3: dense + CUDA graphs 1.43-1.68x on tm_react at batch 1024 (31 % padding waste in the test batches), 1.66-2.2x on H7 (6-11 % waste), 4-5x lower peak memory. End to end in the real training step (batch norm on, bucketing, 8 workers): 5,038 to 7,755 samples/s at hidden 128 (1.54x, data-bound), 3,805 to 5,310 at hidden 256 (1.40x), 769 to 979 on H7 at hidden 256 / batch 64 (1.27x); eager dense without compile is 1.0-1.3x | implemented as `conv_fn: "ResidualBlockDense"` with `compiled: true`; bucketed sampler (E4) removes most of the remaining waste |
| A3 bucketing scheme | E4: rounding atoms and bonds up to multiples of 16 gives 11 % waste on tm_react (23 shapes), 2.5 % on H7; quantile buckets are worse because bond counts vary inside an atom bucket | `BucketBatchSampler` with grid 16 |
| A4 collate | E5: direct collate 3.6-4.1x faster than `Batch.from_data_list` (163 to 40 ms per 1024 graphs) | implemented in `DataLoaderLMDB` |
| A4 tensor-dict serialization | E5: `weights_only=True` load of a flat tensor dict is 0.45x the speed of the pickled HeteroData (0.77 vs 0.35 ms per record) | not adopted; security-only argument remains (todos/005) |
| A5 neighbor build | E6: per-molecule dense cdist exact and 65x (batch 512) to 180x (batch 1024) faster, one sync instead of 114-235 | implemented, dispatched automatically for batched inputs |
| A5 DimeNet++ | E2: cutoff 5 / cap 32 needs 9.4 GB at batch 128 and OOMs at 512; cutoff 4 / cap 16 is 1.8x faster at 3.6 GB. Accuracy (40 epochs, tm_react, batch 128, bn_first): cutoff 4 / cap 16 reaches test MAE 0.1779 and R2 0.917, the best tm_react model so far (no encoder: 0.1938); cutoff 5 / cap 32 reaches 0.1947 with three eval-mode spikes and is 2.3x slower | `encoder_max_neighbors` default 16; cutoff 4.0 for dimenetpp |
| A5 equivariant | E2: fully connected per-edge tensor product OOMs at hidden 64 / batch 128; 16,384 weights per edge. A5-2: hidden 64 / batch 128 runs at 1,261 samples/s and 2.3 GB (was OOM); at hidden 32 it is 0.88x the fully connected speed at equal memory | channel-wise `uvu` tensor product default (256 weights per edge); fully connected kept behind `encoder_tp` |
| E1 accuracy (40 epochs, tm_react) | batch 128 / lr 1e-3 with batch norm on: val R2 0.72 at epochs 1-2, then eval-mode val MSE diverges (1e3 at epoch 3, 4e11 by epoch 23) while train MSE stalls at 0.64. Cause: post-ReLU batch norm channels collapse their running variance. With `bn_before_activation: true` the same recipe trains 40 epochs cleanly to val R2 0.87, and so do both batch-1024 arms (see A1) | `bn_before_activation` is the config default since 2026-09-09; the batch-size gate passed with it on |

## Correctness finding: batch_norm must be on (blocks the E1 accuracy check)

The first E1 accuracy run (batch 128, lr 1e-3, `batch_norm: False` as in every
default config) reported a train loss of 2.5e14 in epoch 0 and then sat at the
label variance (val MSE 0.975, val R2 0.000) for the rest of training. A
300-step diagnostic on cached tm_react batches reproduces it at lr 1e-3, 3e-4,
and 1e-4, with and without clamping the 11 feature columns whose values reach
157: the first-step MSE is 1e16 and the model then predicts the mean. Per-block
activation maxima at initialization are 1e3 after block 0, 8e4 after block 1,
1e7 after block 2, 2e9 after block 3: the sum aggregation into the single
global node (about 60 atoms plus 60 bonds per molecule) and the residual adds
compound across blocks because PyG's `GraphConv` has no degree normalization,
unlike DGL's `norm="both"` which the `norm` config key still names. With
`batch_norm: True` the same model reaches a held-out MSE of 0.30 in 300 steps
(0.27 with lr 1e-2).

Decision: `batch_norm` defaults to True in all default configs and every bench
and training config in `profiling/`; `ResidualBlockDense` uses `MaskedBatchNorm`
with matching statistics (parity tested). The E1 accuracy runs and the A3
end-to-end throughput rows use batch norm on. Accuracy claims from PyG-era runs
that used batch norm off should be treated as unverified.

### Second failure with batch norm on: eval-mode divergence from epoch 3

The rerun of E1 (batch 128, lr 1e-3, `batch_norm: True`, gradient clip 5.0,
`profiling/train_configs/tm_react_b128_lr0.001.json`) trains normally for two
epochs (val MSE 0.28, val R2 0.72 at epoch 1) and then the validation loss
explodes: 1e3 at epoch 3, 4e11 by epoch 23, while the train MSE stalls at
0.64 for the rest of the run. Diagnostics on the epoch-1 checkpoint
(`profiling/train_runs/b128_lr0.001/`):

- eval-mode batch norm and batch-statistics batch norm give the same held-out
  MSE at epoch 1 (0.36 vs 0.35), so the running statistics are still healthy
  at that point;
- the `global` node type's batch norm running variance is 400-500 after
  epoch 1 versus 1-50 for atoms and bonds, because the global node receives an
  unnormalized sum over all atoms and bonds of the molecule (60 + 60 terms on
  tm_react, up to 350 + 370 on H7);
- weights are small; the divergence is in the running statistics, not the
  parameters.

Diagnosed 2026-09-09 (`docs/research/2026-09-tm-react-eval-divergence.md`,
scripts in `profiling/divergence/`): the weights do not diverge (batch-statistic
validation MSE improves monotonically 0.368 to 0.358 over the two continued
epochs, gradient norms 0.2-0.3, train loss flat). What diverges is BatchNorm's
running variance on ReLU channels that are inactive for every global node of a
batch: `GraphConvDropoutBatch` normalizes after the activation, a channel dead
for a whole batch has batch variance exactly zero, and `running_var` decays as
0.9 per step to the float32 floor within an epoch (7 such channels at epoch 1,
47 after one more epoch, 129 after two, all on the `*->global` relations plus
one atom channel). In eval mode those channels apply a gain of 100-260, and
the rare held-out molecule that activates one (1-16 of 6,400) is amplified
through the residual and the g2a/g2b edges; hence the erratic per-epoch val
values (3.5, 0.38, 4e4, 38 at consecutive probes). Resetting the collapsed
channels or raising BN eps to 1e-2 on a diverged state restores val MSE 0.335
with no weight change. The unnormalized a2g/b2g sums (running_var up to
1.7e4 in block 3, dominated by rare-element one-hot columns scaled to 100-160
rather than by molecule size) supply the amplifier's input: with mean
aggregation on the global relations the same dead channels only see O(1)
inputs and the run stays at 0.337-0.344. Gradient clipping is irrelevant
(norms never reach it); lr 3e-4 only delays it; BN momentum 0.01 slows the
collapse ten-fold. Both fixes are implemented as opt-in flags (`model.bn_before_activation`,
`model.global_aggr: "mean"`, sparse and dense twins with parity tests) and
were trained from scratch with the reference recipe
(`profiling/run_divergence_fix.sh`). `bn_before_activation` (with the final
prediction layer made linear) trains 40 epochs without any eval-mode
excursion and reaches val R2 0.87 / RMSE 0.41 against the reference's 0.72 /
0.55 at its best epoch before diverging; test R2 0.94 / 0.97 / 0.97 / 0.87 on
ADCH / CM5 / Loewdin / Mulliken. Mean global aggregation on top changes
nothing (0.870 vs 0.867), and on its own it still diverges (from epoch 3
instead of 2). The flag is the config default since 2026-09-09.

E1 accuracy gate with it on (40 epochs each, `profiling/run_divergence_fix.sh`
with the `tm_react_b1024_*` configs, runs under `profiling/train_runs/`):

| arm | best val loss (epoch) | val R2 / RMSE at epoch 39 | test MAE | test RMSE | test R2 ADCH / CM5 / Loewdin / Mulliken |
|---|---|---|---|---|---|
| batch 128, lr 1e-3 (`b128_bnfirst`) | 1.463 (30) | 0.867 / 0.413 | 0.1938 | 0.419 | 0.939 / 0.973 / 0.968 / 0.870 |
| batch 1024, lr 8e-3, 1 warmup epoch | 1.466 (39) | 0.863 / 0.422 | 0.1925 | 0.418 | 0.940 / 0.972 / 0.975 / 0.868 |
| batch 1024, lr 1e-3 | 1.505 (39) | 0.853 / 0.422 | 0.1939 | 0.424 | 0.936 / 0.973 / 0.975 / 0.855 |

Linear LR scaling with one warmup epoch keeps batch 1024 within 0.2 % of the
batch-128 val loss (and 0.7 % better test MAE), so the gate passes; the
unscaled lr is 3 % worse and still improving at epoch 39. Batch 1024 needs
about 20 s per epoch against 65 s at batch 128 on the same GPU (raw-step
numbers are in E1 and A3). Adopted in the node-level default config: batch
1024, lr 8e-3, `warmup_epochs: 1`, 8 workers. Small datasets should go back
to 128 / 1e-3 / no warmup.

Also found there: the tm_react shard LMDBs carry no QTAIM `extra_feat_*`
columns despite `dataset.extra_keys` in the configs (atom features are degree,
H count, ring flags and 83 element one-hots), which bears on the training
plateau at RMSE 0.64 but not on the divergence.

## What the probe changed about the plan's diagnosis

The plan assumed a launch-bound model. On 60-atom molecules that is true only
at batch 128: step time then scales linearly with atoms (tm_react at batch 1024
and H7 at batch 128 both give about 585K atoms/s with cached batches), and the
device-time breakdown at batch 512 is 22 ms of index_select / index_add against
9 ms of matmul in a 56 ms step. Two consequences:

1. Batch size alone buys 1.4x with cached batches and nothing beyond batch 256
   with 4 DataLoader workers; the data path (0.45 ms deserialization plus 95 ms
   collate per 1024 graphs, per worker) becomes the limiter first.
2. Removing gather / scatter traffic matters more than fusing matmuls. The dense
   padded formulation does that and also delivers the static shapes that CUDA
   graphs need; the fused-sparse formulation keeps the traffic and loses.

## E1: batch, precision, workers

Raw mode (plain fwd/bwd/step loop over the real DataLoader, Lightning
`training_step` including metric updates), 100 warmup + 500 measured steps.

- tm_react, hidden 128, 4 workers: batch 128 gives 3,006-3,408 samples/s; 256
  gives 4,983; 512 / 1024 / 2048 stay at 4,650-4,870 with the data-wait fraction
  at 0.40 / 0.47 / 0.53. GPU utilization 50-67 %.
- 8 workers at batch 1024: 7,163 samples/s, data wait 0.10, GPU 84 %. 16 workers:
  6,815 (no further gain; wait 0.14). pin_memory off: wait 0.66 at 4 workers.
- hidden 256 at batch 1024, 4 workers: 4,281 samples/s at 90 % GPU utilization,
  gather+scatter 120 ms against matmul 47 ms per step. This is the regime the
  dense conv addresses.
- fp32 vs bf16-mixed: 4,865 vs 4,823 at batch 1024. Kernels per step rise from
  2,700 to 3,400 under autocast (cast kernels) while the fp32 residual stream
  keeps the gather traffic unchanged.
- TMQM graph level, Lightning mode, batch 128, 4 workers: 29.1 it/s (baseline
  25.6 it/s reproduced; pin_memory and bf16-mixed account for the difference).

Direct-collate rerun (after A4) and the E1 accuracy comparison (batch 128 at
lr 1e-3 vs batch 1024 at lr 8e-3 with one warmup epoch vs batch 1024 at lr
1e-3, 40 epochs, tm_react validation MAE) are recorded in the appendix as they
complete.

## E2: encoder table

Before the dense neighbor build (the A5-1 table below shows the after state):

- schnet: 2,303 samples/s at batch 128 and 2,652 at 512 (vs 3,006 / 4,784 with
  no encoder). At batch 512 the neighbor build alone was 84 ms of a 193 ms step
  with 220 device syncs.
- dimenetpp: cutoff 5 / cap 32 gives 695 samples/s at 9.4 GB (batch 128) and
  OOMs at 512; cutoff 4 / cap 16 gives 1,226 at 3.6 GB and 1,293 at 13.6 GB at
  batch 512. Encoder share 37-123 ms; triplets (sum of deg^2) dominate.
- equivariant (fully connected, hidden 32, lmax 1): 574 samples/s at 9.4 GB for
  batch 128; the per-edge radial MLP output (4,096 weights per edge) is 70 ms of
  matmul per step. hidden 64 OOMs at 23 GB. lmax 2 at batch 32: 113 samples/s.
- H7 (288 atoms): none 1,341 samples/s at batch 64; schnet 556; dimenetpp
  (cutoff 4 / cap 16) 254 at 9.4 GB.

## E3: conv-stack formulations

`profiling/bench_layers.py`: fwd + bwd of 8 hetero conv layers with identical
weights (fp32 parity asserted, TF32 off, before timing), real batches formed
from graphs of similar size, dense shape rounded to 16. bf16 rows:

| dataset | batch | hidden | ref ms | fused | dense eager | dense compile | dense + CUDA graphs | ref peak GB | CUDA-graph peak GB | pad waste |
|---|---|---|---|---|---|---|---|---|---|---|
| tm_react | 128 | 128 | 24.1 | 1.32x | 1.46x | 2.21x | 2.41x | 0.55 | 0.15 | 0.30 |
| tm_react | 128 | 256 | 38.6 | 1.14x | 1.26x | 1.91x | 1.98x | 1.13 | 0.40 | 0.30 |
| tm_react | 512 | 256 | 109 | 0.86x | 0.94x | 1.43x | 1.49x | 3.80 | 0.89 | 0.32 |
| tm_react | 1024 | 128 | 110 | 0.87x | 0.99x | 1.56x | 1.58x | 3.77 | 0.78 | 0.31 |
| tm_react | 1024 | 256 | 211 | 0.83x | 0.91x | 1.40x | 1.43x | 7.47 | 1.56 | 0.31 |
| tm_react | 1024 | 512 | 631 | OOM | 1.05x | 1.51x | 1.68x | 15.1 | 3.29 | 0.31 |
| H7 | 32 | 256 | 38.0 | 0.99x | 1.40x | 2.15x | 2.20x | 1.23 | 0.41 | 0.06 |
| H7 | 128 | 256 | 123 | 0.85x | 1.18x | 1.80x | 1.87x | 4.38 | 0.98 | 0.11 |
| H7 | 128 | 512 | 280 | 0.86x | 1.11x | 1.60x | 1.66x | 8.92 | 2.28 | 0.11 |

fp32 rows gain less (1.05-1.65x for dense + CUDA graphs) because the bmm on the
incidence matrix is then a full-precision GEMM; bf16 is the intended mode. The
tm_react test batches carry 31 % padding because the middle slice of a sorted
3x pool spans 45-75 atoms; the bucketed sampler brings that to 11 %, which is
worth another 1.3x of useful work per step on top of the ratios above. Kernel
count per fwd+bwd drops from 2,600 to about 600.

## E4: padding waste of static-shape schemes

Padding waste = fraction of padded atom+bond rows that are padding, on 6,000
sampled graphs per dataset.

| scheme | tm_react | H7 | TMQM | shapes (tm_react) |
|---|---|---|---|---|
| single shape (dataset max) | 0.44 | 0.21 | 0.42 | 1 |
| 4 quantile buckets on atoms | 0.22 | 0.09 | 0.23 | 4 |
| 8 quantile buckets on atoms | 0.16 | 0.06 | 0.18 | 8 |
| round atoms and bonds up to 32 | 0.20 | 0.05 | 0.21 | 10 |
| round atoms and bonds up to 16 | 0.11 | 0.025 | 0.11 | 23 |
| round atoms and bonds up to 8 | 0.055 | 0.012 | 0.055 | 54 |
| sorted batches padded to batch max (dynamic shapes), batch 512 | 0.14 | 0.055 | 0.17 | n/a |
| random batches padded to batch max, batch 512 | 0.43 | 0.19 | 0.39 | n/a |

Quantile buckets on atoms leave bond waste at 0.2-0.3 because bond counts vary
inside an atom bucket; the 2D grid at 16 meets the 15 % target with a shape
count that CUDA graphs handle (one recording per shape). `BucketBatchSampler`
merges classes smaller than half a batch into the next larger class so ragged
remainder batches stay rare.

## E5: data path

One CPU thread (what a DataLoader worker pays), tm_react:

| batch | `Batch.from_data_list` | direct collate | speedup |
|---|---|---|---|
| 128 | 14.2 ms | 3.7 ms | 3.8x |
| 512 | 67.4 ms | 18.8 ms | 3.6x |
| 1024 | 163 ms | 40.0 ms | 4.1x |

Deserialization per record: pickled HeteroData via `torch.load` 0.345 ms
(54 KB); flat tensor dict via `torch.load(weights_only=True)` 0.768 ms (53 KB).
The restricted unpickler is slower per tensor, so the serialization change
brings no speed and is not adopted here. LMDB `get` itself is 0.56 ms per record
cold, so with 8 workers deserialization sustains about 15K graphs/s.

## E6: radius-graph build

Edge sets asserted identical against the chunked builder and brute force.

| dataset | batch | atoms | edges (cutoff 5) | chunked GPU | dense GPU | speedup | nonzero calls chunked / dense |
|---|---|---|---|---|---|---|---|
| tm_react | 128 | 7,285 | 167K | 6.6 ms | 0.60 ms | 11x | 29 / 1 |
| tm_react | 512 | 29,110 | 680K | 65 ms | 1.0 ms | 65x | 114 / 1 |
| tm_react | 1024 | 60,138 | 1.42M | 255 ms | 1.6 ms | 160x | 235 / 1 |
| H7 | 32 | 9,405 | 310K | 9.9 ms | 0.76 ms | 13x | 37 / 1 |

The chunked builder compared every atom against the whole batch (N^2 distance
pairs, 880M at batch 512); the dense builder compares within each molecule's
padded block (G x N_b^2, 5M). On one CPU thread the dense variant is 33-72 ms
per batch versus 2.7-16 s, so building neighbors in DataLoader workers is
possible but not needed. DimeNet++ triplets at cap 32: 15.7M for batch 512 at
cutoff 5 (10 ms), 6.9M at cutoff 4.

## A5-1: dense neighbor build in the real step

Same configs re-measured after `radius_neighbors` switched to the per-molecule
dense block:

| run | samples/s before | after | neighbor ms before | after | syncs before | after |
|---|---|---|---|---|---|---|
| tm_react schnet batch 128 | 2,303 | 2,538 | 8.7 | 1.6 | 132 | 111 |
| tm_react schnet batch 512 | 2,652 | 4,182 | 84 | 1.9 | 220 | 111 |
| tm_react dimenetpp (4.0 / 16) batch 512 | 1,293 | 1,583 | 79 | 2.2 | 236 | 128 |
| H7 schnet batch 16 | 346 | 337 | 5.0 | 1.5 | 121 | 111 |

The remaining 111 syncs per step come from torchmetrics updates in
`training_step` and the bincount-based shape reads, not from the neighbor build.

## A5-2: channel-wise equivariant tensor product

Weights per edge at lmax 1: 256 (hidden 64) instead of 16,384; at lmax 2: 704
instead of 45,056. Full-step rows on tm_react (raw mode, 4 workers). The fully connected
column is measured with the A5-1 dense neighbor build already in place, so the
two columns differ only in the tensor product:

| lmax | hidden | batch | fully connected samples/s | peak GB | channel-wise samples/s | peak GB | encoder ms (cw) |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 128 | 2,017 (574 before A5-1, 9.43 GB) | 1.30 | 1,767 | 1.42 | 11.9 |
| 1 | 64 | 128 | OOM at 23 GB | | 1,261 | 2.29 | 22.7 |
| 1 | 64 | 512 | OOM | | 1,173 | 8.42 | 106 |
| 2 | 32 | 128 | OOM (113 samples/s at batch 32, 7.2 GB) | | 676 | 3.34 | 49.7 |

At hidden 32 the channel-wise product is 0.88x the speed of the fully connected
one (the extra `o3.Linear` mix costs more than the smaller per-edge weight
saves at this width); its value is the memory scaling. The hidden 64 / lmax 1
row is the configuration the plan wanted to train and it now fits at batch 128
with 2.3 GB where the fully connected version needs 16 GB of per-edge weights
alone. Batch 512 does not help (the encoder is
compute-bound at 98 % GPU utilization), so batch 128-256 is the working point
for the equivariant encoder. The rotation tests in `tests/test_equivariance.py`
cover both `encoder_tp` modes; no accuracy comparison between the two has been
run.

## A3: dense conv stack end to end (training step, batch norm on)

`qtaim-embed-bench` raw mode, tm_react batch 1024 with 8 workers and H7 batch
64 with 4 workers, `dataset.bucketing: true` for the dense rows (12 shape
classes at batch 1024 after merging classes smaller than half a batch, 17 %
padding), compiled rows measured after one full warmup epoch so no
recompilation lands in the measured window (`--warmup_epoch`):

| dataset | hidden | conv_fn | compiled | samples/s | vs ref | stream step ms | peak GB | GPU util % | data wait | kernels/step |
|---|---|---|---|---|---|---|---|---|---|---|
| tm_react | 128 | ResidualBlock | no | 5,038 | 1.00x | 199 | 3.64 | 94 | 0.08 | 3,909 |
| tm_react | 128 | ResidualBlockDense | no | 6,745 | 1.34x | 132 | 7.36 | 88 | 0.08 | 2,659 |
| tm_react | 128 | ResidualBlockDense | CUDA graphs | 7,755 | 1.54x | 94 | 4.40 | 63 | 0.26 | 2,056 |
| tm_react | 256 | ResidualBlock | no | 3,805 | 1.00x | 265 | 7.03 | 96 | 0.06 | 3,909 |
| tm_react | 256 | ResidualBlockDense | no | 3,692 | 0.97x | 253 | 14.6 | 94 | 0.04 | 2,653 |
| tm_react | 256 | ResidualBlockDense | CUDA graphs | 5,310 | 1.40x | 165 | 8.61 | 84 | 0.10 | 2,047 |
| tm_react | 128 + schnet | ResidualBlockDense | CUDA graphs (conv stack only) | 4,501 | 1.08x vs 4,182 sparse schnet | 199 | 12.2 | 82 | 0.09 | 2,424 |
| H7 | 256 | ResidualBlock | no | 769 | 1.00x | 81 | 2.17 | 93 | 0.02 | 3,641 |
| H7 | 256 | ResidualBlockDense | no | 797 | 1.04x | 76 | 2.95 | 89 | 0.02 | 2,423 |
| H7 | 256 | ResidualBlockDense | CUDA graphs | 979 | 1.27x | 60 | 1.82 | 71 | 0.05 | 1,778 |

Reading the table:

- The compiled dense path is the only variant that beats the reference at
  every size. At hidden 128 it is data-bound: the GPU stream needs 94 ms per
  batch (10,900 samples/s of capacity) but 8 workers deliver a batch every
  126 ms, so 26 % of the loop is spent waiting and GPU utilization is 63 %.
  More workers or the dropped tensor-dict serialization would be needed to
  realize the rest.
- Eager dense (no compile) is not worth using at hidden 256: the padded bmm
  and masked batch norm run as separate kernels and the padding (17 %) is paid
  in full, so it ties the reference at twice the memory. Its memory is 7.4 GB
  at hidden 128 after the cuDNN index-path batch norm (13.3 GB before).
- The compiled rows use less memory than eager dense (4.4 vs 7.4 GB) because
  inductor fuses the mask multiplies and batch norm into the bmm epilogues;
  they still use more than the sparse reference at hidden 128-256 because the
  padded activations are stored per shape class.
- The end-to-end ratios (1.27-1.54x) are below the E3 microbenchmark ratios
  (1.4-2.2x) because the conv stack is 55-65 % of the step; heads, pooling,
  metrics, optimizer, and the data path are unchanged. The remaining
  121 syncs per step are torchmetrics updates and the `ptr`-based shape reads.
- Steady-state check in Lightning-free epochs (`profiling/bench_results/dense_steady_h128.log`):
  compile epoch 0 costs 220 s (12 shape classes, one graph each) and epoch 1
  runs at 7,486 samples/s; CUDA graphs epoch 0 costs 53 s and epoch 1 runs at
  7,524 samples/s at 4.5 GB. Recompilation is bounded by the number of shape
  classes, which `BucketBatchSampler` fixes per dataset.

## A4: direct collate in the real step

`collate_hetero_direct` replaces `Batch.from_data_list` in `DataLoaderLMDB`
(4x faster in isolation, E5). In the training step the gain depends on how
far the data path is from the critical path:

| batch | workers | hidden | samples/s before | after | data wait before | after |
|---|---|---|---|---|---|---|
| 512 | 4 | 128 | 4,784 | 5,621 | 0.40 | 0.33 |
| 1024 | 4 | 128 | 4,823 | 5,090 | 0.47 | 0.49 |
| 1024 | 8 | 128 | 7,163 | 7,369 | 0.10 | 0.12 |
| 1024 | 8 | 256 | 4,281 | 4,337 | 0.06 | 0.06 |

With 4 workers the loop is data-bound before and after (collate was 40 % of a
worker's time per batch, deserialization the rest at 0.45-0.54 ms per graph),
so the collate change alone buys 1.06-1.17x. With 8 workers the model is the
bottleneck and the gain is 1.02-1.03x. The measurement that matters is the
data-wait floor: at batch 1024 a worker now needs about 590 ms per batch
(1024 x 0.54 ms deserialization + 40 ms collate), so 8 workers sustain one
batch per 74 ms, which is enough for the reference model (199 ms per step)
and just short of the compiled dense model (94 ms per step, see A3).

## Defaults and batching audit against the plan (2026-09-09)

Checked after the Track A snapshot (`d753264`) and the trainer refactor
(`3bdaa5f`):

| plan item | state | note |
|---|---|---|
| A1 `precision: "bf16-mixed"` in all four default configs | done | node, graph, link, bond |
| A1 `torch.set_float32_matmul_precision("high")` at script import | done | all ten training and bayes-opt scripts, link included |
| A1 `pin_memory: True`, 8 workers in the optim blocks | done | `dataset.num_workers` (pickle datamodules) stays 1: workers hurt on in-memory datasets |
| A1 batch / lr from the accuracy sweep | node: 1024 / 8e-3 / warmup 1 (gate passed); graph: 128 / 1e-3 (TMQM gate failed for 1024) | TMQM graph level, 100 epochs, `bn_before_activation` on: batch 128 / lr 1e-3 reaches test MAE 0.1211 (R2 0.969, best val loss 0.0229 at epoch 93); batch 1024 / lr 8e-3 / 1 warmup epoch reaches 0.1561 (R2 0.955, best val 0.0369 at epoch 76), 29 % worse and still improving: 48K graphs give only 48 steps per epoch at batch 1024. Graph default set to 128 / 1e-3 (was the test leftover 2 / 1e-2) |
| A1 warmup wired into every Trainer | done | `build_trainer` adds `LinearWarmup` from `optim.warmup_epochs` |
| A3 bucketing opt-in, paired with the dense path | done | round-up-16 shape classes instead of the plan's 8 total-size buckets (E4: 11 % waste vs the 15 % target) |
| A3 "no accuracy change vs unbucketed at equal epochs" | FAILED as configured | `tm_react_b1024_dense_compiled` (ResidualBlockDense, bucketing, CUDA graphs, same batch / lr / warmup) reaches best val loss 1.589 vs 1.466 and test MAE 0.2189 vs 0.1925 (14 % worse; test loss 2.04 vs 1.51, ADCH test R2 0.71 vs 0.94). Confirmed to be the bucketing, not the dense math: `ResidualBlock` + bucketing at the same recipe lands at best val 1.580 and test MAE 0.2146 (11 % worse), the dense path adds 2 % on top. Not a batch-norm train/eval effect: on the dense checkpoint eval-mode running statistics and batch statistics give the same test MSE (0.273 vs 0.268), bucketed and plain test loaders too. Class-homogeneous batches (one size class per optimizer step) simply optimize worse. Mitigation measured: batch 256 per class with `accumulate_grad_batches: 4` (four classes per optimizer step, static shapes) reaches best val 1.550 and test MAE 0.2014, recovering about half of the gap (4.6 % worse than unbucketed instead of 14 %), still outside the 2 % gate. Bucketing therefore stays opt-in and is a throughput tool for exploration, not for a final model, until batches can mix classes fully. Found on the way: CUDA graphs (`compile_mode: "reduce-overhead"`) cannot be used with gradient accumulation, the accumulated `.grad` tensors are outputs of the compiled backward graph and the next replay overwrites them before AccumulateGrad reads them (reproduced outside Lightning); `compile_mode: "default"` (plain inductor, same steady-state throughput within 1 %) is required and `build_trainer` now raises on the bad combination |
| A3 GPU utilization >= 70 % at batch 1024 | tm_react only | 84 % at hidden 256, 63 % at hidden 128 (data-bound); TMQM not measured |
| A4 direct collate; tensor-dict serialization | done; rejected by gate | |
| A6 DDP + bucketing | smoke passed (2 GPUs, 2 epochs + test, `TORCH_DISTRIBUTED_DEBUG=DETAIL`) | first attempt deadlocked: the rank-0 progress bar all-reduced the `sync_dist=True` epoch loss before the model's epoch-end hook while rank 1 was already in torchmetrics' all-gather (collective mismatch ALLREDUCE vs ALLGATHER at the same sequence number). The three GCN models now log the progress-bar loss rank-local; the synced metrics come from torchmetrics. The 1.8x throughput acceptance is not measured |
| `batch_norm` / `bn_before_activation` | both True everywhere | constructors keep `bn_before_activation=False` for old checkpoints |

Inconsistencies found, not changed (need a decision):

- Two batch / worker sources. LMDB datamodules read `optim.train_batch_size`
  and `optim.num_workers`; the pickle datamodules read
  `dataset.train_batch_size` (128 graph, 512 node) and `dataset.num_workers`
  (1). The `--num_workers` CLI flag now overrides both when given (it used
  to set only the dataset one, and unconditionally, with a default of 1).
- `gradient_clip_val` defaults to 5.0 in all configs (E1 set it after the
  unclipped run diverged); the bench configs use 0.0. CLAUDE.md now shows 5.0.
- Bench configs inherit `bn_before_activation: True` from the defaults, so
  bench rows recorded from here on run BN-first; every row in this document
  was measured with the original order (same kernels, one fewer fusion).

## A5-3: DimeNet++ cutoff and neighbor cap, accuracy

Reference recipe (batch 128, lr 1e-3, 40 epochs, `bn_before_activation`),
`encoder_fn: dimenetpp`, `encoder_hidden` 64, both arms sharing a GPU with
another run:

| arm | best val loss (epoch) | test MAE | test RMSE | test R2 ADCH / CM5 / Loewdin / Mulliken | s per epoch (shared GPU) |
|---|---|---|---|---|---|
| no encoder (`b128_bnfirst`) | 1.463 (30) | 0.1938 | 0.419 | 0.939 / 0.973 / 0.968 / 0.870 | 65 (alone) |
| dimenetpp cutoff 4.0, cap 16 | 1.368 (32) | 0.1779 | 0.397 | 0.940 / - / 0.977 / 0.917 | 115 |
| dimenetpp cutoff 5.0, cap 32 | 1.413 (29) | 0.1947 | 0.419 | 0.921 / - / 0.973 / 0.899 | 275 (180 alone) |

The cheaper setting is both 2.3x faster and more accurate (test MAE 0.1779
vs 0.1947; the cutoff-5 model is no better than no encoder at all on the
final-epoch weights Lightning tests with); the E2 memory argument (cutoff 5 /
cap 32 needs 9.4 GB at batch 128) and this accuracy result agree, so cutoff
4.0 / cap 16 is the DimeNet++ setting. The cutoff-5 trace also had three
eval-mode spikes (val loss 3.57, 5.23, 2.66 at epochs 21, 35, 36, back to
1.42-1.51 the following epoch) while cutoff 4 had none; the encoder's own
layers sit before the batch-norm-first conv stack and are not protected by it,
so a denser radius graph seems to make the eval pass more fragile. Not
investigated further.

## Acceptance against the plan

- A0: `qtaim-embed-bench --config profiling/bench_configs/tmqm_graph.json --mode lightning`
  gives 29.1 it/s (target 25 +/- 3 with the old settings; bf16-mixed and pinned
  memory explain the gain). Raw mode reproduces the cached-batch probe within
  the data-wait fraction.
- A1: 7,163 samples/s at batch 1024 / 8 workers versus 3,006-3,408 at batch 128,
  a 2.1-2.4x on throughput before A3; the accuracy half of the gate is blocked
  by the batch-norm divergence described above.
- A3: 1.54x end to end at hidden 128 (5,038 to 7,755 samples/s, data-bound
  at 8 workers), 1.40x at hidden 256, 1.27x on H7; the plan's 1.5x gate was
  set on the E3 microbenchmark, which passed at 1.43-2.2x. Combined with A1,
  tm_react hidden 128 goes from 3,006 samples/s (batch 128, fp32, 4 workers,
  no batch norm) to 7,755 (batch 1024, bf16, 8 workers, dense + CUDA graphs,
  batch norm), 2.6x, against the plan's 3x target.
- A4: direct collate 4x; tensor-dict serialization rejected by its gate.
- A5: schnet at batch 512 is 0.87x of no-encoder throughput (target 0.6x) and
  the neighbor build is under 2 % of step time (target 10 %); dimenetpp at
  cutoff 4 / cap 16 is 0.33x (target 0.3x).

## Appendix: generated tables

Regenerate with `python profiling/report_track_a.py > /tmp/tables.md` after new
runs; the sections below are that output as of 2026-09-09 (after the A3 and
A5-2 reruns). Caveat from the code review
(`docs/research/2026-09-track-a-review-findings.md`): the per-category
device-time columns (gather+scatter ms, matmul ms, batch norm ms) in these
rows were produced before the profiler categories were switched to exact
leaf-op keys and double count nested ops (`aten::matmul` with its `mm`
children, `Optimizer.step` with `_fused_adam_`); samples/s, step ms, memory,
GPU utilization, data wait, kernels and syncs per step are unaffected.

### E1 batch, precision, workers (tm_react node, hidden 128/256, encoder none, raw mode)

| run | batch | workers | precision | samples/s | atoms/s | step ms | data wait | GPU util % | peak GB | kernels/step | gather+scatter ms | matmul ms | collate ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tm_react__none__h128__b128__fp32 | 128 | 4 | 32 | 3,408 | 198,452 | 37.6 | 0.0144 | 56.6 | 0.569 | 2,647 | 7.11 | 6.73 | 12.6 |
| tm_react__none__h128__b128__bf16 | 128 | 4 | bf16-mixed | 3,006 | 174,910 | 42.6 | 0.0131 | 48.1 | 0.432 | 3,335 | 7.7 | 3.87 | 12.9 |
| tm_react__none__h128__b256__bf16 | 256 | 4 | bf16-mixed | 4,983 | 289,975 | 51.4 | 0.115 | 66.8 | 0.777 | 3,336 | 15.1 | 5.69 | 24 |
| tm_react__none__h128__b512__fp32 | 512 | 4 | 32 | 4,969 | 289,229 | 103 | 0.349 | 64.8 | 2.11 | 2,695 | 26.9 | 18.5 | 47.9 |
| tm_react__none__h128__b512__bf16 | 512 | 4 | bf16-mixed | 4,784 | 278,493 | 107 | 0.401 | 61.1 | 1.49 | 3,336 | 29.8 | 10.6 | 48.9 |
| tm_react__none__h128__b512__bf16__w4__direct_collate | 512 | 4 | bf16-mixed | 5,621 | 327,228 | 90.9 | 0.327 | 59.6 | 1.49 | 3,324 | 29.7 | 10.7 | 49.1 |
| tm_react__none__h128__b1024__fp32 | 1024 | 4 | 32 | 4,865 | 283,271 | 209 | 0.447 | 61.9 | 4 | 2,738 | 53.2 | 34.6 | 108 |
| tm_react__none__h128__b1024__bf16 | 1024 | 4 | bf16-mixed | 4,823 | 280,824 | 211 | 0.472 | 53.3 | 2.91 | 3,402 | 57.1 | 19 | 94.7 |
| tm_react__none__h128__b1024__bf16__nopin | 1024 | 4 | bf16-mixed | 4,831 | 281,297 | 210 | 0.662 | 52.2 | 2.84 | 3,402 | 58.6 | 19 | 97.4 |
| tm_react__none__h128__b1024__bf16__w4__direct_collate | 1024 | 4 | bf16-mixed | 5,090 | 296,385 | 200 | 0.487 | 57.8 | 2.9 | 3,390 | 59 | 19.6 | 112 |
| tm_react__none__h128__b1024__bf16__w8 | 1024 | 8 | bf16-mixed | 7,163 | 417,104 | 142 | 0.1 | 84.4 | 2.91 | 3,402 | 57.2 | 18.7 | 95 |
| tm_react__none__h128__b1024__bf16__w8__direct_collate | 1024 | 8 | bf16-mixed | 7,369 | 429,150 | 138 | 0.117 | 88.5 | 2.91 | 3,390 | 58.2 | 19.9 | 98.2 |
| tm_react__none__h128__b1024__bf16__w16 | 1024 | 16 | bf16-mixed | 6,815 | 396,822 | 149 | 0.139 | 97.1 | 2.9 | 3,402 | 57.8 | 19.6 | 95.5 |
| tm_react__none__h128__b2048__bf16 | 2048 | 4 | bf16-mixed | 4,651 | 270,794 | 433 | 0.531 | 50.9 | 5.61 | 3,414 | 121 | 37.7 | 208 |
| tm_react__none__h256__b128__bf16 | 128 | 4 | bf16-mixed | 2,984 | 173,767 | 42.9 | 0.0189 | 83.3 | 0.891 | 3,377 | 14.4 | 6.96 | 13 |
| tm_react__none__h256__b256__bf16 | 256 | 4 | bf16-mixed | 3,482 | 202,699 | 73.5 | 0.0266 | 90.1 | 1.56 | 3,379 | 28 | 13.7 | 24 |
| tm_react__none__h256__b512__bf16 | 512 | 4 | bf16-mixed | 4,018 | 233,897 | 127 | 0.0336 | 91.8 | 2.98 | 3,336 | 55.6 | 22.4 | 47.4 |
| tm_react__none__h256__b1024__bf16 | 1024 | 4 | bf16-mixed | 4,281 | 249,318 | 237 | 0.0618 | 90.2 | 5.66 | 3,402 | 120 | 47.4 | 95.3 |
| tm_react__none__h256__b1024__bf16__w8__direct_collate | 1024 | 8 | bf16-mixed | 4,337 | 252,518 | 234 | 0.0599 | 95.6 | 5.64 | 3,390 | 117 | 44.7 | 103 |
| tm_react__none__h256__b2048__bf16 | 2048 | 4 | bf16-mixed | 4,367 | 254,284 | 462 | 0.093 | 91.3 | 11.1 | 3,414 | 239 | 86.9 | 204 |

### E1 TMQM graph-level continuity (baseline 25.6 it/s at batch 128)

| run | mode | batch | it/s | samples/s | data wait | GPU util % | peak GB |
|---|---|---|---|---|---|---|---|
| tmqm_graph__b128__lightning | lightning | 128 | 29.1 | 3,725 |  | 51.4 | 0.499 |

### E2 encoder table (batch 128/512 tm_react, 16/64 H7; before the dense neighbor build)

| run | batch | samples/s | atoms/s | step ms | peak GB | GPU util % | syncs/step | encoder ms | neighbor ms | gather+scatter ms | matmul ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h7__dimenetpp_c4_m16__b16 | 16 | 203 | 58,516 | 78.8 | 2.53 | 85.4 | 138 | 29 | 6.73 | 9.31 | 13.8 |
| h7__dimenetpp_c4_m16__b64 | 64 | 254 | 73,216 | 251 | 9.37 | 96.7 | 192 | 87.6 | 35.4 | 37 | 45.2 |
| h7__dimenetpp_c5_m32__b16 | 16 | 105 | 30,204 | 153 | 7.23 | 92.9 | 137 | 62.8 | 5.78 | 20.4 | 25.6 |
| h7__equivariant_l1_h32__b8 | 8 | 75.3 | 21,731 | 106 | 4.19 | 87.1 | 112 | 14.3 | 3.45 | 3.95 | 28.3 |
| h7__none__b16 | 16 | 440 | 126,588 | 36.4 | 0.268 | 39.3 | 102 | 0 | 0 | 4.4 | 3.15 |
| h7__none__b64 | 64 | 1,341 | 386,477 | 47.6 | 0.885 | 85.4 | 102 | 0 | 0 | 16.7 | 7.92 |
| h7__schnet__b16 | 16 | 346 | 99,858 | 46.2 | 0.732 | 56.4 | 121 | 3.78 | 4.95 | 7.49 | 4.95 |
| h7__schnet__b64 | 64 | 556 | 160,212 | 115 | 2.61 | 91.6 | 175 | 14.6 | 35.8 | 28.6 | 16.9 |
| tm_react__dimenetpp_c4_m16__b128 | 128 | 1,226 | 71,468 | 104 | 3.58 | 89.9 | 149 | 37.5 | 9.69 | 13.4 | 17 |
| tm_react__dimenetpp_c4_m16__b512 | 512 | 1,293 | 75,305 | 395 | 13.6 | 96.7 | 236 | 123 | 79.3 | 57.3 | 67.3 |
| tm_react__dimenetpp_c4_m32__b128 | 128 | 1,091 | 63,376 | 117 | 4.48 | 92.5 | 150 | 46 | 9.29 | 16.9 | 21.8 |
| tm_react__dimenetpp_c5_m16__b128 | 128 | 1,102 | 64,145 | 116 | 4.43 | 92.8 | 149 | 43.8 | 8.83 | 16.1 | 21.2 |
| tm_react__dimenetpp_c5_m32__b128 | 128 | 695 | 40,443 | 184 | 9.38 | 94.7 | 149 | 74.3 | 10.3 | 25.9 | 31.3 |
| tm_react__equivariant_l1_h32__b128 | 128 | 574 | 33,393 | 223 | 9.43 | 96.6 | 132 | 33.9 | 9.01 | 10.4 | 70.1 |
| tm_react__equivariant_l1_h32__b32 | 32 | 429 | 25,011 | 74.7 | 2.68 | 78.2 | 110 | 9.61 | 2.3 | 3.29 | 19.2 |
| tm_react__equivariant_l2_h32__b32 | 32 | 113 | 6,590 | 282 | 7.2 | 95.7 | 111 | 28.4 | 2.82 | 3.94 | 55.7 |
| tm_react__bn__dense_compiled_schnet__h128__b1024__w8 | 1024 | 4,501 | 261,447 | 217 | 12.2 | 82 | 130 | 25.6 | 2.02 | 49.1 | 7.95 |
| tm_react__schnet__b128 | 128 | 2,303 | 134,084 | 55.6 | 1.01 | 68.5 | 132 | 4.56 | 8.69 | 12.1 | 6.38 |
| tm_react__schnet__b512 | 512 | 2,652 | 154,429 | 193 | 3.57 | 92.9 | 220 | 17.9 | 84.1 | 47.6 | 28 |

### A5-1 dense per-molecule neighbor build: before vs after (same config)

| run | samples/s before | samples/s after | gain | neighbor ms before | neighbor ms after | syncs before | syncs after | peak GB before | peak GB after |
|---|---|---|---|---|---|---|---|---|---|
| h7__dimenetpp_c4_m16__b16 | 203 | 196 | 0.966 | 6.73 | 2.73 | 138 | 128 | 2.53 | 2.49 |
| h7__dimenetpp_c4_m16__b64 | 254 | 280 | 1.1 | 35.4 | 1.98 | 192 | 128 | 9.37 | 9.44 |
| h7__schnet__b16 | 346 | 337 | 0.971 | 4.95 | 1.54 | 121 | 111 | 0.732 | 0.738 |
| h7__schnet__b64 | 556 | 716 | 1.29 | 35.8 | 2.1 | 175 | 111 | 2.61 | 2.62 |
| tm_react__dimenetpp_c4_m16__b128 | 1,226 | 1,272 | 1.04 | 9.69 | 1.42 | 149 | 128 | 3.58 | 3.83 |
| tm_react__dimenetpp_c4_m16__b512 | 1,293 | 1,583 | 1.22 | 79.3 | 2.2 | 236 | 128 | 13.6 | 13.8 |
| tm_react__equivariant_l1_h32__b128 | 574 | 2,017 | 3.51 | 9.01 | 1.57 | 132 | 111 | 9.43 | 1.3 |
| tm_react__schnet__b1024__w8 |  | 4,390 |  |  | 2.67 |  | 111 |  | 6.9 |
| tm_react__schnet__b128 | 2,303 | 2,538 | 1.1 | 8.69 | 1.62 | 132 | 111 | 1.01 | 0.981 |
| tm_react__schnet__b512 | 2,652 | 4,182 | 1.58 | 84.1 | 1.92 | 220 | 111 | 3.57 | 3.53 |

### A4 direct collate in the real step (__direct_collate) vs Batch.from_data_list

| run | samples/s before | samples/s after | gain | data wait before | data wait after | GPU util before | GPU util after |
|---|---|---|---|---|---|---|---|
| tm_react__none__h128__b1024__bf16__w4 |  | 5,090 |  |  | 0.487 |  | 57.8 |
| tm_react__none__h128__b1024__bf16__w8 | 7,163 | 7,369 | 1.03 | 0.1 | 0.117 | 84.4 | 88.5 |
| tm_react__none__h128__b512__bf16__w4 |  | 5,621 |  |  | 0.327 |  | 59.6 |
| tm_react__none__h256__b1024__bf16__w8 |  | 4,337 |  |  | 0.0599 |  | 95.6 |

### A3 end-to-end (batch_norm on): ResidualBlock vs ResidualBlockDense with bucketing, eager and compiled (CUDA graphs)

| run | conv_fn | batch | hidden | encoder | samples/s | atoms/s | step ms | data wait | GPU util % | peak GB | kernels/step | gather+scatter ms | matmul ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h7__bn__dense_compiled__h256__b64 | ResidualBlockDense | 64 | 256 | none | 979 | 282,587 | 63.1 | 0.0483 | 70.6 | 1.82 | 1,778 | 0.305 | 0.2 |
| h7__bn__dense_eager__h256__b64 | ResidualBlockDense | 64 | 256 | none | 797 | 229,713 | 77.2 | 0.019 | 88.8 | 2.95 | 2,423 | 18.2 | 25.3 |
| h7__bn__ref__h256__b64 | ResidualBlock | 64 | 256 | none | 769 | 221,574 | 83 | 0.0168 | 92.7 | 2.17 | 3,641 | 31 | 16 |
| tm_react__bn__dense_compiled__h128__b1024__w8 | ResidualBlockDense | 1024 | 128 | none | 7,755 | 450,418 | 126 | 0.258 | 62.6 | 4.4 | 2,056 | 0.404 | 0.306 |
| tm_react__bn__dense_compiled__h256__b1024__w8 | ResidualBlockDense | 1024 | 256 | none | 5,310 | 308,419 | 184 | 0.103 | 84.4 | 8.61 | 2,047 | 0.558 | 0.499 |
| tm_react__bn__dense_compiled_schnet__h128__b1024__w8 | ResidualBlockDense | 1024 | 128 | schnet | 4,501 | 261,447 | 217 | 0.0853 | 82 | 12.2 | 2,424 | 49.1 | 7.95 |
| tm_react__bn__dense_eager__h128__b1024__w8 | ResidualBlockDense | 1024 | 128 | none | 6,745 | 394,435 | 144 | 0.0777 | 87.9 | 7.36 | 2,659 | 32.7 | 42.2 |
| tm_react__bn__dense_eager__h256__b1024__w8 | ResidualBlockDense | 1024 | 256 | none | 3,692 | 215,894 | 264 | 0.0414 | 94.3 | 14.6 | 2,653 | 67.2 | 92.9 |
| tm_react__bn__ref__h128__b1024__w8 | ResidualBlock | 1024 | 128 | none | 5,038 | 293,363 | 202 | 0.0824 | 94.4 | 3.64 | 3,909 | 79.7 | 22.4 |
| tm_react__bn__ref__h256__b1024__w8 | ResidualBlock | 1024 | 256 | none | 3,805 | 221,624 | 267 | 0.0583 | 96.4 | 7.03 | 3,909 | 127 | 46 |

### A5-2 channel-wise equivariant tensor product (__cw) vs fully connected

| run | batch | samples/s | step ms | peak GB | encoder ms | matmul ms |
|---|---|---|---|---|---|---|
| h7__equivariant_l1_h32__b8 | 8 | 75.3 | 106 | 4.19 | 14.3 | 28.3 |
| tm_react__equivariant_l1_h32__b128 | 128 | 574 | 223 | 9.43 | 33.9 | 70.1 |
| tm_react__equivariant_l1_h32__b128__cw | 128 | 1,767 | 72.4 | 1.42 | 11.9 | 10.7 |
| tm_react__equivariant_l1_h32__b32 | 32 | 429 | 74.7 | 2.68 | 9.61 | 19.2 |
| tm_react__equivariant_l1_h64__b128__cw | 128 | 1,261 | 102 | 2.29 | 22.7 | 18.3 |
| tm_react__equivariant_l1_h64__b512__cw | 512 | 1,173 | 436 | 8.42 | 106 | 76.8 |
| tm_react__equivariant_l2_h32__b128__cw | 128 | 676 | 189 | 3.34 | 49.7 | 52.1 |
| tm_react__equivariant_l2_h32__b32 | 32 | 113 | 282 | 7.2 | 28.4 | 55.7 |

### E3 conv-stack formulations, fwd+bwd of 8 hetero conv layers (same weights, parity asserted at fp32)

| dataset | batch | hidden | dtype | impl | fwd+bwd ms | speedup vs ref | peak GB | kernels/step | dense shape | pad waste | error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h7 | 32 | 128 | bf16 | ref | 35.3 | 1 | 0.607 | 2,623 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | bf16 | fused | 20.7 | 1.7 | 0.873 | 1,049 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | bf16 | dense | 14.9 | 2.37 | 0.673 | 1,167 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | bf16 | dense_compile | 9.37 | 3.76 | 0.499 | 585 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | bf16 | dense_cudagraph | 8.81 | 4 | 0.164 | 589 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | fp32 | ref | 26.7 | 1 | 0.796 | 1,873 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | fp32 | fused | 23.8 | 1.12 | 1.13 | 791 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | fp32 | dense | 18.7 | 1.43 | 0.824 | 817 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | fp32 | dense_compile | 14.8 | 1.81 | 0.574 | 501 | 304x288 | 0.0612 |  |
| h7 | 32 | 128 | fp32 | dense_cudagraph | 14.3 | 1.87 | 0.164 | 505 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | bf16 | ref | 38 | 1 | 1.23 | 2,599 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | bf16 | fused | 38.4 | 0.992 | 1.8 | 1,057 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | bf16 | dense | 27.3 | 1.4 | 1.34 | 1,175 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | bf16 | dense_compile | 17.7 | 2.15 | 1.06 | 593 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | bf16 | dense_cudagraph | 17.3 | 2.2 | 0.413 | 597 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | fp32 | ref | 44.3 | 1 | 1.65 | 1,939 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | fp32 | fused | 49.4 | 0.898 | 2.31 | 791 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | fp32 | dense | 38.3 | 1.16 | 1.72 | 817 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | fp32 | dense_compile | 31.3 | 1.42 | 1.23 | 501 | 304x288 | 0.0612 |  |
| h7 | 32 | 256 | fp32 | dense_cudagraph | 30.9 | 1.44 | 0.413 | 505 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | bf16 | ref | 74.9 | 1 | 2.76 | 2,689 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | bf16 | fused | 84.1 | 0.891 | 3.94 | 1,057 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | bf16 | dense | 62.7 | 1.2 | 3.03 | 1,175 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | bf16 | dense_compile | 44.1 | 1.7 | 2.54 | 593 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | bf16 | dense_cudagraph | 42.8 | 1.75 | 1.24 | 597 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | fp32 | ref | 95.1 | 1 | 3.71 | 1,963 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | fp32 | fused | 114 | 0.837 | 4.98 | 799 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | fp32 | dense | 97.1 | 0.98 | 3.8 | 825 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | fp32 | dense_compile | 79 | 1.2 | 2.84 | 509 | 304x288 | 0.0612 |  |
| h7 | 32 | 512 | fp32 | dense_cudagraph | 82.5 | 1.15 | 1.24 | 513 | 304x288 | 0.0612 |  |
| h7 | 128 | 128 | bf16 | ref | 72.8 | 1 | 2.24 | 2,599 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | bf16 | fused | 79.6 | 0.915 | 3.29 | 1,042 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | bf16 | dense | 53.2 | 1.37 | 2.57 | 1,160 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | bf16 | dense_compile | 33.4 | 2.18 | 1.83 | 578 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | bf16 | dense_cudagraph | 35.3 | 2.07 | 0.501 | 583 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | fp32 | ref | 81.3 | 1 | 2.97 | 1,873 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | fp32 | fused | 95.5 | 0.851 | 4.24 | 791 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | fp32 | dense | 69.3 | 1.17 | 3.21 | 817 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | fp32 | dense_compile | 55 | 1.48 | 2.16 | 501 | 304x320 | 0.106 |  |
| h7 | 128 | 128 | fp32 | dense_cudagraph | 54.1 | 1.5 | 0.501 | 506 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | bf16 | ref | 123 | 1 | 4.38 | 2,641 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | bf16 | fused | 146 | 0.845 | 6.51 | 1,049 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | bf16 | dense | 105 | 1.18 | 4.75 | 1,167 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | bf16 | dense_compile | 68.5 | 1.8 | 3.58 | 585 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | bf16 | dense_cudagraph | 66.1 | 1.87 | 0.983 | 591 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | fp32 | ref | 152 | 1 | 5.86 | 1,963 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | fp32 | fused | 194 | 0.784 | 8.44 | 791 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | fp32 | dense | 143 | 1.07 | 6.38 | 817 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | fp32 | dense_compile | 113 | 1.34 | 4.28 | 501 | 304x320 | 0.106 |  |
| h7 | 128 | 256 | fp32 | dense_cudagraph | 109 | 1.39 | 0.983 | 507 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | bf16 | ref | 280 | 1 | 8.92 | 2,599 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | bf16 | fused | 327 | 0.855 | 13.3 | 1,049 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | bf16 | dense | 251 | 1.11 | 9.49 | 1,167 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | bf16 | dense_compile | 175 | 1.6 | 7.45 | 585 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | bf16 | dense_cudagraph | 168 | 1.66 | 2.28 | 593 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | fp32 | ref | 359 | 1 | 12 | 1,963 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | fp32 | fused | 453 | 0.794 | 17.2 | 799 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | fp32 | dense | 375 | 0.959 | 13 | 825 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | fp32 | dense_compile | 319 | 1.13 | 8.87 | 509 | 304x320 | 0.106 |  |
| h7 | 128 | 512 | fp32 | dense_cudagraph | 309 | 1.16 | 2.28 | 517 | 304x320 | 0.106 |  |
| tm_react | 128 | 128 | bf16 | ref | 24.1 | 1 | 0.545 | 2,599 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | bf16 | fused | 18.2 | 1.32 | 0.798 | 1,042 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | bf16 | dense | 16.5 | 1.46 | 0.69 | 1,160 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | bf16 | dense_compile | 10.9 | 2.21 | 0.539 | 578 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | bf16 | dense_cudagraph | 10 | 2.41 | 0.151 | 582 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | fp32 | ref | 27.5 | 1 | 0.718 | 1,873 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | fp32 | fused | 21.2 | 1.3 | 1.02 | 791 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | fp32 | dense | 21.5 | 1.28 | 0.936 | 817 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | fp32 | dense_compile | 17.2 | 1.6 | 0.65 | 501 | 80x96 | 0.3 |  |
| tm_react | 128 | 128 | fp32 | dense_cudagraph | 16.7 | 1.65 | 0.151 | 505 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | bf16 | ref | 38.6 | 1 | 1.13 | 2,641 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | bf16 | fused | 33.9 | 1.14 | 1.64 | 1,049 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | bf16 | dense | 30.6 | 1.26 | 1.43 | 1,167 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | bf16 | dense_compile | 20.2 | 1.91 | 1.16 | 585 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | bf16 | dense_cudagraph | 19.6 | 1.98 | 0.402 | 589 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | fp32 | ref | 37 | 1 | 1.5 | 1,963 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | fp32 | fused | 42.1 | 0.878 | 2.1 | 791 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | fp32 | dense | 39 | 0.95 | 1.95 | 817 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | fp32 | dense_compile | 31 | 1.19 | 1.38 | 501 | 80x96 | 0.3 |  |
| tm_react | 128 | 256 | fp32 | dense_cudagraph | 30.7 | 1.21 | 0.402 | 505 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | bf16 | ref | 65.8 | 1 | 2.54 | 2,599 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | bf16 | fused | 74.4 | 0.884 | 3.64 | 1,049 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | bf16 | dense | 70.1 | 0.939 | 3.28 | 1,167 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | bf16 | dense_compile | 49.1 | 1.34 | 2.76 | 585 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | bf16 | dense_cudagraph | 48.9 | 1.35 | 1.23 | 589 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | fp32 | ref | 80.7 | 1 | 3.39 | 1,963 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | fp32 | fused | 95.9 | 0.841 | 4.59 | 799 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | fp32 | dense | 100 | 0.804 | 4.32 | 825 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | fp32 | dense_compile | 84.3 | 0.957 | 3.18 | 509 | 80x96 | 0.3 |  |
| tm_react | 128 | 512 | fp32 | dense_cudagraph | 83.2 | 0.97 | 1.23 | 513 | 80x96 | 0.3 |  |
| tm_react | 512 | 128 | bf16 | ref | 58.3 | 1 | 1.91 | 2,599 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | bf16 | fused | 63.7 | 0.915 | 2.84 | 1,042 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | bf16 | dense | 57.3 | 1.02 | 2.49 | 1,160 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | bf16 | dense_compile | 37 | 1.57 | 1.9 | 594 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | bf16 | dense_cudagraph | 35.9 | 1.62 | 0.422 | 598 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | fp32 | ref | 63.7 | 1 | 2.55 | 1,921 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | fp32 | fused | 77.5 | 0.821 | 3.66 | 799 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | fp32 | dense | 76.5 | 0.833 | 3.5 | 825 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | fp32 | dense_compile | 60.9 | 1.05 | 2.37 | 525 | 80x96 | 0.318 |  |
| tm_react | 512 | 128 | fp32 | dense_cudagraph | 59.4 | 1.07 | 0.422 | 529 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | bf16 | ref | 109 | 1 | 3.8 | 2,600 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | bf16 | fused | 127 | 0.858 | 5.69 | 1,042 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | bf16 | dense | 116 | 0.939 | 4.88 | 1,161 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | bf16 | dense_compile | 76.4 | 1.43 | 3.81 | 595 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | bf16 | dense_cudagraph | 73.3 | 1.49 | 0.892 | 601 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | fp32 | ref | 131 | 1 | 5.11 | 1,922 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | fp32 | fused | 157 | 0.837 | 7.37 | 791 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | fp32 | dense | 158 | 0.832 | 7.03 | 818 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | fp32 | dense_compile | 126 | 1.04 | 4.74 | 518 | 80x96 | 0.318 |  |
| tm_react | 512 | 256 | fp32 | dense_cudagraph | 122 | 1.08 | 0.892 | 524 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | bf16 | ref | 247 | 1 | 7.84 | 2,600 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | bf16 | fused | 287 | 0.862 | 11.7 | 1,042 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | bf16 | dense | 282 | 0.877 | 10.1 | 1,161 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | bf16 | dense_compile | 197 | 1.26 | 8 | 595 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | bf16 | dense_cudagraph | 189 | 1.3 | 2.17 | 603 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | fp32 | ref | 317 | 1 | 10.5 | 1,874 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | fp32 | fused | 392 | 0.808 | 15.1 | 791 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | fp32 | dense | 416 | 0.761 | 14.4 | 818 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | fp32 | dense_compile | 350 | 0.906 | 9.85 | 518 | 80x96 | 0.318 |  |
| tm_react | 512 | 512 | fp32 | dense_cudagraph | 342 | 0.927 | 2.17 | 526 | 80x96 | 0.318 |  |
| tm_react | 1024 | 128 | bf16 | ref | 110 | 1 | 3.77 | 2,672 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | bf16 | fused | 127 | 0.867 | 5.66 | 1,058 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | bf16 | dense | 111 | 0.993 | 4.88 | 1,177 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | bf16 | dense_compile | 70.7 | 1.56 | 3.7 | 603 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | bf16 | dense_cudagraph | 69.7 | 1.58 | 0.784 | 609 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | fp32 | ref | 122 | 1 | 5.07 | 1,970 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | fp32 | fused | 155 | 0.787 | 7.33 | 807 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | fp32 | dense | 149 | 0.815 | 6.92 | 834 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | fp32 | dense_compile | 118 | 1.03 | 4.64 | 526 | 80x96 | 0.31 |  |
| tm_react | 1024 | 128 | fp32 | dense_cudagraph | 117 | 1.04 | 0.784 | 532 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | bf16 | ref | 211 | 1 | 7.47 | 2,672 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | bf16 | fused | 255 | 0.829 | 11.3 | 1,050 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | bf16 | dense | 231 | 0.914 | 9.51 | 1,169 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | bf16 | dense_compile | 151 | 1.4 | 7.36 | 595 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | bf16 | dense_cudagraph | 147 | 1.43 | 1.56 | 603 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | fp32 | ref | 327 | 1 | 10 | 1,970 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | fp32 | fused | 393 | 0.833 | 14.6 | 799 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | fp32 | dense | 382 | 0.857 | 13.8 | 826 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | fp32 | dense_compile | 303 | 1.08 | 9.24 | 518 | 80x96 | 0.31 |  |
| tm_react | 1024 | 256 | fp32 | dense_cudagraph | 307 | 1.06 | 1.56 | 526 | 80x96 | 0.31 |  |
| tm_react | 1024 | 512 | bf16 | ref | 631 | 1 | 15.1 | 2,624 | 80x96 | 0.31 |  |
| tm_react | 1024 | 512 | bf16 | fused | nan |  | nan | nan | 80x96 | 0.31 | OutOfMemoryError('CUDA out of memory. Tried to allocate 190.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 152.50 MiB is free. Including non-PyTorch memory, this process has 21.86 GiB memory |
| tm_react | 1024 | 512 | bf16 | dense | 602 | 1.05 | 19 | 1,177 | 80x96 | 0.31 |  |
| tm_react | 1024 | 512 | bf16 | dense_compile | 416 | 1.51 | 14.9 | 603 | 80x96 | 0.31 |  |
| tm_react | 1024 | 512 | bf16 | dense_cudagraph | 376 | 1.68 | 3.29 | 616 | 80x96 | 0.31 |  |
| tm_react | 1024 | 512 | fp32 | ref | 632 | 1 | 20.2 | 1,946 | 80x96 | 0.31 |  |
| tm_react | 1024 | 512 | fp32 | fused | nan |  | nan | nan | 80x96 | 0.31 | OutOfMemoryError('CUDA out of memory. Tried to allocate 378.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 314.75 MiB is free. Including non-PyTorch memory, this process has 22.47 GiB memory |
| tm_react | 1024 | 512 | fp32 | dense | nan |  | nan | nan | 80x96 | 0.31 | OutOfMemoryError('CUDA out of memory. Tried to allocate 576.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 152.75 MiB is free. Including non-PyTorch memory, this process has 22.63 GiB memory |
| tm_react | 1024 | 512 | fp32 | dense_compile | 697 | 0.908 | 18.6 | 518 | 80x96 | 0.31 |  |
| tm_react | 1024 | 512 | fp32 | dense_cudagraph | nan |  | nan | nan | 80x96 | 0.31 | OutOfMemoryError('CUDA out of memory. Tried to allocate 576.00 MiB. GPU 0 has a total capacity of 23.54 GiB of which 185.62 MiB is free. Including non-PyTorch memory, this process has 22.39 GiB memory |

### E4 padding waste of static-shape schemes (fraction of padded atom+bond rows that are padding)


droplet: n_total 14933, atoms mean 59.3 max 130, bonds mean 50.5 max 133

| scheme | static shapes | waste atoms | waste bonds | waste total | ragged b256 | ragged b512 | ragged b1024 |
|---|---|---|---|---|---|---|---|
| quantile k=1 | 1 | 0.544 | 0.62 | 0.582 | 0.0187 | 0.0613 | 0.147 |
| quantile k=2 | 2 | 0.333 | 0.5 | 0.422 | 0.0187 | 0.0613 | 0.147 |
| quantile k=4 | 4 | 0.199 | 0.432 | 0.326 | 0.104 | 0.232 | 0.317 |
| quantile k=8 | 8 | 0.107 | 0.388 | 0.263 | 0.232 | 0.317 | 1 |
| quantile k=16 | 16 | 0.0513 | 0.336 | 0.207 | 0.317 | 1 | 1 |

floors: sorted_waste_total_b256 0.193, sorted_waste_total_b512 0.229, sorted_waste_total_b1024 0.293, random_waste_total_b256 0.566, random_waste_total_b512 0.569, random_waste_total_b1024 0.573

h7: n_total 18071, atoms mean 288.6 max 350, bonds mean 281.3 max 369

| scheme | static shapes | waste atoms | waste bonds | waste total | ragged b256 | ragged b512 | ragged b1024 |
|---|---|---|---|---|---|---|---|
| quantile k=1 | 1 | 0.175 | 0.238 | 0.207 | 0.00836 | 0.00836 | 0.0367 |
| quantile k=2 | 2 | 0.0875 | 0.165 | 0.127 | 0.00836 | 0.00836 | 0.0367 |
| quantile k=4 | 4 | 0.0424 | 0.125 | 0.085 | 0.0225 | 0.065 | 0.0934 |
| quantile k=8 | 8 | 0.0203 | 0.101 | 0.0617 | 0.0651 | 0.0934 | 0.15 |
| quantile k=16 | 16 | 0.00893 | 0.0887 | 0.05 | 0.136 | 0.235 | 0.377 |
| round-up 8 | 97 | 0.0116 | 0.0119 | 0.0117 | 0.405 | 0.717 | 1 |
| round-up 16 | 34 | 0.0241 | 0.0253 | 0.0247 | 0.164 | 0.348 | 0.547 |
| round-up 32 | 13 | 0.0503 | 0.0504 | 0.0503 | 0.0649 | 0.122 | 0.207 |

floors: sorted_waste_total_b256 0.046, sorted_waste_total_b512 0.055, sorted_waste_total_b1024 0.069, random_waste_total_b256 0.189, random_waste_total_b512 0.192, random_waste_total_b1024 0.198

tm_react: n_total 119019, atoms mean 58.0 max 100, bonds mean 61.9 max 114

| scheme | static shapes | waste atoms | waste bonds | waste total | ragged b256 | ragged b512 | ragged b1024 |
|---|---|---|---|---|---|---|---|
| quantile k=1 | 1 | 0.42 | 0.457 | 0.44 | 0.00197 | 0.00197 | 0.00197 |
| quantile k=2 | 2 | 0.26 | 0.344 | 0.306 | 0.00197 | 0.00628 | 0.0106 |
| quantile k=4 | 4 | 0.151 | 0.272 | 0.218 | 0.00412 | 0.0106 | 0.0192 |
| quantile k=8 | 8 | 0.0764 | 0.217 | 0.155 | 0.00842 | 0.0149 | 0.0278 |
| quantile k=16 | 16 | 0.0352 | 0.195 | 0.125 | 0.0192 | 0.0278 | 0.0536 |
| round-up 8 | 54 | 0.0561 | 0.0533 | 0.0546 | 0.0536 | 0.101 | 0.165 |
| round-up 16 | 23 | 0.115 | 0.11 | 0.112 | 0.0235 | 0.0407 | 0.0622 |
| round-up 32 | 10 | 0.206 | 0.197 | 0.201 | 0.00629 | 0.0192 | 0.0278 |

floors: sorted_waste_total_b256 0.117, sorted_waste_total_b512 0.143, sorted_waste_total_b1024 0.183, random_waste_total_b256 0.425, random_waste_total_b512 0.429, random_waste_total_b1024 0.435

tmqm: n_total 48585, atoms mean 55.7 max 85, bonds mean 66.4 max 125

| scheme | static shapes | waste atoms | waste bonds | waste total | ragged b256 | ragged b512 | ragged b1024 |
|---|---|---|---|---|---|---|---|
| quantile k=1 | 1 | 0.345 | 0.469 | 0.419 | 0.00414 | 0.00941 | 0.00941 |
| quantile k=2 | 2 | 0.206 | 0.398 | 0.323 | 0.00414 | 0.00941 | 0.00941 |
| quantile k=4 | 4 | 0.113 | 0.306 | 0.229 | 0.00939 | 0.0305 | 0.0515 |
| quantile k=8 | 8 | 0.0576 | 0.252 | 0.175 | 0.0147 | 0.041 | 0.0515 |
| quantile k=16 | 16 | 0.0268 | 0.232 | 0.15 | 0.0516 | 0.104 | 0.199 |
| round-up 8 | 50 | 0.0614 | 0.0496 | 0.055 | 0.0884 | 0.167 | 0.283 |
| round-up 16 | 20 | 0.122 | 0.102 | 0.111 | 0.041 | 0.0726 | 0.157 |
| round-up 32 | 7 | 0.227 | 0.19 | 0.208 | 0.00941 | 0.0305 | 0.0726 |

floors: sorted_waste_total_b256 0.142, sorted_waste_total_b512 0.170, sorted_waste_total_b1024 0.199, random_waste_total_b256 0.385, random_waste_total_b512 0.393, random_waste_total_b1024 0.403

### E5 data path on one CPU thread (tm_react)

| batch | Batch.from_data_list ms | direct collate ms | speedup |
|---|---|---|---|
| 128 | 14.2 | 3.74 | 3.79 |
| 512 | 67.4 | 18.8 | 3.59 |
| 1024 | 163 | 40 | 4.08 |

deserialization per record (1000 records): pickled HeteroData torch.load 0.345 ms (54 KB); flat tensor dict with weights_only=True 0.768 ms (53 KB); ratio 0.45x

### E6 radius-graph build (edge sets asserted identical)

| dataset | batch | cutoff | atoms | edges | mean deg | chunked GPU ms | dense GPU ms | speedup | nonzero calls chunked | nonzero calls dense | chunked CPU1 ms | dense CPU1 ms | dense peak GB | triplets maxn16 | triplets maxn32 | triplets maxn32 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tm_react | 128 | 4 | 7285 | 107168 | 14.7 | 6.06 | 0.477 | 12.7 | 29 | 1 | 153 | 6.77 | 0.0291 | 1262259 | 1657539 | 1.76 |
| tm_react | 128 | 5 | 7285 | 166951 | 22.9 | 6.64 | 0.601 | 11 | 29 | 1 | 143 | 9.26 | 0.039 | 1644701 | 3800350 | 2.95 |
| tm_react | 512 | 4 | 29110 | 436314 | 15 | 64.8 | 0.907 | 71.5 | 114 | 1 | 2,828 | 41.2 | 0.0946 | 5159866 | 6876233 | 5.06 |
| tm_react | 512 | 5 | 29110 | 679731 | 23.4 | 65.1 | 1 | 64.7 | 114 | 1 | 2,740 | 33.4 | 0.131 | 6597620 | 15655171 | 10.3 |
| tm_react | 1024 | 4 | 60138 | 910356 | 15.1 | 255 | 1.42 | 179 | 235 | 1 | 15,718 | 56 | 0.192 | 10749611 | 14496713 | 10.1 |
| tm_react | 1024 | 5 | 60138 | 1422517 | 23.7 | 255 | 1.59 | 161 | 235 | 1 | 16,170 | 72.2 | 0.258 | 13673119 | 32965378 | 21 |
| h7 | 32 | 4 | 9405 | 177864 | 18.9 | 9.91 | 0.774 | 12.8 | 37 | 1 | 251 | 14.8 | 0.0791 | 1980181 | 3535145 | 2.81 |
| h7 | 32 | 5 | 9405 | 310278 | 33 | 9.91 | 0.761 | 13 | 37 | 1 | 252 | 18 | 0.0726 | 2237398 | 7378008 | 5.06 |
| h7 | 128 | 4 | 37041 | 693688 | 18.7 | 100 | 1.69 | 59.5 | 145 | 1 | 5,461 | 83.7 | 0.182 | 7745426 | 13661689 | 9.28 |
| h7 | 128 | 5 | 37041 | 1206556 | 32.6 | 101 | 1.86 | 54.4 | 145 | 1 | 5,322 | 89.9 | 0.253 | 8793757 | 28760133 | 18.3 |
