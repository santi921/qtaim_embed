---
title: "tm_react eval-mode divergence: BatchNorm running variance collapse on dead ReLU channels"
date: 2026-09-09
status: diagnosed and fixed 2026-09-09; model.bn_before_activation trains 40 epochs from scratch without divergence (val R2 0.87 vs 0.72 for the reference before it diverged); model.global_aggr mean adds nothing; default True in the default configs and loaders since 2026-09-09 (constructor default stays False for old checkpoints)
related: docs/research/2026-09-track-a-measurements.md (correctness section), docs/research/2026-09-track-a-review-findings.md
---

# Validation divergence of GCNNodePred on tm_react (b128, lr 1e-3): diagnosis

Run: `profiling/train_runs/b128_lr0.001/` (config `profiling/train_configs/tm_react_b128_lr0.001.json`, branch feat/3d-encoders, uncommitted). All experiments below are read-only with respect to the repo; scripts and outputs live in `profiling/divergence/`. GPU 1 only.

## Verdict

The divergence is an eval-mode-only failure of BatchNorm running statistics, not a weight divergence, and not a data or metric problem.

Mechanism (each step is measured below):

1. `GraphConvDropoutBatch` (`qtaim_embed/models/layers.py`) applies conv -> ReLU -> dropout -> BatchNorm1d, so BN normalises post-ReLU activations.
2. On the `global` node type, the conv input is an un-normalised sum over all atoms (a2g) and all bonds (b2g) of the molecule (PyG `GraphConv`, `aggr="add"`, no degree normalisation). Pre-BN activations of global nodes have per-channel variance 1e2 to 1e4 (block 0) and running_var up to 17,340 (block 3) at epoch 1.
3. A ReLU channel that is inactive for every global node in a batch (128 nodes per batch, easy to achieve at these magnitudes) has batch variance exactly 0. `running_var` then decays as 0.9^t (momentum 0.1): 1e-3 after 66 steps, the float32 denormal floor 5.6e-45 within one epoch. At epoch 1 there were already 7 such channels; the baseline continuation grows this to 47 after one more epoch and 129 after two (all on `global`, plus one `atom` channel in the output block).
4. In eval mode those channels apply a gain of |gamma|/sqrt(running_var + 1e-5) = up to 316 x gamma (measured gains 100 to 260). Whenever a held-out molecule activates such a channel (1 to 16 of 6400 molecules per channel), the leaked value is multiplied by 100+, passed through the residual add and the g2a/g2b edges to every atom and bond of that molecule, and out through the head. Example on the diverged state: `b3.l0.a2g->global` channel 53, running_var 1.5e-10, gain 131, input 8.4e4 -> output 1.1e7 on 5 of 6400 molecules.
5. In train mode BN uses batch statistics, so the same channel is normalised by its own batch spread and stays bounded (batch-stat eval activations never exceed 307 in any run). Training therefore sees nothing wrong: train loss keeps decreasing slowly (2.79 -> 2.61 summed MSE over the original run) while eval-mode val MSE jumps between 0.34 and 4e5 from probe to probe.

Causality check on a diverged state (baseline continued 1050 steps, eval val MSE 62.5): resetting the 59 channels with running_var < 1e-3 to running_var 1, running_mean 0, weights untouched, gives val MSE 0.3352; keeping all buffers but raising BN eps from 1e-5 to 1e-2 gives 0.3356; batch-stat BN on the same weights gives 0.3613. Same on the clip1 state (600.7 -> 0.3375 / 0.3377 / 0.3616).

## 1. Timeline

Source: TensorBoard events in `test_logs/version_0` (train.log progress lines agree; both parsed by `parse_timeline.py`, full tables in Appendix A). Note that `train_mse`/`val_mse` are `MeanSquaredError(squared=False)`, i.e. RMSE; `*_loss` is the sum of six per-target MSEs. Label std in scaled units is 0.88 to 1.05 per target.

| epoch | val_loss | val_r2 | val_mae | val_mse (RMSE) | train_loss | train_r2 | train_mse (RMSE) | lr |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.167 | 0.692 | 0.284 | 0.564 | 3.461 | 0.442 | 0.744 | 1e-3 |
| 1 | 2.088 | 0.720 | 0.279 | 0.551 | 2.832 | 0.567 | 0.664 | 1e-3 |
| 2 | 1555 | -320.6 | 0.398 | 15.4 | 2.772 | 0.584 | 0.656 | 1e-3 |
| 3 | 2.1e5 | -4.8e4 | 3.03 | 176 | 2.744 | 0.592 | 0.653 | 1e-3 |
| 4 | 3.0e7 | -4.5e6 | 26.3 | 1989 | 2.726 | 0.598 | 0.650 | 1e-3 |
| 5 | 1.2e9 | -4.0e7 | 135 | 1.03e4 | 2.711 | 0.602 | 0.648 | 1e-3 |
| 8 | 7.5e10 | -9.3e9 | 1801 | 1.0e5 | 2.683 | 0.611 | 0.645 | 1e-3 |
| 12 | 1.2e10 | -2.0e9 | 663 | 4.3e4 | 2.664 | 0.617 | 0.642 | 1e-3 |
| 13 | 1.4e10 | -2.4e9 | 539 | 4.3e4 | 2.651 | 0.620 | 0.641 | 6e-4 |
| 17 | 1.9e7 | -1.4e6 | 26.7 | 1547 | 2.637 | 0.623 | 0.639 | 6e-4 |
| 21 | 1.9e6 | -3.6e5 | 7.76 | 543 | 2.630 | 0.625 | 0.638 | 6e-4 |
| 23 | 4.2e11 | -7.0e10 | 3686 | 2.6e5 | 2.630 | 0.625 | 0.638 | 6e-4 |
| 24 | 1.5e9 | -1.9e8 | 103 | 1.2e4 | 2.618 | 0.626 | 0.637 | 3.6e-4 |
| 26 | 1.6e12 | -6.7e8 | 1799 | 2.3e5 | 2.610 | 0.627 | 0.636 | 3.6e-4 |

- Onset is inside epoch 2 (first bad validation at the end of epoch 2, step 2789). Validation values afterwards are erratic over 6 orders of magnitude (epochs 21-22 at 5e2, epoch 23 at 2.6e5), which is what a handful of leaking channels hit by different molecules gives; diverged weights would trend monotonically.
- Learning rate (LearningRateMonitor, `lr-Adam`): 1e-3 until step 12099, 6e-4 until 22349, then 3.6e-4. ReduceLROnPlateau (factor 0.6, patience 10, monitor `val_mae`) fired twice, after epochs 12 and 23, i.e. patience+1 epochs after the last improvement at epoch 1 and again 11 epochs later. Training loss did not react.
- `last.ckpt` is byte-identical to `model_lightning_epoch=001-val_loss=2.0877.ckpt` (epoch 1, step 1860): `ModelCheckpoint(save_last=True)` only writes `last.ckpt` when a monitored checkpoint is written, and `val_mae` never improved after epoch 1. No diverged weights were saved; the divergence had to be reproduced (section 2).
- Side effect of `on_train_epoch_end` logging `val_mae = 1e7` at epoch 0: the scheduler state in the checkpoint has `best = 1e7` after epoch 0. Harmless here.

## 2. Onset localisation: continuation from the epoch-1 checkpoint

`probe_continue.py baseline --steps 1860`: manual loop with Adam state loaded from the checkpoint (lr 1e-3, wd 1e-5, fused=False as `build_adam` does when the trainer clips), bf16 autocast, norm clipping 5.0, same loader (shuffle seeded). Every 150 steps: fixed 50-batch val subset (6400 molecules, 369,752 atoms) in eval mode and in batch-stat mode (BN layers in train mode with momentum 0 so buffers are untouched, dropout off), BN buffer summary, per-block max |activation| (forward hooks on each ResidualBlock), gradient norm before clipping. Full tables in Appendix B.

| step (0 = end of epoch 1) | train loss | grad norm pre-clip | eval-mode val MSE | eval fp32 | batch-stat val MSE | max eval act atom / bond / global | max batch-stat act | dead BN ch (rv < 1e-3) | global rv max |
|---|---|---|---|---|---|---|---|---|---|
| 0 | - | - | 0.3428 | 0.3428 | 0.3675 | 100 / 279 / 128 | 278 | 7 | 1.73e4 |
| 150 | 2.789 | 0.30 | 0.3426 | 0.3426 | 0.3672 | 100 / 277 / 128 | 277 | 8 | 1.46e4 |
| 300 | 2.770 | 0.22 | 0.3418 | 0.3419 | 0.3662 | 94 / 266 / 511 | 266 | 14 | 1.59e4 |
| 450 | 2.807 | 0.26 | 3.469 | 3.213 | 0.3639 | 9730 / 8717 / 8.9e4 | 262 | 24 | 1.47e4 |
| 600 | 2.725 | 0.23 | 0.3756 | 0.372 | 0.3633 | 94 / 252 / 5421 | 252 | 30 | 1.40e4 |
| 750 | 2.807 | 0.24 | 0.3516 | 0.3524 | 0.3636 | 143 / 693 / 4779 | 238 | 38 | 1.14e4 |
| 900 | 2.762 | 0.18 | 0.4092 | 0.4143 | 0.361 | 98 / 874 / 2574 | 231 | 47 | 9849 |
| 1050 | 2.747 | 0.23 | 4.2e4 | 4.0e4 | 0.361 | 2.4e4 / 3.0e5 / 8.8e4 | 224 | 64 | 9335 |
| 1200 | 2.749 | 0.28 | 37.8 | 36.8 | 0.361 | 777 / 3.2e4 / 8965 | 225 | 69 | 7155 |
| 1350 | 2.728 | 0.23 | 49.5 | 45.9 | 0.3608 | 1303 / 3.5e4 / 2.6e4 | 222 | 88 | 7582 |
| 1500 | 2.743 | 0.21 | 1.7e4 | 1.7e4 | 0.3616 | 3.1e4 / 6.3e5 / 3.3e5 | 231 | 96 | 1.14e4 |
| 1650 | 2.735 | 0.26 | 3.9e5 | 3.7e5 | 0.3609 | 6.9e4 / 1.1e7 / 4.9e6 | 234 | 124 | 9783 |
| 1800 | 2.792 | 0.24 | 47.7 | 48.9 | 0.3581 | 3938 / 9536 / 4.5e5 | 225 | 129 | 1.00e4 |

Reading:

- Weights do not diverge: batch-stat val MSE improves monotonically 0.3675 -> 0.3581, training loss and gradient norms (0.18 to 0.30, far below the clip of 5.0) are flat, batch-stat activations shrink (278 -> 225), max training-mode activation stays 56 to 208.
- Running statistics do diverge: the count of BN channels with running_var < 1e-3 grows 7 -> 47 -> 129 (Appendix B: all in `*->global` modules of blocks 0 to 3 plus 1 in `b3.atom`), and min running_var reaches the float32 floor 5.6e-45 in `b2.global` at step 600 and in `b0`, `b1`, `b3.global` by step 1500.
- Onset is reproduced at step 450 of epoch 2 (eval MSE 3.5, max eval activation 8.9e4 on global nodes in block 3), matching the original run (first bad validation at the end of epoch 2). bf16 vs fp32 eval gives the same numbers, so autocast is not involved.
- The leak is intermittent (3.5 at step 450, 0.38 at 600, 4.2e4 at 1050, 37.8 at 1200) because it depends on which of the 6400 molecules hit which of the dead channels; this reproduces the erratic per-epoch values of section 1.

Per-channel evidence (`dead_channel_analysis.py`, Appendix C):

| state | eval val MSE | channels rv < 1e-3 | channels with eval gain > 100 | worst leaking channel (module, ch, rv, gain, max in -> max out, molecules hit) |
|---|---|---|---|---|
| epoch 1 | 0.3428 | 7 | 0 | `b3.l0.a2g->global` 40, rv 7.9e-6, gain 21.7, 6.0 -> 131, 4/6400 |
| baseline +1050 | 62.5 | 59 | 12 | `b3.l0.a2g->global` 53, rv 1.5e-10, gain 131, 8.4e4 -> 1.1e7, 5/6400 |
| clip1 +930 | 600.7 | 64 | 14 | `b3.l0.b2g->global` 68, rv 1.5e-27, gain 18.1, 1.25e4 -> 2.3e5, 1/6400 |

The 1.25e4 and 8.4e4 inputs to block 3 are themselves products of leaks in block 2 (`b2.l0.b2g->global` ch 126: rv 1.4e-15, gain 82.5, 264 -> 2.2e4) propagated through the residual connection and g2a/g2b, so the amplification compounds across blocks.

## 3. Input to the global node

`global_input_stats.py`, 50 val batches, epoch-1 model, fp32 (Appendix D).

Aggregated sums entering conv layer 0 (sum over the embedding output of all atoms / bonds of a molecule; per-atom embedding L2 norm 9.2, per-bond 12.1):

| quantity | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| atoms per molecule | 57 | 87 | 99 | 100 | 100 |
| bonds per molecule | 62 | 93 | 105 | 112 | 119 |
| a2g sum L2 norm | 137 | 201 | 278 | 339 | 393 |
| a2g sum max component | 33.9 | 51.3 | 72.3 | 87.6 | 97.4 |
| b2g sum L2 norm | 163 | 259 | 391 | 775 | 967 |
| b2g sum max component | 39.9 | 67 | 105 | 197 | 247 |

Pearson r(atom count, |a2g sum|) = 0.27 (R2 0.07); r(bond count, |b2g sum|) = 0.39 (R2 0.15). Fit |a2g sum| = 0.50 n_atoms + 117. The largest a2g components (96 to 97) come from molecules with 24 to 49 atoms, not the largest molecules: the sums are dominated by a few atoms with extreme scaled features, not by size.

Extreme scaled feature columns (max |x| > 50): 54 atom columns and 14 bond columns, 0 global columns. All 54 atom columns are standard-scaled one-hot element indicators (`chemical_symbol_*`); a one-hot with frequency p scales to about 1/sqrt(p), so the rarest elements reach Rb 162.3, Cs 157.1, Nb 152.3, Ag 145.2, Ta 142.8, Sb 127.0, K 121.0, Na 119.1, Hf 112.0, Tm 111.2, Er 108.7, Li 106.8, Ho 102.7 (13 columns at or above 100, values identical in train and val). The 14 bond columns are the long-distance tail of the 50-bin Gaussian RBF of bond length (`rbf_gaussian_50_38` to `_49`, `_5`, `_6`), reaching 363 (val) / 376 (train). 65% of molecules (4191/6400 val, 4149/6400 train) contain at least one atom with a scaled feature above 50. These values pass through `UnifySize` (max |w| 1.66) and are the source of the 280-magnitude activations on bond nodes visible in every probe, and of the tail of the global sums.

Global BN batch statistics vs molecule size (pre-BN activations of global nodes at conv layer 0, a2g and b2g modules, per batch of 128 molecules):

| | a2g | b2g |
|---|---|---|
| within-batch variance per channel, median / max | 84.6 / 281 | 104 / 376 |
| batch-to-batch variance of the batch mean per channel, median / max | 17 / 150 | 27.8 / 136 |
| ratio batch-to-batch / within-batch (median over channels) | 0.23 | 0.28 |
| r(batch mean molecule size, batch mean activation) | 0.45 (R2 0.21) | 0.42 (R2 0.18) |
| per molecule r(size, abs pre-BN activation) | 0.30 (R2 0.09) | 0.42 (R2 0.17) |
| running_var max (checkpoint) vs val batch-var max | 491 vs 281 | 609 vs 376 |

So the batch-to-batch variation of the global BN statistics is NOT dominated by molecule-size variation: size explains 18 to 21% of the batch-mean variation and 9 to 17% of the per-molecule activation magnitude. The dominant driver is feature content (rare-element one-hots and long-bond RBF bins summed without normalisation). At layer 0 no global channel is dead (min running_var 15); the dead channels appear from block 1 onward where running_var has grown to 1e3 to 1.7e4 and entire channels sit below the ReLU threshold.

## 4. Hypothesis tests (one continued epoch each, same seed and data order; `probe_continue.py <variant>`)

Eval-mode val MSE on the fixed 50-batch subset. Step 0 is the unmodified checkpoint evaluated under the modified architecture (mean_g, mean_all, sym_all change the function, hence 0.36 to 0.59 at step 0 before any adaptation).

| step | baseline | (a) bn_mom001 | (b) mean_g (a2g,b2g mean) | (c1) mean_all | (c2) sym_all (1/sqrt(deg_s deg_d)) | (d) clip1 | (e) lr3e-4 | extra: bn_eps1e-2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3428 | 0.3428 | 0.3615 | 0.5644 | 0.5855 | 0.3428 | 0.3428 | 0.3431 |
| 150 | 0.3426 | 0.3424 | 0.3437 | 0.3524 | 0.3505 | 0.3425 | 0.3414 | 0.3422 |
| 300 | 0.3418 | 0.3436 | 0.3432 | 0.3474 | 0.3472 | 0.357 | 0.3397 | 0.3419 |
| 450 | 3.469 | 0.3397 | 0.3419 | 0.3432 | 0.3433 | 0.3482 | 0.3393 | 0.3394 |
| 600 | 0.3756 | 0.3407 | 0.3407 | 0.3432 | 0.3434 | 0.5363 | 0.3608 | 0.3398 |
| 750 | 0.3516 | 0.3389 | 0.3366 | 0.3397 | 0.3397 | 0.341 | 0.3388 | 0.3361 |
| 900 | 0.4092 | 0.3366 | 0.3367 | 0.3376 | 0.3378 | 10.36 | 0.3516 | 0.3369 |
| stayed < 1 | no | yes | yes | yes | yes | no | yes | yes |
| dead ch at 900 | 47 | 10 | 52 | 62 | 67 | 55 | 12 | 48 |
| max eval act at 900 | 2574 | 305 | 233 | 245 | 250 | 1.1e4 | 3810 | 967 |

Batch-stat val MSE is 0.361 to 0.376 for every variant at every step; train loss and pre-clip gradient norms are indistinguishable across variants (Appendix B), confirming that none of them changes optimisation, only the eval-mode BN behaviour.

Interpretation:

- (d) clip 1.0: no effect (gradient norms are 0.2 to 0.3, clipping never engages at 5.0 or 1.0); diverges like the baseline (10.4 at step 900, 55 dead channels, 600.7 on the saved state).
- (e) lr 3e-4: delays the symptom (eval MSE stays below 0.37) but the mechanism is present: 12 dead channels and eval activation spikes of 7249 at step 600 and 3810 at step 900. Lower lr slows the drift of pre-activations into the dead region; it does not stop it.
- (a) BN momentum 0.01: passes the window with eval MSE 0.337 to 0.344 and max activation 305; dead channel count stays 6 to 10. This works only because running_var now decays as 0.99^t (690 steps to reach 1e-3 instead of 66); channels that stay dead will still collapse over a few epochs. Symptomatic relief, and it also makes the running stats lag genuinely changing statistics.
- (b) mean a2g/b2g and (c) mean or symmetric normalisation on all relations: eval MSE 0.337 to 0.344 through the window and global running_var max drops from 1.7e4 to 20 to 30 (mean) / 140 to 205 (symmetric). Dead channels nevertheless grow to 52 to 67 (as many as the baseline). The per-channel analysis of the mean_g state shows why they no longer hurt: with a normalised aggregation the input leaking into a collapsed channel is O(1) (max 0.97 -> output 59.8, gain 218 still present on `b3.l0.b2g->global` ch 41) instead of O(1e4). Normalising the global aggregation removes the amplifier's fuel, not the amplifier.
- extra bn_eps1e-2: caps the eval gain at |gamma|/0.1; eval MSE 0.336 to 0.342 with 48 dead channels (same as baseline) and a max eval activation of 967 at step 900. The repair test (section 2) shows eps 1e-2 alone recovers already-diverged states to 0.336.

Ranking on the evidence: the cause is the collapse of post-ReLU BN running variance; (b)/(c) remove the un-normalised global sums that make the collapse catastrophic, bn_eps directly caps the gain, bn_mom slows the collapse, lr delays it, clip is irrelevant. The structural fix is to normalise before the non-linearity (conv -> BN -> ReLU -> dropout), which cannot be tested by continuing from this checkpoint because it changes the function the weights were trained for; this report does not implement or recommend any repo edit.

## 5. Mundane causes (all ruled out; `mundane_checks.py`, Appendix E)

- NaN/inf: 0 non-finite entries in atom/bond/global features and in labels over the full val split (14,634 molecules) and 25,600 train molecules.
- Unusual molecules: val atom count 9 to 100 (median 58), train 9 to 100 (median 57); max bonds 119 vs 118. Feature maxima are identical between splits (atom 162.3 in both, Rb one-hot; bond 363 vs 376, RBF tail). Val is not out of distribution.
- Same scaler: features and labels are scaled once at LMDB conversion (every shard has `scaled=True`, the same `feature_names`, `feature_size`, 83-element `element_set` and `target_dict`); neither `LMDBDataModule`/`TransformMol` nor `shared_step` applies a scaler, so train and val go through identical scaling by construction. Statistics agree: label mean 0.008 to 0.040 and std 0.876 to 1.048 in both splits, max |mean_train - mean_val| over all feature columns 0.014, over labels 0.0006. Label |y| max 59.5 (val) / 68.5 (train): heavy tails but finite.
- Metric accumulation: `compute_metrics` calls `.reset()` on `val_r2`, `val_torch_l1`, `val_torch_mse` after every `compute()` and is called once per epoch from `on_validation_epoch_end`; an empirical update/compute/reset/update check confirms the wrapper resets. `val_loss` uses Lightning's per-epoch `self.log` mean. `self.loss` (a torchmetrics `MultioutputWrapper(MeanSquaredError)` called via forward) accumulates internal state for the whole run, but the value it returns is batch-local, so the training loss is unaffected. Lightning evaluates in `model.eval()`, i.e. the running-statistics path that this report shows is the faulty one.
- Precision: eval in fp32 and bf16 autocast agree to 3 digits at every probe (Appendix B).
- The LMDB carries no QTAIM `extra_feat_*` columns despite `dataset.extra_keys` in the config (atom features are degree, H count, ring membership and 83 element one-hots; bond features are ring flags, bond-order one-hot and a 50-bin RBF of bond length). Unrelated to the divergence, but relevant to the training plateau at RMSE 0.64.

## Scripts (all in `profiling/divergence/`)

- `common.py`: config/datamodule/checkpoint loading (`weights_only=False`), fixed val subset, eval in eval-mode and batch-stat BN mode, BN buffer summaries, activation hooks, Adam rebuilt from the checkpoint optimizer state.
- `parse_timeline.py` -> `timeline.md`: per-epoch tables from train.log and TensorBoard, lr changes.
- `probe_continue.py <variant>` -> `probe_<variant>.jsonl`, `probe_<variant>.log`, `state_<variant>.pt`: continuation with probes every 150 steps; variants baseline, bn_mom001, mean_g, mean_all, sym_all, clip1, lr3e-4, bn_eps1e-2. `probe_baseline.jsonl` / `probe_baseline_2ep.log` is the 1860-step run; `probe_baseline_1050.*` is the 1050-step rerun that produced `state_baseline.pt` (the first run predates the state-saving line).
- `summarize_probes.py` -> `probe_summary.md`: all comparison tables.
- `global_input_stats.py` -> `global_input_stats.md`: section 3.
- `dead_channel_analysis.py [--state ... --variant ... --repair]` -> `dead_channels_<tag>.md`: per-channel BN gain / leak tables and the buffer-reset and eps causality tests (`epoch1`, `state_baseline`, `state_clip1`, `state_mean_g`).
- `mundane_checks.py` -> `mundane_checks.md`: section 5.

Runtime: each continuation took 96 to 300 s on GPU 1 (several ran concurrently); no experiment had to be shortened.

# Appendix A: timeline tables (parse_timeline.py)

## Q1a. Timeline parsed from train.log (last 100% progress line of each epoch)

| epoch | val_loss | val_r2 | val_mse (RMSE) | train_loss | train_mse (RMSE) |
|---|---|---|---|---|---|
| 0 | 2.170 | 0.692 | 0.564 | 3.460 | None |
| 1 | 2.090 | 0.720 | 0.551 | 2.830 | 0.743 |
| 2 | 1.55e+3 | -321. | 15.40 | 2.770 | 0.664 |
| 3 | 2.12e+5 | -4.79e+4 | 176.0 | 2.740 | 0.656 |
| 4 | 3.03e+7 | -4.49e+6 | 1.99e+3 | 2.730 | 0.653 |
| 5 | 1.17e+9 | -3.99e+7 | 1.03e+4 | 2.710 | 0.650 |
| 6 | 2.58e+8 | -2.98e+7 | 5.95e+3 | 2.700 | 0.648 |
| 7 | 2.57e+8 | -3.8e+7 | 6.12e+3 | 2.690 | 0.647 |
| 8 | 7.45e+10 | -9.26e+9 | 1e+5 | 2.680 | 0.646 |
| 9 | 2.16e+7 | -2.48e+6 | 1.69e+3 | 2.680 | 0.645 |
| 10 | 3.56e+11 | -6.26e+10 | 2.23e+5 | 2.680 | 0.644 |
| 11 | 6.75e+9 | -1.48e+9 | 3.23e+4 | 2.670 | 0.644 |
| 12 | 1.17e+10 | -2e+9 | 4.26e+4 | 2.660 | 0.643 |
| 13 | 1.39e+10 | -2.44e+9 | 4.34e+4 | 2.650 | 0.642 |
| 14 | 1.94e+10 | -2.4e+9 | 5.07e+4 | 2.650 | 0.641 |
| 15 | 6.05e+8 | -6.04e+7 | 8.66e+3 | 2.640 | 0.640 |
| 16 | 6.07e+9 | -8.75e+8 | 2.64e+4 | 2.640 | 0.640 |
| 17 | 1.93e+7 | -1.37e+6 | 1.55e+3 | 2.640 | 0.639 |
| 18 | 5.78e+8 | -9.4e+7 | 9.05e+3 | 2.630 | 0.639 |
| 19 | 1.08e+11 | -1.64e+10 | 1.05e+5 | 2.630 | 0.639 |
| 20 | 1.92e+10 | -1.19e+9 | 3.91e+4 | 2.630 | 0.639 |
| 21 | 1.93e+6 | -3.63e+5 | 543.0 | 2.630 | 0.639 |
| 22 | 1.8e+6 | -1.55e+5 | 493.0 | 2.630 | 0.638 |
| 23 | 4.18e+11 | -7e+10 | 2.58e+5 | 2.630 | 0.638 |
| 24 | 1.52e+9 | -1.86e+8 | 1.18e+4 | 2.620 | 0.638 |
| 25 | 2e+9 | -1.46e+6 | 8.66e+3 | 2.610 | 0.637 |
| 26 | 1.57e+12 | -6.74e+8 | 2.26e+5 | 2.610 | 0.636 |
| 27 | 1.57e+12 | -6.74e+8 | 2.26e+5 | 2.610 | 0.636 |

Note: the progress bar for epoch e shows val metrics from the end of epoch e once the bar reaches 100%; epoch 27 has no 100% line (run killed by SIGTERM mid-epoch).

## Q1b. Timeline from TensorBoard events (includes val_mae, the monitored metric, and lr-Adam)

| step | epoch | val_loss | val_r2 | val_mae | val_mse (RMSE) | train_loss | train_r2 | train_mae | train_mse (RMSE) | lr |
|---|---|---|---|---|---|---|---|---|---|---|
| 929 | 0 | 2.167 | 0.6918 | 0.2843 | 0.5636 | 3.461 | 0.4416 | 0.4471 | 0.7435 | 0.001 |
| 1859 | 1 | 2.088 | 0.7204 | 0.2787 | 0.5511 | 2.832 | 0.5665 | 0.3883 | 0.6643 | 0.001 |
| 2789 | 2 | 1555 | -320.6 | 0.398 | 15.4 | 2.772 | 0.5835 | 0.3829 | 0.6563 | 0.001 |
| 3719 | 3 | 2.117e+05 | -4.792e+04 | 3.034 | 176 | 2.744 | 0.5919 | 0.3805 | 0.6526 | 0.001 |
| 4649 | 4 | 3.028e+07 | -4.49e+06 | 26.29 | 1989 | 2.726 | 0.5975 | 0.3789 | 0.6502 | 0.001 |
| 5579 | 5 | 1.165e+09 | -3.99e+07 | 134.9 | 1.027e+04 | 2.711 | 0.6019 | 0.3778 | 0.6483 | 0.001 |
| 6509 | 6 | 2.576e+08 | -2.979e+07 | 63.63 | 5951 | 2.701 | 0.6048 | 0.377 | 0.6471 | 0.001 |
| 7439 | 7 | 2.566e+08 | -3.804e+07 | 62.44 | 6122 | 2.694 | 0.6074 | 0.3764 | 0.6462 | 0.001 |
| 8369 | 8 | 7.454e+10 | -9.256e+09 | 1801 | 1.001e+05 | 2.683 | 0.611 | 0.3757 | 0.6448 | 0.001 |
| 9299 | 9 | 2.16e+07 | -2.483e+06 | 42.08 | 1687 | 2.68 | 0.6124 | 0.3754 | 0.6444 | 0.001 |
| 10229 | 10 | 3.563e+11 | -6.262e+10 | 2478 | 2.228e+05 | 2.676 | 0.6141 | 0.3751 | 0.6439 | 0.001 |
| 11159 | 11 | 6.75e+09 | -1.479e+09 | 364.4 | 3.228e+04 | 2.672 | 0.6153 | 0.3749 | 0.6434 | 0.001 |
| 12089 | 12 | 1.171e+10 | -2.001e+09 | 662.6 | 4.264e+04 | 2.664 | 0.617 | 0.3744 | 0.6424 | 0.001 |
| 13019 | 13 | 1.393e+10 | -2.441e+09 | 538.9 | 4.342e+04 | 2.651 | 0.6202 | 0.3735 | 0.6408 | 0.0006 |
| 13949 | 14 | 1.937e+10 | -2.4e+09 | 725.8 | 5.067e+04 | 2.647 | 0.6213 | 0.3731 | 0.6403 | 0.0006 |
| 14879 | 15 | 6.05e+08 | -6.045e+07 | 98.08 | 8663 | 2.643 | 0.6222 | 0.373 | 0.6398 | 0.0006 |
| 15809 | 16 | 6.067e+09 | -8.746e+08 | 254.2 | 2.638e+04 | 2.64 | 0.6227 | 0.3728 | 0.6394 | 0.0006 |
| 16739 | 17 | 1.933e+07 | -1.366e+06 | 26.66 | 1547 | 2.637 | 0.6231 | 0.3727 | 0.6391 | 0.0006 |
| 17669 | 18 | 5.781e+08 | -9.397e+07 | 102.2 | 9050 | 2.635 | 0.6239 | 0.3726 | 0.6389 | 0.0006 |
| 18599 | 19 | 1.076e+11 | -1.638e+10 | 1176 | 1.054e+05 | 2.633 | 0.6237 | 0.3726 | 0.6388 | 0.0006 |
| 19529 | 20 | 1.917e+10 | -1.192e+09 | 431.5 | 3.909e+04 | 2.632 | 0.6242 | 0.3725 | 0.6385 | 0.0006 |
| 20459 | 21 | 1.925e+06 | -3.634e+05 | 7.761 | 543.2 | 2.63 | 0.6245 | 0.3725 | 0.6384 | 0.0006 |
| 21389 | 22 | 1.803e+06 | -1.55e+05 | 6.263 | 493.2 | 2.63 | 0.6249 | 0.3725 | 0.6384 | 0.0006 |
| 22319 | 23 | 4.176e+11 | -6.999e+10 | 3686 | 2.583e+05 | 2.63 | 0.6248 | 0.3725 | 0.6384 | 0.0006 |
| 23249 | 24 | 1.516e+09 | -1.859e+08 | 103.1 | 1.181e+04 | 2.618 | 0.6259 | 0.372 | 0.637 | 0.00036 |
| 24179 | 25 | 1.999e+09 | -1.462e+06 | 68.07 | 8656 | 2.611 | 0.6271 | 0.3717 | 0.6363 | 0.00036 |
| 25109 | 26 | 1.574e+12 | -6.738e+08 | 1799 | 2.255e+05 | 2.61 | 0.6274 | 0.3717 | 0.6362 | 0.00036 |

lr-Adam changes (step, value): (49, 0.001), (12099, 0.0006), (22349, 0.00036)
930 optimizer steps per epoch, so the drops at steps 12099 and 22349 are the ReduceLROnPlateau (factor 0.6, patience 10, monitor val_mae) firing after epochs 12 and 23: 1e-3 -> 6e-4 -> 3.6e-4.

Per-target val R2 (epochs 0,1,2,3):

| target | ep0 | ep1 | ep2 | ep3 |
|---|---|---|---|---|
| charge_adch | 0.7914 | 0.8011 | -320.6 | -2.989e+04 |
| charge_hirshfeld | 0.3623 | 0.3704 | -242.2 | -5.995e+04 |
| charge_cm5 | 0.8565 | 0.8621 | -461.7 | -4.792e+04 |
| charge_becke | 0.1653 | 0.181 | -36.68 | -2.243e+04 |
| charge_mulliken_orca | 0.6918 | 0.7204 | -398 | -3226 |
| charge_loewdin_orca | 0.8707 | 0.8827 | -200.1 | -6.03e+04 |
# Appendix B: continuation probes (summarize_probes.py)

### Baseline continuation from the epoch-1 checkpoint (step 0 = end of epoch 1; 930 steps per epoch)

| step | train loss (sum of 6 MSE) | grad norm pre-clip | eval-mode val MSE | eval fp32 | batch-stat val MSE | max eval act atom/bond/global | max batch-stat act | max train act | BN dead ch (rv<1e-3) | global rv max | global rm max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | - | - | 0.3428 | 0.3428 | 0.3675 | 100.3 / 279.3 / 128.3 | 277.7 | - | 7 | 1.734e+04 | 127 |
| 150 | 2.789 | 0.3038 | 0.3426 | 0.3426 | 0.3672 | 100 / 277.3 / 128.4 | 277 | 68.86 | 8 | 1.462e+04 | 118 |
| 300 | 2.77 | 0.22 | 0.3418 | 0.3419 | 0.3662 | 93.79 / 265.8 / 510.8 | 265.8 | 82.28 | 14 | 1.594e+04 | 114.1 |
| 450 | 2.807 | 0.2599 | 3.469 | 3.213 | 0.3639 | 9730 / 8717 / 8.864e+04 | 261.9 | 90.66 | 24 | 1.473e+04 | 109.4 |
| 600 | 2.725 | 0.2325 | 0.3756 | 0.372 | 0.3633 | 94.27 / 251.6 / 5421 | 251.8 | 82.9 | 30 | 1.404e+04 | 105.7 |
| 750 | 2.807 | 0.2416 | 0.3516 | 0.3524 | 0.3636 | 143 / 692.7 / 4779 | 237.9 | 184.5 | 38 | 1.135e+04 | 101.3 |
| 900 | 2.762 | 0.1826 | 0.4092 | 0.4143 | 0.361 | 97.88 / 873.9 / 2574 | 230.6 | 171.9 | 47 | 9849 | 95.16 |
| 1050 | 2.747 | 0.2296 | 4.201e+04 | 3.995e+04 | 0.361 | 2.414e+04 / 2.952e+05 / 8.766e+04 | 224.3 | 207.8 | 64 | 9335 | 92.31 |
| 1200 | 2.749 | 0.2801 | 37.81 | 36.78 | 0.361 | 776.7 / 3.162e+04 / 8965 | 225.1 | 77.36 | 69 | 7155 | 91.28 |
| 1350 | 2.728 | 0.2325 | 49.51 | 45.86 | 0.3608 | 1303 / 3.456e+04 / 2.561e+04 | 222.3 | 201.5 | 88 | 7582 | 95.03 |
| 1500 | 2.743 | 0.2109 | 1.7e+04 | 1.691e+04 | 0.3616 | 3.052e+04 / 6.318e+05 / 3.256e+05 | 230.5 | 82.89 | 96 | 1.143e+04 | 118.4 |
| 1650 | 2.735 | 0.2594 | 3.883e+05 | 3.723e+05 | 0.3609 | 6.941e+04 / 1.075e+07 / 4.85e+06 | 234.3 | 70.25 | 124 | 9783 | 119.6 |
| 1800 | 2.792 | 0.2432 | 47.68 | 48.89 | 0.3581 | 3938 / 9536 / 4.485e+05 | 224.9 | 55.98 | 129 | 1.001e+04 | 114.1 |

Per-block max |activation| in eval mode on the val subset (baseline):

| step | b0.atom | b0.bond | b0.global | b1.atom | b1.bond | b1.global | b2.atom | b2.bond | b2.global | b3.atom | b3.bond | b3.global |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 87.24 | 265.7 | 19.99 | 91.32 | 279 | 19.04 | 100.3 | 279.3 | 128.3 | 11.69 | 19.9 | 7.004 |
| 150 | 83.02 | 265 | 87.78 | 87.14 | 276.7 | 89.78 | 100 | 277.3 | 128.4 | 11.06 | 75.78 | 50.49 |
| 300 | 81.02 | 254.2 | 36.5 | 85.9 | 264.5 | 40.42 | 93.79 | 265.8 | 510.8 | 11.5 | 19.62 | 25.97 |
| 450 | 81.47 | 252.4 | 22.59 | 86.67 | 261.9 | 116.6 | 9730 | 8717 | 5.154e+04 | 371.3 | 695.6 | 8.864e+04 |
| 600 | 82.89 | 243.4 | 63.82 | 82.8 | 251.6 | 66.85 | 94.27 | 251.6 | 5421 | 31.3 | 211.4 | 301.6 |
| 750 | 83.23 | 235.8 | 150.6 | 82.29 | 242 | 156.3 | 143 | 241.4 | 4779 | 33.88 | 692.7 | 423.7 |
| 900 | 77.91 | 228.8 | 55.82 | 79.31 | 233.1 | 75.3 | 97.88 | 232.2 | 2574 | 22.95 | 873.9 | 404.1 |
| 1050 | 76.61 | 219.5 | 145 | 86.95 | 224.5 | 421.7 | 5988 | 5579 | 1.921e+04 | 2.414e+04 | 2.952e+05 | 8.766e+04 |
| 1200 | 75.79 | 218.2 | 34.6 | 88.27 | 222.2 | 114.8 | 776.7 | 765.4 | 8965 | 775.6 | 3.162e+04 | 6060 |
| 1350 | 78.97 | 216.4 | 188.1 | 80.78 | 222 | 191.3 | 292.4 | 290.2 | 2.561e+04 | 1303 | 3.456e+04 | 7764 |
| 1500 | 77.11 | 227.1 | 681.1 | 181.3 | 231.6 | 4451 | 1.292e+04 | 1.481e+04 | 2.949e+05 | 3.052e+04 | 6.318e+05 | 3.256e+05 |
| 1650 | 76.02 | 226.6 | 930.9 | 228.8 | 232.9 | 1.743e+04 | 1.108e+04 | 1.072e+04 | 4.85e+06 | 6.941e+04 | 1.075e+07 | 6.221e+05 |
| 1800 | 81.69 | 220.8 | 318 | 104.7 | 227 | 2117 | 732.6 | 409.3 | 4.485e+05 | 3938 | 9536 | 4.198e+04 |

BN dead channels (running_var < 1e-3) per block and node type (baseline):

| step | b0.atom | b0.bond | b0.global | b1.atom | b1.bond | b1.global | b2.atom | b2.bond | b2.global | b3.atom | b3.bond | b3.global |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 2 |
| 150 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 1 |
| 300 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 3 |
| 450 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 20 | 0 | 0 | 2 |
| 600 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 22 | 1 | 0 | 5 |
| 750 | 0 | 0 | 3 | 0 | 0 | 5 | 0 | 0 | 23 | 1 | 0 | 6 |
| 900 | 0 | 0 | 3 | 0 | 0 | 7 | 0 | 0 | 30 | 1 | 0 | 6 |
| 1050 | 0 | 0 | 5 | 0 | 0 | 12 | 0 | 0 | 37 | 1 | 0 | 9 |
| 1200 | 0 | 0 | 6 | 0 | 0 | 12 | 0 | 0 | 40 | 0 | 0 | 11 |
| 1350 | 0 | 0 | 10 | 0 | 0 | 15 | 0 | 0 | 45 | 0 | 0 | 18 |
| 1500 | 0 | 0 | 11 | 0 | 0 | 16 | 0 | 0 | 51 | 1 | 0 | 17 |
| 1650 | 0 | 0 | 16 | 0 | 0 | 27 | 0 | 0 | 63 | 1 | 0 | 17 |
| 1800 | 0 | 0 | 15 | 0 | 0 | 22 | 0 | 0 | 68 | 1 | 0 | 23 |

BN running_var min per block and node type (baseline):

| step | b0.atom | b0.bond | b0.global | b1.atom | b1.bond | b1.global | b2.atom | b2.bond | b2.global | b3.atom | b3.bond | b3.global |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.2239 | 0.1914 | 0.04453 | 0.1698 | 0.04639 | 0.1031 | 0.02169 | 0.04009 | 2.652e-21 | 0.1508 | 0.04086 | 2.349e-08 |
| 150 | 0.1328 | 0.1801 | 0.00294 | 0.1151 | 0.07306 | 0.1104 | 0.01145 | 0.04999 | 3.63e-28 | 0.2707 | 0.01376 | 3.215e-15 |
| 300 | 0.1086 | 0.1287 | 2.133e-08 | 0.07197 | 0.07909 | 0.1016 | 0.0139 | 0.02578 | 4.97e-35 | 0.114 | 0.007323 | 4.401e-22 |
| 450 | 0.09712 | 0.1501 | 1.899e-08 | 0.04263 | 0.04899 | 4.574e-05 | 0.01428 | 0.02205 | 6.803e-42 | 0.02771 | 0.04029 | 6.025e-29 |
| 600 | 0.06364 | 0.1195 | 6.407e-10 | 0.01036 | 0.07753 | 0.0003561 | 0.02669 | 0.01653 | 5.605e-45 | 0.0002097 | 0.02762 | 8.247e-36 |
| 750 | 0.05501 | 0.1744 | 8.77e-17 | 0.02113 | 0.02605 | 5.7e-07 | 0.01024 | 0.03236 | 5.605e-45 | 0.0002921 | 0.01426 | 1.128e-42 |
| 900 | 0.09172 | 0.1697 | 1.201e-23 | 0.03611 | 0.01814 | 7.803e-14 | 0.01349 | 0.01097 | 5.605e-45 | 4.836e-07 | 0.01008 | 5.605e-45 |
| 1050 | 0.1711 | 0.1385 | 1.643e-30 | 0.1047 | 0.02056 | 1.068e-20 | 0.007205 | 0.01014 | 5.605e-45 | 3.888e-08 | 0.01659 | 7.051e-30 |
| 1200 | 0.07136 | 0.1155 | 2.25e-37 | 0.01527 | 0.04287 | 8.334e-26 | 0.0212 | 0.02116 | 5.605e-45 | 0.1544 | 0.004939 | 9.652e-37 |
| 1350 | 0.08488 | 0.1227 | 2.943e-44 | 0.007635 | 0.008128 | 1.141e-32 | 0.01051 | 0.03823 | 5.605e-45 | 0.001572 | 0.004321 | 1.317e-43 |
| 1500 | 0.08926 | 0.112 | 5.605e-45 | 0.006449 | 0.004145 | 1.562e-39 | 0.006672 | 0.007611 | 5.605e-45 | 8.67e-06 | 0.007298 | 5.605e-45 |
| 1650 | 0.07905 | 0.09749 | 5.605e-45 | 0.005277 | 0.01551 | 5.605e-45 | 0.002271 | 0.01694 | 5.605e-45 | 1.153e-06 | 0.004523 | 5.605e-45 |
| 1800 | 0.06726 | 0.07034 | 5.605e-45 | 0.01702 | 0.01436 | 3.354e-28 | 0.0128 | 0.01586 | 5.605e-45 | 2.811e-05 | 0.004627 | 5.605e-45 |

### Hypothesis tests: eval-mode val MSE (fixed 50-batch val subset) per probe step

| step | baseline | bn_mom001 | mean_g | mean_all | sym_all | clip1 | lr3e-4 | bn_eps1e-2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3428 | 0.3428 | 0.3615 | 0.5644 | 0.5855 | 0.3428 | 0.3428 | 0.3431 |
| 150 | 0.3426 | 0.3424 | 0.3437 | 0.3524 | 0.3505 | 0.3425 | 0.3414 | 0.3422 |
| 300 | 0.3418 | 0.3436 | 0.3432 | 0.3474 | 0.3472 | 0.357 | 0.3397 | 0.3419 |
| 450 | 3.469 | 0.3397 | 0.3419 | 0.3432 | 0.3433 | 0.3482 | 0.3393 | 0.3394 |
| 600 | 0.3756 | 0.3407 | 0.3407 | 0.3432 | 0.3434 | 0.5363 | 0.3608 | 0.3398 |
| 750 | 0.3516 | 0.3389 | 0.3366 | 0.3397 | 0.3397 | 0.341 | 0.3388 | 0.3361 |
| 900 | 0.4092 | 0.3366 | 0.3367 | 0.3376 | 0.3378 | 10.36 | 0.3516 | 0.3369 |
| 1050 | 4.201e+04 |  |  |  |  |  |  |  |
| 1200 | 37.81 |  |  |  |  |  |  |  |
| 1350 | 49.51 |  |  |  |  |  |  |  |
| 1500 | 1.7e+04 |  |  |  |  |  |  |  |
| 1650 | 3.883e+05 |  |  |  |  |  |  |  |
| 1800 | 47.68 |  |  |  |  |  |  |  |

Batch-stat BN val MSE per probe step (same runs):

| step | baseline | bn_mom001 | mean_g | mean_all | sym_all | clip1 | lr3e-4 | bn_eps1e-2 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.3675 | 0.3675 | 0.3756 | 0.4557 | 0.4462 | 0.3675 | 0.3675 | 0.3673 |
| 150 | 0.3672 | 0.3669 | 0.3683 | 0.376 | 0.3725 | 0.3669 | 0.3649 | 0.3666 |
| 300 | 0.3662 | 0.3661 | 0.367 | 0.3722 | 0.3707 | 0.3663 | 0.364 | 0.3659 |
| 450 | 0.3639 | 0.3641 | 0.3645 | 0.368 | 0.3668 | 0.3634 | 0.3638 | 0.3631 |
| 600 | 0.3633 | 0.3637 | 0.3635 | 0.3677 | 0.3659 | 0.3629 | 0.3631 | 0.3625 |
| 750 | 0.3636 | 0.3636 | 0.3647 | 0.3684 | 0.3673 | 0.3635 | 0.3642 | 0.3632 |
| 900 | 0.361 | 0.3611 | 0.3624 | 0.3653 | 0.3646 | 0.361 | 0.3612 | 0.3609 |
| 1050 | 0.361 |  |  |  |  |  |  |  |
| 1200 | 0.361 |  |  |  |  |  |  |  |
| 1350 | 0.3608 |  |  |  |  |  |  |  |
| 1500 | 0.3616 |  |  |  |  |  |  |  |
| 1650 | 0.3609 |  |  |  |  |  |  |  |
| 1800 | 0.3581 |  |  |  |  |  |  |  |

Train loss (running mean over the 150 steps before the probe) and grad norm before clipping:

| step | baseline loss / gnorm | bn_mom001 loss / gnorm | mean_g loss / gnorm | mean_all loss / gnorm | sym_all loss / gnorm | clip1 loss / gnorm | lr3e-4 loss / gnorm | bn_eps1e-2 loss / gnorm |
|---|---|---|---|---|---|---|---|---|
| 0 | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - |
| 150 | 2.789 / 0.3038 | 2.788 / 0.7324 | 2.808 / 0.2959 | 2.881 / 0.2978 | 2.875 / 0.2939 | 2.787 / 0.2999 | 2.782 / 0.3001 | 2.787 / 0.3038 |
| 300 | 2.77 / 0.22 | 2.77 / 0.2203 | 2.781 / 0.2252 | 2.812 / 0.2199 | 2.807 / 0.2256 | 2.77 / 0.2208 | 2.767 / 0.2283 | 2.769 / 0.2237 |
| 450 | 2.807 / 0.2599 | 2.808 / 0.2693 | 2.816 / 0.2543 | 2.839 / 0.2924 | 2.834 / 0.2767 | 2.807 / 0.2558 | 2.807 / 0.2657 | 2.807 / 0.2543 |
| 600 | 2.725 / 0.2325 | 2.725 / 0.2517 | 2.731 / 0.232 | 2.747 / 0.2437 | 2.745 / 0.2345 | 2.725 / 0.2428 | 2.727 / 0.2423 | 2.724 / 0.2404 |
| 750 | 2.807 / 0.2416 | 2.807 / 0.2339 | 2.812 / 0.2189 | 2.825 / 0.2464 | 2.823 / 0.2279 | 2.806 / 0.224 | 2.811 / 0.2335 | 2.805 / 0.2209 |
| 900 | 2.762 / 0.1826 | 2.762 / 0.1906 | 2.766 / 0.1802 | 2.777 / 0.1902 | 2.776 / 0.1954 | 2.762 / 0.1891 | 2.77 / 0.1966 | 2.76 / 0.1879 |
| 1050 | 2.747 / 0.2296 |  |  |  |  |  |  |  |
| 1200 | 2.749 / 0.2801 |  |  |  |  |  |  |  |
| 1350 | 2.728 / 0.2325 |  |  |  |  |  |  |  |
| 1500 | 2.743 / 0.2109 |  |  |  |  |  |  |  |
| 1650 | 2.735 / 0.2594 |  |  |  |  |  |  |  |
| 1800 | 2.792 / 0.2432 |  |  |  |  |  |  |  |

Dead BN channels (running_var < 1e-3, all 72 BN modules) and max eval-mode activation:

| step | baseline dead / act | bn_mom001 dead / act | mean_g dead / act | mean_all dead / act | sym_all dead / act | clip1 dead / act | lr3e-4 dead / act | bn_eps1e-2 dead / act |
|---|---|---|---|---|---|---|---|---|
| 0 | 7 / 279.3 | 7 / 279.4 | 7 / 279.5 | 7 / 282 | 7 / 283.6 | 7 / 279.4 | 7 / 279.3 | 7 / 279.2 |
| 150 | 8 / 277.3 | 7 / 270.7 | 17 / 274.1 | 18 / 274.9 | 21 / 285.6 | 12 / 277.7 | 7 / 278.1 | 14 / 277.5 |
| 300 | 14 / 510.8 | 6 / 252.4 | 21 / 263.8 | 23 / 270.1 | 21 / 268.9 | 23 / 4735 | 7 / 276.8 | 18 / 269.3 |
| 450 | 24 / 8.864e+04 | 6 / 264.8 | 24 / 257.4 | 27 / 261.6 | 28 / 258.7 | 36 / 5253 | 11 / 890.1 | 31 / 281.2 |
| 600 | 30 / 5421 | 7 / 259.6 | 27 / 246.9 | 44 / 255.1 | 41 / 244 | 37 / 5.268e+04 | 12 / 7249 | 39 / 266.9 |
| 750 | 38 / 4779 | 9 / 249.9 | 40 / 240.3 | 51 / 246.1 | 51 / 257.3 | 48 / 3885 | 10 / 270.8 | 42 / 268.3 |
| 900 | 47 / 2574 | 10 / 304.5 | 52 / 233.3 | 62 / 244.5 | 67 / 250.1 | 55 / 1.12e+04 | 12 / 3810 | 48 / 966.9 |
| 1050 | 64 / 2.952e+05 |  |  |  |  |  |  |  |
| 1200 | 69 / 3.162e+04 |  |  |  |  |  |  |  |
| 1350 | 88 / 3.456e+04 |  |  |  |  |  |  |  |
| 1500 | 96 / 6.318e+05 |  |  |  |  |  |  |  |
| 1650 | 124 / 1.075e+07 |  |  |  |  |  |  |  |
| 1800 | 129 / 4.485e+05 |  |  |  |  |  |  |  |

Summary per variant: max eval-mode val MSE over probes at steps 150..930, and whether it stayed below 1:

| variant | max eval MSE (steps 150-930) | min | final (930) | stayed < 1 | max batch-stat MSE |
|---|---|---|---|---|---|
| baseline | 3.469 | 0.3418 | - | no | 0.3672 |
| bn_mom001 | 0.3436 | 0.3366 | - | yes | 0.3669 |
| mean_g | 0.3437 | 0.3366 | - | yes | 0.3683 |
| mean_all | 0.3524 | 0.3376 | - | yes | 0.376 |
| sym_all | 0.3505 | 0.3378 | - | yes | 0.3725 |
| clip1 | 10.36 | 0.341 | - | no | 0.3669 |
| lr3e-4 | 0.3608 | 0.3388 | - | yes | 0.3649 |
| bn_eps1e-2 | 0.3422 | 0.3361 | - | yes | 0.3666 |
# Appendix C: dead-channel analyses

## Dead-channel analysis: epoch1

eval-mode val MSE on 50 batches: 0.3428; max |pred| 11.7

BN channels total: 8088; running_var < 1e-3: 7; < 1e-2: 8; eval gain gamma/sqrt(rv+eps) > 10: 6, > 100: 0

Top 20 channels by eval-mode output max |y| on the val subset:

| module | ch | running_var | running_mean | gain=|gamma|/sqrt(rv+eps) | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|---|
| b0.l0.g2b->bond | 9 | 1.5 | 0.484 | 0.703 | 304 | 213 | 164143/394279 |
| b0.l0.b2b->bond | 63 | 1.83 | 0.501 | 0.486 | 332 | 161 | 104086/394279 |
| b0.l0.b2b->bond | 124 | 4.03 | 0.703 | 0.433 | 364 | 157 | 199232/394279 |
| b0.l0.g2b->bond | 47 | 2.02 | 0.726 | 0.617 | 250 | 154 | 219280/394279 |
| b0.l0.b2b->bond | 120 | 4.3 | 1.1 | 0.295 | 456 | 134 | 188705/394279 |
| b0.l0.g2b->bond | 38 | 2.52 | 0.608 | 0.407 | 328 | 133 | 184636/394279 |
| b0.l0.g2b->bond | 92 | 2.23 | 0.611 | 0.569 | 232 | 132 | 145377/394279 |
| b3.l0.a2g->global | 40 | 7.87e-06 | 1.37e-06 | 21.7 | 6.03 | 131 | 4/6400 |
| b2.l1.a2g->global | 94 | 0.0717 | 0.00323 | 2.2 | 56.2 | 124 | 12/6400 |
| b0.l0.g2b->bond | 32 | 3.97 | 0.576 | 0.326 | 348 | 114 | 122304/394279 |
| b0.l0.g2b->bond | 127 | 1.08 | 0.557 | 0.844 | 135 | 114 | 183631/394279 |
| b1.l0.b2b->bond | 81 | 4.32 | 1.11 | 0.311 | 348 | 108 | 225268/394279 |
| b0.l0.g2b->bond | 85 | 2.06 | 0.541 | 0.622 | 168 | 104 | 144810/394279 |
| b0.l0.b2b->bond | 37 | 3.69 | 0.849 | 0.453 | 221 | 100 | 245564/394279 |
| b0.l0.g2b->bond | 76 | 3.45 | 0.569 | 0.409 | 240 | 98 | 112522/394279 |
| b0.l0.b2b->bond | 55 | 2.28 | 0.67 | 0.581 | 156 | 90 | 195000/394279 |
| b2.l0.g2b->bond | 51 | 6.83 | 1.02 | 0.312 | 282 | 88 | 118037/394279 |
| b0.l1.b2b->bond | 58 | 5.68 | 1.01 | 0.388 | 226 | 87.5 | 211153/394279 |
| b1.l0.b2b->bond | 65 | 4.13 | 1.07 | 0.368 | 234 | 85.5 | 201311/394279 |
| b1.l0.g2b->bond | 7 | 2.64 | 0.836 | 0.36 | 234 | 84 | 193511/394279 |

Channels with running_var < 1e-3 (dead on training data) that receive non-zero input on held-out molecules:

| module | ch | running_var | gain | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|
| b3.l0.a2g->global | 40 | 7.87e-06 | 21.7 | 6.03 | 131 | 4/6400 |
(1 such channels in total)

Max eval-mode BN output per destination node type and block:
b0.atom: 70, b0.bond: 213, b0.global: 17.1, b1.atom: 55.2, b1.bond: 108, b1.global: 31.5, b2.atom: 42, b2.bond: 88, b2.global: 124, b3.atom: 42.5, b3.bond: 81, b3.global: 131
## Dead-channel analysis: state_baseline

eval-mode val MSE on 50 batches: 62.54; max |pred| 1.26e+03

BN channels total: 8088; running_var < 1e-3: 59; < 1e-2: 72; eval gain gamma/sqrt(rv+eps) > 10: 54, > 100: 12

Top 20 channels by eval-mode output max |y| on the val subset:

| module | ch | running_var | running_mean | gain=|gamma|/sqrt(rv+eps) | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|---|
| b3.l0.a2g->global | 53 | 1.48e-10 | 2.83e-11 | 131 | 8.45e+04 | 1.11e+07 | 5/6400 |
| b3.l1.g2g->global | 0 | 0.618 | 0.336 | 0.997 | 6.49e+06 | 6.46e+06 | 1681/6400 |
| b3.l0.a2g->global | 42 | 3.29e-40 | 1.49e-41 | 177 | 2.71e+04 | 4.82e+06 | 1/6400 |
| b3.l0.a2g->global | 93 | 0.000121 | 5.49e-06 | 14.1 | 1.04e+05 | 1.47e+06 | 4/6400 |
| b3.l0.a2g->global | 52 | 0.281 | 0.00973 | 1.51 | 1.17e+05 | 1.77e+05 | 1/6400 |
| b3.l0.a2g->global | 15 | 0.0432 | 0.00402 | 3.06 | 5.45e+04 | 1.67e+05 | 14/6400 |
| b2.l1.a2g->global | 12 | 2.3e-13 | 2.07e-14 | 170 | 588 | 1e+05 | 4/6400 |
| b2.l1.a2g->global | 111 | 1.03e-11 | 5.9e-13 | 131 | 612 | 8.04e+04 | 2/6400 |
| b3.l0.a2g->global | 31 | 0.172 | 0.0163 | 0.423 | 1.79e+05 | 7.58e+04 | 41/6400 |
| b3.l0.b2g->global | 122 | 0.0211 | 0.00223 | 0.521 | 1.4e+05 | 7.32e+04 | 22/6400 |
| b2.l0.a2g->global | 36 | 7.63e-35 | 2.57e-36 | 134 | 300 | 4.02e+04 | 1/6400 |
| b2.l0.b2g->global | 95 | 4.96e-08 | 1.51e-09 | 171 | 220 | 3.76e+04 | 2/6400 |
| b2.l1.a2g->global | 20 | 0.00018 | 0.000155 | 26.7 | 1.41e+03 | 3.76e+04 | 10/6400 |
| b2.l1.a2g->global | 8 | 3.81e-19 | 4.05e-20 | 44.7 | 524 | 2.34e+04 | 7/6400 |
| b2.l1.a2g->global | 66 | 1.8e-25 | 6.54e-27 | 102 | 202 | 2.06e+04 | 3/6400 |
| b3.l0.b2g->global | 59 | 4.06e-17 | 5.3e-18 | 46.9 | 372 | 1.74e+04 | 5/6400 |
| b3.l0.b2g->global | 56 | 135 | 2.53 | 0.0625 | 2.4e+05 | 1.5e+04 | 399/6400 |
| b3.l0.b2g->global | 12 | 229 | 3.54 | 0.0325 | 4.3e+05 | 1.4e+04 | 628/6400 |
| b2.l1.a2g->global | 9 | 2.13e-16 | 4.89e-17 | 36.6 | 374 | 1.37e+04 | 7/6400 |
| b3.l0.a2g->global | 81 | 26.8 | 0.66 | 0.104 | 1.27e+05 | 1.32e+04 | 172/6400 |

Channels with running_var < 1e-3 (dead on training data) that receive non-zero input on held-out molecules:

| module | ch | running_var | gain | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|
| b3.l0.a2g->global | 53 | 1.48e-10 | 131 | 8.45e+04 | 1.11e+07 | 5/6400 |
| b3.l0.a2g->global | 42 | 3.29e-40 | 177 | 2.71e+04 | 4.82e+06 | 1/6400 |
| b3.l0.a2g->global | 93 | 0.000121 | 14.1 | 1.04e+05 | 1.47e+06 | 4/6400 |
| b2.l1.a2g->global | 12 | 2.3e-13 | 170 | 588 | 1e+05 | 4/6400 |
| b2.l1.a2g->global | 111 | 1.03e-11 | 131 | 612 | 8.04e+04 | 2/6400 |
| b2.l0.a2g->global | 36 | 7.63e-35 | 134 | 300 | 4.02e+04 | 1/6400 |
| b2.l0.b2g->global | 95 | 4.96e-08 | 171 | 220 | 3.76e+04 | 2/6400 |
| b2.l1.a2g->global | 20 | 0.00018 | 26.7 | 1.41e+03 | 3.76e+04 | 10/6400 |
| b2.l1.a2g->global | 8 | 3.81e-19 | 44.7 | 524 | 2.34e+04 | 7/6400 |
| b2.l1.a2g->global | 66 | 1.8e-25 | 102 | 202 | 2.06e+04 | 3/6400 |
| b3.l0.b2g->global | 59 | 4.06e-17 | 46.9 | 372 | 1.74e+04 | 5/6400 |
| b2.l1.a2g->global | 9 | 2.13e-16 | 36.6 | 374 | 1.37e+04 | 7/6400 |
| b2.l1.a2g->global | 90 | 4.83e-13 | 59.2 | 184 | 1.09e+04 | 4/6400 |
| b2.l0.b2g->global | 93 | 0.000228 | 51.4 | 198 | 1.02e+04 | 1/6400 |
| b2.l1.a2g->global | 15 | 1.7e-06 | 49.7 | 162 | 8.06e+03 | 5/6400 |
| b3.l0.a2g->global | 26 | 6.66e-06 | 15.7 | 512 | 8.03e+03 | 3/6400 |
| b2.l1.a2g->global | 83 | 1.42e-42 | 21.8 | 229 | 4.99e+03 | 7/6400 |
| b2.l1.a2g->global | 98 | 2.18e-08 | 25.4 | 163 | 4.13e+03 | 9/6400 |
| b3.l0.a2g->global | 40 | 5.61e-45 | 25.7 | 137 | 3.52e+03 | 16/6400 |
| b2.l0.b2g->global | 83 | 2.36e-09 | 16.9 | 147 | 2.5e+03 | 1/6400 |
| b2.l0.b2g->global | 126 | 9.85e-21 | 82.3 | 21.9 | 1.8e+03 | 3/6400 |
| b2.l0.b2g->global | 80 | 0.000884 | 10.6 | 156 | 1.65e+03 | 10/6400 |
| b2.l0.a2g->global | 76 | 1.15e-22 | 30.9 | 23.2 | 720 | 1/6400 |
| b1.l1.b2g->global | 52 | 0.000329 | 13.9 | 37.5 | 524 | 10/6400 |
| b2.l0.b2g->global | 75 | 1.35e-15 | 11.1 | 31.2 | 346 | 1/6400 |
(32 such channels in total)

Max eval-mode BN output per destination node type and block:
b0.atom: 69.5, b0.bond: 120, b0.global: 22.1, b1.atom: 45, b1.bond: 81, b1.global: 524, b2.atom: 8.45e+03, b2.bond: 6.46e+03, b2.global: 1e+05, b3.atom: 1.1e+04, b3.bond: 8.45e+03, b3.global: 1.11e+07

### Repair test: causality check
reset 59 channels with running_var < 1e-3 to running_var=1, running_mean=0 (weights untouched): eval-mode val MSE 62.54 -> 0.3352; max |pred| 1.26e+03 -> 11.3
alternative, original buffers but BN eps 1e-5 -> 1e-2 (caps gain at |gamma|/0.1): eval-mode val MSE 0.3356; max |pred| 11.3
reference, batch-stat BN on the same weights: val MSE 0.3613
## Dead-channel analysis: state_clip1

eval-mode val MSE on 50 batches: 600.7; max |pred| 2.93e+03

BN channels total: 8088; running_var < 1e-3: 64; < 1e-2: 77; eval gain gamma/sqrt(rv+eps) > 10: 54, > 100: 14

Top 20 channels by eval-mode output max |y| on the val subset:

| module | ch | running_var | running_mean | gain=|gamma|/sqrt(rv+eps) | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|---|
| b3.l0.b2g->global | 68 | 1.52e-27 | 1.29e-28 | 18.1 | 1.25e+04 | 2.27e+05 | 1/6400 |
| b3.l0.a2g->global | 29 | 0.969 | 0.0318 | 0.521 | 1.05e+05 | 5.5e+04 | 15/6400 |
| b3.l0.b2g->global | 39 | 0.514 | 0.0226 | 1.05 | 4.79e+04 | 5.02e+04 | 1/6400 |
| b3.l1.g2g->global | 0 | 0.59 | 0.335 | 1.02 | 2.43e+04 | 2.48e+04 | 1904/6400 |
| b2.l0.b2g->global | 126 | 1.38e-15 | 8.85e-17 | 82.5 | 264 | 2.18e+04 | 1/6400 |
| b3.l0.b2g->global | 18 | 3.78 | 0.0456 | 0.232 | 3.79e+04 | 8.83e+03 | 16/6400 |
| b3.l0.a2g->global | 66 | 76.4 | 1.8 | 0.0733 | 7.53e+04 | 5.5e+03 | 630/6400 |
| b2.l1.g2g->global | 17 | 0.511 | 0.231 | 1.1 | 4.16e+03 | 4.61e+03 | 1630/6400 |
| b3.l0.a2g->global | 0 | 206 | 3.75 | 0.0543 | 7.68e+04 | 4.16e+03 | 797/6400 |
| b2.l1.g2g->global | 32 | 1.44 | 0.695 | 0.611 | 5.89e+03 | 3.6e+03 | 3261/6400 |
| b3.l0.a2g->global | 6 | 312 | 5.74 | 0.0408 | 8.29e+04 | 3.39e+03 | 877/6400 |
| b2.l1.g2a->atom | 85 | 0.442 | 0.148 | 0.834 | 4.02e+03 | 3.34e+03 | 50913/369752 |
| b3.l0.b2g->global | 12 | 265 | 4.62 | 0.0302 | 1.03e+05 | 3.1e+03 | 965/6400 |
| b2.l1.g2b->bond | 106 | 1.14 | 0.271 | 0.803 | 3.79e+03 | 3.04e+03 | 49553/394279 |
| b3.l0.a2g->global | 57 | 95.1 | 2.41 | 0.0524 | 5.68e+04 | 2.98e+03 | 953/6400 |
| b3.l0.a2g->global | 30 | 60.5 | 1.64 | 0.0664 | 4.45e+04 | 2.96e+03 | 523/6400 |
| b3.l0.a2g->global | 120 | 198 | 4.27 | 0.0452 | 6.55e+04 | 2.96e+03 | 856/6400 |
| b3.l1.g2a->atom | 4 | 64.7 | 9.38 | 0.0389 | 7.48e+04 | 2.91e+03 | 310275/369752 |
| b3.l0.a2g->global | 2 | 1.19e+03 | 20 | 0.0234 | 1.14e+05 | 2.67e+03 | 2350/6400 |
| b3.l0.a2g->global | 116 | 0.0338 | 0.00441 | 0.477 | 5.6e+03 | 2.67e+03 | 4/6400 |

Channels with running_var < 1e-3 (dead on training data) that receive non-zero input on held-out molecules:

| module | ch | running_var | gain | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|
| b3.l0.b2g->global | 68 | 1.52e-27 | 18.1 | 1.25e+04 | 2.27e+05 | 1/6400 |
| b2.l0.b2g->global | 126 | 1.38e-15 | 82.5 | 264 | 2.18e+04 | 1/6400 |
| b2.l0.a2g->global | 58 | 1.64e-11 | 66.3 | 30.9 | 2.05e+03 | 1/6400 |
| b2.l1.a2g->global | 67 | 5.35e-06 | 116 | 3.19 | 372 | 3/6400 |
| b2.l1.b2g->global | 41 | 0.000871 | 6.42 | 53.5 | 344 | 5/6400 |
| b1.l0.b2g->global | 108 | 6.2e-06 | 117 | 2.52 | 294 | 1/6400 |
| b3.l0.g2b->bond | 84 | 0.000307 | 12.5 | 18.2 | 229 | 224/394279 |
| b3.l0.g2b->bond | 34 | 1.14e-06 | 110 | 1.06 | 116 | 6/394279 |
| b2.l1.b2g->global | 61 | 1.47e-06 | 73.1 | 1.41 | 103 | 1/6400 |
| b2.l1.a2g->global | 97 | 9.69e-14 | 181 | 0.523 | 94.5 | 1/6400 |
| b1.l1.a2g->global | 4 | 0.000415 | 39.8 | 1.32 | 52.5 | 3/6400 |
| b2.l1.b2g->global | 101 | 0.000332 | 6.11 | 6.19 | 38 | 4/6400 |
| b2.l1.a2g->global | 73 | 0.000173 | 9.43 | 2.73 | 25.8 | 4/6400 |
| b3.l0.a2g->global | 115 | 0.000567 | 7.72 | 3.25 | 25.2 | 2/6400 |
| b2.l1.a2g->global | 89 | 0.000635 | 4.33 | 3.86 | 16.8 | 1/6400 |
| b2.l1.a2g->global | 43 | 0.000235 | 4.48 | 1.87 | 8.31 | 1/6400 |
| b2.l1.b2g->global | 39 | 7.83e-16 | 16 | 0.161 | 2.42 | 1/6400 |
| b2.l1.b2g->global | 99 | 0.000478 | 3.92 | 0.041 | 0.149 | 1/6400 |
(18 such channels in total)

Max eval-mode BN output per destination node type and block:
b0.atom: 68, b0.bond: 144, b0.global: 26.4, b1.atom: 43, b1.bond: 75, b1.global: 294, b2.atom: 3.34e+03, b2.bond: 3.04e+03, b2.global: 2.18e+04, b3.atom: 2.91e+03, b3.bond: 648, b3.global: 2.27e+05

### Repair test: causality check
reset 64 channels with running_var < 1e-3 to running_var=1, running_mean=0 (weights untouched): eval-mode val MSE 600.7 -> 0.3375; max |pred| 2.93e+03 -> 12.8
alternative, original buffers but BN eps 1e-5 -> 1e-2 (caps gain at |gamma|/0.1): eval-mode val MSE 0.3377; max |pred| 12.7
reference, batch-stat BN on the same weights: val MSE 0.3616
## Dead-channel analysis: state_mean_g

eval-mode val MSE on 50 batches: 0.3381; max |pred| 13.5

BN channels total: 8088; running_var < 1e-3: 52; < 1e-2: 66; eval gain gamma/sqrt(rv+eps) > 10: 52, > 100: 19

Top 20 channels by eval-mode output max |y| on the val subset:

| module | ch | running_var | running_mean | gain=|gamma|/sqrt(rv+eps) | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|---|
| b0.l0.b2b->bond | 124 | 3.79 | 0.721 | 0.425 | 330 | 140 | 181451/394279 |
| b0.l0.g2b->bond | 9 | 2 | 0.642 | 0.591 | 213 | 126 | 171405/394279 |
| b0.l0.g2b->bond | 38 | 2.45 | 0.579 | 0.415 | 256 | 106 | 167802/394279 |
| b0.l0.g2b->bond | 92 | 3 | 0.582 | 0.476 | 212 | 101 | 133216/394279 |
| b0.l0.b2b->bond | 63 | 3.89 | 0.632 | 0.323 | 294 | 95 | 109753/394279 |
| b0.l0.g2b->bond | 47 | 3.28 | 0.561 | 0.454 | 207 | 94 | 178816/394279 |
| b0.l0.g2b->bond | 127 | 1.76 | 0.632 | 0.628 | 140 | 87.5 | 173588/394279 |
| b0.l0.g2b->bond | 76 | 3.63 | 0.509 | 0.388 | 224 | 87 | 108978/394279 |
| b0.l0.g2b->bond | 32 | 4.71 | 0.705 | 0.284 | 296 | 84 | 167064/394279 |
| b0.l0.g2b->bond | 85 | 2.43 | 0.629 | 0.559 | 147 | 82 | 157341/394279 |
| b0.l0.b2b->bond | 120 | 5.28 | 1.17 | 0.274 | 300 | 82 | 185414/394279 |
| b1.l0.b2b->bond | 76 | 3.74 | 0.914 | 0.256 | 284 | 72.5 | 187504/394279 |
| b2.l1.b2a->atom | 47 | 0.904 | 0.18 | 0.929 | 71 | 66 | 29275/369752 |
| b2.l1.b2a->atom | 50 | 1.51 | 0.145 | 0.779 | 84.5 | 66 | 13208/369752 |
| b2.l0.a2b->bond | 104 | 0.419 | 0.0463 | 0.918 | 71.5 | 65.5 | 5107/394279 |
| b0.l0.g2b->bond | 36 | 1.91 | 0.595 | 0.637 | 102 | 64.5 | 163654/394279 |
| b0.l1.b2b->bond | 58 | 4.53 | 0.77 | 0.398 | 161 | 64 | 208368/394279 |
| b0.l0.g2b->bond | 4 | 3 | 0.703 | 0.346 | 184 | 63.5 | 186433/394279 |
| b0.l0.b2b->bond | 94 | 6.9 | 1.23 | 0.251 | 254 | 63.5 | 153112/394279 |
| b0.l0.b2b->bond | 0 | 6.53 | 1.17 | 0.312 | 200 | 62 | 161276/394279 |

Channels with running_var < 1e-3 (dead on training data) that receive non-zero input on held-out molecules:

| module | ch | running_var | gain | max in | max out | nodes with in!=0 / total |
|---|---|---|---|---|---|---|
| b2.l1.a2g->global | 42 | 8.11e-20 | 61.7 | 0.969 | 59.8 | 2/6400 |
| b2.l0.a2g->global | 107 | 0.00029 | 22.5 | 2.2 | 49.8 | 7/6400 |
| b3.l0.b2g->global | 41 | 6.17e-07 | 218 | 0.0625 | 13.7 | 1/6400 |
| b2.l1.a2g->global | 112 | 2.04e-07 | 61.7 | 0.0352 | 2.12 | 2/6400 |
| b2.l0.a2g->global | 65 | 8.27e-38 | 24.6 | 0.043 | 1.14 | 1/6400 |
| b2.l0.g2g->global | 124 | 0.000362 | 1.51 | 0.496 | 0.738 | 2/6400 |
| b1.l1.b2g->global | 89 | 0.000607 | 0.41 | 0.398 | 0.104 | 644/6400 |
(7 such channels in total)

Max eval-mode BN output per destination node type and block:
b0.atom: 61.5, b0.bond: 140, b0.global: 11.9, b1.atom: 47.8, b1.bond: 72.5, b1.global: 31.4, b2.atom: 66, b2.bond: 65.5, b2.global: 59.8, b3.atom: 48.2, b3.bond: 57, b3.global: 25.6
# Appendix D: global-node input statistics

## Q3a. Scaled feature columns with extreme values (50 val batches + 50 train batches)

val atom: 369752 nodes, 54 columns with max |x| > 50, overall max 162.28
    col   9 chemical_symbol_Ag           max   145.15 min  -0.007 nodes>50: 4
    col  10 chemical_symbol_Al           max    65.08 min  -0.015 nodes>50: 113
    col  12 chemical_symbol_As           max    71.03 min  -0.014 nodes>50: 61
    col  13 chemical_symbol_Au           max    63.81 min  -0.016 nodes>50: 115
    col  15 chemical_symbol_Ba           max    71.34 min  -0.014 nodes>50: 71
    col  16 chemical_symbol_Be           max    61.93 min  -0.016 nodes>50: 109
    col  17 chemical_symbol_Bi           max    64.56 min  -0.015 nodes>50: 117
    col  20 chemical_symbol_Ca           max    67.62 min  -0.015 nodes>50: 89
    col  21 chemical_symbol_Cd           max    69.67 min  -0.014 nodes>50: 113
    col  22 chemical_symbol_Ce           max    56.58 min  -0.018 nodes>50: 158
    col  24 chemical_symbol_Co           max    51.26 min  -0.020 nodes>50: 199
    col  25 chemical_symbol_Cr           max    53.93 min  -0.019 nodes>50: 136
    col  26 chemical_symbol_Cs           max   157.08 min  -0.006 nodes>50: 8
    col  27 chemical_symbol_Cu           max    62.69 min  -0.016 nodes>50: 85
    col  28 chemical_symbol_Dy           max    95.31 min  -0.010 nodes>50: 65
    col  29 chemical_symbol_Er           max   108.73 min  -0.009 nodes>50: 50
    col  30 chemical_symbol_Eu           max    61.40 min  -0.016 nodes>50: 128
    col  33 chemical_symbol_Ga           max    68.82 min  -0.015 nodes>50: 113
    col  34 chemical_symbol_Gd           max    73.02 min  -0.014 nodes>50: 83
    col  38 chemical_symbol_Hf           max   111.95 min  -0.009 nodes>50: 18
    col  39 chemical_symbol_Hg           max    64.13 min  -0.016 nodes>50: 103
    col  40 chemical_symbol_Ho           max   102.69 min  -0.010 nodes>50: 49
    col  42 chemical_symbol_In           max    58.72 min  -0.017 nodes>50: 105
    col  44 chemical_symbol_K            max   121.03 min  -0.008 nodes>50: 5
    col  46 chemical_symbol_La           max    64.09 min  -0.016 nodes>50: 132
    col  47 chemical_symbol_Li           max   106.80 min  -0.009 nodes>50: 5
    col  48 chemical_symbol_Lu           max    64.45 min  -0.016 nodes>50: 104
    col  49 chemical_symbol_Mg           max    66.75 min  -0.015 nodes>50: 99
    col  51 chemical_symbol_Mo           max    71.63 min  -0.014 nodes>50: 102
    col  53 chemical_symbol_Na           max   119.10 min  -0.008 nodes>50: 3
    col  54 chemical_symbol_Nb           max   152.34 min  -0.007 nodes>50: 21
    col  55 chemical_symbol_Nd           max    94.58 min  -0.011 nodes>50: 45
    col  57 chemical_symbol_Ni           max    72.29 min  -0.014 nodes>50: 94
    col  59 chemical_symbol_Os           max    56.55 min  -0.018 nodes>50: 158
    col  63 chemical_symbol_Pm           max    92.03 min  -0.011 nodes>50: 66
    col  64 chemical_symbol_Pr           max    88.91 min  -0.011 nodes>50: 58
    col  66 chemical_symbol_Rb           max   162.28 min  -0.006 nodes>50: 2
    col  67 chemical_symbol_Re           max    66.92 min  -0.015 nodes>50: 130
    col  68 chemical_symbol_Rh           max    66.43 min  -0.015 nodes>50: 124
    col  71 chemical_symbol_Sb           max   127.03 min  -0.008 nodes>50: 26
    col  72 chemical_symbol_Sc           max    69.30 min  -0.014 nodes>50: 102
    col  75 chemical_symbol_Sm           max    85.55 min  -0.012 nodes>50: 68
    col  77 chemical_symbol_Sr           max    68.79 min  -0.015 nodes>50: 98
    col  78 chemical_symbol_Ta           max   142.76 min  -0.007 nodes>50: 21
    col  79 chemical_symbol_Tb           max    88.39 min  -0.011 nodes>50: 64
    col  80 chemical_symbol_Tc           max    73.83 min  -0.014 nodes>50: 90
    col  81 chemical_symbol_Te           max    70.45 min  -0.014 nodes>50: 103
    col  83 chemical_symbol_Tl           max    62.74 min  -0.016 nodes>50: 128
    col  84 chemical_symbol_Tm           max   111.21 min  -0.009 nodes>50: 42
    col  86 chemical_symbol_W            max    56.79 min  -0.018 nodes>50: 156
    col  88 chemical_symbol_Y            max    62.82 min  -0.016 nodes>50: 125
    col  89 chemical_symbol_Yb           max    69.39 min  -0.014 nodes>50: 75
    col  90 chemical_symbol_Zn           max    66.78 min  -0.015 nodes>50: 82
    col  91 chemical_symbol_Zr           max    63.29 min  -0.016 nodes>50: 105
val bond: 394279 nodes, 14 columns with max |x| > 50, overall max 363.05
    col  17 rbf_gaussian_50_5            max    53.46 min  -0.012 nodes>50: 3
    col  18 rbf_gaussian_50_6            max    77.68 min  -0.039 nodes>50: 79
    col  50 rbf_gaussian_50_38           max    62.18 min  -0.026 nodes>50: 50
    col  51 rbf_gaussian_50_39           max    78.92 min  -0.019 nodes>50: 48
    col  52 rbf_gaussian_50_40           max    96.76 min  -0.015 nodes>50: 41
    col  53 rbf_gaussian_50_41           max   124.15 min  -0.012 nodes>50: 26
    col  54 rbf_gaussian_50_42           max   168.72 min  -0.009 nodes>50: 26
    col  55 rbf_gaussian_50_43           max   202.99 min  -0.007 nodes>50: 15
    col  56 rbf_gaussian_50_44           max   221.15 min  -0.006 nodes>50: 11
    col  57 rbf_gaussian_50_45           max   285.74 min  -0.005 nodes>50: 9
    col  58 rbf_gaussian_50_46           max   320.27 min  -0.004 nodes>50: 7
    col  59 rbf_gaussian_50_47           max   363.05 min  -0.004 nodes>50: 5
    col  60 rbf_gaussian_50_48           max   297.23 min  -0.004 nodes>50: 8
    col  61 rbf_gaussian_50_49           max   341.68 min  -0.004 nodes>50: 8
val global: 6400 nodes, 0 columns with max |x| > 50, overall max 2.16

train atom: 372258 nodes, 54 columns with max |x| > 50, overall max 162.28
    col   9 chemical_symbol_Ag           max   145.15 min  -0.007 nodes>50: 9
    col  10 chemical_symbol_Al           max    65.08 min  -0.015 nodes>50: 111
    col  12 chemical_symbol_As           max    71.03 min  -0.014 nodes>50: 63
    col  13 chemical_symbol_Au           max    63.81 min  -0.016 nodes>50: 116
    col  15 chemical_symbol_Ba           max    71.34 min  -0.014 nodes>50: 93
    col  16 chemical_symbol_Be           max    61.93 min  -0.016 nodes>50: 89
    col  17 chemical_symbol_Bi           max    64.56 min  -0.015 nodes>50: 119
    col  20 chemical_symbol_Ca           max    67.62 min  -0.015 nodes>50: 102
    col  21 chemical_symbol_Cd           max    69.67 min  -0.014 nodes>50: 81
    col  22 chemical_symbol_Ce           max    56.58 min  -0.018 nodes>50: 141
    col  24 chemical_symbol_Co           max    51.26 min  -0.020 nodes>50: 186
    col  25 chemical_symbol_Cr           max    53.93 min  -0.019 nodes>50: 157
    col  26 chemical_symbol_Cs           max   157.08 min  -0.006 nodes>50: 6
    col  27 chemical_symbol_Cu           max    62.69 min  -0.016 nodes>50: 95
    col  28 chemical_symbol_Dy           max    95.31 min  -0.010 nodes>50: 58
    col  29 chemical_symbol_Er           max   108.73 min  -0.009 nodes>50: 54
    col  30 chemical_symbol_Eu           max    61.40 min  -0.016 nodes>50: 116
    col  33 chemical_symbol_Ga           max    68.82 min  -0.015 nodes>50: 101
    col  34 chemical_symbol_Gd           max    73.02 min  -0.014 nodes>50: 88
    col  38 chemical_symbol_Hf           max   111.95 min  -0.009 nodes>50: 20
    col  39 chemical_symbol_Hg           max    64.13 min  -0.016 nodes>50: 89
    col  40 chemical_symbol_Ho           max   102.69 min  -0.010 nodes>50: 47
    col  42 chemical_symbol_In           max    58.72 min  -0.017 nodes>50: 111
    col  44 chemical_symbol_K            max   121.03 min  -0.008 nodes>50: 4
    col  46 chemical_symbol_La           max    64.09 min  -0.016 nodes>50: 105
    col  47 chemical_symbol_Li           max   106.80 min  -0.009 nodes>50: 10
    col  48 chemical_symbol_Lu           max    64.45 min  -0.016 nodes>50: 109
    col  49 chemical_symbol_Mg           max    66.75 min  -0.015 nodes>50: 81
    col  51 chemical_symbol_Mo           max    71.63 min  -0.014 nodes>50: 104
    col  53 chemical_symbol_Na           max   119.10 min  -0.008 nodes>50: 3
    col  54 chemical_symbol_Nb           max   152.34 min  -0.007 nodes>50: 23
    col  55 chemical_symbol_Nd           max    94.58 min  -0.011 nodes>50: 56
    col  57 chemical_symbol_Ni           max    72.29 min  -0.014 nodes>50: 82
    col  59 chemical_symbol_Os           max    56.55 min  -0.018 nodes>50: 155
    col  63 chemical_symbol_Pm           max    92.03 min  -0.011 nodes>50: 49
    col  64 chemical_symbol_Pr           max    88.91 min  -0.011 nodes>50: 63
    col  66 chemical_symbol_Rb           max   162.28 min  -0.006 nodes>50: 12
    col  67 chemical_symbol_Re           max    66.92 min  -0.015 nodes>50: 105
    col  68 chemical_symbol_Rh           max    66.43 min  -0.015 nodes>50: 110
    col  71 chemical_symbol_Sb           max   127.03 min  -0.008 nodes>50: 16
    col  72 chemical_symbol_Sc           max    69.30 min  -0.014 nodes>50: 119
    col  75 chemical_symbol_Sm           max    85.55 min  -0.012 nodes>50: 70
    col  77 chemical_symbol_Sr           max    68.79 min  -0.015 nodes>50: 84
    col  78 chemical_symbol_Ta           max   142.76 min  -0.007 nodes>50: 21
    col  79 chemical_symbol_Tb           max    88.39 min  -0.011 nodes>50: 68
    col  80 chemical_symbol_Tc           max    73.83 min  -0.014 nodes>50: 93
    col  81 chemical_symbol_Te           max    70.45 min  -0.014 nodes>50: 85
    col  83 chemical_symbol_Tl           max    62.74 min  -0.016 nodes>50: 111
    col  84 chemical_symbol_Tm           max   111.21 min  -0.009 nodes>50: 32
    col  86 chemical_symbol_W            max    56.79 min  -0.018 nodes>50: 164
    col  88 chemical_symbol_Y            max    62.82 min  -0.016 nodes>50: 106
    col  89 chemical_symbol_Yb           max    69.39 min  -0.014 nodes>50: 95
    col  90 chemical_symbol_Zn           max    66.78 min  -0.015 nodes>50: 99
    col  91 chemical_symbol_Zr           max    63.29 min  -0.016 nodes>50: 116
train bond: 397746 nodes, 14 columns with max |x| > 50, overall max 376.42
    col  17 rbf_gaussian_50_5            max    64.56 min  -0.012 nodes>50: 3
    col  18 rbf_gaussian_50_6            max    85.24 min  -0.039 nodes>50: 83
    col  50 rbf_gaussian_50_38           max    62.18 min  -0.026 nodes>50: 59
    col  51 rbf_gaussian_50_39           max    78.87 min  -0.019 nodes>50: 35
    col  52 rbf_gaussian_50_40           max    96.74 min  -0.015 nodes>50: 27
    col  53 rbf_gaussian_50_41           max   124.01 min  -0.012 nodes>50: 20
    col  54 rbf_gaussian_50_42           max   144.89 min  -0.009 nodes>50: 16
    col  55 rbf_gaussian_50_43           max   202.58 min  -0.007 nodes>50: 8
    col  56 rbf_gaussian_50_44           max   231.06 min  -0.006 nodes>50: 8
    col  57 rbf_gaussian_50_45           max   291.24 min  -0.005 nodes>50: 7
    col  58 rbf_gaussian_50_46           max   301.81 min  -0.004 nodes>50: 4
    col  59 rbf_gaussian_50_47           max   346.28 min  -0.004 nodes>50: 5
    col  60 rbf_gaussian_50_48           max   376.42 min  -0.004 nodes>50: 6
    col  61 rbf_gaussian_50_49           max   335.99 min  -0.004 nodes>50: 5
train global: 6400 nodes, 0 columns with max |x| > 50, overall max 2.09

val: molecules containing an atom with a scaled feature > 50: 4191 / 6400
train: molecules containing an atom with a scaled feature > 50: 4149 / 6400

## Q3b. a2g / b2g aggregated sums entering conv layer 0 (val, 50 batches, embedding output, fp32)

molecules: 6400; atom count percentiles (50/90/99/99.9/100): 57 / 87 / 99 / 100 / 100; bond count: 62 / 93 / 105 / 112 / 119
per-atom embedding L2 norm (batch mean): 9.21;  per-bond: 12.1
a2g sum L2 norm percentiles: 137 / 201 / 278 / 339 / 393
a2g sum max|component| percentiles: 33.9 / 51.3 / 72.3 / 87.6 / 97.4
b2g sum L2 norm percentiles: 163 / 259 / 391 / 775 / 967
b2g sum max|component| percentiles: 39.9 / 67 / 105 / 197 / 247
Pearson r(atom count, |a2g sum|) = 0.268  (R2 0.072);  r(bond count, |b2g sum|) = 0.392  (R2 0.154)
fit |a2g sum| ~ 0.501 * n_atoms + 117; mean atom count 57.8, mean |a2g sum| 146
top-5 molecules by a2g max component: (n_atoms, |sum|, max comp) (49, 292, 97.4); (47, 366, 96.9); (24, 393, 96); (24, 393, 96); (56, 346, 91.5)

## Q3c. Global-node pre-BN activations at conv layer 0 (a2g and b2g modules): batch statistics vs molecule size

a2g: pre-BN activation, mean over channels of batch mean 6.84, max channel batch mean 59.8
a2g: within-batch variance per channel: median 84.6, max 281
a2g: batch-to-batch variance of the batch mean per channel: median 17, max 150; ratio b2b/within median 0.233
a2g: r(batch mean molecule size, batch mean activation) = 0.454 (R2 0.206); batch mean size range 33.0-86.4
a2g: per molecule, r(size, |pre-BN activation|) = 0.302 (R2 0.091); |act| percentiles 130 / 203 / 286 / 345 / 356
a2g: linear fit |act| ~ 0.614*size + 104; fraction of variance explained by size 0.091

b2g: pre-BN activation, mean over channels of batch mean 7.55, max channel batch mean 63.8
b2g: within-batch variance per channel: median 104, max 376
b2g: batch-to-batch variance of the batch mean per channel: median 27.8, max 136; ratio b2b/within median 0.281
b2g: r(batch mean molecule size, batch mean activation) = 0.421 (R2 0.177); batch mean size range 34.8-91.3
b2g: per molecule, r(size, |pre-BN activation|) = 0.417 (R2 0.174); |act| percentiles 151 / 233 / 332 / 556 / 726
b2g: linear fit |act| ~ 1.06*size + 92.1; fraction of variance explained by size 0.174

## Q3d. Layer-0 global BN: checkpoint running stats vs the val-subset batch stats

a2g: running_mean max 14.7 vs val batch-mean max 14.7; running_var max 491 vs val batch-var max 281; running_var min 15.9, channels with running_var < 1e-3: 0, channels with val batch var < 1e-3: 0
a2g: channels dead in running stats (rv<1e-3) but with val batch var > 1e-3: []
b2g: running_mean max 17.7 vs val batch-mean max 18.6; running_var max 609 vs val batch-var max 376; running_var min 15.3, channels with running_var < 1e-3: 0, channels with val batch var < 1e-3: 0
b2g: channels dead in running stats (rv<1e-3) but with val batch var > 1e-3: []
# Appendix E: mundane checks

## Q5a. LMDB metadata per split (scaler applied at conversion time?)

train: 119019 graphs, 4 shards; meta keys ['allowed_charges', 'allowed_ring_size', 'allowed_spins', 'element_set', 'feature_names', 'feature_size', 'length', 'scaled', 'target_dict']
    scaled=True length=29755 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=29755 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=29755 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=29754 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaler-related metadata keys: ['scaled']
val: 14634 graphs, 4 shards; meta keys ['allowed_charges', 'allowed_ring_size', 'allowed_spins', 'element_set', 'feature_names', 'feature_size', 'length', 'scaled', 'target_dict']
    scaled=True length=3659 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=3659 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=3658 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=3658 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaler-related metadata keys: ['scaled']
test: 14726 graphs, 4 shards; meta keys ['allowed_charges', 'allowed_ring_size', 'allowed_spins', 'element_set', 'feature_names', 'feature_size', 'length', 'scaled', 'target_dict']
    scaled=True length=3682 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=3682 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=3681 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaled=True length=3681 feature_size={'atom': 92, 'bond': 62, 'global': 7} target_dict={'atom': ['charge_adch', 'charge_hirshfeld', 'charge_cm5', 'charge_becke', 'charge_mulliken_orca', 'charge_loewdin_orca'], 'bond': [], 'global': []} n_elements=83
    scaler-related metadata keys: ['scaled']

Interpretation: features and labels are scaled once at LMDB conversion (scaled=True), no scaler object is applied in LMDBDataModule/TransformMol or in validation_step, so train and val necessarily go through the same (frozen) scaling iff the shards were produced by the same conversion. The statistics below test that.

## Q5b. Full val split scan and a 200-batch train sample

### val (all batches): 14634 molecules
non-finite feature entries: {'atom': 0, 'bond': 0, 'global': 0}; non-finite label entries: 0
atoms per molecule: min 9 median 58 mean 58.5 max 100; max bonds 119
label mean per target: 0.040, 0.018, 0.022, 0.012, 0.008, 0.021
label std  per target: 0.938, 0.983, 0.876, 0.971, 1.033, 1.047
label max |y|: 59.54
atom features: max |x| 162.3 (col 66 chemical_symbol_Rb); columns with max>50: 54; mean of column means 0.004, mean of column stds 0.978
bond features: max |x| 363.0 (col 59 rbf_gaussian_50_47); columns with max>50: 14; mean of column means 0.002, mean of column stds 0.857
global features: max |x| 2.2 (col 2 molecule weight); columns with max>50: 0; mean of column means -0.141, mean of column stds 0.421

### train (200 batches, 25600 molecules): 25600 molecules
non-finite feature entries: {'atom': 0, 'bond': 0, 'global': 0}; non-finite label entries: 0
atoms per molecule: min 9 median 57 mean 58.2 max 100; max bonds 118
label mean per target: 0.039, 0.017, 0.022, 0.011, 0.008, 0.020
label std  per target: 0.939, 1.006, 0.879, 0.994, 1.034, 1.048
label max |y|: 68.51
atom features: max |x| 162.3 (col 66 chemical_symbol_Rb); columns with max>50: 54; mean of column means 0.004, mean of column stds 0.979
bond features: max |x| 376.4 (col 60 rbf_gaussian_50_48); columns with max>50: 14; mean of column means 0.002, mean of column stds 0.865
global features: max |x| 2.3 (col 2 molecule weight); columns with max>50: 0; mean of column means -0.146, mean of column stds 0.425

### train vs val column statistics (same scaler check)
atom: max |mean_train - mean_val| over columns 0.009 (col 5 ring_size_5); max |std_train - std_val| 0.527
bond: max |mean_train - mean_val| over columns 0.009 (col 4 ring size_5); max |std_train - std_val| 0.187
global: max |mean_train - mean_val| over columns 0.014 (col 2 molecule weight); max |std_train - std_val| 0.008
labels: max |mean_train - mean_val| 0.0006, max |std_train - std_val| 0.0239

Val columns with max > 50 (name, val max, train-sample max):
    atom col 9 chemical_symbol_Ag: val 145.2, train 145.2
    atom col 10 chemical_symbol_Al: val 65.1, train 65.1
    atom col 12 chemical_symbol_As: val 71.0, train 71.0
    atom col 13 chemical_symbol_Au: val 63.8, train 63.8
    atom col 15 chemical_symbol_Ba: val 71.3, train 71.3
    atom col 16 chemical_symbol_Be: val 61.9, train 61.9
    atom col 17 chemical_symbol_Bi: val 64.6, train 64.6
    atom col 20 chemical_symbol_Ca: val 67.6, train 67.6
    atom col 21 chemical_symbol_Cd: val 69.7, train 69.7
    atom col 22 chemical_symbol_Ce: val 56.6, train 56.6
    atom col 24 chemical_symbol_Co: val 51.3, train 51.3
    atom col 25 chemical_symbol_Cr: val 53.9, train 53.9
    atom col 26 chemical_symbol_Cs: val 157.1, train 157.1
    atom col 27 chemical_symbol_Cu: val 62.7, train 62.7
    atom col 28 chemical_symbol_Dy: val 95.3, train 95.3
    atom col 29 chemical_symbol_Er: val 108.7, train 108.7
    atom col 30 chemical_symbol_Eu: val 61.4, train 61.4
    atom col 33 chemical_symbol_Ga: val 68.8, train 68.8
    atom col 34 chemical_symbol_Gd: val 73.0, train 73.0
    atom col 38 chemical_symbol_Hf: val 112.0, train 112.0
    atom col 39 chemical_symbol_Hg: val 64.1, train 64.1
    atom col 40 chemical_symbol_Ho: val 102.7, train 102.7
    atom col 42 chemical_symbol_In: val 58.7, train 58.7
    atom col 44 chemical_symbol_K: val 121.0, train 121.0
    atom col 46 chemical_symbol_La: val 64.1, train 64.1
    atom col 47 chemical_symbol_Li: val 106.8, train 106.8
    atom col 48 chemical_symbol_Lu: val 64.4, train 64.4
    atom col 49 chemical_symbol_Mg: val 66.7, train 66.7
    atom col 51 chemical_symbol_Mo: val 71.6, train 71.6
    atom col 53 chemical_symbol_Na: val 119.1, train 119.1
    atom col 54 chemical_symbol_Nb: val 152.3, train 152.3
    atom col 55 chemical_symbol_Nd: val 94.6, train 94.6
    atom col 57 chemical_symbol_Ni: val 72.3, train 72.3
    atom col 59 chemical_symbol_Os: val 56.6, train 56.6
    atom col 63 chemical_symbol_Pm: val 92.0, train 92.0
    atom col 64 chemical_symbol_Pr: val 88.9, train 88.9
    atom col 66 chemical_symbol_Rb: val 162.3, train 162.3
    atom col 67 chemical_symbol_Re: val 66.9, train 66.9
    atom col 68 chemical_symbol_Rh: val 66.4, train 66.4
    atom col 71 chemical_symbol_Sb: val 127.0, train 127.0
    atom col 72 chemical_symbol_Sc: val 69.3, train 69.3
    atom col 75 chemical_symbol_Sm: val 85.5, train 85.5
    atom col 77 chemical_symbol_Sr: val 68.8, train 68.8
    atom col 78 chemical_symbol_Ta: val 142.8, train 142.8
    atom col 79 chemical_symbol_Tb: val 88.4, train 88.4
    atom col 80 chemical_symbol_Tc: val 73.8, train 73.8
    atom col 81 chemical_symbol_Te: val 70.4, train 70.4
    atom col 83 chemical_symbol_Tl: val 62.7, train 62.7
    atom col 84 chemical_symbol_Tm: val 111.2, train 111.2
    atom col 86 chemical_symbol_W: val 56.8, train 56.8
    atom col 88 chemical_symbol_Y: val 62.8, train 62.8
    atom col 89 chemical_symbol_Yb: val 69.4, train 69.4
    atom col 90 chemical_symbol_Zn: val 66.8, train 66.8
    atom col 91 chemical_symbol_Zr: val 63.3, train 63.3
    bond col 17 rbf_gaussian_50_5: val 71.8, train 69.4
    bond col 18 rbf_gaussian_50_6: val 89.6, train 88.2
    bond col 50 rbf_gaussian_50_38: val 62.2, train 62.2
    bond col 51 rbf_gaussian_50_39: val 78.9, train 78.9
    bond col 52 rbf_gaussian_50_40: val 96.8, train 96.8
    bond col 53 rbf_gaussian_50_41: val 124.1, train 124.1
    bond col 54 rbf_gaussian_50_42: val 168.7, train 168.7
    bond col 55 rbf_gaussian_50_43: val 203.0, train 203.4
    bond col 56 rbf_gaussian_50_44: val 221.1, train 233.9
    bond col 57 rbf_gaussian_50_45: val 285.7, train 291.4
    bond col 58 rbf_gaussian_50_46: val 320.3, train 336.3
    bond col 59 rbf_gaussian_50_47: val 363.0, train 362.7
    bond col 60 rbf_gaussian_50_48: val 357.5, train 376.4
    bond col 61 rbf_gaussian_50_49: val 342.1, train 342.1

## Q5c. torchmetrics accumulation / reset

RMSE after two updates of error 1: [1.0, 1.0]; after reset and one update of error 3: [3.0, 3.0] -> reset works
Code check (qtaim_embed/models/node_level/base_gcn.py): compute_metrics() calls .reset() on val_r2/val_torch_l1/val_torch_mse after every compute; on_validation_epoch_end calls compute_metrics('val') once per epoch. So val metrics do not accumulate across epochs. val_loss is logged with on_epoch=True via self.log (Lightning's own per-epoch mean, reset every epoch). self.loss (MultioutputWrapper(MeanSquaredError)) is called via forward(); its internal state accumulates for the whole run but the returned per-batch value is batch-local, so this does not affect training or logging. Note 'train_mse'/'val_mse' are MeanSquaredError(squared=False), i.e. RMSE, so val_mse=176 at epoch 3 means MSE about 3.1e4.

## Q5d. Does validation use the same scaler / mode as training?

validation_step and training_step both call shared_step; no scaler is applied in either (LMDB graphs are pre-scaled, labels come from graph['atom'].labels via _get_ndata). Lightning runs validation under model.eval() (BN running stats, no dropout) and train under model.train(). The label statistics above show train and val labels are both standardised to mean~0, std~1 with the same scaler.

## From-scratch test of the fixes (2026-09-09, in progress)

Flags `model.bn_before_activation` and `model.global_aggr: "mean"` (sparse and
dense twins, parity tested) trained from scratch with the reference recipe via
`profiling/run_divergence_fix.sh`: `b128_bnfirst`, `b128_meang`,
`b128_bnfirst_meang` under `profiling/train_runs/`.

Two false starts worth recording:

1. The first launch trained plain reference models because the loader edit had
   not landed; both arms reproduced the reference divergence exactly (val RMSE
   1.09 at epoch 2, 44 at epoch 3, 1e3 at epoch 4), which doubles as a
   reproduction of the original failure.
2. The second launch had the flags but plateaued at val R2 0.29 from epoch 0
   with no divergence through epoch 3. Cause: the output block's final conv
   keeps the ReLU, and with batch norm moved before it every prediction is
   clamped at zero while half the scaled charge targets are negative (the
   original order hides this because the trailing batch norm re-centres the
   ReLU output). `get_layer_args` now makes the final prediction layer linear
   (no activation, no batch norm) when `bn_before_activation` is set.

Third launch, interim (epochs 0-10 of 40; the reference diverged during
epoch 2). Validation R2 / RMSE on the full val split, train RMSE from the
progress bar:

| epoch | reference R2 | bn_first R2 | bn_first RMSE | bn_first + mean global R2 | RMSE | train RMSE (bn_first) |
|---|---|---|---|---|---|---|
| 0 | 0.692 | 0.761 | 0.501 | 0.771 | 0.488 | 0.933 |
| 1 | 0.720 | 0.793 | 0.470 | 0.797 | 0.465 | 0.580 |
| 2 | -320 | 0.811 | 0.456 | 0.811 | 0.458 | 0.559 |
| 3 | -4.8e4 | 0.821 | 0.447 | 0.815 | 0.453 | 0.549 |
| 5 | -4.0e7 | 0.837 | 0.433 | 0.836 | 0.443 | 0.538 |
| 7 | | 0.847 | 0.427 | 0.844 | 0.434 | 0.532 |
| 9 | | 0.855 | 0.422 | 0.853 | 0.426 | 0.528 |
| 10 | | 0.854 | 0.426 | 0.853 | 0.429 | 0.526 |

Both arms are stable where the reference failed and fit better than the
reference ever did (its best val RMSE was 0.551 at epoch 1; train RMSE 0.526
vs the reference's 0.638 plateau). Mean aggregation on the global relations
adds nothing measurable on top of batch norm before the activation.

Final, 40 epochs (checkpoint at the best val loss, test split, scaled units):

| | bn_first | bn_first + mean global |
|---|---|---|
| best val loss (epoch) | 1.463 (30) | 1.457 (37) |
| val R2 / RMSE at epoch 39 | 0.867 / 0.413 | 0.870 / 0.414 |
| train RMSE at epoch 39 | 0.514 | 0.514 |
| test loss / MAE / RMSE | 1.504 / 0.194 / 0.419 | 1.510 / 0.194 / 0.420 |
| test R2 ADCH | 0.939 | 0.939 |
| test R2 CM5 | 0.973 | 0.972 |
| test R2 Loewdin | 0.968 | 0.971 |
| test R2 Mulliken | 0.870 | 0.867 |
| test R2 Hirshfeld | 0.448 | 0.448 |
| test R2 Becke | 0.281 | 0.277 |

No epoch of either run showed an eval-mode excursion (val RMSE monotone
0.50 -> 0.41 with noise below 0.01). The two arms are indistinguishable, so
mean aggregation on the global relations is not needed once the batch norm
is moved before the activation. The weak Hirshfeld and Becke targets are a
data question (these shards carry no QTAIM columns), not an eval-mode one.
The `test_r2` aggregate logged by the model equals the last target's R2
(the MultioutputWrapper reports per-output values and the summary picks the
final one); use the per-target rows.

Mean global aggregation alone (`b128_meang`, batch norm still after the
activation) is neither necessary nor sufficient: it tracks the reference for
three epochs (val R2 0.67, 0.72, 0.74) and then shows the same erratic
eval-mode excursions one epoch later than the reference (val RMSE 4.2, 1.1,
624, 11.9, 0.60, 3.0, 5.8, 1.2 at epochs 3-10) while train RMSE keeps
improving (0.63). This matches section 4: removing the un-normalized global
sums slows the amplifier down but the running-variance collapse of post-ReLU
batch norm channels still happens. Conclusion: `bn_before_activation` is the
fix and is now the config default; `global_aggr` stays available but is not needed.

