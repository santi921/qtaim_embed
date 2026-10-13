# OMol4M atomic charges: 3D encoder and conv-function comparison

Six architectures scored on the same test split, each from the checkpoint its
own run selected. This is the Phase 1 lever-ordering experiment from
`docs/roadmap_v4.md` section 3.3, on known topology (task T2).

## What was run

- **Data**: `OMol4M_charge_canonical`, train/val/test under
  `/pscratch/sd/s/santiago/qtaim_embed_experiments/data/node/`. Features are
  pre-scaled on disk; atom features 92, bond 62, global 7.
- **Targets**: six atomic charge definitions (ADCH, Hirshfeld, CM5, Becke,
  Mulliken/ORCA, Loewdin/ORCA), node-level regression, MSE loss.
- **Training**: Perlmutter, 4x A100, `bf16-mixed`, `bn_before_activation: true`,
  `gradient_clip_val: 5`, `warmup_epochs: 1`, `compiled: false`, no bucketing.
  Configs and checkpoints under
  `experiments/node_level/OMol4M_single/new_scalers/`.
- **Scoring**: one `--test_only` pass per checkpoint on a single GPU
  (`qtaim-embed-train-node -config <cfg> --use_lmdb --num_devices 1
  --test_only <ckpt>`), wandb project `OMol4m_rescore_best`, 2026-10-13.

Scoring separately matters: `trainer.test` had been running on the **last**
epoch's weights rather than the checkpoint selected on `val_mae`, so the
numbers here supersede the `OMol4m_charges_final_models` summaries. The
re-scored values are 0.1-2 % better, and the equivariant row is new, since that
run never produced test metrics at all.

## Results

| rank | encoder | conv_fn | width | L | batch | test MAE | test R2 |
|---|---|---|---|---|---|---|---|
| 1 | **equivariant** (lmax 2, cutoff 4) | GraphConvDropoutBatch | hidden 512 | 9 | 256 | **0.1040** | **0.9446** |
| 2 | schnet | GraphConvDropoutBatch | hidden 384 | 5 | 1024 | 0.1235 | 0.9296 |
| 3 | schnet | GATConv | hidden 128 | 4 | 512 | 0.1375 | 0.9127 |
| 4 | none | GraphConvDropoutBatch | hidden 768 | 9 | 1024 | 0.1558 | 0.8171 |
| 5 | none | ResidualBlock | **embedding 512** | 7 | 1024 | 0.1573 | 0.8144 |
| 6 | none | GATConv | hidden 128 | 4 | 512 | 0.1842 | 0.7670 |

Width is reported as the knob that actually controls it: `ResidualBlock` reads
`embedding_size` and ignores `hidden_size`, every other conv_fn is the
opposite. Run 5's config says `hidden_size: 256`, which is inert; it is a
512-wide model.

### Per-target R2

| encoder / conv | ADCH | Hirshfeld | CM5 | Becke | Loewdin | Mulliken |
|---|---|---|---|---|---|---|
| equivariant / GCN | 0.978 | **0.789** | 0.992 | **0.728** | 0.994 | **0.945** |
| schnet / GCN | 0.975 | 0.691 | 0.992 | 0.542 | 0.994 | 0.930 |
| schnet / GAT | 0.960 | 0.662 | 0.987 | 0.496 | 0.991 | 0.913 |
| none / GCN | 0.969 | 0.655 | 0.991 | 0.450 | 0.990 | 0.817 |
| none / ResidualBlock | 0.969 | 0.656 | 0.990 | 0.453 | 0.991 | 0.814 |
| none / GAT | 0.941 | 0.623 | 0.981 | 0.419 | 0.980 | 0.767 |

## Findings

**Geometry is the dominant lever, by a wide margin.** The best encoder-free
model reaches test MAE 0.1558; adding an encoder to the same conv family takes
it to 0.1040, a 33 % reduction. This is the lever ordering from section 3.3
confirmed on real data: an encoder is worth far more than any conv_fn or width
choice.

**Conv_fn barely matters once geometry is absent.** `GraphConvDropoutBatch` at
hidden 768 (0.1558) and `ResidualBlock` at embedding 512 (0.1573) are within
1 % of each other despite different architectures and parameter counts. GAT is
the clear loser at 0.1842. So the encoder-free rows mostly measure the same
thing, and the 9-layer 768-wide model buys nothing over the 7-layer one.

**Within encoders, angular resolution pays.** The lmax-2 equivariant encoder
beats SchNet's invariant filters by 16 % on MAE (0.1040 vs 0.1235) at the same
conv_fn, which is lever 4 in section 3.3 paying off on top of lever 1.

**Three of the six targets are already saturated and carry no signal.** CM5 and
Loewdin sit at R2 0.98-0.994 for *every* model including no encoder, and ADCH
at 0.94-0.98. All of the discrimination lives in **Becke** (0.419 to 0.728),
**Hirshfeld** (0.623 to 0.789) and **Mulliken** (0.767 to 0.945). Reporting a
mean R2 over six targets therefore understates the differences between models
and should be read alongside the per-target table.

**Becke and Hirshfeld remain the weakest targets even for the best model**
(0.728 and 0.789 against 0.99 for CM5 and Loewdin). That is consistent with the
open question of whether the shard LMDBs carry the QTAIM `extra_feat_*` columns
their configs request; these two definitions are the ones most dependent on
real-space partitioning, so a missing feature block would show up exactly here.
Worth checking before concluding this is a model-capacity limit.

## Caveats

- **Single seed per architecture.** No error bars; the 1 % gap between rows 4
  and 5 is not meaningful, the 33 % gap between rows 1 and 4 clearly is.
- **Checkpoints were selected on `val_mae`**, which is torchmetrics-synced and
  therefore correct under DDP. The `val_loss` in each checkpoint filename was
  rank-local when these runs trained and should be ignored.
- **DimeNet++ is absent from this table.** Its best result, test MAE 0.1008,
  was measured on `OMol4M_charge_full` at cutoff 5 with last-epoch weights
  (`OMol4M_desc_charges`, run `effortless-deluge-65`), so it is not comparable
  to the rows above. `profiling/train_configs/omol4m_canonical_dimenetpp_c4.json`
  runs it on `canonical` at cutoff 4 to close that gap.
- **Topology is given.** These are T2 numbers on known graphs; nothing here
  speaks to T3 bond prediction.
