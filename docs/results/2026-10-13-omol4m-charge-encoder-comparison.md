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

| rank | encoder | conv_fn | width | L | batch | test MAE | test R2 (median) | test R2 (mean) |
|---|---|---|---|---|---|---|---|---|
| 1 | **equivariant** (lmax 2, cutoff 4) | GraphConvDropoutBatch | hidden 512 | 9 | 256 | **0.1040** | **0.9446** | 0.9043 |
| 2 | **dimenetpp** (cutoff 4, cap 16) | GraphConvDropoutBatch | hidden 176 | 9 | 512 | 0.1062 | 0.9398 | **0.9098** |
| 3 | schnet | GraphConvDropoutBatch | hidden 384 | 5 | 1024 | 0.1235 | 0.9296 | 0.8538 |
| 4 | schnet | GATConv | hidden 128 | 4 | 512 | 0.1375 | 0.9127 | 0.8349 |
| 5 | none | GraphConvDropoutBatch | hidden 768 | 9 | 1024 | 0.1558 | 0.8171 | 0.8120 |
| 6 | none | ResidualBlock | **embedding 512** | 7 | 1024 | 0.1573 | 0.8144 | 0.8122 |
| 7 | none | GATConv | hidden 128 | 4 | 512 | 0.1842 | 0.7670 | 0.7853 |

`test_r2` as logged is `r2.median()` over the six targets, and torch returns
the **lower** median for an even count, so the headline R2 is literally the
third-smallest per-target value: Mulliken, for every model here. The mean is
given alongside because the two disagree at the top of the table.

Width is reported as the knob that actually controls it: `ResidualBlock` reads
`embedding_size` and ignores `hidden_size`, every other conv_fn is the
opposite. Run 5's config says `hidden_size: 256`, which is inert; it is a
512-wide model.

### Per-target R2

| encoder / conv | ADCH | Hirshfeld | CM5 | Becke | Loewdin | Mulliken |
|---|---|---|---|---|---|---|
| equivariant / GCN | 0.978 | 0.789 | 0.992 | 0.728 | 0.994 | **0.945** |
| dimenetpp / GCN | **0.979** | **0.790** | 0.992 | **0.764** | 0.993 | 0.940 |
| schnet / GCN | 0.975 | 0.691 | 0.992 | 0.542 | 0.994 | 0.930 |
| schnet / GAT | 0.960 | 0.662 | 0.987 | 0.496 | 0.991 | 0.913 |
| none / GCN | 0.969 | 0.655 | 0.991 | 0.450 | 0.990 | 0.817 |
| none / ResidualBlock | 0.969 | 0.656 | 0.990 | 0.453 | 0.991 | 0.814 |
| none / GAT | 0.941 | 0.623 | 0.981 | 0.419 | 0.980 | 0.767 |

### Training length

| run | best ckpt epoch | last epoch | ended |
|---|---|---|---|
| equivariant | 120 | 124 | **crashed** |
| dimenetpp_c4 | - | 163 | **crashed** |
| schnet | 75 | 146 | finished |
| schnet_gat | 221 | 261 | finished |
| gat | 140 | 158 | finished |
| resid | 46 | 67 | finished |
| gcn | 53 | 55 | finished |

The two best models are the two that crashed rather than early-stopped.
Equivariant died four epochs past its best checkpoint and DimeNet++ at epoch
163, so neither reached its stopping criterion and both may be undertrained.
Every model that finished ran well past its best (schnet best at 75 of 146,
gat at 140 of 158, resid at 46 of 67), which is early stopping working and
evidence those five converged. The gap between the top two and SchNet is
therefore a lower bound, and the equivariant/DimeNet++ tie should not be read
as final.

Note also how much longer geometry-aware models keep improving: the
encoder-free models stopped at 55 and 67 epochs, against 146-261 with an
encoder. An encoder does not only lower the error, it extends the useful
training horizon.

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

**Within encoders, angular information pays, and the two angular encoders
tie.** Both the lmax-2 equivariant encoder and DimeNet++ beat SchNet's
invariant filters by 14-16 % on MAE (0.1040 and 0.1062 vs 0.1235) at the same
conv_fn. That is lever 4 in section 3.3 paying off on top of lever 1, and it
is angular resolution rather than equivariance specifically: DimeNet++ is
invariant and gets there with directional triplets.

**The top two trade places depending on the aggregate, and the disagreement is
informative.** Equivariant wins on MAE (0.1040 vs 0.1062) and on the logged
median R2; DimeNet++ wins on mean R2 (0.9098 vs 0.9043) because it is clearly
better on the hardest target, Becke (0.764 vs 0.728), while giving back a
little on Mulliken (0.940 vs 0.945). Treat them as tied overall and pick on
the target that matters: DimeNet++ for the real-space partitioning
definitions, equivariant for Mulliken and for the mean absolute error. Note
DimeNet++ is also the cheaper model, hidden 176 against 512.

**Three of the six targets are already saturated and carry no signal.** CM5 and
Loewdin sit at R2 0.98-0.994 for *every* model including no encoder, and ADCH
at 0.94-0.98. All of the discrimination lives in **Becke** (0.419 to 0.764),
**Hirshfeld** (0.623 to 0.790) and **Mulliken** (0.767 to 0.945). Any single
aggregate over six targets therefore hides most of the difference between
models, and the logged `test_r2` hides it twice over by collapsing to the
Mulliken column. Read the per-target table, not the headline.

**Becke and Hirshfeld remain the weakest targets even for the best model**
(0.728 and 0.789 against 0.99 for CM5 and Loewdin). That is consistent with the
open question of whether the shard LMDBs carry the QTAIM `extra_feat_*` columns
their configs request; these two definitions are the ones most dependent on
real-space partitioning, so a missing feature block would show up exactly here.
Worth checking before concluding this is a model-capacity limit.

## Per-vertical breakdown (2026-10-16)

`qtaim-embed-eval-stratified --group_depth 1 --stride 20` over the canonical
test split: 20,184 molecules spread evenly across all 403,680, the identical
index set for every model, 33 verticals. CSVs under `strat_<arch>/`.

The aggregate is mostly an `omol` score: that vertical is 36 % of the split
and among the harder ones, so the single test MAE answers "how good is this
model on omol" more than anything else.

### The encoder's value is not uniform: 9 % to 50 % depending on chemistry

Reduction in MAE from the best encoder-free model (`gcn`) to the better of the
two angular encoders, by vertical:

| gain | verticals |
|---|---|
| 43-50 % | ml_elytes, droplet, protein_interface, ml_protein_interface, 5A_elytes, pdb_fragments, pdb_pockets |
| 30-38 % | rpmd, dna, rna, nakb, electrolytes_scaled_sep, geom_orca6, scaled_separations_exp, orbnet_denali, **omol**, spice, electrolytes_redox |
| 21-28 % | ml_mo, **tm_react**, low_spin_23, pmechdb, ani2x, ani1xbb, rgd_uks, trans1x |
| 9-15 % | **mo_hydrides**, electrolytes_reactivity |

Median 35 %, against the 33 % the aggregate reports. Geometry pays most where
the chemistry is conformational, meaning the same bonding in different spatial
arrangements: proteins, electrolytes, droplets, PDB fragments. It pays least
where the chemistry is bond-making and unusual metal oxidation states:
`mo_hydrides` (9 %) and `electrolytes_reactivity` (15 %) are nearly
encoder-proof, and `tm_react` and `low_spin_23` sit near the bottom at 26-28 %.

The mechanism is visible in the per-target numbers: the encoder-free model's
**Mulliken** predictions collapse on exactly the condensed-phase verticals
where the encoder gains most. `gcn` Mulliken MAE reaches 0.408 on ml_elytes
(R2 0.588), 0.576 on electrolytes_scaled_sep (R2 0.449), 0.346 on
ml_protein_interface (R2 0.675), against 0.146, 0.376 and 0.141 for
equivariant. Most of that 50 % headline gain is Mulliken rescue.

GAT without an encoder is the same failure, further along: Mulliken R2 0.390
on electrolytes_scaled_sep, 0.474 on droplet, 0.558 on ml_elytes, 0.619 on
pdb_pockets_400K. Against the best encoder it loses 53 % on droplet, ml_elytes
and protein_interface, but only 25 % on mo_hydrides, the same chemistry split
as the `gcn` comparison. It also has essentially no predictive power on metal
Becke charges: R2 0.208 on tm_react, 0.230 on low_spin_23, 0.179 on
mo_hydrides.

### Equivariant and DimeNet++ split by chemistry

They tie overall but not per vertical. DimeNet++ is better on the
protein and PDB condensed systems (protein_interface by 7.5 %,
pdb_fragments_300K 4.9 %, pdb_pockets_300K 4.0 %, pdb_fragments_400K 3.7 %).
Equivariant is better on reactive chemistry and on metals: tm_react by 8.6 %,
electrolytes_reactivity 9.0 %, rmechdb 9.8 %, pmechdb 7.1 %, low_spin_23
6.1 %, ml_mo 3.6 %, plus the nucleic acids (nakb 10.8 %, dna 7.5 %, rna 6.1 %).

For this project's purpose, QTAIM descriptors on transition-metal complexes,
that is the relevant signal: **equivariant is the better choice on tm_react,
ml_mo and low_spin_23**, and the overall tie is driven by DimeNet++ winning
verticals that are not the target chemistry.

### Hardest verticals, consistent across all models

`mo_hydrides` is worst for every model (MAE 0.21-0.23, Hirshfeld R2 0.33-0.44,
Becke R2 0.17-0.55) and barely responds to an encoder. `low_spin_23`,
`tm_react`, `ml_mo` and `pmechdb` follow. These are the metal and
reactive-chemistry verticals, which is precisely where this project needs the
model to work, and they are the ones the aggregate flatters.

Becke remains the weakest target in nearly every vertical, but the spread is
large: equivariant gets Becke R2 0.97 on noble_gas, 0.94 on droplet and 0.92
on nakb, against 0.64 on tm_react and 0.45 on mo_hydrides. So the global
"Becke is hard" conclusion is really "Becke is hard on metals and reactions".

## Caveats

- **Single seed per architecture.** No error bars; the 1 % gap between rows 4
  and 5 is not meaningful, the 33 % gap between rows 1 and 4 clearly is.
- **Checkpoints were selected on `val_mae`**, which is torchmetrics-synced and
  therefore correct under DDP. The `val_loss` in each checkpoint filename was
  rank-local when these runs trained and should be ignored.
- **The DimeNet++ row is from a run that crashed at epoch 163** (wandb
  `OMol4m_charges_final_models/dimenetpp_c4`, 2026-10-13) and was scored
  afterwards with `--test_only`, so it may not have converged as far as the
  others. Its earlier 0.1008 on `OMol4M_charge_full` at cutoff 5 with
  last-epoch weights is a different dataset and not comparable to anything in
  this table.
- **Validation losses are not comparable across these runs.** `val_loss` was
  rank-local until 2026-10-13; the DimeNet++ run is the only one here trained
  after that fix, so its 0.5099 is a whole-validation-set value while the
  others' summaries are rank 0's quarter. Test metrics are unaffected: they
  come from torchmetrics and always synced.
- **The per-vertical tables cover six of the seven models** (equivariant,
  dimenetpp, schnet, schnet_gat, gcn, gat). `resid` was not captured; it tracks
  `gcn` to within 1 % in the aggregate, so it does not change the conclusions.
  `gcn` is the better encoder-free baseline on 32 of 33 verticals, GAT edging
  it only on `noble_gas` (12 molecules), so the gain figures above use `gcn`.
- **Small verticals are flagged, not dropped.** `mo_hydrides`, `nakb`,
  `noble_gas`, `rmechdb` and `pdb_pockets_400K` have 12-18 molecules in this
  sample; treat them as directional. Rerun without `--stride` for the full
  403,680 before quoting them.
- **Topology is given.** These are T2 numbers on known graphs; nothing here
  speaks to T3 bond prediction.
