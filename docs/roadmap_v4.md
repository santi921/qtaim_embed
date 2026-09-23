# Architectural Review & Roadmap: Foundational Model for Quantum Chemical Descriptors

## CHANGELOG

**v4 (2026-09-08) - Synchronized with the state of qtaim_embed (branch feat/3d-encoders) and qtaim_generator (branch feat/horton-charge-engine); added gap analysis and the bond-prediction validation program:**

- Rewrote Section 1 (What You Have Now) against the code as of 2026-09-08: PyG migration complete, torch 2.11 / CUDA 13 / PyG 2.7, 3D encoders (SchNet, DimeNet++, MACE-style) live behind `encoder_fn`, `atom.pos`/`atom.z` on graphs, composition splitting, sharded LMDB directories, OMol-Descriptors-4M as the training corpus.
- Added Section 1.5 (What Is Missing): a ranked gap list. The top gap is a validated bond (topology) prediction model measured against the distance-rule baseline from the NeurIPS rebuttal (macro-F1 0.971).
- Added Section 2.9 (Training Data): OMol-Descriptors-4M scale, level of theory, composition split, five stress suites, known data-quality defects and their repair tooling.
- Annotated Section 3.3 lever ordering with implementation status.
- Added Section 3.5 (Bond Prediction Validation Program): the staged path from T3 bond classification to the full geometry-to-labeled-graph model.
- Replaced Section 3.4 scaling notes with a pointer to `docs/plans/2026-09-08-feat-performance-engineering-plan.md` (training, inference, packaging into ASE).
- Rewrote Section 4 phases with status and re-ordered so that bond prediction validation precedes the iterative loop.
- Rewrote Section 5 next steps.
- Added open questions on candidate recall for non-covalent BCPs and on threshold calibration across chemistries.
- Superseded documents: `docs/FM_model_report.md` (v1), `docs/roadmap_v3.md` (v3). Both kept for history.
---

**v3 - Updated with findings from IDGL deep-dive and large-molecule inference planning:**

- Added Section 2.7: Long-Range Interactions & Large-Molecule Inference (dual-cutoff strategy, GemNet-OC reference)
- Updated Section 3.4 Scaling Considerations: dual-cutoff as the preferred large-molecule scaling path over anchor machinery
- Updated Phase 3: added large-molecule benchmark (MOF/polynuclear complex) as explicit evaluation target
- Updated Section 5 Next Steps: added dual-cutoff implementation note
- Added GemNet-OC dual-cutoff reference to Section 6
---

**v2 - Updated with findings from PaiNN deep-dive and angular/radial expressiveness analysis:**

- Revised Section 2.2 with detailed PaiNN analysis (strengths, limitations for this project)
- Added Section 2.6: Radial Basis Function Design Space
- Replaced Section 3.1 encoder recommendation: hybrid l_max strategy instead of "start PaiNN, upgrade later"
- Added Section 3.3: Expressiveness Lever Ordering (new)
- Revised Phase 1 to include RBF/angular baseline experiments before building iterative loop
- Added ablation plan for learnable RBFs
- Revised Section 7 with sharper open questions on angular expressiveness for TM/lanthanide/actinide chemistry
---

## 1. What You Have Now (2026-09-08)

Both repos moved a long way since v3. This section replaces the DGL-era description.

**qtaim_generator** (branch `feat/horton-charge-engine`) is the data factory and has produced the corpus this roadmap assumed would exist:

- **OMol-Descriptors-4M** (`santi921/OMol-Descriptors-4M` on the Hugging Face Hub): 3,986,738 structures across 34 verticals and 80+ elements. Geometries and wavefunctions inherited from OMol25 (wB97M-V / def2-TZVPD), post-processed with ORCA 6.0.0 and Multiwfn 3.8. Descriptor families: six partial-charge schemes (Hirshfeld, CM5, ADCH, Becke, Mulliken, Loewdin), four bond-order definitions (Mayer, Loewdin, fuzzy, QTAIM bond presence), 26 QTAIM scalars per bond critical point, fuzzy-atom integrations, ORCA globals. Eight typed LMDBs per vertical (`structure, charge, bond, qtaim, fuzzy, other, orca, timings`).
- **Composition-based split**, not random: `sha256(Composition.formula)` into 0.8/0.1/0.1, giving 3,125,019 / 398,982 / 403,951 train/val/test after excluding 58,786 held-out structures. Zero test compositions appear in train (a random row split would leak 80.9 %).
- **Five stress suites** (60,578 records): H1 metal-ligand pairs (15,025), H3 reactivity (12,506), H6 lanthanide-ligand (2,589), H7 large systems with n_atoms > 250 (18,096), H8 large net charge |q| > 4 (12,362). These are the "hard subsets" v1 asked for and they are built.
- **Sharded graph LMDB directories** via `multi-vertical-merge`: plan, build, scale phases; train-only scaler fitting; each split is a directory of shard LMDBs that `LMDBMoleculeDataset` reads directly.
- **Independent cross-validation engines**: HORTON charges (becke, becke_csd, hirshfeld, is) and Critic2 bond critical points, both standalone benchmarks against the shipped Multiwfn values. Not in the released corpus.
- **QTAIM repair loop**: `audit-qtaim-connectivity`, `select-qtaim-rerun`, `verify-qtaim-rerun`. Built after finding that truncated Multiwfn `CPprop.txt` files silently drop bond critical points while passing validation. See Section 2.9 for the defect list.
- **NeurIPS 2026 Datasets and Benchmarks submission 3446** (reviews 3/2/2). The decisive reviewer gap: tasks T1 (joint charge regression), T2 (BCP property regression), T3 (bond classification) were defined with no baseline. Ten rebuttal plans exist under `docs/plans/2026-07-2*-neurips-r1-*`. For T3 a tuned one-parameter distance rule reaches macro-F1 0.971 at the paper's 2x covalent-radius cutoff.
- Open generator items that gate this roadmap: `atom.pos`/`atom.z` are patched onto graphs post-hoc by `add_pos_z_to_graphs.py`, not emitted by the converter; ECP-driven extreme charge magnitudes are unresolved; OMol25 energies and forces are not yet merged (`energy.lmdb`, generator todo #9).

**qtaim_embed** (branch `feat/3d-encoders`, 43 commits ahead of local `main`; `origin/main` is at 2026-06-04) is PyG-native:

- **PyG migration complete** (2026-02 to 2026-03). torch 2.11.0+cu130, torch-geometric 2.7.0, lightning 2.6, e3nn 0.6. No `torch_cluster`/`torch_sparse`; neighbor lists are torch-only. 228 tests in 24 files. Details in `docs/PyG_Research_Report.md` Status Addendum and `MIGRATION_GUIDE.md`.
- **3D geometric encoders** (commit 766849d, 2026-08-03) behind `encoder_fn`: `schnet` (PyG blocks, 50 Gaussians, cutoff 5 A), `dimenetpp` (Bessel radial 6, spherical 7, torch-only triplets capped at 32 neighbors), `equivariant` (MACE-style e3nn, lmax 1, l=0 readout). Output width `encoder_hidden` is concatenated onto `atom.feat` before `UnifySize`. Supported by graph-level regression, graph-level classification, and node-level models. Parity and equivariance tests exist. Not supported by the link model or `FullPredictor`.
- **Geometry on graphs**: `atom.pos` float32 [N,3] and `atom.z` int64 [N] since commit 9084345 (2026-07-27); mixed-schema shard directories fail fast.
- **RBF bond-length features** in the featurizer (`rbf_bessel_<n>` with a p=5 polynomial envelope, `rbf_gaussian_<n>`), fixed and non-learnable, cutoff 5 A default.
- **Composition-based LMDB splitting** in `split_lmdb_file`, mirroring the generator.
- **Link model** `GCNLinkPred`: homogeneous graph via `hetero_to_homo`, negative sampling with PyG `negative_sampling`, Dot/MLP/Attention predictors on node embeddings only. Logs accuracy, F1, AUROC on raw logits (implicit threshold at logit 0). No geometry input, no distance feature on candidate edges, no precision/recall, no calibrated threshold, and no test asserts an accuracy number.
- **FullPredictor**: still the v1 skeleton. Loads two pretrained checkpoints, `.eval()`, `@torch.no_grad()`, fixed 3 iterations, all-pairs candidate edges in O(N^2) Python loops, threshold 0.5 on sigmoid. Not trainable end-to-end.
- **`candidate_pairs`** (covalent-radius pooling, `d <= 2.0 * (rcov_i + rcov_j)`) exists in `models/encoders/neighbors.py` and is tested, but nothing uses it yet.
- **Multi-task loss weighting**: planned (`docs/plans/2026-03-11-feat-multi-task-loss-weighting-plan.md`), not implemented; the branch is empty.
- **Performance**: fused Adam, bf16 configs, DDP hang fixed, LMDB worker stability fixed. Throughput is flat at 25-28 it/s (about 3300 samples/s at batch 128) across QM8, QM9, and TMQM with under 0.5 GB GPU memory, which means the model is launch-bound. See `docs/plans/2026-09-08-feat-performance-engineering-plan.md`.

What v3 predicted correctly: geometry had to enter the model (done), the link model's homogeneous collapse is a problem (still true), the O(N^2) loops are a bottleneck (still true, todos/009). What v3 got wrong: it recommended PaiNN as the backbone; the implementation went SchNet / DimeNet++ / MACE-style instead, which covers levers 0, 1 (via `encoder_num_gaussians`/`encoder_num_radial`), and 4 (via `encoder_lmax`) but skips PaiNN's l=1 Cartesian path. That is acceptable; the lever ordering in Section 3.3 is what matters, not the specific backbone.

---

## 1.5 What Is Missing

Ranked by how much each blocks the geometry-to-labeled-graph model.

| Rank | Gap | Evidence | Unblocks |
|------|-----|----------|----------|
| 1 | **A validated bond (topology) predictor.** No model in the repo has been shown to beat the distance rule (macro-F1 0.971 at 2x covalent cutoff) on T3, per stress suite, with a calibrated threshold. | `link_model.py` has no geometry input; no numeric assertion in `tests/test_link_models.py`; `neurips_status.md` lists T3 baseline as the missing result. | Everything downstream: BCP property prediction on predicted topology, the iterative loop, the ASE calculator's `bond_pairs` output, the NeurIPS rebuttal. |
| 2 | **Encoder validation on known topology at scale (Phase 1 baseline).** Encoders exist and pass parity tests, but there is no reported T1/T2 result on OMol-Descriptors-4M or the H-suites for any `encoder_fn`. | No benchmark configs or results under `profiling/` or `docs/` for encoder runs; the memory benchmarks predate encoders. | Choosing the encoder lever (Section 3.3); the NeurIPS T1/T2 baselines. |
| 3 | **Trainable iterative loop.** `FullPredictor` composes frozen checkpoints; no gradient from node quality to edge quality; no weight sharing; no convergence criterion. | `full.py:94-95, 156`. | The "neural SCF" thesis of this roadmap. |
| 4 | **Multi-task loss weighting.** Atom targets dominate bond targets by count; six charge schemes at different scales are summed unweighted. | Plan exists, branch empty. | Stable joint T1 + T2 training; the iterative loop's multi-term loss. |
| 5 | **Inference and packaging surface.** No batched predictor, no bundle format, no ASE calculator, no export, `pyproject.toml` dependencies commented out. | See performance plan Track B and C. | Anyone using the model outside a training script. |
| 6 | **Training throughput.** 3300 samples/s means one epoch of the 3.1M train split is ~16 min on one GPU; 100 epochs is 26 hours per GPU per experiment. | Memory benchmarks. | Running the ablations in Sections 3.3 and 4 at all. |
| 7 | **Energy and forces head.** Required for a true ASE calculator and the MLIP benchmark; gated on generator todo #9 (`energy.lmdb`). | Generator `master_list_todo.md` items 8, 9. | ASE `get_potential_energy`/`get_forces`; fairchem comparison. |
| 8 | **Higher-order critical points and non-nuclear attractors.** Unchanged from v1; the graph framing covers BCPs only. | Section 7. | Full QTAIM topology, not just the bond graph. |
| 9 | **Uncertainty quantification.** Nothing implemented. | | Trusting predictions on H6/H8 chemistry. |
| 10 | **Dual cutoff, envelope on the encoder path, learnable RBFs.** Absent; per Section 3.3 these are only pulled when a plateau is observed. | Encoder survey. | Large-system (H7, MOF) accuracy. |

Not missing, despite what v3 assumed: geometry on graphs, hard-subset benchmarks, composition splits, sharded 100M-scale data plumbing, mixed precision, DDP.

---

## 2. Literature Map

### 2.1 Scaling GNNs to Hundreds of Millions of Molecules

The scaling problem has been solved in practice by a few groups. The key references:

**OC20/OC22 (Meta FAIR)**: Trained on ~130M DFT relaxation frames. Their scaling playbook: LMDB data storage (you already have this), distributed data-parallel training, and architectures that balance expressiveness with throughput. The progression was GemNet-T → GemNet-OC → eSCN → EquiformerV2. EquiformerV2 (Liao & Smidt, ICLR 2024) is currently SOTA on OC20 - uses equivariant Transformers with SO(3) convolutions, but is expensive. **GemNet-OC** (Gasteiger et al., 2022) is probably the better cost-performance tradeoff and more relevant to your case since you care about throughput at inference.

**MACE** (Batatia et al., NeurIPS 2022; Batatia et al., 2024 for MACE-MP-0): Higher-order equivariant message passing using ACE (Atomic Cluster Expansion) body-ordered descriptors. MACE-MP-0 was trained on ~150k materials but shows extraordinary transfer. The architecture is relevant because it's very efficient (few message passing steps needed) and has clean multi-body interaction terms. **Critically, MACE's ACE descriptor framework computes rotationally invariant contractions of l=0,1,2,... neighbor sums - these scalar invariants encode angular information at orders higher than the equivariant features carried through message passing. This property is directly exploited in our hybrid encoder strategy (Section 3.3).**

**Practical scaling lessons from these efforts**:

- LMDB with memory-mapped reads (you have this)
- bf16/mixed precision training
- Graph batching with dynamic padding rather than fixed batch sizes
- Distributed training with DDP - PyG's `DistributedDataParallel` or DeepSpeed
- For 100M+ molecules: epoch-level sharding where each worker sees a different data shard per epoch
**For your PyG migration specifically**: Look at PyG's `LargeGraphDataset` and their LMDB integration patterns from OCP (Open Catalyst Project). The OCP codebase (github.com/Open-Catalyst-Project/ocp) is the reference implementation for training molecular GNNs at this scale.

### 2.2 Geometry-Aware Architectures (Critical for Your Use Case)

Since your input is geometry and your output includes QM topology, you need architectures that properly encode 3D structure. The current GCN/GAT layers don't do this.

**SchNet** (Schütt et al., 2017): Continuous-filter convolutions on interatomic distances. Simple but effective baseline. The key idea - using radial basis function expansions of distances as edge features - should be in your architecture regardless of what else you choose.

**DimeNet / DimeNet++** (Gasteiger et al., 2020): Adds directional information via angles between triplets. DimeNet++ is 10x faster. Relevant because bond angles carry information about the QM topology you're trying to predict. However, angular message terms scale O(|N|²) with neighbors, which is a throughput concern at scale.

**PaiNN** (Schütt et al., 2021): Equivariant message passing with scalar (l=0) and vector (l=1) features in Cartesian space. Good balance of expressiveness and speed - ~3.5x faster than DimeNet++ at comparable or better accuracy on standard benchmarks.

**PaiNN strengths for this project:**

- Conceptually simple equivariance in Cartesian space (no CG coefficients, no spherical harmonics machinery in the core message passing)
- Equivariant messages recover angular information at O(|N|) cost, avoiding the O(|N|²) angle enumeration of DimeNet
- Propagation of directional information across message passes - critical for resolving distant substituent effects on BCP properties
- Natural framework for tensorial property prediction via rank-1 decomposition (Eq. 11-14 in the paper)
- Fast inference: 13ms per batch of 50 molecules on V100, important for 100M-scale training
- ~600k parameters (vs 1.8M for DimeNet++), good parameter efficiency
**PaiNN limitations for this project (identified in deep-dive):**

- **l=1 angular resolution is insufficient for TM/lanthanide/actinide coordination chemistry.** The vector (l=1) channel captures dipolar angular information only. Octahedral vs trigonal prismatic coordination (same distances, zero net l=1 moment for homoleptic complexes) cannot be distinguished in a single message pass. For lanthanide coordination numbers 8-12 with subtle geometry differences, even l=2 may be marginal.
- **BCP properties depend on multi-body arrangements** that l=1 captures indirectly (through stacking layers) rather than directly. The density at a BCP is sensitive to the full coordination environment of both atoms, not just the pairwise direction.
- **The 20 fixed sinusoidal RBF default provides ~0.25 Å resolution**, which is coarse for BCP prediction where ρ can change by a factor of 2 over a 0.2 Å distance change (e.g., C-C single vs double bond). This is easily fixed by increasing RBF count (see Section 2.6).
- **Upgrading from l=1 to l=2 later is architecturally non-trivial** - it changes the tensor algebra of every layer, not just a hyperparameter swap.
**Recommended role for PaiNN in this project:** Use as the backbone of a hybrid encoder (Section 3.3) where l=1 equivariant features handle directional propagation and l=2 scalar invariants (computed separately) provide higher angular resolution. This preserves PaiNN's speed while addressing the angular limitation.

**TorchMD-Net** (Thölke & De Fabritiis, 2022): Equivariant Transformer that combines the benefits of attention mechanisms with geometric priors. Has a clean PyTorch implementation and good scaling properties.

**EGNN** (Satorras et al., 2021): E(n) equivariant graph neural networks. Simpler than the above, uses coordinate updates during message passing. Worth considering as a baseline because it's very fast.

### 2.3 Link Prediction / Graph Structure Learning

This is the core novelty of your approach - predicting the QM topological graph. The relevant literature spans several subfields:

**Graph Structure Learning (GSL)**:

- **IDGL** (Chen et al., NeurIPS 2020): Iterative Deep Graph Learning - learns graph structure and node representations jointly. Very close to your neural SCF idea. It maintains a differentiable adjacency matrix that's refined alongside node embeddings. Key paper for your architecture.
- **SLAPS** (Fatemi et al., NeurIPS 2021): Simultaneously Learning graph structure And Predictions. Uses a generative model for graph structure that's trained end-to-end with the downstream task.
- **NodeFormer** (Wu et al., NeurIPS 2022): All-pair message passing with kernelized softmax - efficiently computes attention over all node pairs without explicit edge construction. Could replace your O(N²) candidate edge enumeration.
**Chemical bond prediction specifically**:

- **BondNet** (Wen et al., 2021): Predicts bond dissociation energies using a heterogeneous graph similar to yours (atom + bond + global nodes). Not link prediction per se, but the architecture is very relevant.
- The bond topology prediction problem also appears in coarse-grained molecular dynamics, where people predict which CG beads should be bonded. **CGSchNet** and similar approaches use distance-based edge prediction with learned cutoffs.
**For QTAIM specifically**: The number of bond critical points is relatively small and their existence is strongly correlated with internuclear distance (most BCPs are between atoms within ~1.5-3Å). This means a distance-initialized graph with learned refinement is a reasonable prior - your current approach of distance cutoff → iterative refinement is scientifically sound.

### 2.4 Joint Prediction (Links + Features Simultaneously)

This is where it gets interesting and where I want to be honest about uncertainty in the science.

**Graph Generation / Diffusion Models**:

- **DiGress** (Vignac et al., ICLR 2023): Discrete denoising diffusion for graph generation. Generates node types and edge types jointly by denoising a corrupted graph. Very relevant - you could frame your problem as denoising a distance-cutoff graph into the QTAIM graph, with continuous node/edge features generated alongside.
- **GDSS** (Jo et al., ICML 2022): Score-based diffusion for simultaneous node and edge generation. Handles continuous features naturally.
- **MiDi** (Vignac et al., 2023): Extension of DiGress to 3D molecular generation with continuous coordinates. Shows that diffusion can handle mixed discrete (topology) + continuous (positions/features) generation.
**Flow Matching**:

- **Flow Matching for Generative Modeling** (Lipman et al., ICLR 2023): Simpler and more stable than diffusion, with comparable quality. Applied to molecules in several follow-ups.
- **EquiFM** (Song et al., 2023): Equivariant flow matching for molecular conformation generation. The equivariant flow matching framework could be adapted to your setting where you flow from an initial distance-cutoff graph to the QTAIM graph.
- **Riemannian Flow Matching** (Chen & Lipman, 2024): For generation on manifolds and mixed spaces. Relevant because your output space is mixed: discrete (edge existence) + continuous (descriptor values).
**Denoising / Iterative Refinement (closest to your neural SCF)**:

- **AlphaFold2's recycling** (Jumper et al., 2021): The most successful example of iterative refinement in structure prediction. Key insight: share weights across iterations and pass structure module outputs back as inputs to the representation module. Your `FullPredictor` loop is conceptually identical.
- **Iterative refinement in SE(3)-Transformers**: Several works show that recycling predictions back through equivariant networks improves predictions. **RoseTTAFold** also uses this pattern.
- **Deep Equilibrium Models (DEQ)** (Bai et al., NeurIPS 2019): Instead of a fixed number of iterations, find the fixed point of an implicit layer. Could make your neural SCF converge to a self-consistent solution rather than running a fixed number of steps. The physics analogy to actual SCF convergence is appealing here.
### 2.5 Multi-Task / Multi-Scale Prediction

Since you're predicting atom, bond, and global features simultaneously:

**Multi-task learning on graphs**:

- **GPS** (Rampášek et al., NeurIPS 2022): General, Powerful, Scalable graph Transformer. Combines message passing with global attention. Relevant for your heterogeneous prediction tasks.
- **Graphormer** (Ying et al., NeurIPS 2021): Graph Transformer with structural encodings. Won OGB-LSC. The spatial encoding (using shortest path distances) could encode your distance information naturally.
**Heterogeneous graph transformers**:

- **HGT** (Hu et al., 2020): Heterogeneous Graph Transformer - uses type-specific attention. Direct replacement for your current hetero message passing layers.
- **HEAT** (Chen et al., 2023): Heterogeneous Edge Attribute Transformer. Handles edge features natively.
### 2.6 Radial Basis Function Design Space (NEW)

The choice of radial basis functions is a critical and underappreciated design decision for this project. Standard energy/force models use 20 fixed sinusoidal RBFs - this is insufficient for predicting QTAIM descriptors across chemically diverse data.

**Why RBF design matters more here than for energy/force prediction:**

BCP properties (ρ, ∇²ρ, ellipticity) vary rapidly with internuclear distance in the bonding region. The electron density at a C-C BCP changes by roughly a factor of 2 between 1.34 Å (double bond) and 1.54 Å (single bond) - a 0.2 Å range covered by less than one basis function at the default 20 RBFs over [0, 5Å]. The Laplacian ∇²ρ is even more sensitive, being a second derivative of the density.

For energy/force prediction, the relevant distance scales are well-understood (bond lengths, van der Waals radii) and the energy landscape is relatively smooth. For QTAIM descriptors, the property landscape is "rougher" relative to geometric variation, and the relevant distance scales are element-pair-specific across diverse chemistry.

**Fixed sinusoidal RBFs** (Klicpera et al.): Orthogonal on [0, r_cut], uniform resolution. Default 20 → ~0.25 Å resolution. At 50 → ~0.1 Å, at 100 → ~0.05 Å. Cost of increasing count is negligible (wider input to filter network's first linear layer, minimal impact on per-edge message passing cost).

**Fixed Gaussian RBFs** (SchNet original): Evenly spaced Gaussians with fixed width. Hyperparameter-dependent (center spacing, width). Not orthogonal, potential redundancy.

**Learnable Gaussian RBFs**: Trainable centers and widths. Can concentrate resolution where the data needs it - e.g., high resolution around typical bond lengths (1.0-2.0 Å for organics, 1.9-2.2 Å for Fe-N, 2.2-2.8 Å for lanthanide-ligand). Potentially fewer basis functions needed for equivalent performance. Risk: training instability, basis collapse (multiple Gaussians converging to same center).

**Learnable Bernstein or Chebyshev bases** (used in some recent work): Better numerical conditioning than Gaussians, harder to collapse.

**Hybrid fixed + learnable**: Fixed sinusoidal base for stable training, learnable correction for adaptive resolution. Lower risk than pure learnable.

**Recommendation for this project**: Start with 50 fixed sinusoidal RBFs (2.5x default, negligible cost). If BCP prediction plateaus, increase to 100 before trying learnable bases. Learnable RBFs are most valuable when you need to minimize basis count for inference speed, or as an ablation for the methods paper. See Lever Ordering in Section 3.3.

### 2.7 Long-Range Interactions via Dual Cutoff (NEW)

**Primary motivation - descriptor quality, not just scaling**: A single short cutoff (e.g., 4Å) captures bonded interactions and first-shell geometry well but misses longer-range field effects that genuinely influence QTAIM descriptors: the effect of a remote electronegative substituent on BCPs in a conjugated system, oxidation-state-driven modulation of ligand BCPs in TM complexes, or through-space polarization in charged or ionic systems. The dual-cutoff approach is a way to incorporate this context cheaply - without paying for full equivariant message passing at long range, and without the added complexity of learned graph structure (IDGL anchors).

The secondary benefit is inference scaling to large systems (1000+ atoms), but that's a consequence of the approach rather than the reason to use it.

**The GemNet-OC dual-cutoff strategy**: Maintain two separate edge sets with different cutoffs and different message passing treatment:

- **Short-range edges** (e.g., ≤4Å): Full equivariant message passing (PaiNN-style, with angular features and l=1/l=2 features). This is where BCP-relevant geometric information lives.
- **Long-range edges** (e.g., 4–8Å): Cheap scalar-only message passing - just RBF-expanded distance + node scalars, no equivariant features. This propagates coarse-grained electronic context: "there's a charged nitrogen 6Å away," not "the nitrogen is oriented at this angle."
The key physical justification: **angular precision matters at short range, not long range**. The precise coordination geometry of a TM center at 6Å influences a BCP primarily through its scalar field effect (oxidation state, electron density), not through its directional arrangement. Running equivariant MP over those long edges would add cost without adding useful information.

**Implementation sketch**:

```python
# Graph construction
short_edges = radius_graph(pos, r=4.0, batch=batch)   # full equivariant MP
long_edges  = radius_graph(pos, r=8.0, batch=batch)   # scalar-only MP
# Exclude short edges from the long set to avoid double-counting
long_only_mask = ~isin(long_edges, short_edges)        # see note below
long_edges = long_edges[:, long_only_mask]

# Encoder forward pass:
# 1. Short-range PaiNN layers  - equivariant, l=1 (or l=2 at Lever 4)
# 2. Long-range scalar MP      - 1-2 cheap layers, scalar node features only
# 3. Combine: node_feats = short_feats + long_feats  (or concat + project)
```

Note: the `isin` mask needs care with batched graphs in PyG. The OCP codebase has a reference implementation of this pattern.

**Practical cutoff values**:

- Short-range: 4Å (standard; captures all covalent BCPs and most coordination bonds)
- Long-range: 8Å (covers most secondary coordination shells and van der Waals contacts)
- For f-element systems: consider extending long-range to 10–12Å given larger coordination spheres
**Training consideration**: Most 100M training molecules are small (10–50 atoms) where the long-range channel is sparse or empty. This is fine - the long-range edge set contributes zero messages in those cases. Train the full dual-cutoff architecture from the start so the long-range channel is learned, even if rarely exercised on small molecules. Adding it post-hoc as an inference patch won't work well.

**Empirical gate (Phase 3)**: Run the model at single cutoff (4Å) vs dual cutoff (4Å + 8Å) on a held-out set of structures where long-range effects are expected - charged systems, ionic host-guest complexes, large TM or f-element complexes. Measure BCP descriptor errors on atoms near or beyond the short cutoff boundary. If errors are indistinguishable, the single-cutoff model is the deployment default and the long-range channel is dropped.

**Why not IDGL anchors instead**: Anchors solve a different problem - reducing O(n²) all-pairs communication to O(ns). For chemistry, geometry already provides that sparsification for free (BCPs beyond 6Å don't exist), so anchor machinery adds complexity without benefit. The dual-cutoff approach exploits the chemistry directly.

### 2.8 Cutoff Envelope Functions (NEW)

**The problem**: A hard distance cutoff creates a discontinuity in the message passing - atoms just inside the cutoff contribute fully, atoms just outside contribute nothing. For energy/force models this violates conservation laws. For QTAIM descriptor prediction it creates a subtler but real problem: the model can learn to exploit the discontinuity as a feature, leading to artifacts near the cutoff radius and poor generalization when cutoff radii are changed.

**What envelope functions do**: An envelope function e(r) multiplies the RBF expansion (and often the message itself) to smoothly attenuate contributions to zero as r → r_cut:

```
m_ij = e(r_ij) · MLP(RBF(r_ij)) · ...
```

This ensures the message passing function is at least C¹ continuous at the cutoff - no sharp discontinuity in either value or gradient. The cost is essentially zero (one scalar multiply per edge).

**Common choices**:

**Polynomial envelope** (DimeNet, Gasteiger et al. 2020): The standard choice. For polynomial order p:

```python
def envelope(r, r_cut, p=6):
    x = r / r_cut
    return (1 - (p+1)*(p+2)/2 * x**p + p*(p+2) * x**(p+1) - p*(p+1)/2 * x**(p+2)) * (r < r_cut)
```

p=6 is the DimeNet default and a reasonable starting point. Higher p gives a "flatter" envelope (closer to 1.0 for most of the range) with a sharper but still smooth decay near r_cut.

**Cosine envelope** (simpler, used in some SchNet variants):

```python
def envelope(r, r_cut):
    return 0.5 * (1 + torch.cos(torch.pi * r / r_cut)) * (r < r_cut)
```

Smoother but attenuates more aggressively in the mid-range - can hurt performance on interactions between 2–3Å if r_cut is 5Å.

**Recommendation for this project**: Use the polynomial envelope (p=6) on both the short-range and long-range edge sets. Apply it to the full RBF-expanded edge feature before it enters any MLP. This is particularly important for the dual-cutoff setup (Section 2.7) because atoms near the 4Å short-range boundary will also appear in the long-range graph - the smooth handoff between the two edge sets requires that neither discontinuously cuts off. Apply the envelope with r_cut matching each respective cutoff radius.

**Implementation note**: The envelope should be applied *before* the RBF features enter the filter network, not after. Applying it after the MLP (to the full message) also works and is sometimes cleaner, but applying it to the input is more numerically stable since the MLP doesn't see sharp gradients near the cutoff.

---

### 2.9 Training Data: OMol-Descriptors-4M and Its Defects (NEW)

The scale arguments in 2.1 and 3.4 assumed a corpus; it now exists and its shape constrains the model.

**Scale and provenance**: 3,986,738 structures, 34 verticals, 80+ elements, one level of theory (wB97M-V / def2-TZVPD, inherited from OMol25). The single-level-of-theory point matters: Section 7's transferability question is deferred, not answered. Everything the model learns is at this functional and basis.

**Split**: composition-hashed 0.8/0.1/0.1 with 58,786 structures held out into five stress suites (H1 metal-ligand, H3 reactivity, H6 lanthanide-ligand, H7 n_atoms > 250, H8 |q| > 4). 96 % of test compositions have a same-element-set nearest training neighbour differing by a median 5.3 % of atom count, and 17.3 % are one CH2 away, so the main test split measures interpolation. The H-suites are the extrapolation measurement. Every result in this roadmap should be reported on the main test split and per H-suite; a single aggregate number hides exactly the failure modes (TM, lanthanide, charged, large) that Sections 2.2 and 3.3 are about.

**Descriptor definitions available as targets**: bond presence under four definitions (QTAIM bond path, Mayer, Loewdin, fuzzy bond order), so the "bond order ambiguity" question in Section 7 is now a concrete multi-head design choice: one shared encoder, one edge-scoring head per bonding definition, evaluated separately.

**Known defects and their status** (all in qtaim_generator):

| Defect | Mechanism | Detection / fix | Status |
|--------|-----------|-----------------|--------|
| Lost bond critical points | Truncated `CPprop.txt` drops tail CPs; nuclear CPs are numbered first so the prefix passes validation | `audit-qtaim-connectivity` compares to covalent-radius bonding from `structure.lmdb`; `select-qtaim-rerun` / `verify-qtaim-rerun` repair loop with known-good controls | Tooling done; rerun campaign in progress on the local ~199K subset |
| Stolen nuclear CPs | `find_cp_map` took the first same-element CP within 1.0 A; two H atoms 0.99 A apart lost 3-4 BCPs | Fixed (fixture `qtaim_0` corrected from 101 to 102 bonds) | Fixed 2026-09-03 |
| Duplicate Mayer charges | ORCA column mislabel made `mayer_orca` a copy of Mulliken | Excluded from configs; `remove_mayer_charge_dup.py` for existing LMDBs | Fixed |
| Infinite re-queue after walltime kills | Restart gate demanded `qtaim.json`; zip merge kept the larger (spammed) log | Gate now parses each `.out`, checks charge sums and electron counts | Fixed 2026-09-03 |
| ECP-driven extreme charge magnitudes | Unclear; likely core-electron accounting in ECP elements | None yet | Open (generator todo #6) |
| Missing `atom.pos`/`atom.z` on released graphs | Converter never emitted them | `add_pos_z_to_graphs.py` post-hoc patch | Patched; converter should emit natively |

Implication for training: the first-generation graph LMDBs contain some fraction of molecules whose QTAIM bond set is incomplete. For T3 this is label noise concentrated in large molecules (where `CPprop.txt` truncation is likelier). Until the rerun campaign finishes, T3 metrics on H7 should be reported with and without records flagged by the audit.

---

## 3. Architectural Recommendations

### 3.1 The "Neural SCF" Approach - Making It Work

*v4 note*: the diagram below says PaiNN; read it as "the `encoder_fn` encoder selected in Phase 1" (schnet, dimenetpp, or equivariant). The edge-score module is the T3-b scorer from Section 3.5 and the feature heads are the T2 regressor. Nothing else in this section changed.

Your intuition is good. Here's how to make the iterative approach trainable and scalable:

**Architecture: Iterative Equivariant Refinement Network (IERN)**

```
Input: Atomic coordinates + elements
  │
  ▼
[Geometry Encoder] ── Hybrid encoder (Section 3.3):
  │                    PaiNN l=1 equivariant + l=2 scalar invariants
  │                    50+ sinusoidal RBFs + element embeddings
  │
  ▼
[Initial Graph Constructor] ── Distance cutoff (generous, e.g., 4Å)
  │                              Creates dense initial graph
  ▼
┌─────────────── Iteration Block (shared weights, N iterations) ──────────────┐
│                                                                              │
│  [Edge Score Module] ── Predicts edge existence probability                 │
│       │                  Uses node embeddings + distance + l=2 invariants   │
│       │                  Output: continuous edge weights (not hard threshold)│
│       ▼                                                                      │
│  [Weighted Message Passing] ── PaiNN-style l=1 equivariant MP              │
│       │                        Edge weights gate messages                    │
│       │                        l=2 scalar invariants as additional features  │
│       │                        Atom + bond features updated jointly          │
│       ▼                                                                      │
│  [Feature Prediction Heads] ── Per-type MLPs                                │
│       │                        atom features, bond features, global features │
│       ▼                                                                      │
│  [Feedback] ── Predicted features concatenated with embeddings              │
│                for next iteration                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  ▼
[Final Edge Decision] ── Threshold or top-k on converged edge scores
  │
  ▼
[Final Feature Readout] ── Descaled predictions for atom/bond/global descriptors
  │                         Optional: l_max=2 tensor product for rank-2 outputs
  │                         (one expensive pass, amortized over all iterations)
```

**Critical design decisions**:

1. **Soft edges during training, hard edges at inference**: Don't use hard edge thresholds during training - this kills gradients. Instead, use continuous edge weights (sigmoid of edge scores) to gate message passing. At inference, threshold to get the final topology. This is analogous to Gumbel-Softmax for discrete structure learning.
2. **Shared weights across iterations**: Like AlphaFold2 recycling. This dramatically reduces parameters and acts as a strong regularizer. The model learns a "refinement operator" rather than "step 1, step 2, step 3."
3. **Distance-aware edge scoring**: The edge score module should take as input: (a) concatenated node embeddings, (b) interatomic distance (via RBF expansion), (c) l=2 scalar invariants encoding the angular environment of both endpoints, and (d) optionally angular features from triplets. QTAIM bond paths are strongly distance-correlated, so this prior helps enormously. The l=2 invariants are critical here - they allow the edge scorer to distinguish coordination geometries (octahedral vs trigonal prismatic) that determine whether a BCP exists.
4. **Don't convert to homogeneous graphs**: Your current link model collapses the heterograph to a homograph, losing the atom/bond distinction. Instead, score edges in the full heterogeneous representation. An edge (atom_i, atom_j) produces a bond node between them - this is the natural heterogeneous formulation.
5. **Loss function**: Multi-task loss with three components:
   - Binary cross-entropy on edge existence (link prediction)
   - MSE/MAE on node features (atom + bond descriptors)
   - Optional: auxiliary loss on global molecular properties
   Weight the edge loss higher in early training (get topology right first, then refine features). Curriculum learning: start with molecules where QTAIM topology matches distance-cutoff topology (most organic molecules), then introduce harder cases (TM complexes, non-nuclear attractors, etc.).
### 3.2 The Diffusion/Flow Alternative

For comparison, here's the diffusion formulation:

**Architecture: QTAIM-DiGress**

Frame the problem as: given geometry, denoise a corrupted labeled graph into the QTAIM graph.

```
Forward process: QTAIM graph → add noise to edge types + feature values → random graph
Reverse process: random graph → denoise with geometry-conditioned model → QTAIM graph
```

The geometry acts as conditioning (not generated, given). Edge types are discrete (bond/no-bond, or more fine-grained: BCP type, RCP, CCP). Node features are continuous.

**Pros**: Handles mixed discrete/continuous outputs naturally. Can generate diverse samples (useful for uncertainty quantification). Doesn't require iterative refinement at inference - can do it in one shot with enough denoising steps.

**Cons**: Slower inference than the iterative approach (many denoising steps). Harder to train. Less physically interpretable than the SCF analogy. I'm not confident this is scientifically better for your problem - the SCF analogy gives you a strong inductive bias that diffusion doesn't naturally capture.

**My honest assessment**: For a foundational model, I'd start with the iterative approach (3.1) because the physics are cleaner and it's easier to debug and interpret. The diffusion approach is worth exploring as a comparison, but I wouldn't make it the primary architecture. The iterative approach has a natural connection to the actual SCF procedure, which gives you interpretability and a convergence criterion that diffusion lacks.

### 3.3 Expressiveness Lever Ordering

A critical design principle for a foundational model at scale: **pull the cheapest lever first, only escalate when the previous lever plateaus.** This avoids committing to expensive architectural choices before you have empirical evidence they're needed.

**Lever 0 - PaiNN baseline, 20 sinusoidal RBFs (paper default)**

Cost: baseline. The starting point.

**Lever 1 - Increase to 50-100 fixed sinusoidal RBFs**

Cost: negligible (wider input to filter network). Expected impact: significant improvement on BCP predictions where the bottleneck is radial resolution. The default 20 RBFs provide ~0.25 Å resolution, which is too coarse for the bonding region where BCP properties vary most rapidly. At 50 RBFs (~0.1 Å) or 100 RBFs (~0.05 Å), you have dense coverage of the chemically critical distance ranges. This should be the *default* for this project, not an upgrade.

**Lever 2 - Add l=2 scalar invariants to the PaiNN encoder**

Cost: low (~50 lines of code, one-time computation per layer, not per message pass iteration). For each atom i, compute:

```
A_i^(2) = Σ_j R(r_ij) Y_2(r̂_ij)     # l=2 neighbor sum (5 components)
B_i^(2,2) = || A_i^(2) ||²            # scalar encoding quadrupolar environment
```

These B invariants are scalars that encode l=2 angular information. They enter the PaiNN scalar channel as additional features, propagating through message passing at zero extra cost per layer. This gives the model the ability to distinguish coordination geometries (octahedral vs trigonal prismatic, tetrahedral vs square planar) that l=1 vector features cannot resolve in a single pass.

**What you lose vs full l_max=2**: no equivariant l=2 feature propagation between layers. The l=2 information enters as fixed scalar descriptors recomputed from geometry, not as evolving equivariant features. For BCP prediction this is likely sufficient (BCP properties depend primarily on local environment), but for properties requiring propagation of quadrupolar information from distant atoms, it may not be.

**Lever 3 - Learnable radial basis functions**

Cost: moderate (trainable Gaussian centers + widths, some training instability risk). Most valuable when: (a) you need to minimize RBF count for inference speed, or (b) the element-pair diversity of your data means uniform radial resolution is wasteful. At 100 fixed RBFs, the marginal value of learnable over fixed diminishes. **Best positioned as an ablation for the methods paper rather than a core architectural commitment.**

Implementation options:
- Pure learnable: trainable Gaussian centers and widths, risk of basis collapse
- Hybrid: fixed sinusoidal base + learnable correction (lower risk)
- Element-pair-specific: different learnable bases for different atom pair types (expensive but chemically motivated)
**Lever 4 - Full l_max=2 equivariant features**

Cost: high (~3-4x per-edge cost in tensor product operations, changes tensor algebra of every layer). Pull this lever only if Lever 2 (l=2 scalar invariants) proves insufficient for TM/lanthanide/actinide environments. Indicators that you need this: (a) descriptor prediction for homoleptic TM complexes plateaus despite good organic molecule performance, (b) systematic errors correlated with coordination geometry rather than element identity.

If needed, two implementation options:
- MACE-style encoder with l_max=2, 2 interaction layers (clean, well-tested)
- eSCN-style discrete sphere representation (avoids CG coefficients, better scaling, more complex implementation)
**Lever 5 - l_max=3 or higher (unlikely to be needed)**

Cost: very high. Only justified if l_max=2 fails on lanthanide/actinide coordination with CN 8-12 where subtle geometry differences at high angular order determine electronic structure. Empirical evidence required before committing.

**Interface design principle**: The downstream components (edge scoring, feature prediction heads) should consume *scalar* features extracted from the equivariant representation. This allows swapping the encoder (Lever 2 → Lever 4) without rewriting the heads. Design the interface at:

```python
class GeometryEncoder(nn.Module):
    """Returns scalar features per atom and per edge.
    Internal representation may be l=1 equivariant (PaiNN)
    or l=2 equivariant (MACE), but output interface is scalar."""

    def forward(self, Z, pos, edge_index, batch):
        # ... encoder-specific internals ...
        return node_scalars, edge_scalars, node_vectors  # vectors for tensor outputs only
```

**Implementation status of the levers (2026-09-08)**:

| Lever | Status in qtaim_embed |
|-------|-----------------------|
| 0 baseline | `encoder_fn: "schnet"` with 50 Gaussians is the working baseline (not PaiNN with 20 sinusoidal RBFs). |
| 1 more RBFs | Available as `encoder_num_gaussians` (schnet) and `encoder_num_radial` (dimenetpp, equivariant). Not yet swept. |
| 2 l=2 scalar invariants | Absent as a standalone module. `dimenetpp` supplies angular information via triplets instead, at O(sum deg^2) cost capped by `encoder_max_neighbors`. |
| 3 learnable RBFs | Absent (explicitly out of scope in `equivariant_encoder.py` docstring). |
| 4 full l_max=2 | Available as `encoder_fn: "equivariant"` with `encoder_lmax: 2` (e3nn). Slow; no benchmark yet. |
| 5 l_max>=3 | Available in principle via `encoder_lmax`; not recommended. |

The comparison that Phase 1 needs is therefore schnet(50) vs schnet(100) vs dimenetpp vs equivariant(lmax 1) vs equivariant(lmax 2) on T2 with known topology, per H-suite.

### 3.4 Scaling Considerations

Superseded by `docs/plans/2026-09-08-feat-performance-engineering-plan.md`, which has measured baselines and acceptance targets. Summary of what changed since v3:

- The data pipeline items are done: sharded LMDB directories, per-worker LMDB envs, composition splits, mixed precision, DDP.
- The real bottleneck is kernel-launch overhead in the per-edge-type `HeteroConv` stack (six edge types x eight layers), not data loading and not FLOPs. The plan's Track A addresses this with larger batches, a fused typed convolution, and shape bucketing for CUDA graphs. Target: 15,000 samples/s on one A5000 from 3,300 today.
- On-the-fly graph construction from coordinates (v3 item) is not needed at 4M; pre-built shard directories are fine. Revisit above ~50M.
- The dual-cutoff strategy (2.7) remains the plan for H7 and MOF-scale systems and is not implemented. The neighbor builder is a chunked `cdist`, so a cell-list path is a prerequisite for >500-atom systems.
- Inference and packaging (batched predictor, bundle format, ASE calculator, `torch.export`) are Tracks B and C of the same plan.

### 3.5 Bond Prediction Validation Program (NEW)

This is gap #1 from Section 1.5 and the first thing to build.

*Cross-references*: the task definition, measured reference numbers and scoping decision live in qtaim_generator's `docs/plans/2026-07-27-neurips-r1-05-t3-bond-classification-plan.md` (doc 05). The code-level plan is `docs/plans/2026-09-08-feat-t3-bond-classifier-plan.md` in this repo. Two facts from doc 05 that this section depends on:

- **Label gate.** Trustworthy QTAIM labels require graphs built from the regenerated `qtaim.lmdb` (generator doc 10, lock 5 in the dependency map). Every local `qtaim.lmdb` predates the repair campaign. Code and tests proceed on existing fixtures; reported numbers, including the Stage T3-a baselines below, wait for regenerated graphs and are labelled "development" until then.
- **Existing predictor bugs.** `MLPPredictor` in `models/layers_homo.py` ends in a sigmoid while the loss is BCE-with-logits (double sigmoid; `FullPredictor` adds a third), and `AttentionPredictor` softmaxes over a size-1 dimension so the destination node is ignored. The T3 model is a new module (`GCNBondPred`) and does not inherit these; `GCNLinkPred` is left unchanged. The "full wavefunction simulator" is, in graph terms, a function from (Z, pos, charge, spin) to a labeled QTAIM graph. Its first and hardest component is the topology. Everything else is regression on a known graph, which the node and graph models already do. So the program is staged: prove topology first, then properties on predicted topology, then close the loop.

**Stage T3-a: baselines that must be beaten (1 week of code now; numbers after the label gate)**

Reproduce the distance-rule baseline inside qtaim_embed so that every later model is compared on identical candidate sets and identical metrics.

- Candidate set: `candidate_pairs` from `models/encoders/neighbors.py` at `pool_multiplier = 2.0` (the paper's cutoff). Measure candidate recall against QTAIM bond paths per H-suite first. Any positive outside the candidate set is unreachable by every downstream model; if recall on H1/H6 is below 99.5 %, raise the multiplier or add a non-covalent radius table before training anything.
- Baseline 1: one-parameter distance rule `d <= alpha * (rcov_i + rcov_j)`, alpha tuned on train. Expected macro-F1 0.971 on the main split.
- Baseline 2: per-element-pair alpha (a lookup table fit on train). This is the strongest "no learning" baseline and should be reported; if it reaches 0.985 the bar for a GNN is higher than v3 assumed.
- Metrics: precision, recall, macro-F1, AUROC, and calibration error at the operating threshold, on the main test split and each of H1, H3, H6, H7, H8. Also report per-molecule exact-topology rate (all bonds right), since a downstream BCP regressor sees whole graphs.
- Deliverable: `qtaim_embed/scripts/eval/eval_bond_baselines.py` and a results table committed under `docs/results/`.

**Stage T3-b: geometry-aware edge scorer (2-3 weeks)**

A new module, `GCNBondPred`, rather than retrofitting `GCNLinkPred` (see the implementation plan for why: the old model is 3D-blind through `hetero_to_homo`, samples uninformative negatives, and message-passes over the label graph).

- Inputs per candidate edge: RBF-expanded distance (Bessel, 50 basis, p=6 envelope), the two endpoint atom embeddings from a shared `encoder_fn` encoder run over the candidate graph, the ratio `d / (rcov_i + rcov_j)` as a scalar, and for the dimenetpp variant the angular triplet features. This is design decision 3 from Section 3.1.
- Heterogeneous formulation (design decision 4): score edges as prospective bond nodes; do not collapse to a homograph.
- Loss: BCE with positive-class weighting from the candidate set's positive rate; report the calibrated threshold chosen on val, not an implicit logit-0 threshold.
- Ablations: schnet vs dimenetpp encoder; with and without the `d/rcov` scalar; 20 vs 50 vs 100 radial functions (lever 1).
- Acceptance: beats Baseline 2 on macro-F1 on the main split and on every H-suite, with the largest gains expected on H1 and H6 (where distance alone is ambiguous). A test in `tests/test_link_models.py` that trains for a fixed budget on the small TMQM fixture and asserts F1 above a stored floor, so regressions are caught.

**Stage T3-c: topology definition heads (1 week)**

Same encoder, four edge heads: QTAIM bond path, Mayer > threshold, Loewdin > threshold, fuzzy bond order > threshold. Report agreement matrices between heads and against DFT. This answers the Section 7 "bond order ambiguity" question with data and gives the ASE calculator a choice of bonding definition.

**Stage T2-on-T3 (2 weeks)**

Train the node-level BCP regressor (T2) on ground-truth topology as today, then evaluate it on predicted topology from T3-b. The metric gap between the two is the cost of topology error and is the number that justifies (or not) the iterative loop. If the gap is small on the main split but large on H1/H6, that is the curriculum boundary for Phase 2.

**Then Phase 2** (Section 4): the iterative model, with the T3-b scorer as the edge module and the T2 regressor as the feature module, sharing the encoder.

**Why this order**: v3 put the iterative architecture before validating its components. The NeurIPS review made the cost of that concrete: three benchmark tasks with no baselines. Each stage above produces a reportable number on the released dataset, and each is a component the iterative model reuses unchanged.

---

## 4. Recommended Phased Roadmap (revised)

### Phase 0: Infrastructure catch-up (done, 2026-02 to 2026-08)

- PyG migration, LMDB hardening, composition splits, DDP, 3D encoders, `atom.pos`/`atom.z`, OMol-Descriptors-4M build, stress suites. Status: complete except the QTAIM rerun campaign and the ECP charge issue (generator).

### Phase 1: Bond prediction validation and encoder validation (now, 6-8 weeks)

Runs the program in Section 3.5 and the Phase 1 encoder baseline from v3 in parallel, on the same hardware after the performance plan's Track A lands (they are otherwise 26-hour-per-experiment jobs).

- T3-a baselines and candidate recall per H-suite (week 1).
- Performance plan A0-A2 (weeks 1-2) so that the sweeps below fit in a day each.
- T3-b geometry-aware edge scorer with encoder and RBF ablations (weeks 2-4).
- T1 and T2 on known topology for `schnet(50)`, `schnet(100)`, `dimenetpp`, `equivariant(lmax 1)`, `equivariant(lmax 2)` on one vertical (tm_react, 158K), then the full split (weeks 3-6). This is the lever-ordering experiment from Section 3.3; the decision rule is unchanged.
- Multi-task loss weighting (existing plan) before T1's six-scheme joint regression (week 3).
- T3-c definition heads and T2-on-T3 (weeks 6-8).
- Deliverables: results tables under `docs/results/`, a bond-prediction bundle loadable by the new predictor API, NeurIPS rebuttal numbers for T1/T2/T3.

### Phase 2: Iterative architecture (8-12 weeks after Phase 1)

Unchanged in substance from v3 Section 3.1 with these updates:

- The edge module is the T3-b scorer; the feature module is the T2 regressor; both share the encoder chosen in Phase 1.
- Soft edges in training, hard edges at inference, shared weights across iterations, curriculum from main split to H-suites.
- Compare end-to-end joint training against the Phase 1 pretrained-then-composed baseline on T2-on-T3. This is the experiment that decides whether the iterative idea earns its complexity.
- Convergence: log per-iteration edge-set change and feature deltas; try a DEQ-style fixed-point variant only if fixed 3-5 iterations show non-monotone behaviour.

### Phase 3: Scale, large systems, alternatives (after Phase 2)

- Full 3.1M train split with DDP on a cluster (budget in the performance plan A6).
- H7 and MOF-scale inference: cell-list neighbors, dual cutoff (2.7), envelope on the encoder path (2.8), measured against single-cutoff on the H7 suite as v3 specified.
- Diffusion or flow-matching comparison (3.2) only if Phase 2 convergence is poor.
- Uncertainty via ensembles over the Phase 2 model.

### Phase 4: Foundation model and release (ongoing)

- Bundles on the Hugging Face Hub next to the dataset card; ASE calculator with bonding-definition choice; energy and forces head once `energy.lmdb` exists (generator todo #9), enabling the fairchem MLIP benchmark (todo #8).
- Fine-tuning protocols per chemistry; transfer to other levels of theory once a second-functional subset exists.

---

## 5. Concrete Next Steps for the Codebase

Ordered. Items 1-3 are independent and can start today.

1. **T3-a baseline script** (`qtaim_embed/scripts/eval/eval_bond_baselines.py`): candidate recall per H-suite with `candidate_pairs`, then the one-parameter and per-element-pair distance rules with precision/recall/macro-F1/AUROC/calibration. Read holdout membership from the generator's `manifest_holdout.parquet`. Commit the table.
2. **Performance plan A0 and A1**: benchmark harness with real GPU utilization, then the batch-size and bf16 sweep. Without this every Phase 1 sweep costs a day of GPU per point.
3. **Vectorize `FullPredictor` candidate edges and switch to `candidate_pairs`** (todos/009, performance plan B1). Small, isolated, and it makes the existing predictor usable on the H7 suite.
4. **Geometry-aware edge scorer** (`qtaim_embed/models/link_pred/edge_scorer.py`): heterogeneous, RBF distance plus `d/rcov` scalar, shared `encoder_fn` encoder, calibrated threshold, precision and recall logged. Add the accuracy-floor test.
5. **Multi-task loss weighting** per the existing plan; needed before joint T1.
6. **Encoder sweep configs** for T1/T2 on known topology (`profiling/graph_configs/encoder_*.json`), one per lever, run on tm_react then the full split.
7. **Predictor API, bundle format, ASE calculator** (performance plan B2, B3, C2). The T3-b model is the first bundle.
8. **Generator side**: emit `atom.pos`/`atom.z` from the converter natively; finish the QTAIM rerun campaign and publish the audit flags so T3 on H7 can be reported with and without flagged records; `energy.lmdb` merge.
9. **Trainable iterative module** (Phase 2): design the interface now (shared encoder, edge module, feature module, iteration block), implement after items 4 and 6 report.

---

## 6. Key References (Consolidated)

**Scaling**:

- Gasteiger et al. "GemNet-OC" (2022) - efficient geometric message passing at scale; **reference implementation for dual-cutoff strategy (short-range equivariant MP at ~5Å + long-range scalar MP at ~12Å) enabling O(n) inference on large systems**
- Chen et al. "IDGL" (NeurIPS 2020) - iterative graph structure learning; anchor-based scaling reviewed but dual-cutoff preferred for chemistry
- Liao & Smidt. "EquiformerV2" (ICLR 2024) - SOTA equivariant transformer
- Batatia et al. "MACE" (NeurIPS 2022) - efficient higher-order equivariant MP
- Open Catalyst Project codebase - reference for training at 100M+ scale
**Geometry-Aware Architectures**:

- Schütt et al. "PaiNN" (ICML 2021) - recommended base encoder architecture
- Schütt et al. "SchNet" (2017) - continuous-filter convolutions, RBF distance expansion
- Gasteiger et al. "DimeNet++" (2020) - directional message passing
- Satorras et al. "EGNN" (ICML 2021) - simple equivariant GNN
**Link Prediction / Structure Learning**:

- Fatemi et al. "SLAPS" (NeurIPS 2021) - joint structure + prediction learning
- Wu et al. "NodeFormer" (NeurIPS 2022) - efficient all-pair attention
**Diffusion / Flow on Graphs**:

- Vignac et al. "DiGress" (ICLR 2023) - discrete graph diffusion
- Jo et al. "GDSS" (ICML 2022) - score-based graph generation
- Lipman et al. "Flow Matching" (ICLR 2023) - simpler alternative to diffusion
**Iterative Refinement**:

- Jumper et al. "AlphaFold2" (Nature 2021) - recycling mechanism
- Bai et al. "Deep Equilibrium Models" (NeurIPS 2019) - implicit fixed-point layers
**Heterogeneous / Multi-Task**:

- Hu et al. "HGT" (2020) - heterogeneous graph transformer
- Rampášek et al. "GPS" (NeurIPS 2022) - general graph transformer
- Wen et al. "BondNet" (2021) - hetero graphs for bond property prediction
**Radial Basis Functions / Distance Representations**:

- Klicpera et al. (2020) - sinusoidal RBFs (orthogonal on [0, r_cut])
- Schütt et al. (2017) - Gaussian RBFs (SchNet original)
- Gasteiger et al. "GemNet" (2021) - discusses radial basis design choices
- Gasteiger et al. "DimeNet" (2020) - polynomial envelope function (p=6 default); reference implementation for smooth cutoffs
---

**Data and benchmarks (NEW in v4)**:

- OMol-Descriptors-4M dataset card, `santi921/OMol-Descriptors-4M` (Hugging Face Hub); NeurIPS 2026 Datasets and Benchmarks submission 3446 sources under `qtaim_generator/docs/neurips/`
- OMol25 (Meta FAIR, 2025) - source geometries and wavefunctions at wB97M-V / def2-TZVPD
- Multiwfn 3.8 (Lu and Chen) and Critic2 (Otero-de-la-Roza et al.) - QTAIM engines; HORTON (Verstraelen et al.) - charge cross-validation

---

## 7. Open Scientific Questions (Where I'm Less Sure)

I want to flag a few things where my confidence is lower and the science is genuinely open:

- **Will the iterative approach actually converge?** Real SCF converges because the Fock operator has mathematical properties (variational principle, Aufbau). Your learned iteration operator has no such guarantee. DEQ-style fixed-point finding might help, but convergence is not assured. You may need to add explicit convergence regularization.
- **Non-nuclear attractors and ring/cage critical points**: Most QTAIM learning work focuses on BCPs between atoms. NNAs and RCPs are rare but chemically important. The link prediction framing handles BCPs naturally (edges between atoms) but RCPs/CCPs require predicting higher-order structures (faces, volumes of the graph). This might need a simplicial/cellular complex rather than a graph.
- **Transferability across levels of theory**: QTAIM descriptors depend on the level of DFT used. A model trained on B3LYP descriptors may not transfer to ωB97X-D. This is a data curation question more than an architecture question, but worth thinking about early.
- **Bond order ambiguity**: NBO and QTAIM can disagree on bonding, and "fuzzy bonds" give yet another picture. Training a single model to predict all of these simultaneously might be asking for trouble if the different definitions are inconsistent. Consider: shared encoder with separate prediction heads per bonding definition.
- **Is l=2 sufficient for lanthanide/actinide coordination? (NEW, high uncertainty)** Coordination geometries with CN 8-12 (square antiprismatic, tricapped trigonal prismatic, bicapped square antiprismatic, etc.) may differ only at l=3 or l=4 in the angular power spectrum. The l=2 scalar invariant approach (Lever 2) can detect some of these differences, but we don't have empirical evidence for how well it works on f-element chemistry specifically. The Phase 1 diagnostic subset (homoleptic lanthanide complexes with different coordination geometries) is designed to answer this question early. If l=2 is insufficient and full l_max=2 equivariance is needed, the scaling implications are significant - roughly 3-4x cost increase in the message passing hot loop. This is the highest-risk architectural uncertainty in the project.
- **Radial resolution vs angular resolution: which is the actual bottleneck? (NEW)** For organic molecules, the answer is likely radial (BCP properties vary rapidly with bond length, angular environments are stereotyped). For TM/lanthanide systems, the answer is likely angular (distances are similar across different coordination geometries, angular arrangement determines electronic structure). The Phase 1 experiments are designed to disentangle these two sources of error. If both matter for different subsets of the data, you may want property-head-specific radial bases (different RBF parameterization for organic BCP prediction vs TM BCP prediction), which is a form of Lever 3 (learnable RBFs) with chemical motivation.
- **Learnable RBFs: genuine improvement or overfitting risk? (NEW)** With 100M diverse training molecules, overfitting the radial basis is unlikely. But with the smaller Phase 1 datasets (1-10M), learnable bases might overfit to the training distribution's distance statistics. The hybrid fixed+learnable approach mitigates this. Ablate carefully on held-out chemistry (train on organics + TM, test on lanthanides) to test generalization of learned radial bases.
- **Candidate recall for non-covalent bond paths (NEW, v4)**. QTAIM bond paths exist for hydrogen bonds, agostic interactions, metal-metal contacts, and some van der Waals contacts, at distances well beyond 2x the sum of covalent radii in some cases. The candidate pool sets an upper bound on recall that no model can exceed. Measuring this per H-suite is step 1 of Section 3.5; if H1 or H6 recall is poor at 2x, the pool needs a second radius table (van der Waals) and the class imbalance gets worse. Confidence: moderate that 2x suffices for the main split, low for H1/H6.
- **One threshold or many? (NEW, v4)**. A single calibrated edge threshold across all chemistries is convenient for the ASE calculator, but calibration almost certainly differs between organics and lanthanide complexes. Report calibration error per H-suite; if it diverges, the predictor should expose per-element-pair or per-suite thresholds, or output probabilities and leave the decision to the caller (which is what `bond_probs` in the ASE calculator does).
- **Label noise from lost BCPs (NEW, v4)**. Until the QTAIM rerun campaign finishes, some training positives are missing, mostly in large molecules. A model that learns the truncation pattern would look worse than it is on H7 after repair. Track the audit flag through training and evaluation.
