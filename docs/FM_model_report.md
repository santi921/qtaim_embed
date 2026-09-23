# Architectural Review & Roadmap: Foundational Model for Quantum Chemical Descriptors

> **Status (2026-09-08): this is the v1 snapshot (2026-02), kept for history. It is superseded by `docs/roadmap_v4.md`, which carries the current state of both repos, the gap analysis, and the bond-prediction validation program. `docs/roadmap_v3.md` is the intermediate revision.**
>
> What changed between this document and v4, in one screen:
>
> - The PyG migration it recommends is complete (2026-03). Environment is torch 2.11 / CUDA 13 / PyG 2.7.
> - "Geometry-aware encoder" is done as `encoder_fn` in {schnet, dimenetpp, equivariant}, not PaiNN. Graphs carry `atom.pos` and `atom.z`.
> - The training corpus is OMol-Descriptors-4M (3.99M structures, 34 verticals, composition split, five stress suites), built by qtaim_generator and submitted to NeurIPS 2026 Datasets and Benchmarks.
> - The O(N^2) candidate-edge loops and the inference-only `FullPredictor` described in Section 1 are still there (todos/009). The link model still collapses to a homogeneous graph and has no geometry input.
> - The top gap is now a validated bond (topology) predictor measured against the distance-rule baseline (macro-F1 0.971 at 2x covalent radii). See roadmap v4 Sections 1.5 and 3.5.
> - Performance work (training throughput, inference API, ASE packaging) is planned in `docs/plans/2026-09-08-feat-performance-engineering-plan.md`.

## 1. What You Have Now

After reading both repos, here's my understanding of the current state:

**qtaim_generator**: A mature pipeline for computing QTAIM/NBO/Multiwfn descriptors at scale from QM calculations. Supports sharded LMDB output, handles global/atom/bond-level features, and has robust parsing for ORCA, Multiwfn, and JSON formats. This is the data factory.

**qtaim_embed**: A heterogeneous GNN framework (currently DGL-based) with three node types (atom, bond, global) and three task heads: graph-level regression/classification, node-level prediction, and link prediction. The `feature/iterative-link-node-predictor` branch has the skeleton of the "neural SCF" idea — a `FullPredictor` that alternates between link prediction (on a homogenized graph) and node prediction (on the heterograph), iterating a fixed number of steps.

Key observations from the code:

- The link model operates on homogeneous graphs (via `hetero_to_homo` transform), while the node model operates on heterographs. This asymmetry means you're losing bond-type information during link prediction.
- The iterative loop in `FullPredictor` is currently inference-only (pretrained link + node models composed post-hoc). The two models aren't trained jointly, so there's no gradient signal flowing from node prediction quality back to link prediction quality.
- Candidate edge generation in `_get_bidirectional_candidate_edges` is O(N²) with a Python loop — this will be a hard bottleneck for anything beyond small molecules.
- The current GCN/GAT/GraphSAGE layers are isotropic message passing — they don't exploit 3D geometry (distances, angles) which are your primary input.

---

## 2. Literature Map

### 2.1 Scaling GNNs to Hundreds of Millions of Molecules

The scaling problem has been solved in practice by a few groups. The key references:

**OC20/OC22 (Meta FAIR)**: Trained on ~130M DFT relaxation frames. Their scaling playbook: LMDB data storage (you already have this), distributed data-parallel training, and architectures that balance expressiveness with throughput. The progression was GemNet-T → GemNet-OC → eSCN → EquiformerV2. EquiformerV2 (Liao & Smidt, ICLR 2024) is currently SOTA on OC20 — uses equivariant Transformers with SO(3) convolutions, but is expensive. **GemNet-OC** (Gasteiger et al., 2022) is probably the better cost-performance tradeoff and more relevant to your case since you care about throughput at inference.

**MACE** (Batatia et al., NeurIPS 2022; Batatia et al., 2024 for MACE-MP-0): Higher-order equivariant message passing using ACE (Atomic Cluster Expansion) body-ordered descriptors. MACE-MP-0 was trained on ~150k materials but shows extraordinary transfer. The architecture is relevant because it's very efficient (few message passing steps needed) and has clean multi-body interaction terms. However, I should be upfront: MACE is designed for energy/force prediction on known topologies — adapting it to predict topology is non-trivial.

**Practical scaling lessons from these efforts**:
- LMDB with memory-mapped reads (you have this)
- bf16/mixed precision training
- Graph batching with dynamic padding rather than fixed batch sizes
- Distributed training with DDP — PyG's `DistributedDataParallel` or DeepSpeed
- For 100M+ molecules: epoch-level sharding where each worker sees a different data shard per epoch

**For your PyG migration specifically**: Look at PyG's `LargeGraphDataset` and their LMDB integration patterns from OCP (Open Catalyst Project). The OCP codebase (github.com/Open-Catalyst-Project/ocp) is the reference implementation for training molecular GNNs at this scale.

### 2.2 Geometry-Aware Architectures (Critical for Your Use Case)

Since your input is geometry and your output includes QM topology, you need architectures that properly encode 3D structure. The current GCN/GAT layers don't do this.

**SchNet** (Schütt et al., 2017): Continuous-filter convolutions on interatomic distances. Simple but effective baseline. The key idea — using radial basis function expansions of distances as edge features — should be in your architecture regardless of what else you choose.

**DimeNet / DimeNet++** (Gasteiger et al., 2020): Adds directional information via angles between triplets. DimeNet++ is 10x faster. Relevant because bond angles carry information about the QM topology you're trying to predict.

**PaiNN** (Schütt et al., 2021): Equivariant message passing with scalar and vector features. Good balance of expressiveness and speed. This is probably the sweet spot for your use case — fast enough for 100M-scale training, expressive enough to capture the geometric information that determines QTAIM topology.

**TorchMD-Net** (Thölke & De Fabritiis, 2022): Equivariant Transformer that combines the benefits of attention mechanisms with geometric priors. Has a clean PyTorch implementation and good scaling properties.

**EGNN** (Satorras et al., 2021): E(n) equivariant graph neural networks. Simpler than the above, uses coordinate updates during message passing. Worth considering as a baseline because it's very fast.

### 2.3 Link Prediction / Graph Structure Learning

This is the core novelty of your approach — predicting the QM topological graph. The relevant literature spans several subfields:

**Graph Structure Learning (GSL)**:
- **IDGL** (Chen et al., NeurIPS 2020): Iterative Deep Graph Learning — learns graph structure and node representations jointly. Very close to your neural SCF idea. It maintains a differentiable adjacency matrix that's refined alongside node embeddings. Key paper for your architecture.
- **SLAPS** (Fatemi et al., NeurIPS 2021): Simultaneously Learning graph structure And Predictions. Uses a generative model for graph structure that's trained end-to-end with the downstream task.
- **NodeFormer** (Wu et al., NeurIPS 2022): All-pair message passing with kernelized softmax — efficiently computes attention over all node pairs without explicit edge construction. Could replace your O(N²) candidate edge enumeration.

**Chemical bond prediction specifically**:
- **BondNet** (Wen et al., 2021): Predicts bond dissociation energies using a heterogeneous graph similar to yours (atom + bond + global nodes). Not link prediction per se, but the architecture is very relevant.
- The bond topology prediction problem also appears in coarse-grained molecular dynamics, where people predict which CG beads should be bonded. **CGSchNet** and similar approaches use distance-based edge prediction with learned cutoffs.

**For QTAIM specifically**: The number of bond critical points is relatively small and their existence is strongly correlated with internuclear distance (most BCPs are between atoms within ~1.5-3Å). This means a distance-initialized graph with learned refinement is a reasonable prior — your current approach of distance cutoff → iterative refinement is scientifically sound.

### 2.4 Joint Prediction (Links + Features Simultaneously)

This is where it gets interesting and where I want to be honest about uncertainty in the science.

**Graph Generation / Diffusion Models**:
- **DiGress** (Vignac et al., ICLR 2023): Discrete denoising diffusion for graph generation. Generates node types and edge types jointly by denoising a corrupted graph. Very relevant — you could frame your problem as denoising a distance-cutoff graph into the QTAIM graph, with continuous node/edge features generated alongside.
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
- **HGT** (Hu et al., 2020): Heterogeneous Graph Transformer — uses type-specific attention. Direct replacement for your current hetero message passing layers.
- **HEAT** (Chen et al., 2023): Heterogeneous Edge Attribute Transformer. Handles edge features natively.

---

## 3. Architectural Recommendations

### 3.1 The "Neural SCF" Approach — Making It Work

Your intuition is good. Here's how to make the iterative approach trainable and scalable:

**Architecture: Iterative Equivariant Refinement Network (IERN)**

```
Input: Atomic coordinates + elements
  │
  ▼
[Geometry Encoder] ── RBF distance expansion + element embeddings
  │
  ▼
[Initial Graph Constructor] ── Distance cutoff (generous, e.g., 4Å)
  │                              Creates dense initial graph
  ▼
┌─────────────── Iteration Block (shared weights, N iterations) ──────────────┐
│                                                                              │
│  [Edge Score Module] ── Predicts edge existence probability                 │
│       │                  Uses node embeddings + distance + angle features    │
│       │                  Output: continuous edge weights (not hard threshold)│
│       ▼                                                                      │
│  [Weighted Message Passing] ── PaiNN-style equivariant MP                   │
│       │                        Edge weights gate messages                    │
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
```

**Critical design decisions**:

1. **Soft edges during training, hard edges at inference**: Don't use hard edge thresholds during training — this kills gradients. Instead, use continuous edge weights (sigmoid of edge scores) to gate message passing. At inference, threshold to get the final topology. This is analogous to Gumbel-Softmax for discrete structure learning.

2. **Shared weights across iterations**: Like AlphaFold2 recycling. This dramatically reduces parameters and acts as a strong regularizer. The model learns a "refinement operator" rather than "step 1, step 2, step 3."

3. **Distance-aware edge scoring**: The edge score module should take as input: (a) concatenated node embeddings, (b) interatomic distance (via RBF expansion), and (c) optionally angular features from triplets. QTAIM bond paths are strongly distance-correlated, so this prior helps enormously.

4. **Don't convert to homogeneous graphs**: Your current link model collapses the heterograph to a homograph, losing the atom/bond distinction. Instead, score edges in the full heterogeneous representation. An edge (atom_i, atom_j) produces a bond node between them — this is the natural heterogeneous formulation.

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

**Pros**: Handles mixed discrete/continuous outputs naturally. Can generate diverse samples (useful for uncertainty quantification). Doesn't require iterative refinement at inference — can do it in one shot with enough denoising steps.

**Cons**: Slower inference than the iterative approach (many denoising steps). Harder to train. Less physically interpretable than the SCF analogy. I'm not confident this is scientifically better for your problem — the SCF analogy gives you a strong inductive bias that diffusion doesn't naturally capture.

**My honest assessment**: For a foundational model, I'd start with the iterative approach (3.1) because the physics are cleaner and it's easier to debug and interpret. The diffusion approach is worth exploring as a comparison, but I wouldn't make it the primary architecture. The iterative approach has a natural connection to the actual SCF procedure, which gives you interpretability and a convergence criterion that diffusion lacks.

### 3.3 Scaling Considerations

For 100M+ molecules:

**Data pipeline**:
- Your LMDB infrastructure in qtaim_generator is already good
- Add: on-the-fly graph construction from stored coordinates + descriptors (don't pre-build graphs, they're too large to store for 100M molecules)
- Use PyG's `InMemoryDataset` with lazy loading, or better, their `LargeGraphDataset` pattern

**Model efficiency**:
- Use **PyG** (not DGL) — better ecosystem for distributed training, better integration with PyTorch's DDP and FSDP
- For the message passing layers: PaiNN or EGNN give you equivariance without the cost of higher-order spherical harmonics (EquiformerV2 is overkill for descriptor prediction)
- **Quantization**: Post-training INT8 quantization for inference. Descriptors have finite precision from DFT anyway
- **Graph coarsening**: For very large molecules (100+ atoms), consider hierarchical message passing — local interactions first, then global

**Distributed training**:
- DDP across GPUs (straightforward with PyTorch Lightning, which you already use)
- For truly massive scale: consider DeepSpeed ZeRO Stage 2
- Gradient accumulation to simulate large batch sizes

**Inference at scale**:
- Batch inference with dynamic batching (group molecules by size)
- TorchScript or torch.compile for the inference path
- Consider ONNX export for deployment

---

## 4. Recommended Phased Roadmap

### Phase 1: Foundation (1-2 months)
- Complete the PyG migration of qtaim_embed
- Implement a geometry-aware encoder (SchNet-style RBF + element embeddings as the base, then upgrade to PaiNN)
- Set up distributed training infrastructure with the LMDB pipeline
- Train baseline models: separate link prediction + node prediction on a moderate dataset (1-10M molecules)

### Phase 2: Iterative Architecture (2-3 months)
- Implement the IERN architecture from 3.1 with soft edges and shared weights
- Key experiment: compare joint end-to-end training vs. pretrained-then-composed (your current approach)
- Implement curriculum learning (easy molecules → hard molecules)
- Benchmark against separate link + node prediction baselines
- Scale to 10-50M molecules

### Phase 3: Scale & Compare (2-3 months)
- Scale to full 100M+ dataset
- Implement and benchmark the diffusion alternative
- Ablation studies: number of iterations, edge scoring mechanisms, equivariant vs. invariant layers
- Uncertainty quantification (ensemble or MC dropout for the iterative model, native sampling for diffusion)

### Phase 4: Foundation Model (ongoing)
- Pre-train on massive diverse dataset
- Fine-tuning protocols for specific chemistry (TM complexes, reaction paths, etc.)
- Transfer learning experiments
- Public release

---

## 5. Concrete Next Steps for the Codebase

1. **PyG migration priority**: Focus on `layers.py` and the message passing infrastructure first. PyG's `MessagePassing` base class with `edge_attr` support replaces your custom hetero message passing. Use `HeteroData` for the atom/bond/global node types.

2. **Replace the O(N²) candidate edge enumeration**: Use `torch_cluster.radius_graph` for distance-based edge construction — it's GPU-accelerated and handles batched graphs.

3. **Make the iterative loop differentiable**: The current `FullPredictor` uses `@torch.no_grad()` and hardcodes pretrained models. Rewrite it as a single `nn.Module` with shared-weight iteration blocks that can be trained end-to-end.

4. **Add distance/angle features to edge representations**: Even before the full PaiNN migration, adding RBF-expanded distances to your edge features in the current GCN will give significant improvements.

5. **Benchmark data**: Create a standardized benchmark split from your generator output — fixed train/val/test with both "easy" (organic) and "hard" (TM, charged, high-spin) subsets.

---

## 6. Key References (Consolidated)

**Scaling**:
- Gasteiger et al. "GemNet-OC" (2022) — efficient geometric message passing at scale
- Liao & Smidt. "EquiformerV2" (ICLR 2024) — SOTA equivariant transformer
- Batatia et al. "MACE" (NeurIPS 2022) — efficient higher-order equivariant MP
- Open Catalyst Project codebase — reference for training at 100M+ scale

**Geometry-Aware Architectures**:
- Schütt et al. "PaiNN" (ICML 2021) — recommended base architecture
- Schütt et al. "SchNet" (2017) — continuous-filter convolutions
- Gasteiger et al. "DimeNet++" (2020) — directional message passing
- Satorras et al. "EGNN" (ICML 2021) — simple equivariant GNN

**Link Prediction / Structure Learning**:
- Chen et al. "IDGL" (NeurIPS 2020) — iterative graph structure learning (most relevant)
- Fatemi et al. "SLAPS" (NeurIPS 2021) — joint structure + prediction learning
- Wu et al. "NodeFormer" (NeurIPS 2022) — efficient all-pair attention

**Diffusion / Flow on Graphs**:
- Vignac et al. "DiGress" (ICLR 2023) — discrete graph diffusion
- Jo et al. "GDSS" (ICML 2022) — score-based graph generation
- Lipman et al. "Flow Matching" (ICLR 2023) — simpler alternative to diffusion

**Iterative Refinement**:
- Jumper et al. "AlphaFold2" (Nature 2021) — recycling mechanism
- Bai et al. "Deep Equilibrium Models" (NeurIPS 2019) — implicit fixed-point layers

**Heterogeneous / Multi-Task**:
- Hu et al. "HGT" (2020) — heterogeneous graph transformer
- Rampášek et al. "GPS" (NeurIPS 2022) — general graph transformer
- Wen et al. "BondNet" (2021) — hetero graphs for bond property prediction

---

## 7. Open Scientific Questions (Where I'm Less Sure)

I want to flag a few things where my confidence is lower and the science is genuinely open:

- **Will the iterative approach actually converge?** Real SCF converges because the Fock operator has mathematical properties (variational principle, Aufbau). Your learned iteration operator has no such guarantee. DEQ-style fixed-point finding might help, but convergence is not assured. You may need to add explicit convergence regularization.

- **Non-nuclear attractors and ring/cage critical points**: Most QTAIM learning work focuses on BCPs between atoms. NNAs and RCPs are rare but chemically important. The link prediction framing handles BCPs naturally (edges between atoms) but RCPs/CCPs require predicting higher-order structures (faces, volumes of the graph). This might need a simplicial/cellular complex rather than a graph.

- **Transferability across levels of theory**: QTAIM descriptors depend on the level of DFT used. A model trained on B3LYP descriptors may not transfer to ωB97X-D. This is a data curation question more than an architecture question, but worth thinking about early.

- **Bond order ambiguity**: NBO and QTAIM can disagree on bonding, and "fuzzy bonds" give yet another picture. Training a single model to predict all of these simultaneously might be asking for trouble if the different definitions are inconsistent. Consider: shared encoder with separate prediction heads per bonding definition.
