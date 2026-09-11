---
title: "feat: Performance engineering for training, inference, and packaging (ASE)"
type: feat
date: 2026-09-08
status: track A executed 2026-09-09, see docs/research/2026-09-track-a-measurements.md for measurements and gate decisions (A2 dropped; A3 shipped as ResidualBlockDense + BucketBatchSampler, 1.3-1.5x end to end, 2.6x with A1; A4 collate shipped, serialization dropped; A5 neighbor build and channel-wise TP shipped; E1 accuracy gate open: training diverges at epoch 3 even with batch norm on)
---

# Performance Engineering Plan: Training, Inference, Packaging

## Summary

- Training is launch-bound, not compute-bound or data-bound. Throughput is flat at 25-28 it/s across QM8, QM9, and TMQM with under 0.5 GB of GPU memory in use. The fix is fewer, larger kernels per step (fused per-edge-type matmuls, CUDA graphs on bucketed shapes), not more workers or bigger GPUs.
- Inference has no product surface. The only entry points are `FullPredictor.predict_from_geometry` (O(N^2) Python loops, inference-only, three separate checkpoints plus scalers loaded by path) and `evaluate_manually`. There is no batched predictor, no model bundle format, no ASE calculator, no export path.
- Packaging is incomplete. `pyproject.toml` has its dependency list commented out, version `0.0.0`, and no extras. ASE 3.28 is already in the environment and is the natural integration target.

The plan is three tracks that can run in parallel after a shared measurement step. Each item lists the file it touches, the expected gain, and an acceptance test.

## Baseline (measured, do not re-derive)

| Source | Setting | Result |
|--------|---------|--------|
| profiling/config_c_analysis.md (2026-02) | GAT + MLP link model | 3.30 it/s, optimizer 31.5 % of CPU time before fused Adam |
| Memory, 2026-03-11 | ResidualBlock 8 layers, batch 128, 4 workers, one GPU | QM8 25.8 it/s (3290 samples/s), QM9 28.1 it/s, TMQM 25.6 it/s |
| Same | GPU peak memory | 0.15 GB (QM8/QM9), 0.49 GB (TMQM) |
| Same | CUDA dispatch ratio | 43 % non-QTAIM, 70 % TMQM (this is a dispatch ratio, not utilization; see todos/012) |
| Memory, small test set | torch.compile | 0 % gain, 25 s compile overhead |
| Memory | persistent_workers=True | SIGSEGV after ~34K samples; keep False |

Hardware available now: 2x RTX A5000 (24 GB each), torch 2.11.0+cu130, torch-geometric 2.7.0, triton 3.6.

**Every number above was measured with `encoder_fn: "none"`.** The 3D encoders landed 2026-08-03, after all benchmarks, and there is no throughput or memory measurement for schnet, dimenetpp, or equivariant on any dataset. Since the encoders are the training path going forward (roadmap v4 Phase 1 and the T3 bond classifier both run through them), the encoder path is the primary optimization target of this plan, not the hetero conv stack alone. Concretely: A0 must produce encoder baselines before anything else is tuned, and A5 moves from "later" to weeks 1-2. Expect the bottleneck to shift from launch overhead to (a) the chunked `cdist` neighbor build, which runs eagerly on every forward and cannot be compiled, (b) DimeNet++ triplet memory, sum(deg^2) per molecule, and (c) e3nn tensor products for the equivariant path, which are slow and not compile-friendly.

### Diagnosis

Per training step the model runs `ResidualBlock` x 8, each an `HeteroConv` over six edge types (`qtaim_embed/models/layers.py:147-173`). Each `GraphConv` is two linears, one gather, one scatter, plus batch norm and dropout per node type. That is on the order of 400-600 small kernels per forward and the same again backward, on tensors of a few thousand rows. At that size a kernel takes about as long to launch as to run. Doubling molecule size or feature width barely moves it/s, which is what the flat QM8/QM9/TMQM numbers show. Data loading is not the limiter: memory note says 4 workers give 1.8-2.3x on the DGL-era pipeline and no further gain after that.

Consequences for prioritization:

- Batch size is the cheapest lever and has not been pushed. At 0.15-0.49 GB peak, batch 128 could go to 1024-2048 on an A5000 before memory matters. Expect near-linear samples/s gains until kernels become compute-bound, then retune LR (linear scaling rule with warmup).
- Kernel count is the second lever. Fusing the six per-edge-type linears into one grouped matmul cuts the conv kernel count by roughly 6x.
- CUDA graphs remove launch overhead entirely but need static shapes. Bucketing batches by total atom count makes that feasible.
- More workers, faster LMDB reads, and mixed precision give little here. bf16 is still worth turning on as the default because it is free and helps once batches are large.

## Track A: Training throughput

### A0. Measurement harness (prerequisite, 1-2 days)

Files: `profiling/profile_pyg_graph.py`, `profiling/analyze_pyg_graph_results.py`, new `qtaim_embed/scripts/bench/bench_train.py`, new entry point `qtaim-embed-bench`.

- Replace `cuda_pct` with real GPU utilization sampled from `nvidia-smi --query-gpu=utilization.gpu` in a background thread, and with `torch.cuda.Event` busy time per step (todos/012).
- Steady-state over at least 4 epochs after 2 warmup epochs; report mean and std (todos/016).
- Emit one JSON per run with: samples/s, it/s, GPU util, peak memory, kernel count per step (from `torch.profiler` `key_averages`), dataloader wait fraction (time in `next(iter)` vs step).
- Fixed benchmark configs checked into `profiling/graph_configs/`: TMQM 48K node-level with QTAIM features, and one OMol-Descriptors-4M vertical (tm_react, 158K) once the shard directory is local. Both at batch 128 to reproduce the baseline, then swept.
- **Encoder baselines (first deliverable)**: the same two datasets at batch 128 for each of `none`, `schnet` (50 Gaussians, cutoff 5), `dimenetpp` (max_neighbors 32), `equivariant` (lmax 1) and `equivariant` (lmax 2). Report samples/s, peak memory, and the encoder's share of step time from `record_function` ranges around the encoder call in `encode_atom_inputs`. This table does not exist today and every later decision depends on it.
- Report neighbor-build time separately from encoder message passing (wrap `radius_neighbors` and `build_triplets` in `record_function`).
- Acceptance: running `qtaim-embed-bench --config profiling/graph_configs/tmqm_baseline.json` reproduces 25 +/- 2 it/s on one A5000 and writes the JSON.

### A1. Batch size, precision, matmul precision (1 day, expected 3-6x samples/s)

Files: `qtaim_embed/utils/data.py` defaults, training scripts, `docs/`.

- Sweep batch 128, 256, 512, 1024, 2048 on TMQM with `precision="bf16-mixed"`. Record samples/s and val MAE at fixed epoch budget. Pick the largest batch whose val MAE at 100 epochs is within noise of batch 128; retune LR with linear scaling and a 1-epoch warmup.
- Set `torch.set_float32_matmul_precision("high")` once at training-script import.
- Default `precision` to `"bf16-mixed"` in all four default configs. Do not use `16`: fp16 NaNs silently zero out `R2Score` (memory note).
- Acceptance: TMQM samples/s at chosen batch >= 3x the 3270 baseline with val MAE within 2 % of baseline.

### A2. Fused hetero convolution (3-5 days, expected 1.5-2.5x on top of A1)

Files: `qtaim_embed/models/layers.py` (`ResidualBlock`, `GraphConvDropoutBatch`), new `qtaim_embed/models/layers_fused.py`, tests in `tests/test_layers.py`.

- Replace the six independent `GraphConv` modules per layer with one `FusedTypedGraphConv` that holds stacked weights `W_rel[E, in, out]` and `W_root[T, in, out]`. Apply `W_rel` per edge type with `torch.bmm` on padded per-type segments or `pyg_lib.ops.segment_matmul` when `pyg_lib` is importable (it is optional; keep the bmm fallback). Aggregate with one `scatter_add` on a concatenated destination index with per-type offsets.
- Keep `HeteroConv` as the reference implementation behind `conv_fn: "ResidualBlock"`; register the fused one as `conv_fn: "ResidualBlockFused"`. Weight-copy parity test between the two at fp32, tolerance 1e-5.
- Batch norm per node type stays; dropout stays.
- Acceptance: parity test passes; kernel count per forward from A0 drops by >= 4x; samples/s improves >= 1.5x at batch 1024.

### A3. Shape bucketing and CUDA graphs (3-5 days, expected 1.3-2x on top of A2)

Files: new `qtaim_embed/data/bucketing.py`, `qtaim_embed/core/datamodule.py` (`LMDBDataModule`), model `compiled_forward` paths.

- Bucketed batch sampler: group graphs so that total atoms, total bonds, and total edges per batch fall in one of ~8 fixed buckets; pad node and edge tensors to the bucket ceiling with masked dummy nodes attached to a dummy graph. Padding waste target < 15 %.
- With static shapes, `torch.compile(mode="reduce-overhead")` captures CUDA graphs. The encoder neighbor build stays eager (it is data-dependent); compile only the post-encoder stack, which is where the kernels are. Lift the current "compiled + encoder" assertion to "compile the conv stack only" (`qtaim_embed/models/graph_level/base_gcn.py:150-152` and node equivalent).
- Fallback: if compile fails or on ROCm, bucketing alone still helps allocator reuse.
- Acceptance: GPU utilization from A0 >= 70 % at batch 1024 on TMQM; no accuracy change vs unbucketed at equal epochs.

### A4. Data pipeline hygiene (1-2 days, small gain, removes future cliffs)

Files: `qtaim_embed/data/dataloader.py`, `qtaim_embed/core/dataset.py`, `qtaim_embed/data/lmdb.py`.

- `Batch.from_data_list` on `HeteroData` with three node types and six edge types is Python-heavy. Measure its share in A0. If > 15 % of step time at batch 1024, write a direct collate that concatenates the known stores without PyG's generic `collate` introspection.
- `pin_memory=True` and `non_blocking=True` transfers in `transfer_batch_to_device`; currently pin_memory is False in graph and node defaults.
- Move link-model negative sampling (`get_negative_graph`, `dataloader.py:161-189`) from collate to GPU inside the training step.
- `torch.load(weights_only=False)` on every LMDB read (`lmdb.py:99`, todos/005): switch graph serialization to a plain tensor dict so `weights_only=True` works. Also drops pickle overhead per sample. Requires an LMDB version bump and a converter change in qtaim_generator; coordinate with the `add_pos_z_to_graphs.py` re-write pass since both rewrite every record.
- Acceptance: dataloader wait fraction < 5 % at batch 1024 with 4 workers.

### A5. 3D encoder cost (weeks 1-2, primary target; 4-6 days)

Files: `qtaim_embed/models/encoders/*.py`.

- Measure first (A0 encoder table). Expected order of cost: equivariant >> dimenetpp > schnet >> none.
- **Neighbor build**: `radius_neighbors` runs a chunked `cdist` over the whole batch on every forward, including backward-free eval. Options in order of cost: cache the radius graph per batch in the collate (positions do not change between forward calls, so build once on CPU workers and ship `edge_index`/`d_ij` with the batch, which also makes the post-encoder stack compile-able again); build once per batch and reuse across the encoder's interaction blocks (already the case, verify); cell-list builder for N > 500.
- **SchNet**: `InteractionBlock` is PyG reference code and compiles when `edge_index` is an input rather than built inside. With collate-time neighbors, wrap the interaction stack in `torch.compile(dynamic=True)` and measure.
- **DimeNet++**: triplet count is sum(deg^2). At cutoff 5 A in solvated systems degree is 30+, so 1000 triplets per atom. Sweep `encoder_cutoff` 4.0 vs 5.0 and `encoder_max_neighbors` 16 vs 32 against T2 accuracy before optimizing kernels; the cheapest fix is a smaller candidate set. Move `build_triplets` to collate as well.
- **Equivariant (e3nn)**: `FullyConnectedTensorProduct` with per-edge weights is the known hot spot. Measure lmax 1 vs 2. If lmax 2 is required by accuracy, evaluate `cuequivariance` (NVIDIA) or `openequivariance` as drop-in tensor-product backends before writing anything custom. If lmax 1 suffices, e3nn is tolerable and the neighbor caching above is the main gain.
- SchNet and DimeNet++ are fine for molecules under ~150 atoms. Profile them on the H7 large-system suite (n_atoms > 250) where chunked `cdist` is O(N^2) per molecule and DimeNet++ triplets scale with sum(deg^2).
- If the H7 suite is a training target, add a cell-list neighbor builder behind the same `radius_neighbors` signature, selected when N > 500.
- The `EquivariantEncoder` uses `e3nn` `FullyConnectedTensorProduct` with per-edge weights; e3nn is slow and not compile-friendly. Defer replacing it with `cuequivariance` or `openequivariance` until lmax >= 2 is empirically required (roadmap lever 4).
- Acceptance: encoder table exists for all five settings; schnet samples/s >= 0.6x of `none` at batch 256 on TMQM after neighbor caching; dimenetpp >= 0.3x; neighbor build < 10 % of step time.

### A6. Multi-GPU and scale-out (1-2 days once A1-A3 land)

Files: training scripts, `qtaim_embed/core/datamodule.py`.

- DDP on the two A5000s (hang fixed 2026-03-10). Sharded LMDB directories map naturally to `DistributedSampler`; verify each rank opens its own LMDB envs after fork.
- Gradient accumulation is not needed once batch 1024+ fits; remove it from the default configs to avoid confusion.
- For OMol-Descriptors-4M full training on a cluster: one epoch is ~3.1M graphs. At 10K samples/s (A1-A3 target on one GPU) that is 5 min per epoch per GPU; a 100-epoch run on 4 GPUs is about 2 hours. Write this budget into the run configs rather than guessing.
- Acceptance: 2-GPU DDP samples/s >= 1.8x single GPU at batch 1024.

## Track B: Inference

### B1. Vectorize and shrink candidate edges (1 day)

Files: `qtaim_embed/models/full_predictor/full.py:262-316`, `qtaim_embed/models/encoders/neighbors.py:57-79`.

- Implement todos/009: `torch.meshgrid` plus a score matrix and `triu_indices`. Identical output, test against the loop version on a 20-atom molecule.
- Replace all-pairs candidates with `candidate_pairs` (covalent radius pooling, `pool_multiplier=2.0`) which already exists in `neighbors.py` and is tested but unused. This is the same 2x covalent cutoff the NeurIPS T3 baseline uses, so candidate recall is known (see roadmap v4, bond prediction section).
- Acceptance: candidate generation < 1 ms for N=200 on GPU; predicted edge set unchanged on the test molecule.

### B2. Batched predictor API (2-3 days)

Files: new `qtaim_embed/inference/predictor.py`, `qtaim_embed/inference/__init__.py`.

```python
class QTAIMPredictor:
    @classmethod
    def from_bundle(cls, path: str, device: str = "cuda") -> "QTAIMPredictor": ...
    def predict(self, structures: list[tuple[np.ndarray, np.ndarray]] | list[ase.Atoms],
                batch_size: int = 64) -> list[PredictedGraph]: ...
```

- Runs under `torch.inference_mode()` and bf16 autocast on CUDA.
- Sorts inputs by atom count and batches contiguous groups so padding waste stays low; reuses A3 bucketing if it lands first.
- Returns a plain dataclass per structure: `edges (M,2) int64`, `edge_prob (M,)`, `atom_props {name: (N,)}`, `bond_props {name: (M,)}`, `global_props {name: float}`, with descaled units. No `HeteroData` in the public return type.
- Wraps `FullPredictor` today; the same API fronts the trainable iterative model later, so callers do not change.
- Acceptance: 1000 TMQM molecules predicted in < 10 s on one A5000; results match `FullPredictor.predict_from_geometry` per molecule.

### B3. Model bundle format (1-2 days)

Files: new `qtaim_embed/inference/bundle.py`, `qtaim_embed/models/utils.py` (save side), training scripts (write bundle at end of fit).

A bundle is a directory:

```
bundle/
  manifest.json      # schema_version, model class, target names and units, element_set,
                     # grapher config, encoder_fn, cutoff, git sha, train dataset id
  model.ckpt         # Lightning checkpoint (weights only)
  feature_scaler.pt  # torch.save with weights_only-compatible tensors
  label_scaler.pt
```

- Today the scaler, the grapher config, and the checkpoint live in separate paths that must be kept in sync by hand (`FullPredictor` config takes `link_model_path`, `node_model_path`, scaler paths). The bundle removes that failure mode.
- Loading uses `torch.load(weights_only=True)` throughout (todos/005 for the scaler side is already done; checkpoints need the same).
- A `FullPredictor` bundle is a bundle containing sub-bundles `link/` and `node/` plus loop settings in its manifest.
- Acceptance: `QTAIMPredictor.from_bundle` round-trips a trained node model and a trained link model; `manifest.json` validated against a JSON schema in tests.

### B4. Export (2-3 days, after B2 and A2)

- TorchScript is not viable: `HeteroConv` dict-of-dict modules and PyG's typed dispatch do not script cleanly, and PyG has deprecated scripting paths. Do not pursue.
- `torch.export` + AOTInductor on the fused conv stack from A2 with bucketed shapes from A3 is viable and produces a `.so` loadable without Python module imports. Target use: CPU inference nodes and the ASE calculator hot loop.
- ONNX: low priority. Only if an external consumer asks.
- Acceptance: exported node model matches eager to 1e-4 on 100 molecules; CPU latency for one 50-atom molecule < 20 ms.

### B5. CPU inference path (1 day)

- Verify `QTAIMPredictor(device="cpu")` works with `torch.set_num_threads`, fp32, no autocast. Many downstream users (ASE workflows on login nodes) will not have a GPU.
- Acceptance: TMQM 50-atom molecule < 100 ms on 8 CPU threads eager; < 20 ms exported.

## Track C: Packaging and ASE

### C1. pyproject.toml (0.5 day)

- Uncomment and complete `dependencies`: `torch>=2.4`, `torch-geometric>=2.6`, `lightning>=2.4`, `numpy`, `scipy`, `scikit-learn`, `rdkit`, `pymatgen`, `lmdb<2`, `torchmetrics`, `tqdm`, `pandas`.
- Extras: `encoders = ["e3nn>=0.5"]`, `ase = ["ase>=3.22"]`, `wandb = ["wandb"]`, `dev = [...]` (already present).
- Version from git tags via `setuptools_scm`, or at minimum bump to `0.1.0` and record in `manifest.json` of every bundle.
- Move `qtaim_embed/scripts/notebooks/old/` out of the package tree (it is being installed today).
- Acceptance: `pip install .` in a clean venv imports `qtaim_embed` and runs `pytest tests/test_layers.py`.

### C2. ASE calculator (2-3 days, after B2 and B3)

Files: new `qtaim_embed/ase/calculator.py`.

```python
from ase.calculators.calculator import Calculator, all_changes

class QTAIMCalculator(Calculator):
    implemented_properties = ["qtaim_atoms", "qtaim_bonds", "bond_pairs", "bond_probs"]

    def __init__(self, bundle: str, device: str = "cuda", **kwargs): ...

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        out = self._predictor.predict([atoms])[0]
        self.results["bond_pairs"] = out.edges
        self.results["bond_probs"] = out.edge_prob
        self.results["qtaim_atoms"] = out.atom_props   # dict name -> (N,)
        self.results["qtaim_bonds"] = out.bond_props   # dict name -> (M,)
```

- ASE permits arbitrary keys in `results`; `atoms.calc.get_property("qtaim_atoms")` works. Energy and forces are not implemented until a trained energy head exists (below), and `get_potential_energy()` raises `PropertyNotImplementedError` cleanly, which is the correct behaviour.
- Attach also to `atoms.arrays["qtaim_<name>"]` for per-atom scalars so they travel with `ase.io.write` to extxyz.
- Periodic systems: the encoder neighbor builder ignores `cell` and `pbc`. Raise `NotImplementedError` if `any(atoms.pbc)` until a minimum-image neighbor build exists (this is the MOF target in roadmap Phase 3).
- Acceptance: `ase.build.molecule("H2O")` with the calculator attached yields 2 bond pairs with prob > 0.9 and per-atom charge predictions; extxyz round trip preserves the arrays.

### C3. Energy and forces (optional, gated on qtaim_generator todo #9)

- qtaim_generator's master todo #9 merges OMol25 total energies and per-atom forces into `energy.lmdb` under the same keys as the descriptor LMDBs. Once that exists, a graph-level energy head with force by autograd on `atom.pos` makes `QTAIMCalculator` a real ASE calculator with `energy` and `forces`, and enables the fairchem MLIP benchmark (todo #8).
- Not a performance item, but it is the reason to keep `atom.pos` with `requires_grad` reachable from the graph-level readout. Do not detach positions in the encoder path.

### C4. Distribution (1 day)

- Publish bundles to the Hugging Face Hub alongside the `santi921/OMol-Descriptors-4M` dataset card; `QTAIMPredictor.from_pretrained("santi921/<bundle>")` via `huggingface_hub.snapshot_download`. Keep it in the `ase`/`hf` extras so core install stays light.
- One shared conda env with qtaim_generator is already the convention (`env.yml` aligned 2026-05-26). Keep `env.yml` as the reproducible research env and `pyproject.toml` as the user install.

### C5. CI (1 day)

- CPU-only GitHub Actions matrix: py3.11, torch CPU wheel, `pytest -m "not gpu"`. Mark the encoder parity, equivariance, and DDP tests `gpu`.
- Nightly self-hosted GPU job runs `qtaim-embed-bench` on the TMQM baseline config and posts samples/s so regressions in A1-A3 are caught.

## Sequencing

| Week | Track A | Track B | Track C |
|------|---------|---------|---------|
| 1 | A0 harness incl. encoder table, A1 batch/bf16 sweep | B1 vectorize candidates | C1 pyproject |
| 2 | A5 encoder cost (neighbor caching, cutoff sweep) | B2 predictor API, B3 bundle | |
| 3 | A2 fused typed conv | B5 CPU path | C2 ASE calculator |
| 4 | A3 bucketing + CUDA graphs, A4, A6 DDP | B4 torch.export | C4 HF distribution, C5 CI |
| later | equivariant tensor-product backend if lmax 2 is needed | | C3 energy/forces (after generator todo #9) |

Targets at the end of week 4, TMQM node-level, one A5000 (add a second row per metric for `encoder_fn: schnet` once the A0 table exists):

| Metric | Baseline | Target |
|--------|----------|--------|
| samples/s (train) | 3270 | >= 15000 |
| GPU utilization (sampled) | unknown (dispatch ratio 70 %) | >= 70 % |
| 1000-molecule inference | not measured | < 10 s GPU, < 60 s 8-thread CPU |
| Install | editable only, deps commented out | `pip install qtaim_embed[ase]` works in a clean venv |

## Risks

- Bucketing with padding changes batch-norm statistics if dummy nodes are not masked out. Use masked batch norm or exclude padded rows explicitly, and test that per-type running stats match unpadded training.
- `torch.compile(mode="reduce-overhead")` with PyG scatter ops has had recompilation storms in the past; the flat-shape bucketing is the mitigation, and the eager fallback must stay one config flag away.
- Fused typed conv changes parameter layout, so old checkpoints will not load into `ResidualBlockFused`. Provide a one-time weight converter and keep `ResidualBlock` loadable.
- Switching LMDB serialization to a tensor dict (A4) touches every dataset on disk and the generator converter. Version the record format and keep the pickle reader behind `weights_only=False` for one release with a warning.
- The ASE calculator will be used on periodic cells sooner than expected. Fail loudly on `pbc` rather than silently ignoring the cell.

## References in repo

- `docs/research/performance-optimization-index.md` (2026-02 optimizer work; fused Adam done)
- `docs/solutions/performance-issues/torch-split-shared-storage-lmdb-bloat.md`
- `docs/todos/005`, `009`, `012`, `016`
- `profiling/config_c_analysis.md`, `profiling/pipeline_analysis.md`
- `docs/roadmap_v4.md` section 3.5 for the bond-prediction validation that inference Track B serves
- `docs/plans/2026-09-08-feat-t3-bond-classifier-plan.md` (the T3 model that trains through the encoder path this plan optimizes)
