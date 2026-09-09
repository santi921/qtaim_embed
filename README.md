# QTAIM-Embed
<img src="https://github.com/santi921/qtaim_embed/blob/main/data/plots/TOC.png" width=80% height=80%>
A GNN package for molecular properties. These models can handle spin and charged species as well as complex atom, bond features. 
QTAIM features are compatible via a sister package <a href="https://github.com/santi921/qtaim_generator">Generator</a> and can add performance and robustness to existing models 
Currently only structure-to-property models are supported but we are working on structure-to-node level models. 
<br/>

The current implementation supports a good starting set of message-passing functions:
- GCN
- GAT
- GCN+Residual
  
In addition, several global readout functions are implemented for size-intensive and extensive properties including:
- Mean
- Sum
- WeightedMean
- WeightedSum
- GlobalAttentionPooling
- Set2Set  


TODO: some instructional notebooks

## Training throughput on large molecules (2026-09)

Measured on 60-350 atom systems (details and every number: `docs/research/2026-09-track-a-measurements.md`).
The switches below are plain config keys and combine freely; all are optional and default to the previous behaviour unless noted.

| setting | what it does | when to use |
|---|---|---|
| `model.conv_fn: "ResidualBlockDense"` | same math as `ResidualBlock` on a padded per-molecule layout (atom-bond messages become one `bmm`, no gather/scatter); pair with `model.compiled: true` to run the conv stack as CUDA graphs | 40+ atom molecules, bf16-mixed, hidden 128-512 |
| `dataset.bucketing: true` (LMDB loaders) | batches graphs whose atom and bond counts round up to the same multiple of `bucket_grid` (default 16), so padded shapes are static; 11-17 % padding on tm_react | always with `ResidualBlockDense` |
| `model.encoder_tp: "channelwise"` (default) | channel-wise tensor product in the equivariant encoder, 64x fewer per-edge weights than the old fully connected one, which OOMed at batch 128 | any `encoder_fn: "equivariant"` run |
| `optim.precision: "bf16-mixed"`, `optim.num_workers: 8`, `optim.pin_memory: true` (defaults now) | halves activation memory; 8 workers are needed above batch 512 or the GPU waits on data | any LMDB training |
| `optim.warmup_epochs: 1` | linear LR warmup, then the usual ReduceLROnPlateau | large batches with a scaled learning rate |
| `qtaim-embed-bench --config profiling/bench_configs/<name>.json` | throughput harness: samples/s, GPU utilization, kernels and syncs per step, data-wait fraction | before and after any performance change |

Measured on one RTX A5000 (bf16-mixed, batch norm on, raw training step):

| dataset, model | before | after | change |
|---|---|---|---|
| tm_react hidden 128, batch 128 fp32 4 workers to batch 1024 dense + CUDA graphs 8 workers | 3,006 samples/s | 7,755 samples/s | 2.6x |
| tm_react hidden 256, batch 1024, ResidualBlock to ResidualBlockDense compiled | 3,805 | 5,310 | 1.4x |
| H7 (250-350 atoms) hidden 256, batch 64, same switch | 769 | 979 | 1.3x |
| tm_react schnet encoder, batch 512 (neighbor build) | 2,652 | 4,182 | 1.6x |
| tm_react equivariant encoder lmax 1 hidden 64, batch 128 (channel-wise TP) | OOM at 23 GB | 1,261 samples/s at 2.3 GB | trains |

Also changed under the hood, no config needed: the radius-graph build for 3D encoders is per molecule (65-180x faster, exact), the LMDB collate skips PyG's generic `Batch.from_data_list` (4x), and `encoder_max_neighbors` defaults to 16.

`model.batch_norm` must stay `True` for the hetero conv stack: without it activations grow 60-100x per block on 40+ atom molecules and training collapses to the label mean. Open issue (2026-09-09): with batch norm on, a tm_react run at batch 128 still diverges in eval mode from epoch 2. Cause and fix (`docs/research/2026-09-tm-react-eval-divergence.md`): batch norm runs after ReLU, so channels that are inactive for a whole batch collapse their running variance and amplify rare inputs 100x at eval time. Setting `model.bn_before_activation: true` (normalize before the activation) trains 40 epochs cleanly and reaches val R2 0.87 where the reference managed 0.72 before diverging. It is the default in every default config since 2026-09-09 (model constructors keep the old order so existing checkpoints load unchanged).

## Security Notice

LMDB data files and scaler files use Python pickle-based serialization (`torch.save`/`torch.load`). Do not load LMDB datasets or scaler files from untrusted sources, as they may contain arbitrary code that executes during deserialization.