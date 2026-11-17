# OMol-Descriptors-4M `Charges` baselines

Training configurations for the `Charges` task baselines (six charge schemes predicted
jointly per atom): `gcn`, `gat`, `schnet`, `dimenetpp`, `equivariant`, and `resgcn`.

- Replace `<DATA_ROOT>` with the directory holding the released `train`, `val` and `test`
  graph LMDBs.
- `train_batch_size` is per device; the runs used 4 GPUs with DDP (`optim.num_devices`).
- Adam optimizer, MSE loss on per-scheme standardized targets, reduce-on-plateau
  schedule, early stopping on validation loss after `model.extra_stop_patience` epochs.
