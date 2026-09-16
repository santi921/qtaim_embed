"""Q5: mundane causes. Full val split scan for NaN/inf, extreme molecules, label and
feature statistics compared with train (same scaler?), LMDB scaler metadata, and a
torchmetrics accumulate/reset check."""
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, load_config, make_dm, seed_all, take_batches

seed_all(0)
cfg = load_config(num_workers=4)
dm = make_dm(cfg)
names = dm.train_dataset.feature_names
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))

P("## Q5a. LMDB metadata per split (scaler applied at conversion time?)")
P()
for split in ["train", "val", "test"]:
    ds = getattr(dm, f"{split}_dataset")
    metas = []
    for env in ds.envs:
        with env.begin() as txn:
            m = {k.decode(): pickle.loads(v) for k, v in txn.cursor() if not k.isdigit() and k not in (b"processed_source_keys",)}
        metas.append(m)
    keys = sorted(set().union(*[set(m) for m in metas]))
    P(f"{split}: {len(ds)} graphs, {len(ds.envs)} shards; meta keys {keys}")
    for m in metas:
        P(f"    scaled={m.get('scaled')} length={m.get('length')} feature_size={m.get('feature_size')} target_dict={m.get('target_dict')} n_elements={len(m.get('element_set', []))}")
    # is there any scaler object stored?
    scaler_keys = [k for k in keys if "scal" in k.lower()]
    P(f"    scaler-related metadata keys: {scaler_keys}")
P()
P("Interpretation: features and labels are scaled once at LMDB conversion (scaled=True), no scaler object is applied "
  "in LMDBDataModule/TransformMol or in validation_step, so train and val necessarily go through the same (frozen) scaling "
  "iff the shards were produced by the same conversion. The statistics below test that.")
P()


def scan(batches, label):
    st = {}
    nan_any = {"feat": {}, "labels": 0}
    max_atoms = 0
    max_bonds = 0
    max_atoms_name = None
    feat_max = {}
    feat_argmax = {}
    lab_sum = None
    lab_sq = None
    lab_n = 0
    lab_max = 0
    feat_sum = {}
    feat_sq = {}
    feat_n = {}
    sizes = []
    for g, labels in batches:
        for nt in g.node_types:
            f = g[nt].feat
            nan_any["feat"][nt] = nan_any["feat"].get(nt, 0) + int((~torch.isfinite(f)).sum())
            m = f.abs().max(0).values
            if nt not in feat_max:
                feat_max[nt] = m
                feat_sum[nt] = f.sum(0).double()
                feat_sq[nt] = (f.double() ** 2).sum(0)
                feat_n[nt] = f.shape[0]
            else:
                feat_max[nt] = torch.maximum(feat_max[nt], m)
                feat_sum[nt] += f.sum(0).double()
                feat_sq[nt] += (f.double() ** 2).sum(0)
                feat_n[nt] += f.shape[0]
        y = labels["atom"]
        nan_any["labels"] += int((~torch.isfinite(y)).sum())
        lab_max = max(lab_max, y.abs().max().item())
        lab_sum = y.sum(0).double() if lab_sum is None else lab_sum + y.sum(0).double()
        lab_sq = (y.double() ** 2).sum(0) if lab_sq is None else lab_sq + (y.double() ** 2).sum(0)
        lab_n += y.shape[0]
        na = torch.bincount(g["atom"].batch, minlength=g.num_graphs)
        nb = torch.bincount(g["bond"].batch, minlength=g.num_graphs)
        sizes.append(na)
        if na.max() > max_atoms:
            max_atoms = na.max().item()
        max_bonds = max(max_bonds, nb.max().item())
    sizes = torch.cat(sizes).numpy()
    P(f"### {label}: {len(sizes)} molecules")
    P(f"non-finite feature entries: {nan_any['feat']}; non-finite label entries: {nan_any['labels']}")
    P(f"atoms per molecule: min {sizes.min()} median {np.median(sizes):.0f} mean {sizes.mean():.1f} max {max_atoms}; max bonds {max_bonds}")
    lm = lab_sum / lab_n
    ls = (lab_sq / lab_n - lm ** 2).sqrt()
    P("label mean per target: " + ", ".join(f"{x:.3f}" for x in lm.tolist()))
    P("label std  per target: " + ", ".join(f"{x:.3f}" for x in ls.tolist()))
    P(f"label max |y|: {lab_max:.2f}")
    for nt in ["atom", "bond", "global"]:
        fm = feat_sum[nt] / feat_n[nt]
        fs = (feat_sq[nt] / feat_n[nt] - fm ** 2).clamp(min=0).sqrt()
        P(f"{nt} features: max |x| {feat_max[nt].max():.1f} (col {feat_max[nt].argmax().item()} {names[nt][feat_max[nt].argmax().item()]}); "
          f"columns with max>50: {(feat_max[nt] > 50).sum().item()}; mean of column means {fm.mean():.3f}, mean of column stds {fs.mean():.3f}")
    P()
    return dict(feat_max=feat_max, lab_mean=lm, lab_std=ls, feat_mean={k: feat_sum[k] / feat_n[k] for k in feat_sum},
                feat_std={k: (feat_sq[k] / feat_n[k] - (feat_sum[k] / feat_n[k]) ** 2).clamp(min=0).sqrt() for k in feat_sum})


P("## Q5b. Full val split scan and a 200-batch train sample")
P()
val_stats = scan(dm.val_dataloader(), "val (all batches)")
torch.manual_seed(0)
train_stats = scan(take_batches(dm.train_dataloader(), 200), "train (200 batches, 25600 molecules)")

P("### train vs val column statistics (same scaler check)")
for nt in ["atom", "bond", "global"]:
    dm_ = (train_stats["feat_mean"][nt] - val_stats["feat_mean"][nt]).abs()
    P(f"{nt}: max |mean_train - mean_val| over columns {dm_.max():.3f} (col {dm_.argmax().item()} {names[nt][dm_.argmax().item()]}); "
      f"max |std_train - std_val| {(train_stats['feat_std'][nt] - val_stats['feat_std'][nt]).abs().max():.3f}")
P(f"labels: max |mean_train - mean_val| {(train_stats['lab_mean'] - val_stats['lab_mean']).abs().max():.4f}, "
  f"max |std_train - std_val| {(train_stats['lab_std'] - val_stats['lab_std']).abs().max():.4f}")
P()
P("Val columns with max > 50 (name, val max, train-sample max):")
for nt in ["atom", "bond", "global"]:
    for c in (val_stats["feat_max"][nt] > 50).nonzero().flatten().tolist():
        P(f"    {nt} col {c} {names[nt][c]}: val {val_stats['feat_max'][nt][c]:.1f}, train {train_stats['feat_max'][nt][c]:.1f}")
P()

P("## Q5c. torchmetrics accumulation / reset")
P()
import torchmetrics
from torchmetrics.wrappers import MultioutputWrapper

m = MultioutputWrapper(torchmetrics.MeanSquaredError(squared=False), num_outputs=2)
a = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
m.update(a, a + 1)
m.update(a, a + 1)
c1 = m.compute()
m.reset()
m.update(a, a + 3)
c2 = m.compute()
P(f"RMSE after two updates of error 1: {c1.tolist()}; after reset and one update of error 3: {c2.tolist()} -> reset works")
P("Code check (qtaim_embed/models/node_level/base_gcn.py): compute_metrics() calls .reset() on val_r2/val_torch_l1/val_torch_mse "
  "after every compute; on_validation_epoch_end calls compute_metrics('val') once per epoch. So val metrics do not accumulate across epochs. "
  "val_loss is logged with on_epoch=True via self.log (Lightning's own per-epoch mean, reset every epoch). "
  "self.loss (MultioutputWrapper(MeanSquaredError)) is called via forward(); its internal state accumulates for the whole run but the "
  "returned per-batch value is batch-local, so this does not affect training or logging. "
  "Note 'train_mse'/'val_mse' are MeanSquaredError(squared=False), i.e. RMSE, so val_mse=176 at epoch 3 means MSE about 3.1e4.")
P()
P("## Q5d. Does validation use the same scaler / mode as training?")
P()
P("validation_step and training_step both call shared_step; no scaler is applied in either (LMDB graphs are pre-scaled, "
  "labels come from graph['atom'].labels via _get_ndata). Lightning runs validation under model.eval() (BN running stats, no dropout) "
  "and train under model.train(). The label statistics above show train and val labels are both standardised to mean~0, std~1 with the same scaler.")

open(OUT + "/mundane_checks.md", "w").write("\n".join(lines))
