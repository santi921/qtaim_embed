"""Q3: characterise the input to the global node with the epoch-1 checkpoint.

- a2g / b2g aggregated sums entering the first conv layer (sum over embedded atom/bond
  features per molecule) vs molecule atom/bond count, over 50 val batches.
- Feature columns whose scaled values reach ~157: node type, column, name.
- Whether batch-to-batch variance of the global BN batch statistics (layer 0, a2g/b2g
  modules, pre-BN activations of global nodes) is dominated by molecule size.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEV, OUT, load_config, load_model, make_dm, seed_all, take_batches, to_dev

torch.set_float32_matmul_precision("high")
seed_all(0)
cfg = load_config(num_workers=4)
dm = make_dm(cfg)
NB = 50
val_batches = take_batches(dm.val_dataloader(), NB)
torch.manual_seed(0)
train_batches = take_batches(dm.train_dataloader(), NB)
names = dm.train_dataset.feature_names

model = load_model().eval()
lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))


def pct(x, qs=(50, 90, 99, 99.9, 100)):
    x = np.asarray(x, dtype=np.float64)
    return " / ".join(f"{np.percentile(x, q):.3g}" for q in qs)


# ---------- 1. feature columns with extreme scaled values ----------
P("## Q3a. Scaled feature columns with extreme values (50 val batches + 50 train batches)")
P()
for split, batches in [("val", val_batches), ("train", train_batches)]:
    for nt in ["atom", "bond", "global"]:
        feats = torch.cat([b[0][nt].feat for b in batches], 0)
        colmax = feats.abs().max(0).values
        big = (colmax > 50).nonzero().flatten().tolist()
        P(f"{split} {nt}: {feats.shape[0]} nodes, {len(big)} columns with max |x| > 50, overall max {colmax.max():.2f}")
        for c in big:
            col = feats[:, c]
            nz = (col > 50).sum().item()
            P(f"    col {c:3d} {names[nt][c]:28s} max {col.max():8.2f} min {col.min():7.3f} nodes>50: {nz}")
    P()

# molecule-level: how many val molecules contain any atom feature > 50 (rare element)
def mols_with_big(batches, thresh=50):
    cnt = 0
    tot = 0
    for g, _ in batches:
        big = (g["atom"].feat.abs() > thresh).any(1)
        per_mol = torch.zeros(g.num_graphs, dtype=torch.bool).scatter_(0, g["atom"].batch, big, reduce="add") if False else \
            torch.zeros(g.num_graphs, dtype=torch.long).index_add_(0, g["atom"].batch, big.long()) > 0
        cnt += per_mol.sum().item()
        tot += g.num_graphs
    return cnt, tot

for split, batches in [("val", val_batches), ("train", train_batches)]:
    c, t = mols_with_big(batches)
    P(f"{split}: molecules containing an atom with a scaled feature > 50: {c} / {t}")
P()

# ---------- 2. aggregated sums entering conv layer 0 ----------
P("## Q3b. a2g / b2g aggregated sums entering conv layer 0 (val, 50 batches, embedding output, fp32)")
P()
rows_a, rows_b = [], []
with torch.no_grad():
    for b in val_batches:
        g, _ = to_dev(b)
        feats = model.embedding({nt: g[nt].feat for nt in g.node_types})
        G = g.num_graphs
        na = torch.bincount(g["atom"].batch, minlength=G).float()
        nb = torch.bincount(g["bond"].batch, minlength=G).float()
        sa = torch.zeros(G, feats["atom"].shape[1], device=DEV).index_add_(0, g["atom"].batch, feats["atom"])
        sb = torch.zeros(G, feats["bond"].shape[1], device=DEV).index_add_(0, g["bond"].batch, feats["bond"])
        # per-node embedding scale for reference
        rows_a.append(torch.stack([na, sa.norm(dim=1), sa.abs().max(1).values, feats["atom"].norm(dim=1).mean().expand(G)], 1).cpu())
        rows_b.append(torch.stack([nb, sb.norm(dim=1), sb.abs().max(1).values, feats["bond"].norm(dim=1).mean().expand(G)], 1).cpu())
A = torch.cat(rows_a).numpy()
B = torch.cat(rows_b).numpy()
P(f"molecules: {A.shape[0]}; atom count percentiles (50/90/99/99.9/100): {pct(A[:,0])}; bond count: {pct(B[:,0])}")
P(f"per-atom embedding L2 norm (batch mean): {A[:,3].mean():.3g};  per-bond: {B[:,3].mean():.3g}")
P(f"a2g sum L2 norm percentiles: {pct(A[:,1])}")
P(f"a2g sum max|component| percentiles: {pct(A[:,2])}")
P(f"b2g sum L2 norm percentiles: {pct(B[:,1])}")
P(f"b2g sum max|component| percentiles: {pct(B[:,2])}")
ra = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
rb = np.corrcoef(B[:, 0], B[:, 1])[0, 1]
P(f"Pearson r(atom count, |a2g sum|) = {ra:.3f}  (R2 {ra**2:.3f});  r(bond count, |b2g sum|) = {rb:.3f}  (R2 {rb**2:.3f})")
# linear fit norm ~ count
ka = np.polyfit(A[:, 0], A[:, 1], 1)
P(f"fit |a2g sum| ~ {ka[0]:.3g} * n_atoms + {ka[1]:.3g}; mean atom count {A[:,0].mean():.1f}, mean |a2g sum| {A[:,1].mean():.3g}")
# atoms of the largest sum molecules
idx = np.argsort(-A[:, 2])[:5]
P("top-5 molecules by a2g max component: (n_atoms, |sum|, max comp) " + "; ".join(f"({int(A[i,0])}, {A[i,1]:.3g}, {A[i,2]:.3g})" for i in idx))
P()

# ---------- 3. pre-BN activations of global nodes in layer 0 vs batch molecule size ----------
P("## Q3c. Global-node pre-BN activations at conv layer 0 (a2g and b2g modules): batch statistics vs molecule size")
P()
store = {}
def mk_hook(name):
    def fn(mod, inp, out):
        store[name] = inp[0].detach().float()
    return fn
h = []
blk = model.conv_layers[0].layers[0]
for et, conv in blk.convs.items():
    if et[1] in ("a2g", "b2g"):
        h.append(conv.batch_norm.register_forward_pre_hook(lambda m, i, n=et[1]: store.__setitem__(n, i[0].detach().float())))

per_batch = {"a2g": [], "b2g": []}
per_mol = {"a2g": [], "b2g": []}
with torch.no_grad():
    for b in val_batches:
        g, _ = to_dev(b)
        na = torch.bincount(g["atom"].batch, minlength=g.num_graphs).float()
        nb = torch.bincount(g["bond"].batch, minlength=g.num_graphs).float()
        model.eval()
        _ = model(g, {nt: g[nt].feat for nt in g.node_types})
        for rel, cnt in [("a2g", na), ("b2g", nb)]:
            x = store[rel]  # [G, 128] pre-BN (post ReLU) global activations
            per_batch[rel].append((cnt.mean().item(), cnt.std().item(), x.mean(0).cpu(), x.var(0, unbiased=True).cpu()))
            per_mol[rel].append(torch.stack([cnt.cpu(), x.norm(dim=1).cpu(), x.mean(1).cpu()], 1))
for hh in h:
    hh.remove()

for rel in ["a2g", "b2g"]:
    M = torch.stack([r[2] for r in per_batch[rel]])  # [50 batches, 128 ch] batch means
    V = torch.stack([r[3] for r in per_batch[rel]])  # batch variances
    cnt_mean = np.array([r[0] for r in per_batch[rel]])
    ch_mean_over_batches = M.mean(0)
    # batch-to-batch variance of the batch mean per channel, relative to the within-batch variance
    b2b_var = M.var(0, unbiased=True)
    within = V.mean(0)
    P(f"{rel}: pre-BN activation, mean over channels of batch mean {ch_mean_over_batches.mean():.3g}, max channel batch mean {M.max():.3g}")
    P(f"{rel}: within-batch variance per channel: median {within.median():.3g}, max {within.max():.3g}")
    P(f"{rel}: batch-to-batch variance of the batch mean per channel: median {b2b_var.median():.3g}, max {b2b_var.max():.3g}; "
      f"ratio b2b/within median {(b2b_var/within.clamp(min=1e-12)).median():.3g}")
    # how much of the batch-mean variation is explained by the batch mean atom count
    scal = M.mean(1).numpy()
    r = np.corrcoef(cnt_mean, scal)[0, 1]
    P(f"{rel}: r(batch mean molecule size, batch mean activation) = {r:.3f} (R2 {r**2:.3f}); batch mean size range {cnt_mean.min():.1f}-{cnt_mean.max():.1f}")
    PM = torch.cat(per_mol[rel]).numpy()
    r2 = np.corrcoef(PM[:, 0], PM[:, 1])[0, 1]
    P(f"{rel}: per molecule, r(size, |pre-BN activation|) = {r2:.3f} (R2 {r2**2:.3f}); |act| percentiles {pct(PM[:,1])}")
    # variance decomposition of per-molecule activation norm: fraction explained by size (linear)
    k = np.polyfit(PM[:, 0], PM[:, 1], 1)
    resid = PM[:, 1] - np.polyval(k, PM[:, 0])
    P(f"{rel}: linear fit |act| ~ {k[0]:.3g}*size + {k[1]:.3g}; fraction of variance explained by size {1 - resid.var()/PM[:,1].var():.3f}")
    P()

# ---------- 4. running stats vs batch stats of the global BN at layer 0 ----------
P("## Q3d. Layer-0 global BN: checkpoint running stats vs the val-subset batch stats")
P()
for et, conv in blk.convs.items():
    if et[1] in ("a2g", "b2g"):
        bn = conv.batch_norm
        M = torch.stack([r[2] for r in per_batch[et[1]]])
        V = torch.stack([r[3] for r in per_batch[et[1]]])
        rm, rv = bn.running_mean.cpu(), bn.running_var.cpu()
        P(f"{et[1]}: running_mean max {rm.abs().max():.3g} vs val batch-mean max {M.mean(0).abs().max():.3g}; "
          f"running_var max {rv.max():.3g} vs val batch-var max {V.mean(0).max():.3g}; running_var min {rv.min():.3g}, "
          f"channels with running_var < 1e-3: {(rv<1e-3).sum().item()}, channels with val batch var < 1e-3: {(V.mean(0)<1e-3).sum().item()}")
        # channels dead in running stats but alive on val
        dead = rv < 1e-3
        alive_val = V.mean(0) > 1e-3
        both = (dead & alive_val).nonzero().flatten().tolist()
        P(f"{et[1]}: channels dead in running stats (rv<1e-3) but with val batch var > 1e-3: {both}")

open(OUT + "/global_input_stats.md", "w").write("\n".join(lines))
