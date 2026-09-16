"""Mechanism test: BN channels whose running_var has collapsed (dead on training
data: ReLU output identically 0, so running_var decays as 0.9^t towards 0 and
eval-mode BN divides by sqrt(eps)=3.2e-3) but which receive non-zero input on
held-out molecules, and the resulting eval-mode amplification.

usage: python dead_channel_analysis.py [--state state_baseline.pt] [--nval 50]
Without --state the epoch-1 checkpoint is analysed.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEV, OUT, bn_modules, eval_subset, forward_logits, load_config, load_model,
                    make_dm, seed_all, take_batches, to_dev)

p = argparse.ArgumentParser()
p.add_argument("--state", default=None)
p.add_argument("--nval", type=int, default=50)
p.add_argument("--tag", default=None)
p.add_argument("--variant", default="baseline", help="apply the same architectural monkeypatch as probe_continue.py (mean_g, mean_all, bn_eps1e-2)")
p.add_argument("--repair", action="store_true", help="after the analysis, reset running_var<1e-3 channels to 1 (and running_mean to 0) and re-evaluate")
args = p.parse_args()
tag = args.tag or (os.path.basename(args.state).replace(".pt", "") if args.state else "epoch1")

torch.set_float32_matmul_precision("high")
seed_all(0)
cfg = load_config(num_workers=4)
dm = make_dm(cfg)
val_batches = take_batches(dm.val_dataloader(), args.nval)
model = load_model()
if args.variant in ("mean_g", "mean_all"):
    from torch_geometric.nn.aggr import MeanAggregation
    for bi, block in enumerate(model.conv_layers):
        for li, hconv in enumerate(block.layers):
            for et, conv in hconv.convs.items():
                if args.variant == "mean_all" or et[1] in ("a2g", "b2g"):
                    conv.graph_conv.aggr = "mean"
                    conv.graph_conv.aggr_module = MeanAggregation()
elif args.variant == "bn_eps1e-2":
    for *_, bn in bn_modules(model):
        bn.eps = 1e-2
elif args.variant != "baseline":
    raise SystemExit("variant not supported here")
if args.state:
    model.load_state_dict(torch.load(args.state, map_location=DEV))
model.eval()

lines = []
P = lambda s="": (lines.append(s), print(s, flush=True))
P(f"## Dead-channel analysis: {tag}")
P()

# per-BN-module hooks: record per-channel max |input| and max |output| over the val subset, eval mode
stats = {}
handles = []
for bi, li, rel, dst, bn in bn_modules(model):
    key = f"b{bi}.l{li}.{rel}->{dst}"
    stats[key] = dict(bn=bn, in_max=None, out_max=None, in_nz=None, n=0)

    def hook(mod, inp, out, key=key):
        x = inp[0].detach().float()
        y = out.detach().float()
        st = stats[key]
        im = x.abs().max(0).values
        om = y.abs().max(0).values
        nz = (x.abs() > 1e-6).sum(0)
        st["in_max"] = im if st["in_max"] is None else torch.maximum(st["in_max"], im)
        st["out_max"] = om if st["out_max"] is None else torch.maximum(st["out_max"], om)
        st["in_nz"] = nz if st["in_nz"] is None else st["in_nz"] + nz
        st["n"] += x.shape[0]

    handles.append(bn.register_forward_hook(hook))

ev = eval_subset(model, val_batches, batch_stats=False, autocast=True)
for h in handles:
    h.remove()
P(f"eval-mode val MSE on {len(val_batches)} batches: {ev['mse']:.4g}; max |pred| {ev['max_abs_pred']:.3g}")
P()

rows = []
for key, st in stats.items():
    bn = st["bn"]
    rv = bn.running_var.float()
    rm = bn.running_mean.float()
    gamma = bn.weight.float()
    scale = gamma.abs() / (rv + bn.eps).sqrt()  # eval-mode per-channel gain
    for c in range(rv.numel()):
        rows.append(dict(module=key, ch=c, rv=rv[c].item(), rm=rm[c].item(), gain=scale[c].item(),
                         in_max=st["in_max"][c].item(), out_max=st["out_max"][c].item(),
                         in_nz=int(st["in_nz"][c].item()), n=st["n"]))

rv_all = np.array([r["rv"] for r in rows])
gain_all = np.array([r["gain"] for r in rows])
P(f"BN channels total: {len(rows)}; running_var < 1e-3: {(rv_all < 1e-3).sum()}; < 1e-2: {(rv_all < 1e-2).sum()}; "
  f"eval gain gamma/sqrt(rv+eps) > 10: {(gain_all > 10).sum()}, > 100: {(gain_all > 100).sum()}")
P()
P("Top 20 channels by eval-mode output max |y| on the val subset:")
P()
P("| module | ch | running_var | running_mean | gain=|gamma|/sqrt(rv+eps) | max in | max out | nodes with in!=0 / total |")
P("|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: -r["out_max"])[:20]:
    P(f"| {r['module']} | {r['ch']} | {r['rv']:.3g} | {r['rm']:.3g} | {r['gain']:.3g} | {r['in_max']:.3g} | {r['out_max']:.3g} | {r['in_nz']}/{r['n']} |")
P()
P("Channels with running_var < 1e-3 (dead on training data) that receive non-zero input on held-out molecules:")
P()
P("| module | ch | running_var | gain | max in | max out | nodes with in!=0 / total |")
P("|---|---|---|---|---|---|---|")
cnt = 0
for r in sorted(rows, key=lambda r: -r["out_max"]):
    if r["rv"] < 1e-3 and r["in_nz"] > 0:
        cnt += 1
        if cnt <= 25:
            P(f"| {r['module']} | {r['ch']} | {r['rv']:.3g} | {r['gain']:.3g} | {r['in_max']:.3g} | {r['out_max']:.3g} | {r['in_nz']}/{r['n']} |")
P(f"({cnt} such channels in total)")
P()
# per node type summary of out_max
P("Max eval-mode BN output per destination node type and block:")
agg = {}
for r in rows:
    b = r["module"].split(".")[0]
    dst = r["module"].split("->")[1]
    k = f"{b}.{dst}"
    agg[k] = max(agg.get(k, 0.0), r["out_max"])
P(", ".join(f"{k}: {v:.3g}" for k, v in sorted(agg.items())))

if args.repair:
    P()
    P("### Repair test: causality check")
    n_fix = 0
    with torch.no_grad():
        for bi, li, rel, dst, bn in bn_modules(model):
            m = bn.running_var < 1e-3
            n_fix += int(m.sum())
            bn.running_var[m] = 1.0
            bn.running_mean[m] = 0.0
    ev2 = eval_subset(model, val_batches, batch_stats=False, autocast=True)
    P(f"reset {n_fix} channels with running_var < 1e-3 to running_var=1, running_mean=0 (weights untouched): eval-mode val MSE {ev['mse']:.4g} -> {ev2['mse']:.4g}; max |pred| {ev['max_abs_pred']:.3g} -> {ev2['max_abs_pred']:.3g}")
    # alternative: large eps only
    model.load_state_dict(torch.load(args.state, map_location=DEV)) if args.state else None
    for *_, bn in bn_modules(model):
        bn.eps = 1e-2
    ev3 = eval_subset(model, val_batches, batch_stats=False, autocast=True)
    P(f"alternative, original buffers but BN eps 1e-5 -> 1e-2 (caps gain at |gamma|/0.1): eval-mode val MSE {ev3['mse']:.4g}; max |pred| {ev3['max_abs_pred']:.3g}")
    bs = eval_subset(model, val_batches, batch_stats=True, autocast=True)
    P(f"reference, batch-stat BN on the same weights: val MSE {bs['mse']:.4g}")

open(f"{OUT}/dead_channels_{tag}.md", "w").write("\n".join(lines))
