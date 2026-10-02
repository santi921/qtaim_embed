"""Q2 + Q4: continue training from the epoch-1 checkpoint with a manual loop that
reproduces the Lightning optimisation (Adam state from ckpt, bf16 autocast,
norm clipping) and probe every K steps on a fixed val subset in eval mode and
in batch-stat BN mode. Variants implement the hypothesis tests by monkeypatching.

usage: python probe_continue.py <variant> [--steps N] [--every K] [--nval 50]
variants: baseline bn_mom001 mean_g mean_all sym_all clip1 lr3e-4 bn_eps1e-2
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CKPT, DEV, EDGE_TYPES, OUT, ActRecorder, Timer, bn_modules, bn_summary,
                    build_optimizer, conv_modules, eval_subset, forward_logits, grad_total_norm,
                    jsonl_append, load_config, load_model, loss_terms, make_dm, seed_all,
                    take_batches, to_dev)

p = argparse.ArgumentParser()
p.add_argument("variant")
p.add_argument("--steps", type=int, default=930)
p.add_argument("--every", type=int, default=150)
p.add_argument("--nval", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--workers", type=int, default=4)
args = p.parse_args()

VAR = args.variant
LOGP = f"{OUT}/probe_{VAR}.jsonl"
if os.path.exists(LOGP):
    os.remove(LOGP)

torch.set_float32_matmul_precision("high")
seed_all(args.seed)
cfg = load_config(num_workers=args.workers)
dm = make_dm(cfg)
val_batches = take_batches(dm.val_dataloader(), args.nval)
print(f"[{VAR}] fixed val subset: {len(val_batches)} batches, {sum(b[0]['atom'].num_nodes for b in val_batches)} atoms", flush=True)

model = load_model()
model.train()

clip = cfg["optim"]["gradient_clip_val"]
lr = None

# ---------------- variants ----------------
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import degree


def set_mean_aggr(conv):
    conv.graph_conv.aggr = "mean"
    conv.graph_conv.aggr_module = MeanAggregation()


if VAR == "baseline":
    pass
elif VAR == "bn_mom001":
    for *_, bn in bn_modules(model):
        bn.momentum = 0.01
elif VAR == "mean_g":
    for bi, li, et, conv in conv_modules(model):
        if et[1] in ("a2g", "b2g"):
            set_mean_aggr(conv)
elif VAR == "mean_all":
    for bi, li, et, conv in conv_modules(model):
        set_mean_aggr(conv)
elif VAR == "sym_all":
    # symmetric normalisation 1/sqrt(deg_src * deg_dst) as GraphConv edge weights,
    # injected through HeteroConv's edge_weight_dict; ResidualBlock.forward patched
    from qtaim_embed.models.layers import ResidualBlock

    def sym_weights(edge_index_dict, x_dict):
        w = {}
        for et, ei in edge_index_dict.items():
            src, rel, dst = et
            ns, nd = x_dict[src].shape[0], x_dict[dst].shape[0]
            ds = degree(ei[0], ns, dtype=torch.float32).clamp(min=1)
            dd = degree(ei[1], nd, dtype=torch.float32).clamp(min=1)
            w[et] = (ds[ei[0]] * dd[ei[1]]).rsqrt()
        return w

    def patched_forward(self, x_dict, edge_index_dict):
        input_feats = x_dict
        w = sym_weights(edge_index_dict, x_dict)
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_weight_dict=w)
        if not self.output_block:
            for k in x_dict.keys():
                x_dict[k].add_(input_feats[k])
        return x_dict

    ResidualBlock.forward = patched_forward
elif VAR == "clip1":
    clip = 1.0
elif VAR == "lr3e-4":
    lr = 3e-4
elif VAR == "bn_eps1e-2":
    # extra control beyond (a)-(e): caps the eval-mode BN gain at |gamma|/0.1 for
    # channels whose running_var collapsed; function change is negligible for live channels
    for *_, bn in bn_modules(model):
        bn.eps = 1e-2
else:
    raise SystemExit(f"unknown variant {VAR}")

opt, ck = build_optimizer(model, lr=lr)
print(f"[{VAR}] lr={opt.param_groups[0]['lr']} clip={clip} start epoch={ck['epoch']} step={ck['global_step']}", flush=True)

rec = ActRecorder(model)


def probe(step, train_loss, gnorm, train_acts):
    t = Timer()
    ev = eval_subset(model, val_batches, batch_stats=False, autocast=True, recorder=rec)
    bs = eval_subset(model, val_batches, batch_stats=True, autocast=True, recorder=rec)
    ev32 = eval_subset(model, val_batches, batch_stats=False, autocast=False)
    bns = bn_summary(model)
    out = dict(variant=VAR, step=step, train_loss_sum=train_loss, grad_norm_preclip=gnorm,
               eval_mse=ev["mse"], eval_rmse=ev["rmse"], eval_max_pred=ev["max_abs_pred"],
               eval32_mse=ev32["mse"],
               bstat_mse=bs["mse"], bstat_rmse=bs["rmse"], bstat_max_pred=bs["max_abs_pred"],
               eval_acts=ev["acts"], bstat_acts=bs["acts"], train_acts=train_acts, bn=bns,
               eval_mse_per_target=ev["mse_per_target"], probe_sec=t.lap())
    jsonl_append(LOGP, out)
    g_rv = max(v["rv_max"] for k, v in bns.items() if k.endswith("global"))
    dead = sum(v["dead"] for v in bns.values())
    ea = max(v for k, v in ev["acts"].items() if not k.endswith("nonfinite"))
    ba = max(v for k, v in bs["acts"].items() if not k.endswith("nonfinite"))
    print(f"[{VAR}] step {step:5d} train_loss {train_loss:.3f} gnorm {gnorm:8.2f} | eval mse {ev['mse']:.4g} (fp32 {ev32['mse']:.4g}) "
          f"bstat mse {bs['mse']:.4g} | eval act max {ea:.3g} bstat act max {ba:.3g} | global rv max {g_rv:.3g} dead ch {dead}", flush=True)


# initial probe (step 0 == end of epoch 1)
probe(0, float("nan"), float("nan"), {})

seed_all(args.seed)
loader = dm.train_dataloader()
step = 0
run_loss = 0.0
n_loss = 0
t0 = time.time()
done = False
while not done:
    torch.manual_seed(args.seed + step)  # deterministic shuffle order across variants
    for b in loader:
        g, labels = to_dev(b)
        model.train()
        rec.start()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_logits(model, g)
        train_acts = rec.stop()
        pred = logits["atom"].float()
        y = labels["atom"].float()
        loss = ((pred - y) ** 2).mean(0).sum()  # == all_loss.sum() in shared_step
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
        opt.step()
        step += 1
        run_loss += loss.item()
        n_loss += 1
        if step % args.every == 0:
            probe(step, run_loss / n_loss, gnorm, train_acts)
            run_loss = 0.0
            n_loss = 0
        if step >= args.steps:
            done = True
            break
    print(f"[{VAR}] epoch boundary at step {step}, {time.time()-t0:.0f}s elapsed", flush=True)
torch.save(model.state_dict(), f"{OUT}/state_{VAR}.pt")
print(f"[{VAR}] done {step} steps in {time.time()-t0:.0f}s; state saved to state_{VAR}.pt", flush=True)
