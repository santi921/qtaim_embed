"""Shared helpers for the divergence investigation. Read-only w.r.t. the repo."""
import json
import os
import random
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

ROOT = "/home/santiagovargas/dev/qtaim_embed"
CFG_PATH = ROOT + "/profiling/train_configs/tm_react_b128_lr0.001.json"
CKPT = ROOT + "/profiling/train_runs/b128_lr0.001/model_lightning_epoch=001-val_loss=2.0877.ckpt"
OUT = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, ROOT)

EDGE_TYPES = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]


def seed_all(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(num_workers=4, batch_size=None):
    cfg = json.load(open(CFG_PATH))
    cfg["optim"]["num_workers"] = num_workers
    if batch_size is not None:
        cfg["optim"]["train_batch_size"] = batch_size
    return cfg


def make_dm(cfg):
    from qtaim_embed.core.datamodule import LMDBDataModule

    dm = LMDBDataModule(cfg)
    dm.setup("fit")
    return dm


def take_batches(loader, n):
    out = []
    for i, b in enumerate(loader):
        if i >= n:
            break
        out.append(b)
    return out


def to_dev(batch, dev=DEV):
    g, labels = batch
    g = g.to(dev)
    labels = {k: v.to(dev) for k, v in labels.items()}
    return g, labels


def load_model(ckpt=CKPT):
    from qtaim_embed.models.node_level.base_gcn import GCNNodePred

    model = GCNNodePred.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
    return model.to(DEV)


def forward_logits(model, g):
    feat_dict = {nt: g[nt].feat for nt in g.node_types}
    return model(g, feat_dict)


def loss_terms(logits, labels):
    """Returns (sum over 6 targets of per-target MSE, i.e. the logged *_loss;
    mean per-target MSE; mean per-target RMSE, i.e. the logged *_mse)."""
    pred = logits["atom"].float()
    y = labels["atom"].float()
    mse_t = ((pred - y) ** 2).mean(0)
    return mse_t.sum().item(), mse_t.mean().item(), mse_t.sqrt().mean().item()


def bn_modules(model):
    """Yield (block_idx, layer_idx, rel, dst_ntype, bn_module)."""
    for bi, block in enumerate(model.conv_layers):
        for li, hconv in enumerate(block.layers):
            for et, conv in hconv.convs.items():
                src, rel, dst = et
                if conv.batch_norm is not None:
                    yield bi, li, rel, dst, conv.batch_norm


def conv_modules(model):
    for bi, block in enumerate(model.conv_layers):
        for li, hconv in enumerate(block.layers):
            for et, conv in hconv.convs.items():
                yield bi, li, et, conv


@torch.no_grad()
def bn_summary(model, dead_thresh=1e-3):
    """Per (block, dst node type): max |running_mean|, max running_var, min running_var,
    number of channels with running_var < dead_thresh, total channels."""
    agg = {}
    for bi, li, rel, dst, bn in bn_modules(model):
        key = f"b{bi}.{dst}"
        d = agg.setdefault(key, {"rm_max": 0.0, "rv_max": 0.0, "rv_min": float("inf"), "dead": 0, "n": 0})
        rm = bn.running_mean.float()
        rv = bn.running_var.float()
        d["rm_max"] = max(d["rm_max"], rm.abs().max().item())
        d["rv_max"] = max(d["rv_max"], rv.max().item())
        d["rv_min"] = min(d["rv_min"], rv.min().item())
        d["dead"] += int((rv < dead_thresh).sum().item())
        d["n"] += rv.numel()
    return agg


@torch.no_grad()
def bn_summary_detail(model, dead_thresh=1e-3):
    rows = []
    for bi, li, rel, dst, bn in bn_modules(model):
        rv = bn.running_var.float()
        rows.append(
            dict(block=bi, layer=li, rel=rel, dst=dst,
                 rm_max=bn.running_mean.abs().max().item(),
                 rv_max=rv.max().item(), rv_min=rv.min().item(),
                 dead=int((rv < dead_thresh).sum().item()), n=rv.numel(),
                 gamma_max=bn.weight.abs().max().item(),
                 beta_max=bn.bias.abs().max().item())
        )
    return rows


class ActRecorder:
    """Forward hooks on each ResidualBlock recording max |activation| per node type."""

    def __init__(self, model):
        self.records = []
        self.handles = []
        for bi, block in enumerate(model.conv_layers):
            self.handles.append(block.register_forward_hook(self._hook(bi)))
        self.cur = None

    def _hook(self, bi):
        def fn(module, inp, out):
            if self.cur is None:
                return
            for k, v in out.items():
                vv = v.detach().float()
                m = vv.abs().max().item()
                key = f"b{bi}.{k}"
                self.cur[key] = max(self.cur.get(key, 0.0), m)
                if not torch.isfinite(vv).all():
                    self.cur[key + ".nonfinite"] = True
        return fn

    def start(self):
        self.cur = {}

    def stop(self):
        r = self.cur
        self.cur = None
        return r

    def remove(self):
        for h in self.handles:
            h.remove()


def set_bn_mode(model, batch_stats: bool):
    """batch_stats=True: BN layers use batch statistics but do not update running
    stats (momentum 0); everything else stays in eval mode (no dropout)."""
    model.eval()
    for _, _, _, _, bn in bn_modules(model):
        if batch_stats:
            bn.train()
            bn._saved_momentum = bn.momentum
            bn.momentum = 0.0
        else:
            bn.eval()
            if hasattr(bn, "_saved_momentum"):
                bn.momentum = bn._saved_momentum
                del bn._saved_momentum


@torch.no_grad()
def eval_subset(model, batches, batch_stats=False, autocast=True, recorder=None):
    """Held-out MSE on the fixed subset. Returns dict with loss-sum, mean MSE,
    mean RMSE, and (if recorder) max activations."""
    was_training = model.training
    set_bn_mode(model, batch_stats)
    nbt = {id(bn): bn.num_batches_tracked.clone() for *_, bn in bn_modules(model)}
    se = None
    n = 0
    maxabs_pred = 0.0
    if recorder is not None:
        recorder.start()
    for b in batches:
        g, labels = to_dev(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = forward_logits(model, g)
        pred = logits["atom"].float()
        y = labels["atom"].float()
        d = (pred - y) ** 2
        se = d.sum(0) if se is None else se + d.sum(0)
        n += y.shape[0]
        maxabs_pred = max(maxabs_pred, pred.abs().max().item())
    acts = recorder.stop() if recorder is not None else None
    # restore
    for *_, bn in bn_modules(model):
        bn.num_batches_tracked.copy_(nbt[id(bn)])
    set_bn_mode(model, False)
    if was_training:
        model.train()
    mse_t = se / n
    out = dict(loss_sum=mse_t.sum().item(), mse=mse_t.mean().item(),
               rmse=mse_t.sqrt().mean().item(), max_abs_pred=maxabs_pred,
               mse_per_target=[round(x, 5) for x in mse_t.tolist()])
    if acts is not None:
        out["acts"] = acts
    return out


def grad_total_norm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += p.grad.detach().float().norm() ** 2
    return float(tot ** 0.5)


def build_optimizer(model, ckpt_path=CKPT, lr=None, weight_decay=None):
    """Adam matching build_adam when the trainer clips (fused=False), with the
    optimizer state loaded from the checkpoint so the continuation is faithful."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    lr = hp["lr"] if lr is None else lr
    wd = hp["weight_decay"] if weight_decay is None else weight_decay
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"], fused=False)
    opt.load_state_dict(ck["optimizer_states"][0])
    for gparam in opt.param_groups:
        gparam["lr"] = lr
        gparam["weight_decay"] = wd
    return opt, ck


def jsonl_append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        now = time.time()
        d = now - self.t
        self.t = now
        return d
