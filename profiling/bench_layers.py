#!/usr/bin/env python3
"""E3: conv-stack formulation microbenchmark on real batched molecules.

Times forward + backward of an 8-layer ResidualBlock-equivalent stack
(4 blocks x 2 hetero convs, 9 edge types, sum aggregation, ReLU, dropout,
no batch norm) for three formulations that compute the same function:

  ref    current HeteroConv of GraphConvDropoutBatch (qtaim_embed.models.layers)
  fused  all node types concatenated with offsets; one gather + one index_add
         over all 9 edge types into a [3 slots, N_all, H] buffer, then per
         destination type one bmm (W_rel) + one mm (W_root), act, dropout,
         sum over slots
  dense  molecules padded to a static (N_b, B_b) shape; a2b/b2a are bmm with a
         per-molecule incidence matrix, global edges are masked sums and
         broadcasts, self loops are identities; eager, torch.compile, and
         torch.compile(mode="reduce-overhead") (CUDA graphs)

Weights are copied from ref into fused and dense and fp32 parity is asserted
(dropout off) before timing. Batches are formed from graphs of similar size
(the middle slice of a sorted sample) to emulate bucketed sampling; the
dense shape is the batch max rounded up to a multiple of --grid.

Usage:
  CUDA_VISIBLE_DEVICES=0 python profiling/bench_layers.py \
      --datasets tm_react h7 --hidden 128 256 512 --dtypes bf16 fp32
"""

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from qtaim_embed.models.layers import EDGE_TYPE_MAP, ResidualBlock

DATASETS = {
    "tm_react": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/splits/tm_react/train",
    "h7": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/holdouts/H7",
}
NODE_TYPES = ["atom", "bond", "global"]
# incoming edge types per destination type; the slot order is the contract
# shared by the fused and dense parameter layouts
SLOTS = {
    "atom": ["b2a", "g2a", "a2a"],
    "bond": ["a2b", "g2b", "b2b"],
    "global": ["a2g", "b2g", "g2g"],
}
SRC_OF = {e: EDGE_TYPE_MAP[e][0] for et in SLOTS.values() for e in et}
EDGE_TYPES = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]


def make_layer_args(hidden: int, act, dropout: float):
    return {
        e: {
            "in_feats": hidden, "out_feats": hidden, "norm": "both", "bias": True,
            "activation": act, "allow_zero_in_degree": True, "dropout": dropout,
            "batch_norm_tf": False,
        }
        for e in EDGE_TYPES
    }


# --------------------------------------------------------------------------- ref
class RefStack(nn.Module):
    def __init__(self, hidden, n_blocks, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualBlock(make_layer_args(hidden, nn.ReLU(), dropout), aggregate="sum",
                          resid_n_graph_convs=2, output_block=False)
            for _ in range(n_blocks)
        )

    def forward(self, x_dict, edge_index_dict):
        for b in self.blocks:
            x_dict = b(x_dict, edge_index_dict)
        return x_dict


# ------------------------------------------------------------------------- fused
class TypedWeights(nn.Module):
    """Shared parameter layout: W_rel[dst, slot, in, out], b_rel[dst, slot, 1, out],
    W_root[dst, in, 3*out] (slot-major columns)."""

    def __init__(self, hidden):
        super().__init__()
        self.W_rel = nn.Parameter(torch.empty(3, 3, hidden, hidden))
        self.b_rel = nn.Parameter(torch.zeros(3, 3, 1, hidden))
        self.W_root = nn.Parameter(torch.empty(3, hidden, 3 * hidden))
        nn.init.xavier_uniform_(self.W_rel)
        nn.init.xavier_uniform_(self.W_root)

    @torch.no_grad()
    def copy_from_hetero(self, hetero_conv, hidden):
        for t, dst in enumerate(NODE_TYPES):
            for s, e in enumerate(SLOTS[dst]):
                gc = hetero_conv.convs[EDGE_TYPE_MAP[e]].graph_conv
                self.W_rel[t, s].copy_(gc.lin_rel.weight.t())
                self.b_rel[t, s, 0].copy_(gc.lin_rel.bias)
                self.W_root[t][:, s * hidden:(s + 1) * hidden].copy_(gc.lin_root.weight.t())


class FusedLayer(TypedWeights):
    def __init__(self, hidden, dropout):
        super().__init__(hidden)
        self.hidden = hidden
        self.drop = nn.Dropout(dropout)

    def forward(self, x_all, src_all, dst_slot, offsets, sizes):
        H = self.hidden
        n_all = x_all.shape[0]
        msg = x_all.index_select(0, src_all)
        agg = x_all.new_zeros(3 * n_all, H).index_add_(0, dst_slot, msg).view(3, n_all, H)
        outs = []
        for t in range(3):
            a, n = offsets[t], sizes[t]
            z = torch.baddbmm(self.b_rel[t], agg[:, a:a + n], self.W_rel[t])
            root = torch.mm(x_all[a:a + n], self.W_root[t]).view(n, 3, H).transpose(0, 1)
            z = self.drop(torch.relu(z + root))
            outs.append(z.sum(0))
        return torch.cat(outs, 0)


class FusedStack(nn.Module):
    def __init__(self, hidden, n_blocks, dropout):
        super().__init__()
        self.layers = nn.ModuleList(FusedLayer(hidden, dropout) for _ in range(2 * n_blocks))

    def forward(self, x_all, idx):
        for i in range(0, len(self.layers), 2):
            h = self.layers[i](x_all, *idx)
            h = self.layers[i + 1](h, *idx)
            x_all = x_all + h
        return x_all


def fused_indices(batch, device):
    sizes = [int(batch[nt].num_nodes) for nt in NODE_TYPES]
    offsets = [0, sizes[0], sizes[0] + sizes[1]]
    n_all = sum(sizes)
    off = dict(zip(NODE_TYPES, offsets))
    src, dst = [], []
    for t, dst_type in enumerate(NODE_TYPES):
        for s, e in enumerate(SLOTS[dst_type]):
            ei = batch[EDGE_TYPE_MAP[e]].edge_index
            src.append(ei[0] + off[SRC_OF[e]])
            dst.append(ei[1] + off[dst_type] + s * n_all)
    return (torch.cat(src).to(device), torch.cat(dst).to(device), offsets, sizes)


# ------------------------------------------------------------------------- dense
class DenseLayer(TypedWeights):
    def __init__(self, hidden, dropout):
        super().__init__(hidden)
        self.hidden = hidden
        self.drop = nn.Dropout(dropout)

    def _head(self, t, agg, X, mask):
        H = self.hidden
        S = agg.shape[1] * agg.shape[2]
        z = torch.baddbmm(self.b_rel[t], agg.reshape(3, S, H), self.W_rel[t])
        root = torch.mm(X.reshape(S, H), self.W_root[t]).view(S, 3, H).transpose(0, 1)
        z = self.drop(torch.relu(z + root)).sum(0).view_as(X)
        return z * mask if mask is not None else z

    def forward(self, Xa, Xb, Xg, A_a2b, A_b2a, ma, mb):
        G, Nb, H = Xa.shape
        Bb = Xb.shape[1]
        agg_a = torch.stack([torch.bmm(A_b2a, Xb), Xg.expand(G, Nb, H), Xa])
        agg_b = torch.stack([torch.bmm(A_a2b, Xa), Xg.expand(G, Bb, H), Xb])
        agg_g = torch.stack([Xa.sum(1, keepdim=True), Xb.sum(1, keepdim=True), Xg])
        return (self._head(0, agg_a, Xa, ma), self._head(1, agg_b, Xb, mb),
                self._head(2, agg_g, Xg, None))


class DenseStack(nn.Module):
    def __init__(self, hidden, n_blocks, dropout):
        super().__init__()
        self.layers = nn.ModuleList(DenseLayer(hidden, dropout) for _ in range(2 * n_blocks))

    def forward(self, Xa, Xb, Xg, A_a2b, A_b2a, ma, mb):
        for i in range(0, len(self.layers), 2):
            ha, hb, hg = self.layers[i](Xa, Xb, Xg, A_a2b, A_b2a, ma, mb)
            ha, hb, hg = self.layers[i + 1](ha, hb, hg, A_a2b, A_b2a, ma, mb)
            Xa, Xb, Xg = Xa + ha, Xb + hb, Xg + hg
        return Xa, Xb, Xg


def dense_inputs(batch, x_dict, grid, device):
    ba, bb = batch["atom"].batch, batch["bond"].batch
    G = int(ba.max()) + 1
    n_a = torch.bincount(ba, minlength=G)
    n_b = torch.bincount(bb, minlength=G)
    Nb = int(np.ceil(int(n_a.max()) / grid) * grid)
    Bb = int(np.ceil(int(n_b.max()) / grid) * grid)
    Xa, ma = to_dense_batch(x_dict["atom"], ba, max_num_nodes=Nb)
    Xb, mb = to_dense_batch(x_dict["bond"], bb, max_num_nodes=Bb)
    Xg = x_dict["global"].view(G, 1, -1)
    ptr_a = torch.cat([n_a.new_zeros(1), n_a.cumsum(0)[:-1]])
    ptr_b = torch.cat([n_b.new_zeros(1), n_b.cumsum(0)[:-1]])

    ei = batch[EDGE_TYPE_MAP["a2b"]].edge_index
    g = ba[ei[0]]
    A_a2b = torch.zeros(G, Bb, Nb, device=device, dtype=Xa.dtype)
    A_a2b[g, ei[1] - ptr_b[g], ei[0] - ptr_a[g]] = 1.0
    ei = batch[EDGE_TYPE_MAP["b2a"]].edge_index
    g = bb[ei[0]]
    A_b2a = torch.zeros(G, Nb, Bb, device=device, dtype=Xa.dtype)
    A_b2a[g, ei[1] - ptr_a[g], ei[0] - ptr_b[g]] = 1.0

    waste = 1.0 - float(n_a.sum() + n_b.sum()) / float(G * (Nb + Bb))
    inputs = (Xa.contiguous(), Xb.contiguous(), Xg.contiguous(), A_a2b, A_b2a,
              ma.unsqueeze(-1).to(Xa.dtype), mb.unsqueeze(-1).to(Xb.dtype))
    return inputs, {"G": G, "N_b": Nb, "B_b": Bb, "pad_waste": waste}


# --------------------------------------------------------------------- helpers
def load_batch(src, bs, seed=0, pool_mult=3):
    from qtaim_embed.core.dataset import LMDBMoleculeDataset
    from qtaim_embed.data.lmdb import TransformMol

    ds = LMDBMoleculeDataset(config={"src": src}, transform=partial(TransformMol, dtype="float32"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(bs * pool_mult, len(ds)), replace=False)
    graphs = [ds[int(i)] for i in idx]
    graphs.sort(key=lambda g: g["atom"].num_nodes)
    mid = (len(graphs) - bs) // 2
    return Batch.from_data_list(graphs[mid:mid + bs])


def rand_feats(batch, hidden, device, gen):
    return {nt: torch.randn(int(batch[nt].num_nodes), hidden, device=device, generator=gen)
            for nt in NODE_TYPES}


def timed(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.fmean(times), statistics.pstdev(times), torch.cuda.max_memory_allocated() / 1e9


def kernel_count(fn, n=3):
    from torch.profiler import DeviceType, ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == DeviceType.CUDA) / n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["tm_react", "h7"])
    p.add_argument("--tm_react_batches", nargs="+", type=int, default=[128, 512, 1024])
    p.add_argument("--h7_batches", nargs="+", type=int, default=[32, 128])
    p.add_argument("--hidden", nargs="+", type=int, default=[128, 256, 512])
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"])
    p.add_argument("--impls", nargs="+", default=["ref", "fused", "dense", "dense_compile", "dense_cudagraph"])
    p.add_argument("--n_blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--out", default="profiling/bench_results/layers_e3.json")
    args = p.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    rows = []
    if Path(args.out).exists():
        rows = json.load(open(args.out))
    done = {(r["dataset"], r["batch_size"], r["hidden"], r["dtype"], r["impl"]) for r in rows}

    for name in args.datasets:
        bss = args.tm_react_batches if name == "tm_react" else args.h7_batches
        for bs in bss:
            batch = load_batch(DATASETS[name], bs).to(dev)
            ei_dict = {et: batch[et].edge_index for et in batch.edge_types}
            n_nodes = {nt: int(batch[nt].num_nodes) for nt in NODE_TYPES}
            n_edges = int(sum(e.shape[1] for e in ei_dict.values()))
            for hidden in args.hidden:
                x = rand_feats(batch, hidden, dev, gen)
                ref = RefStack(hidden, args.n_blocks, args.dropout).to(dev)
                fused = FusedStack(hidden, args.n_blocks, args.dropout).to(dev)
                dense = DenseStack(hidden, args.n_blocks, args.dropout).to(dev)
                for i, blk in enumerate(ref.blocks):
                    for j, hc in enumerate(blk.layers):
                        fused.layers[2 * i + j].copy_from_hetero(hc, hidden)
                        dense.layers[2 * i + j].copy_from_hetero(hc, hidden)
                idx = fused_indices(batch, dev)
                x_all = torch.cat([x[nt] for nt in NODE_TYPES], 0)
                d_in, d_meta = dense_inputs(batch, x, args.grid, dev)

                # parity at full fp32 (TF32 off), dropout off
                for m in (ref, fused, dense):
                    m.eval()
                torch.set_float32_matmul_precision("highest")
                with torch.no_grad():
                    o_ref = ref({k: v.clone() for k, v in x.items()}, ei_dict)
                    o_ref = torch.cat([o_ref[nt] for nt in NODE_TYPES], 0)
                    o_fused = fused(x_all, idx)
                    Xa, Xb, Xg = dense(*d_in)
                    o_dense = torch.cat([Xa[d_in[5].squeeze(-1).bool()], Xb[d_in[6].squeeze(-1).bool()],
                                         Xg.view(-1, hidden)], 0)
                scale = o_ref.abs().max().item()
                err_f = (o_ref - o_fused).abs().max().item() / scale
                err_d = (o_ref - o_dense).abs().max().item() / scale
                assert err_f < 1e-4, f"fused parity failed: {err_f}"
                assert err_d < 1e-4, f"dense parity failed: {err_d}"
                torch.set_float32_matmul_precision("high")
                for m in (ref, fused, dense):
                    m.train()

                for dtype in args.dtypes:
                    amp = dtype == "bf16"

                    def run_ref():
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                            out = ref({k: v.clone() for k, v in x.items()}, ei_dict)
                            loss = sum(v.float().sum() for v in out.values())
                        loss.backward()

                    def run_fused():
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                            loss = fused(x_all, idx).float().sum()
                        loss.backward()

                    def make_dense_runner(mod):
                        def run():
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                                Xa, Xb, Xg = mod(*d_in)
                                loss = Xa.float().sum() + Xb.float().sum() + Xg.float().sum()
                            loss.backward()
                        return run

                    runners = {"ref": run_ref, "fused": run_fused, "dense": make_dense_runner(dense)}
                    if "dense_compile" in args.impls:
                        runners["dense_compile"] = make_dense_runner(torch.compile(dense, dynamic=False))
                    if "dense_cudagraph" in args.impls:
                        runners["dense_cudagraph"] = make_dense_runner(
                            torch.compile(dense, mode="reduce-overhead", dynamic=False))

                    for impl in args.impls:
                        key = (name, bs, hidden, dtype, impl)
                        if key in done:
                            continue
                        fn = runners[impl]
                        try:
                            t0 = time.perf_counter()
                            fn()
                            torch.cuda.synchronize()
                            first_ms = (time.perf_counter() - t0) * 1e3
                            mean, std, peak = timed(fn, warmup=5, reps=args.reps)
                            kps = kernel_count(fn)
                            err = None
                        except Exception as ex:  # OOM or compile failure: record and continue
                            mean = std = peak = kps = first_ms = float("nan")
                            err = repr(ex)[:200]
                            torch.cuda.empty_cache()
                        row = {
                            "dataset": name, "batch_size": bs, "hidden": hidden, "dtype": dtype,
                            "impl": impl, "fwd_bwd_ms": mean, "fwd_bwd_ms_std": std, "peak_gb": peak,
                            "kernels_per_step": kps, "first_call_ms": first_ms, "n_nodes": n_nodes,
                            "n_edges": n_edges, "dense_shape": d_meta, "parity_err_fused": err_f,
                            "parity_err_dense": err_d, "error": err,
                        }
                        rows.append(row)
                        print(f"{name:>8} bs={bs:<5} H={hidden:<4} {dtype:<4} {impl:<16} "
                              f"{mean:8.2f} ms  peak {peak:5.2f} GB  kernels {kps:7.0f}  "
                              f"(dense {d_meta['N_b']}x{d_meta['B_b']} waste {d_meta['pad_waste']:.2f})"
                              + (f"  ERROR {err}" if err else ""), flush=True)
                        with open(args.out, "w") as f:
                            json.dump(rows, f, indent=1)
                    torch._dynamo.reset()
                del ref, fused, dense, x, x_all, d_in
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
