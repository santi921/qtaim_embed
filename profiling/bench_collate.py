#!/usr/bin/env python3
"""E5: data-path costs per batch on one CPU thread (what a DataLoader worker pays).

  collate   Batch.from_data_list vs a direct collate that concatenates the known
            stores and offsets edge_index (no generic PyG introspection)
  deser     torch.load of the pickled HeteroData record vs a flat tensor dict
            loaded with weights_only=True (schema 2 candidate)

Usage:
  python profiling/bench_collate.py --dataset tm_react --batches 128 512 1024
"""

import argparse
import io
import json
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData

DATASETS = {
    "tm_react": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/splits/tm_react/train",
    "h7": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/holdouts/H7",
}
NODE_KEYS = ("feat", "labels", "pos", "z")


def collate_direct(graphs):
    """Batch HeteroData graphs of a fixed schema without PyG's generic collate."""
    out = HeteroData()
    g0 = graphs[0]
    G = len(graphs)
    ptr = {}
    for nt in g0.node_types:
        stores = [g[nt] for g in graphs]
        counts = torch.tensor([s.num_nodes for s in stores])
        cum = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        ptr[nt] = cum
        for key in NODE_KEYS:
            if key in stores[0]:
                out[nt][key] = torch.cat([s[key] for s in stores], 0)
        out[nt].num_nodes = int(cum[-1])
        out[nt].batch = torch.repeat_interleave(torch.arange(G), counts)
        out[nt].ptr = cum
    for et in g0.edge_types:
        src_t, _, dst_t = et
        eis = [g[et].edge_index for g in graphs]
        n_e = torch.tensor([e.shape[1] for e in eis])
        off = torch.stack([ptr[src_t][:-1], ptr[dst_t][:-1]])  # [2, G]
        out[et].edge_index = torch.cat(eis, 1) + torch.repeat_interleave(off, n_e, dim=1)
    out.num_graphs = G
    return out


def check_equal(a, b):
    for nt in a.node_types:
        for key in NODE_KEYS:
            if key in a[nt]:
                assert torch.equal(a[nt][key], b[nt][key]), (nt, key)
        assert torch.equal(a[nt].batch, b[nt].batch), nt
    for et in a.edge_types:
        assert torch.equal(a[et].edge_index, b[et].edge_index), et


def to_tensor_dict(g):
    d = {"__schema__": 2, "node_types": list(g.node_types), "edge_types": [list(e) for e in g.edge_types]}
    for nt in g.node_types:
        d[f"{nt}.num_nodes"] = int(g[nt].num_nodes)
        for key in NODE_KEYS:
            if key in g[nt]:
                d[f"{nt}.{key}"] = g[nt][key]
    for et in g.edge_types:
        d["|".join(et) + ".edge_index"] = g[et].edge_index
    return d


def from_tensor_dict(d):
    g = HeteroData()
    for nt in d["node_types"]:
        g[nt].num_nodes = d[f"{nt}.num_nodes"]
        for key in NODE_KEYS:
            if f"{nt}.{key}" in d:
                g[nt][key] = d[f"{nt}.{key}"]
    for et in d["edge_types"]:
        g[tuple(et)].edge_index = d["|".join(et) + ".edge_index"]
    return g


def timeit(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return statistics.fmean(t), statistics.pstdev(t)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tm_react")
    p.add_argument("--batches", nargs="+", type=int, default=[128, 512, 1024])
    p.add_argument("--n_records", type=int, default=1000)
    p.add_argument("--out", default="profiling/bench_results/collate_e5.json")
    args = p.parse_args(argv)
    torch.set_num_threads(1)

    from qtaim_embed.core.dataset import LMDBMoleculeDataset
    from qtaim_embed.data.lmdb import TransformMol, load_graph_from_serialized, serialize_graph

    ds = LMDBMoleculeDataset(config={"src": DATASETS[args.dataset]},
                             transform=partial(TransformMol, dtype="float32"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), max(args.batches), replace=False)
    graphs = [ds[int(i)] for i in idx]

    result = {"dataset": args.dataset, "collate": [], "deser": {}}
    for bs in args.batches:
        gs = graphs[:bs]
        check_equal(Batch.from_data_list(gs), collate_direct(gs))
        pyg = timeit(lambda: Batch.from_data_list(gs), reps=5)
        direct = timeit(lambda: collate_direct(gs), reps=5)
        row = {"batch_size": bs, "pyg_ms": pyg[0], "direct_ms": direct[0], "speedup": pyg[0] / direct[0]}
        result["collate"].append(row)
        print(f"collate bs={bs:<5} Batch.from_data_list {pyg[0]:8.1f} ms | direct {direct[0]:7.1f} ms | {row['speedup']:.1f}x")

    # deserialization: current pickled HeteroData vs flat tensor dict
    recs = graphs[:args.n_records]
    blobs_hd = [serialize_graph(g) for g in recs]
    blobs_td = []
    for g in recs:
        buf = io.BytesIO()
        torch.save(to_tensor_dict(g), buf)
        blobs_td.append(buf.getvalue())
    g_rt = from_tensor_dict(torch.load(io.BytesIO(blobs_td[0]), weights_only=True))
    check_equal(Batch.from_data_list([recs[0]]), Batch.from_data_list([g_rt]))

    def load_hd():
        for b in blobs_hd:
            load_graph_from_serialized(b)

    def load_td():
        for b in blobs_td:
            from_tensor_dict(torch.load(io.BytesIO(b), weights_only=True))

    hd = timeit(load_hd, reps=3)
    td = timeit(load_td, reps=3)
    result["deser"] = {
        "n_records": len(recs),
        "heterodata_ms_per_record": hd[0] / len(recs),
        "tensor_dict_ms_per_record": td[0] / len(recs),
        "speedup": hd[0] / td[0],
        "heterodata_kb": float(np.mean([len(b) for b in blobs_hd]) / 1024),
        "tensor_dict_kb": float(np.mean([len(b) for b in blobs_td]) / 1024),
    }
    d = result["deser"]
    print(f"deser   HeteroData torch.load {d['heterodata_ms_per_record']:.3f} ms/rec ({d['heterodata_kb']:.0f} KB) | "
          f"tensor dict weights_only {d['tensor_dict_ms_per_record']:.3f} ms/rec ({d['tensor_dict_kb']:.0f} KB) | {d['speedup']:.1f}x")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
