#!/usr/bin/env python3
"""E6: radius-graph construction strategies on real batched molecules.

Compares, on GPU and on a single CPU thread (what a DataLoader worker pays):
  chunked   the pre-A5-1 builder (_radius_neighbors_chunked: cdist of 256-row
            chunks against the whole batch, one nonzero per chunk)
  dense     per-molecule cdist on to_dense_batch positions, one mask, one nonzero
Edge sets are asserted identical. Also times build_triplets on the result for
two dimenetpp settings so E2 has a reference for the triplet cost.

Usage:
  CUDA_VISIBLE_DEVICES=0 python profiling/bench_neighbors.py
"""

import argparse
import json
import math
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from qtaim_embed.models.encoders.neighbors import _CHUNK, _radius_neighbors_chunked, build_triplets, radius_neighbors


def radius_neighbors_chunked(pos, batch, cutoff):
    """The pre-A5-1 batch-wide chunked cdist; radius_neighbors itself now dispatches to the dense path."""
    edge_index = _radius_neighbors_chunked(pos, batch, cutoff, _CHUNK)
    return edge_index, (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

DATASETS = {
    "tm_react": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/splits/tm_react/train",
    "h7": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/holdouts/H7",
}


def radius_neighbors_dense(pos, batch, cutoff):
    """Per-molecule dense distance matrix; same output convention as radius_neighbors."""
    dense, mask = to_dense_batch(pos, batch)  # [G, N_b, 3], [G, N_b]
    d = torch.cdist(dense, dense)
    keep = (d <= cutoff) & mask[:, :, None] & mask[:, None, :]
    n_b = dense.shape[1]
    keep &= ~torch.eye(n_b, dtype=torch.bool, device=pos.device)[None]
    g, i, j = keep.nonzero(as_tuple=True)
    counts = mask.sum(1)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    base = ptr[g]
    edge_index = torch.stack([base + j, base + i])  # [source j, target i]
    d_ij = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
    return edge_index, d_ij


def _cap_neighbors(edge_index, d, max_n):
    dst = edge_index[1]
    order_d = torch.argsort(d, stable=True)
    order = order_d[torch.argsort(dst[order_d], stable=True)]
    dst_sorted = dst[order]
    counts = torch.bincount(dst_sorted)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)])[:-1]
    rank = torch.arange(dst_sorted.numel(), device=d.device) - starts[dst_sorted]
    keep = order[rank < max_n]
    return edge_index[:, keep], d[keep]


def canonical(edge_index):
    n = int(edge_index.max()) + 1
    key = edge_index[1].to(torch.int64) * n + edge_index[0].to(torch.int64)
    return torch.sort(key).values


def time_gpu(fn, reps=20):
    for _ in range(3):
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


def time_cpu(fn, reps=5):
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.fmean(times), statistics.pstdev(times)


def count_nonzero_calls(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ka = {e.key: e for e in prof.key_averages()}
    return ka["aten::nonzero"].count if "aten::nonzero" in ka else 0


def load_batches(src, batch_sizes, n_per=2, seed=0):
    from qtaim_embed.core.dataset import LMDBMoleculeDataset
    from qtaim_embed.data.lmdb import TransformMol

    ds = LMDBMoleculeDataset(config={"src": src}, transform=partial(TransformMol, dtype="float32"))
    rng = np.random.default_rng(seed)
    need = max(batch_sizes) * n_per
    idx = rng.choice(len(ds), min(need, len(ds)), replace=False)
    graphs = [ds[int(i)] for i in idx]
    out = {}
    for bs in batch_sizes:
        out[bs] = [Batch.from_data_list(graphs[k * bs:(k + 1) * bs]) for k in range(n_per)]
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["tm_react", "h7"])
    p.add_argument("--tm_react_batches", nargs="+", type=int, default=[128, 512, 1024])
    p.add_argument("--h7_batches", nargs="+", type=int, default=[32, 128])
    p.add_argument("--cutoffs", nargs="+", type=float, default=[4.0, 5.0])
    p.add_argument("--out_dir", default="profiling/bench_results")
    args = p.parse_args(argv)

    torch.set_num_threads(1)  # CPU timings emulate one DataLoader worker
    dev = torch.device("cuda")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    for name in args.datasets:
        bss = args.tm_react_batches if name == "tm_react" else args.h7_batches
        batches = load_batches(DATASETS[name], bss)
        for bs in bss:
            b = batches[bs][0]
            pos_c, batch_c = b["atom"].pos.float(), b["atom"].batch
            pos, batch = pos_c.to(dev), batch_c.to(dev)
            n = pos.shape[0]
            n_b = int(torch.bincount(batch).max())
            for cutoff in args.cutoffs:
                ei_ref, d_ref = radius_neighbors_chunked(pos, batch, cutoff)
                ei_dn, d_dn = radius_neighbors_dense(pos, batch, cutoff)
                same = torch.equal(canonical(ei_ref), canonical(ei_dn))
                assert same, f"edge set mismatch {name} bs={bs} cutoff={cutoff}"
                assert torch.allclose(torch.sort(d_ref).values, torch.sort(d_dn).values, atol=1e-5)

                g_chunk = time_gpu(lambda: radius_neighbors_chunked(pos, batch, cutoff))
                g_dense = time_gpu(lambda: radius_neighbors_dense(pos, batch, cutoff))
                c_chunk = time_cpu(lambda: radius_neighbors_chunked(pos_c, batch_c, cutoff))
                c_dense = time_cpu(lambda: radius_neighbors_dense(pos_c, batch_c, cutoff))
                nz_chunk = count_nonzero_calls(lambda: radius_neighbors_chunked(pos, batch, cutoff))
                nz_dense = count_nonzero_calls(lambda: radius_neighbors_dense(pos, batch, cutoff))

                trip = {}
                for max_n in (16, 32):
                    ei_c, d_c = _cap_neighbors(ei_ref, d_ref, max_n)
                    t_ms, _, t_gb = time_gpu(lambda: build_triplets(ei_c, n), reps=10)
                    n_trip = int(build_triplets(ei_c, n)[0].numel())
                    trip[f"triplets_maxn{max_n}_ms"] = t_ms
                    trip[f"triplets_maxn{max_n}_count"] = n_trip
                    trip[f"triplets_maxn{max_n}_peak_gb"] = t_gb
                    trip[f"edges_maxn{max_n}"] = int(ei_c.shape[1])

                row = {
                    "dataset": name, "batch_size": bs, "cutoff": cutoff, "atoms": n,
                    "max_atoms_per_mol": n_b, "edges": int(ei_ref.shape[1]),
                    "mean_degree": float(ei_ref.shape[1] / n),
                    "chunked_gpu_ms": g_chunk[0], "chunked_gpu_peak_gb": g_chunk[2],
                    "dense_gpu_ms": g_dense[0], "dense_gpu_peak_gb": g_dense[2],
                    "chunked_cpu1_ms": c_chunk[0], "dense_cpu1_ms": c_dense[0],
                    "chunked_nonzero_calls": nz_chunk, "dense_nonzero_calls": nz_dense,
                    "distance_pairs_chunked": n * n, "distance_pairs_dense": int(len(torch.unique(batch)) * n_b * n_b),
                    "edge_sets_identical": bool(same),
                    **trip,
                }
                rows.append(row)
                print(f"{name:>9} bs={bs:<5} cutoff={cutoff} atoms={n:<6} E={row['edges']:<7} "
                      f"chunked {g_chunk[0]:7.2f} ms ({nz_chunk} nonzero) | dense {g_dense[0]:7.2f} ms "
                      f"({nz_dense} nonzero) | cpu1 chunked {c_chunk[0]:7.1f} ms dense {c_dense[0]:7.1f} ms | "
                      f"triplets maxn32 {trip['triplets_maxn32_count']:,} in {trip['triplets_maxn32_ms']:.1f} ms",
                      flush=True)

    out = Path(args.out_dir) / "neighbors_e6.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
