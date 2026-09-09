#!/usr/bin/env python3
"""E4: padding waste of fixed-shape bucketing schemes on real size histograms.

For each dataset, samples atom and bond counts, then reports for k buckets
(quantile edges on atom count, per-bucket ceilings = max atoms / max bonds in
the bucket) the padding waste on atoms, bonds, and both, plus the fraction of
graphs left in ragged last batches for a given batch size. Also reports the
waste floor of "sort by size, pad to the per-batch max" (dynamic shapes) for
comparison.

Usage:
  python profiling/bench_bucketing.py --datasets tm_react h7 tmqm --n 6000
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np

DATASETS = {
    "tm_react": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/splits/tm_react/train",
    "h7": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/holdouts/H7",
    "tmqm": "data/lmdb/tmqm_train/molecule.lmdb",
    "droplet": "/home/santiagovargas/dev/qtaim_generator/data/graphs_posz_local/splits/droplet/train",
}


def sample_counts(src: str, n: int, seed: int = 0):
    from qtaim_embed.core.dataset import LMDBMoleculeDataset
    from qtaim_embed.data.lmdb import TransformMol

    ds = LMDBMoleculeDataset(config={"src": src}, transform=partial(TransformMol, dtype="float32"))
    idx = np.random.default_rng(seed).choice(len(ds), min(n, len(ds)), replace=False)
    atoms = np.empty(len(idx), dtype=np.int64)
    bonds = np.empty(len(idx), dtype=np.int64)
    for j, i in enumerate(idx):
        g = ds[int(i)]
        atoms[j] = g["atom"].num_nodes
        bonds[j] = g["bond"].num_nodes
    return atoms, bonds, len(ds)


def bucket_scheme(atoms, bonds, k):
    """Quantile edges on atoms; ceilings are the per-bucket maxima."""
    qs = np.linspace(0, 1, k + 1)[1:-1]
    edges = np.quantile(atoms, qs) if k > 1 else np.array([])
    bucket = np.searchsorted(edges, atoms, side="left")
    ceil_a = np.zeros(k, dtype=np.int64)
    ceil_b = np.zeros(k, dtype=np.int64)
    sizes = np.zeros(k, dtype=np.int64)
    for b in range(k):
        m = bucket == b
        sizes[b] = m.sum()
        if sizes[b]:
            ceil_a[b] = atoms[m].max()
            ceil_b[b] = bonds[m].max()
    return bucket, ceil_a, ceil_b, sizes


def waste(actual, padded):
    return 1.0 - actual.sum() / padded.sum()


def grid_scheme(atoms, bonds, step):
    """Round atom and bond counts up to multiples of step; each distinct
    (ceil_atoms, ceil_bonds) pair is one static shape."""
    ca = np.ceil(atoms / step).astype(np.int64) * step
    cb = np.ceil(bonds / step).astype(np.int64) * step
    shapes = np.unique(np.stack([ca, cb], 1), axis=0)
    return ca, cb, len(shapes)


def analyze(atoms, bonds, ks, batch_sizes, n_total=None, grid_steps=(8, 16, 32)):
    n_total = n_total or len(atoms)
    scale = n_total / len(atoms)
    out = []
    for k in ks:
        bucket, ceil_a, ceil_b, sizes = bucket_scheme(atoms, bonds, k)
        pad_a = ceil_a[bucket].astype(np.float64)
        pad_b = ceil_b[bucket].astype(np.float64)
        row = {
            "k": k,
            "ceil_atoms": ceil_a.tolist(),
            "ceil_bonds": ceil_b.tolist(),
            "bucket_frac": (sizes / sizes.sum()).round(3).tolist(),
            "waste_atoms": waste(atoms, pad_a),
            "waste_bonds": waste(bonds, pad_b),
            "waste_total": waste(atoms + bonds, pad_a + pad_b),
        }
        for bs in batch_sizes:
            # graphs that do not fill a complete fixed-size batch inside their bucket
            full = np.round(sizes * scale)
            row[f"ragged_frac_b{bs}"] = float(np.sum(full % bs) / max(full.sum(), 1))
        out.append(row)

    for step in grid_steps:
        ca, cb, n_shapes = grid_scheme(atoms, bonds, step)
        out.append({
            "k": f"grid{step}", "n_shapes": int(n_shapes),
            "waste_atoms": waste(atoms, ca.astype(np.float64)),
            "waste_bonds": waste(bonds, cb.astype(np.float64)),
            "waste_total": waste(atoms + bonds, (ca + cb).astype(np.float64)),
            "ceil_atoms": [int(step)],
        })
        for bs in batch_sizes:
            # graphs per (ceil_atoms, ceil_bonds) shape, scaled to the full set
            _, counts = np.unique(np.stack([ca, cb], 1), axis=0, return_counts=True)
            full = np.round(counts * scale)
            out[-1][f"ragged_frac_b{bs}"] = float(np.sum(full % bs) / max(full.sum(), 1))

    # dynamic-shape floor: sort by atoms, pad each batch of bs to its max
    order = np.argsort(atoms)
    floor = {}
    for bs in batch_sizes:
        pa, pb = [], []
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            pa.append(np.full(len(sel), atoms[sel].max(), dtype=np.float64))
            pb.append(np.full(len(sel), bonds[sel].max(), dtype=np.float64))
        pa, pb = np.concatenate(pa), np.concatenate(pb)
        floor[f"sorted_waste_total_b{bs}"] = waste(atoms[order] + bonds[order], pa + pb)
    # random batches padded to their max (what naive padding without sorting costs)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(atoms))
    for bs in batch_sizes:
        pa, pb = [], []
        for s in range(0, len(perm), bs):
            sel = perm[s:s + bs]
            pa.append(np.full(len(sel), atoms[sel].max(), dtype=np.float64))
            pb.append(np.full(len(sel), bonds[sel].max(), dtype=np.float64))
        pa, pb = np.concatenate(pa), np.concatenate(pb)
        floor[f"random_waste_total_b{bs}"] = waste(atoms[perm] + bonds[perm], pa + pb)
    return out, floor


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["tm_react", "h7", "tmqm"])
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[256, 512, 1024])
    p.add_argument("--out_dir", default="profiling/bench_results")
    args = p.parse_args(argv)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        t0 = time.perf_counter()
        atoms, bonds, n_total = sample_counts(DATASETS[name], args.n)
        rows, floor = analyze(atoms, bonds, args.ks, args.batch_sizes, n_total=n_total)
        result = {
            "dataset": name,
            "src": DATASETS[name],
            "n_sampled": int(len(atoms)),
            "n_total": n_total,
            "atoms": {"mean": float(atoms.mean()), "p50": float(np.median(atoms)),
                      "p90": float(np.percentile(atoms, 90)), "max": int(atoms.max())},
            "bonds": {"mean": float(bonds.mean()), "p50": float(np.median(bonds)),
                      "p90": float(np.percentile(bonds, 90)), "max": int(bonds.max())},
            "buckets": rows,
            "floors": floor,
        }
        out = Path(args.out_dir) / f"bucketing_{name}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=1)

        print(f"\n== {name}: n={len(atoms)} of {n_total}, atoms mean {atoms.mean():.1f} max {atoms.max()}, "
              f"bonds mean {bonds.mean():.1f} max {bonds.max()}  ({time.perf_counter() - t0:.0f}s)")
        print(f"{'k':>6} {'waste_atoms':>12} {'waste_bonds':>12} {'waste_total':>12} "
              + " ".join(f"{'ragged_b' + str(b):>12}" for b in args.batch_sizes) + "  ceil_atoms")
        for r in rows:
            print(f"{str(r['k']):>6} {r['waste_atoms']:>12.3f} {r['waste_bonds']:>12.3f} {r['waste_total']:>12.3f} "
                  + " ".join(f"{r[f'ragged_frac_b{b}']:>12.3f}" for b in args.batch_sizes)
                  + f"  {r['ceil_atoms']}" + (f" shapes={r['n_shapes']}" if 'n_shapes' in r else ""))
        for k, v in floor.items():
            print(f"  {k}: {v:.3f}")
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
