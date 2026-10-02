#!/usr/bin/env python3
"""Geometric reference numbers for T3 bond classification on a graph LMDB.

Reports, for the candidate pool at each requested pool_multiplier:
  candidates/atom, positive fraction, candidate recall (upper bound for any
  model), and the k-sweep of the distance rule d <= k * (rcov_i + rcov_j)
  with the macro-F1-optimal k; plus the per-element-pair rule.

Optional --suite_csv (columns: mol_name, suite) stratifies every number by
suite via graph.mol_name, for the H1/H3/H6/H7/H8 hold-out evaluation.

Example:
    qtaim-embed-eval-bond-baselines --lmdb tests/data/lmdb_link/train \\
        --pool_multipliers 2.0 2.5 3.0 --out_dir ./bond_baselines/
"""

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Batch

from qtaim_embed.core.datamodule import _resolve_lmdb_path
from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.data.bonds import bond_pairs_from_heterograph, candidate_labels, candidate_recall
from qtaim_embed.data.lmdb import TransformMol
from qtaim_embed.models.encoders.neighbors import RCOV, candidate_pairs
from qtaim_embed.models.link_pred.baselines import (
    apply_pairwise_distance_rule,
    binary_scores,
    fit_distance_rule,
    fit_pairwise_distance_rule,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def collect_pairs(
    dataset: LMDBMoleculeDataset, pool_max: float, batch_size: int, max_graphs: Optional[int]
) -> dict:
    """Enumerate candidates at pool_max once; smaller pools are subsets."""
    n = len(dataset) if max_graphs is None else min(len(dataset), max_graphs)
    d_all, r_all, zi_all, zj_all, y_all, gid_all = [], [], [], [], [], []
    names: List[str] = []
    n_bonds, n_atoms = 0, 0
    recall_miss: Dict[float, int] = defaultdict(int)
    for start in range(0, n, batch_size):
        graphs = [dataset[k] for k in range(start, min(start + batch_size, n))]
        b = Batch.from_data_list(graphs)
        atom = b["atom"]
        N = int(atom.num_nodes)
        pairs = bond_pairs_from_heterograph(b)
        i, j, d = candidate_pairs(atom.pos, atom.z, atom.batch, RCOV, pool_multiplier=pool_max)
        r_ref = RCOV[atom.z[i]] + RCOV[atom.z[j]]
        y = candidate_labels(i, j, pairs, N)
        d_all.append(d); r_all.append(r_ref); zi_all.append(atom.z[i]); zj_all.append(atom.z[j]); y_all.append(y)
        gid_all.append(atom.batch[i] + start)
        names.extend(getattr(g, "mol_name", str(start + k)) for k, g in enumerate(graphs))
        n_bonds += int(pairs.shape[0]); n_atoms += N
    out = {
        "d": torch.cat(d_all), "r_ref": torch.cat(r_all), "z_i": torch.cat(zi_all),
        "z_j": torch.cat(zj_all), "y": torch.cat(y_all), "gid": torch.cat(gid_all),
        "n_bonds": n_bonds, "n_atoms": n_atoms, "n_graphs": n, "names": names,
    }
    return out


def summarize(
    pool: dict, multipliers: List[float], k_grid: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tuple[List[dict], dict]:
    """Per-multiplier pool statistics and best distance rule, plus the per-element-pair rule."""
    d, r, zi, zj, y = pool["d"], pool["r_ref"], pool["z_i"], pool["z_j"], pool["y"]
    if mask is not None:
        d, r, zi, zj, y = d[mask], r[mask], zi[mask], zj[mask], y[mask]
    total_pos = int(y.sum())
    rows = []
    for pm in multipliers:
        sel = d <= pm * r
        ds, rs, ys = d[sel], r[sel], y[sel]
        recall = float(ys.sum() / max(total_pos, 1))
        best_k, table = fit_distance_rule(ds, rs, ys, k_grid)
        s = table[best_k]
        rows.append({
            "pool_multiplier": pm, "candidates": int(sel.sum()), "positives": int(ys.sum()),
            "positive_fraction": float(ys.mean()) if ys.numel() else float("nan"),
            "candidate_recall": recall, "best_k": best_k, **{f"rule_{k}": v for k, v in s.as_dict().items()},
        })
    # per element-pair rule at the largest pool
    pm = max(multipliers)
    sel = d <= pm * r
    table, k_global = fit_pairwise_distance_rule(d[sel], r[sel], zi[sel], zj[sel], y[sel], k_grid)
    pred = apply_pairwise_distance_rule(d[sel], r[sel], zi[sel], zj[sel], table, k_global)
    pairwise = {"pool_multiplier": pm, "n_element_pairs": len(table), "k_global": k_global, **binary_scores(pred, y[sel]).as_dict()}
    return rows, pairwise


def sweep_table(pool: dict, pm: float, k_grid: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[dict]:
    """Distance-rule metrics for every k in k_grid at one pool multiplier."""
    d, r, y = pool["d"], pool["r_ref"], pool["y"]
    if mask is not None:
        d, r, y = d[mask], r[mask], y[mask]
    sel = d <= pm * r
    _, table = fit_distance_rule(d[sel], r[sel], y[sel], k_grid)
    return [{"k": k, **s.as_dict()} for k, s in table.items()]


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def md_table(rows: List[dict], cols: List[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for r in rows:
        body += "| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return head + body


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-config", type=str, default=None,
        help="training config JSON; supplies dataset.test_lmdb (or val/train) and model.pool_multiplier "
             "so baselines are computed on the same data and candidate pool as the model",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"], help="which -config split to use")
    p.add_argument("--lmdb", default=None, help="graph LMDB file or directory (shard dirs ok); overrides -config")
    p.add_argument("--out_dir", default="./bond_baselines/")
    p.add_argument("--pool_multipliers", type=float, nargs="+", default=[2.0, 2.5, 3.0])
    p.add_argument("--k_min", type=float, default=0.8)
    p.add_argument("--k_max", type=float, default=2.0)
    p.add_argument("--k_step", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_graphs", type=int, default=None)
    p.add_argument("--suite_csv", type=str, default=None, help="csv with columns mol_name,suite")
    args = p.parse_args(argv)
    if args.config is not None:
        with open(args.config) as f:
            cfg = json.load(f)
        if args.lmdb is None:
            args.lmdb = cfg["dataset"][f"{args.split}_lmdb"]
        pm = cfg.get("model", {}).get("pool_multiplier")
        if pm is not None and pm not in args.pool_multipliers:
            args.pool_multipliers = sorted(set(args.pool_multipliers + [float(pm)]))
    if args.lmdb is None:
        p.error("one of --lmdb or -config is required")

    os.makedirs(args.out_dir, exist_ok=True)
    ds = LMDBMoleculeDataset({"src": _resolve_lmdb_path(args.lmdb)}, transform=partial(TransformMol, dtype="float32"))
    k_grid = torch.arange(args.k_min, args.k_max + 1e-9, args.k_step)
    pool = collect_pairs(ds, max(args.pool_multipliers), args.batch_size, args.max_graphs)
    logger.info("graphs=%d atoms=%d bond_paths=%d candidates(at %.1fx)=%d", pool["n_graphs"], pool["n_atoms"], pool["n_bonds"], max(args.pool_multipliers), pool["d"].numel())

    rows, pairwise = summarize(pool, args.pool_multipliers, k_grid)
    sweep = sweep_table(pool, args.pool_multipliers[0], k_grid)
    write_csv(os.path.join(args.out_dir, "pool_summary.csv"), rows)
    write_csv(os.path.join(args.out_dir, f"k_sweep_pm{args.pool_multipliers[0]}.csv"), sweep)
    write_csv(os.path.join(args.out_dir, "pairwise_rule.csv"), [pairwise])

    md = [f"# Bond baselines: {args.lmdb}\n", f"graphs {pool['n_graphs']}, atoms {pool['n_atoms']}, QTAIM bond paths {pool['n_bonds']}\n",
          "## Candidate pool and best distance rule\n",
          md_table(rows, ["pool_multiplier", "candidates", "positive_fraction", "candidate_recall", "best_k", "rule_precision", "rule_recall", "rule_f1_pos", "rule_macro_f1"]),
          f"\n## Per element-pair rule at {pairwise['pool_multiplier']}x\n",
          md_table([pairwise], ["n_element_pairs", "k_global", "precision", "recall", "f1_pos", "macro_f1"])]

    if args.suite_csv:
        suite_of = {}
        with open(args.suite_csv) as f:
            for r in csv.DictReader(f):
                suite_of[r["mol_name"]] = r["suite"]
        gid_suite = [suite_of.get(str(n), "unassigned") for n in pool["names"]]
        suites = sorted(set(gid_suite))
        md.append("\n## Per suite\n")
        suite_rows = []
        for s in suites:
            keep = torch.tensor([gid_suite[g] == s for g in range(pool["n_graphs"])])
            mask = keep[pool["gid"]]
            if int(mask.sum()) == 0:
                continue
            srows, spair = summarize(pool, args.pool_multipliers, k_grid, mask)
            for r in srows:
                r["suite"] = s
            suite_rows.extend(srows)
        write_csv(os.path.join(args.out_dir, "pool_summary_by_suite.csv"), suite_rows)
        md.append(md_table(suite_rows, ["suite", "pool_multiplier", "candidates", "positive_fraction", "candidate_recall", "best_k", "rule_macro_f1"]))

    with open(os.path.join(args.out_dir, "README.md"), "w") as f:
        f.write("\n".join(md))
    for r in rows:
        logger.info("pm=%.1f cand/atom=%.2f pos_frac=%.3f recall=%.4f best_k=%.2f macro_f1=%.4f",
                    r["pool_multiplier"], r["candidates"] / pool["n_atoms"], r["positive_fraction"], r["candidate_recall"], r["best_k"], r["rule_macro_f1"])
    logger.info("pairwise rule: %d element pairs, macro_f1=%.4f", pairwise["n_element_pairs"], pairwise["macro_f1"])
    logger.info("wrote %s", args.out_dir)


if __name__ == "__main__":
    main()
