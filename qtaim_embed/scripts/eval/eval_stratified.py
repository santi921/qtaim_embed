"""Per-vertical test metrics for a node-level checkpoint.

`trainer.test` reports one number over the whole split. OMol4M mixes verticals
(omol, ani1xbb, 5A_elytes, electrolytes_reactivity, pdb_pockets, ...) whose
chemistry and label spread differ a lot, so the aggregate hides where a model
is actually weak. Graph names carry the vertical:

    5A_elytes__1002_Hg+2_5_group_1_shell_15_0_1
    omol__metal_organics__restart5to6__job_1741276959_dd078151ea78
    ani1xbb__aniBB_010_416098_0_1

Group key is the first `--group_depth` tokens of `mol_name` split on "__".

This reads the LMDB directly rather than going through DataLoaderLMDB, because
`collate_hetero_direct` drops `mol_name` (it keeps only the node tensors the
model reads). Batches are built here so names stay paired with their graphs.

Report MAE first. R2 is normalised by the within-group label variance, so a
vertical with a narrow charge range scores badly on R2 even when its MAE is
fine; that is a property of the group, not of the model.
"""

import argparse
import csv
import logging
import math
import os
import re
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch

from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.data.dataloader import collate_hetero_direct
from qtaim_embed.data.lmdb import TransformMol

logger = logging.getLogger(__name__)

SPLIT_RE = re.compile(r"__")


def group_key(mol_name: str, depth: int) -> str:
    """First `depth` "__"-delimited tokens, e.g. omol__metal_organics."""
    parts = SPLIT_RE.split(str(mol_name))
    return "__".join(parts[:depth]) if parts else "?"


class _Accum:
    """Streaming sums per (group, target): enough for MAE and R2 in one pass."""

    __slots__ = ("n", "sum_abs", "sum_y", "sum_y2", "sum_res2", "n_mols")

    def __init__(self, n_targets: int):
        z = torch.zeros(n_targets, dtype=torch.float64)
        self.n = 0
        self.sum_abs = z.clone()
        self.sum_y = z.clone()
        self.sum_y2 = z.clone()
        self.sum_res2 = z.clone()
        self.n_mols = 0

    def update(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        pred = pred.double()
        y = y.double()
        res = pred - y
        self.n += y.shape[0]
        self.sum_abs += res.abs().sum(0)
        self.sum_y += y.sum(0)
        self.sum_y2 += (y * y).sum(0)
        self.sum_res2 += (res * res).sum(0)

    def mae(self) -> torch.Tensor:
        return self.sum_abs / max(self.n, 1)

    def r2(self) -> torch.Tensor:
        if self.n < 2:
            return torch.full_like(self.sum_y, float("nan"))
        mean = self.sum_y / self.n
        ss_tot = self.sum_y2 - self.n * mean * mean
        out = 1.0 - self.sum_res2 / ss_tot
        # a constant target inside a group has zero variance: R2 undefined
        return torch.where(ss_tot > 1e-12, out, torch.full_like(out, float("nan")))


def _atom_targets(target_dict: dict) -> List[str]:
    names = target_dict.get("atom", []) or []
    names = [n for n in names if n is not None]
    others = {
        k: [n for n in (v or []) if n is not None]
        for k, v in target_dict.items()
        if k != "atom"
    }
    if any(others.values()):
        raise SystemExit(
            f"only atom targets are supported, found {others}; "
            "bond/global targets would need the padded concat shared_step does"
        )
    if not names:
        raise SystemExit("checkpoint has no atom targets")
    return names


@torch.no_grad()
def evaluate(
    ckpt: str,
    lmdb_src: str,
    group_depth: int = 1,
    batch_size: int = 256,
    device: str = "cuda",
    limit: Optional[int] = None,
    stride: int = 1,
    dtype: str = "float32",
) -> Dict[str, _Accum]:
    from qtaim_embed.models.node_level.base_gcn import GCNNodePred

    # weights_only=False: hparams hold an instantiated activation module
    model = GCNNodePred.load_from_checkpoint(ckpt, weights_only=False, map_location="cpu")
    model.eval()
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable, falling back to cpu")
        device = "cpu"
    model.to(device)

    targets = _atom_targets(dict(model.hparams.target_dict))
    logger.info("targets: %s", targets)

    ds = LMDBMoleculeDataset(
        config={"src": lmdb_src}, transform=partial(TransformMol, dtype=dtype)
    )
    # records are stored grouped by source, so the first N graphs are not a
    # representative sample: stride spreads a subsample across the whole file
    indices = list(range(0, len(ds), max(1, stride)))
    if limit is not None:
        indices = indices[:limit]
    logger.info(
        "scoring %d of %d graphs (stride %d) from %s",
        len(indices), len(ds), stride, lmdb_src,
    )

    acc: Dict[str, _Accum] = defaultdict(lambda: _Accum(len(targets)))
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        graphs = [ds[i] for i in idx]
        keys = [group_key(g.mol_name, group_depth) for g in graphs]
        batch = collate_hetero_direct(graphs).to(device)

        feats = {nt: batch[nt].feat for nt in batch.node_types}
        pred = model(batch, feats)["atom"]
        y = batch["atom"].labels
        owner = batch["atom"].batch  # atom -> index within this chunk

        for gi, key in enumerate(keys):
            mask = owner == gi
            if not bool(mask.any()):
                continue
            a = acc[key]
            a.update(pred[mask].cpu(), y[mask].cpu())
            a.n_mols += 1

        if start and start % (batch_size * 50) == 0:
            logger.info("  %d / %d", start, len(indices))
    return dict(acc), targets


def _short_names(targets: Sequence[str]) -> List[str]:
    """Drop a common prefix so the per-target columns fit (charge_adch -> adch)."""
    if len(targets) > 1:
        head = os.path.commonprefix(list(targets))
        head = head[: head.rfind("_") + 1] if "_" in head else ""
    else:
        head = ""
    return [t[len(head):][:9] for t in targets]


def _write(acc: Dict[str, _Accum], targets: Sequence[str], out_dir: str, min_group: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for key, a in sorted(acc.items(), key=lambda kv: -kv[1].n_mols):
        mae, r2 = a.mae(), a.r2()
        rows.append(
            {
                "group": key,
                "n_molecules": a.n_mols,
                "n_atoms": a.n,
                "mae_mean": float(mae.mean()),
                "r2_mean": float(torch.nanmean(r2)),
                **{f"mae_{t}": float(mae[i]) for i, t in enumerate(targets)},
                **{f"r2_{t}": float(r2[i]) for i, t in enumerate(targets)},
            }
        )
    csv_path = os.path.join(out_dir, "stratified.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'group':34s} {'mols':>8s} {'atoms':>9s} {'MAE':>8s} {'R2(mean)':>9s}")
    for r in rows:
        flag = "" if r["n_molecules"] >= min_group else "  (small)"
        print(
            f"{r['group'][:34]:34s} {r['n_molecules']:>8d} {r['n_atoms']:>9d} "
            f"{r['mae_mean']:>8.4f} {r['r2_mean']:>9.4g}{flag}"
        )

    # the aggregate hides which charge definition is failing; these two are the
    # point of the tool, and the mean over six targets is diluted by the three
    # that every model already saturates
    short = _short_names(targets)
    for metric, label in (("mae", "MAE"), ("r2", "R2")):
        print(f"\nper-target {label}")
        print(f"{'group':34s} " + " ".join(f"{n:>9s}" for n in short))
        for r in rows:
            fmt = ".4f" if metric == "mae" else ".4g"
            cells = " ".join(f"{r[f'{metric}_{t}']:>9{fmt}}" for t in targets)
            print(f"{r['group'][:34]:34s} {cells}")

    print(f"\nwrote {csv_path}")
    return csv_path


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, help="node-level checkpoint to score")
    p.add_argument("--lmdb", required=True, help="test LMDB (dir or merged.lmdb)")
    p.add_argument("--group_depth", type=int, default=1,
                   help='"__"-delimited tokens of mol_name to group on (2 splits omol into its sub-verticals)')
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="score at most N graphs")
    p.add_argument("--stride", type=int, default=1,
                   help="score every Nth graph; records are grouped by source, so a "
                        "plain --limit over-samples whatever sits at the start of the file")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--min_group", type=int, default=100,
                   help="groups smaller than this are flagged, not dropped")
    p.add_argument("--out_dir", default="./stratified_eval")
    args = p.parse_args(argv)

    acc, targets = evaluate(
        ckpt=args.ckpt,
        lmdb_src=args.lmdb,
        group_depth=args.group_depth,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        stride=args.stride,
        dtype=args.dtype,
    )
    _write(acc, targets, args.out_dir, args.min_group)
    return None


if __name__ == "__main__":
    main()
