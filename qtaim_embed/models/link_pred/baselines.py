"""Geometric reference rules for bond classification (T3).

The learned classifier is always reported next to these. Numbers here must be
comparable to qtaim_generator's bond_agreement.py, so the covalent radii come
from the same RDKit table (models/encoders/neighbors.RCOV).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def scores_from_counts(
    tp: torch.Tensor, fp: torch.Tensor, tn: torch.Tensor, fn: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Precision, recall, F1 (pos and neg), macro-F1 and accuracy from counts.

    Vectorised over any leading shape, so it serves both the scalar
    BinaryScores below and the per-threshold grid in
    models/link_pred/bond_model.BinnedBinaryStats. Zero-division convention:
    an undefined precision or recall is 0 (denominators clamped to 1), so a
    class with no predictions and no members contributes F1 = 0 to macro-F1.
    """
    tp, fp, tn, fn = (x.to(torch.float64) for x in (tp, fp, tn, fn))
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1_pos = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    p_neg = tn / (tn + fn).clamp_min(1)
    r_neg = tn / (tn + fp).clamp_min(1)
    f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg).clamp_min(1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "macro_f1": 0.5 * (f1_pos + f1_neg),
        "accuracy": (tp + tn) / (tp + fp + tn + fn).clamp_min(1),
    }


@dataclass
class BinaryScores:
    """Confusion counts for one binary decision, with derived metrics.

    Metric definitions and the zero-division convention are those of
    scores_from_counts; macro-F1 is the unweighted mean of the positive-class
    and negative-class F1, matching qtaim_generator's bond_agreement.py.
    """

    tp: int
    fp: int
    tn: int
    fn: int

    def _scores(self) -> Dict[str, float]:
        out = scores_from_counts(*(torch.tensor(x) for x in (self.tp, self.fp, self.tn, self.fn)))
        return {k: float(v) for k, v in out.items()}

    @property
    def precision(self) -> float:
        return self._scores()["precision"]

    @property
    def recall(self) -> float:
        return self._scores()["recall"]

    @property
    def f1_pos(self) -> float:
        return self._scores()["f1_pos"]

    @property
    def f1_neg(self) -> float:
        return self._scores()["f1_neg"]

    @property
    def macro_f1(self) -> float:
        return self._scores()["macro_f1"]

    @property
    def accuracy(self) -> float:
        return self._scores()["accuracy"]

    def as_dict(self) -> Dict[str, float]:
        out = self._scores()
        out.pop("f1_neg")
        out.update(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)
        return out


def binary_scores(pred: torch.Tensor, y: torch.Tensor) -> BinaryScores:
    """Confusion counts of a boolean prediction against boolean labels."""
    pred = pred.bool()
    y = y.bool()
    return BinaryScores(
        tp=int((pred & y).sum()),
        fp=int((pred & ~y).sum()),
        tn=int((~pred & ~y).sum()),
        fn=int((~pred & y).sum()),
    )


def distance_rule(d: torch.Tensor, r_ref: torch.Tensor, k: float) -> torch.Tensor:
    """d <= k * (rcov_i + rcov_j)."""
    return d <= k * r_ref


def fit_distance_rule(
    d: torch.Tensor,
    r_ref: torch.Tensor,
    y: torch.Tensor,
    k_grid: Optional[torch.Tensor] = None,
) -> Tuple[float, Dict[float, BinaryScores]]:
    """Sweep k and return (best_k by macro-F1, {k: scores})."""
    if k_grid is None:
        k_grid = torch.arange(0.80, 2.001, 0.01)
    ratio = d / r_ref.clamp_min(1e-6)
    table: Dict[float, BinaryScores] = {}
    best_k, best = None, -1.0
    for k in k_grid.tolist():
        k = round(k, 4)
        s = binary_scores(ratio <= k, y)
        table[k] = s
        if s.macro_f1 > best:
            best, best_k = s.macro_f1, k
    return best_k, table


def fit_pairwise_distance_rule(
    d: torch.Tensor,
    r_ref: torch.Tensor,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    y: torch.Tensor,
    k_grid: Optional[torch.Tensor] = None,
    min_count: int = 20,
) -> Tuple[Dict[Tuple[int, int], float], float]:
    """Per element-pair k, falling back to the global k for rare pairs.

    Returns ({(z_lo, z_hi): k}, k_global).
    """
    k_global, _ = fit_distance_rule(d, r_ref, y, k_grid)
    z_lo = torch.minimum(z_i, z_j)
    z_hi = torch.maximum(z_i, z_j)
    key = z_lo * 200 + z_hi
    table: Dict[Tuple[int, int], float] = {}
    for kk in torch.unique(key).tolist():
        m = key == kk
        if int(m.sum()) < min_count:
            continue
        yy = y[m]
        if bool(yy.all()) or not bool(yy.any()):
            continue
        k_best, _ = fit_distance_rule(d[m], r_ref[m], yy, k_grid)
        table[(kk // 200, kk % 200)] = k_best
    return table, k_global


def apply_pairwise_distance_rule(
    d: torch.Tensor,
    r_ref: torch.Tensor,
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    table: Dict[Tuple[int, int], float],
    k_global: float,
) -> torch.Tensor:
    """d <= k(z_i, z_j) * r_ref with per element-pair k, k_global as fallback."""
    k = torch.full_like(d, float(k_global))
    z_lo = torch.minimum(z_i, z_j)
    z_hi = torch.maximum(z_i, z_j)
    for (a, b), kv in table.items():
        k[(z_lo == a) & (z_hi == b)] = float(kv)
    return d <= k * r_ref
