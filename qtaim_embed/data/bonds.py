"""Bond-pair extraction and candidate labelling for bond classification (T3).

Labels are derived on the fly from the heterograph's atom->bond connectivity,
so no candidate or label tensors are stored in the LMDBs. Everything here is
pure torch and works on batched HeteroData (atom indices are already offset
by Batch.from_data_list).
"""

from typing import Tuple

import torch
from torch_geometric.data import HeteroData


def bond_pairs_from_heterograph(graph: HeteroData) -> torch.Tensor:
    """Atom-index pairs for every bond node, sorted so pair[:, 0] < pair[:, 1].

    Uses the (atom, a2b, bond) edge_index: a real bond node has exactly two
    incoming atom edges. Zero-bond molecules carry a placeholder bond node with
    a single a2b edge (grapher.build_hetero_graph_skeleton); those are dropped.
    Returns a (B, 2) int64 tensor; (0, 2) when there are no real bonds.
    """
    ei = graph["atom", "a2b", "bond"].edge_index
    if ei.numel() == 0:
        return ei.new_zeros((0, 2))
    atom, bond = ei[0], ei[1]
    num_bonds = int(graph["bond"].num_nodes)
    if ei.shape[1] == 2 * num_bonds:
        # Fast path, no host sync: the grapher emits a2b as [u, v] per bond in
        # bond order and Batch.from_data_list preserves that order.
        pairs = atom.view(-1, 2)
    else:
        # Placeholder bond nodes (single a2b edge) or non-canonical order.
        counts = torch.bincount(bond, minlength=num_bonds)
        if bool((counts > 2).any()):
            raise ValueError(
                f"a bond node has more than two a2b edges (max={int(counts.max())})"
            )
        keep = counts[bond] == 2
        atom, bond = atom[keep], bond[keep]
        if atom.numel() == 0:
            return ei.new_zeros((0, 2))
        order = torch.argsort(bond, stable=True)
        pairs = atom[order].view(-1, 2)
    lo = torch.minimum(pairs[:, 0], pairs[:, 1])
    hi = torch.maximum(pairs[:, 0], pairs[:, 1])
    pairs = torch.stack([lo, hi], dim=1)
    # self-bonds (i == i) can appear when graphs are built with self_loop=True
    return pairs[lo != hi]


def pair_keys(i: torch.Tensor, j: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Unique int64 key per unordered pair; callers pass i < j."""
    return i.to(torch.int64) * int(num_nodes) + j.to(torch.int64)


def candidate_labels(
    i: torch.Tensor, j: torch.Tensor, bond_pairs: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Binary label per candidate pair: 1 if (i, j) is a bond pair.

    Args:
        i, j: (P,) candidate endpoints with i < j (as from candidate_pairs).
        bond_pairs: (B, 2) sorted atom pairs from bond_pairs_from_heterograph.
        num_nodes: total atom count in the (batched) graph.

    Returns:
        (P,) float32 tensor in {0, 1}.
    """
    if bond_pairs.numel() == 0:
        return torch.zeros(i.shape[0], dtype=torch.float32, device=i.device)
    cand = pair_keys(i, j, num_nodes)
    bonds = pair_keys(bond_pairs[:, 0], bond_pairs[:, 1], num_nodes)
    return torch.isin(cand, bonds).to(torch.float32)


def candidate_recall(
    i: torch.Tensor, j: torch.Tensor, bond_pairs: torch.Tensor, num_nodes: int
) -> Tuple[float, int]:
    """Fraction of bond pairs present in the candidate set, and the miss count.

    This is the upper bound on recall for any model scored over (i, j).
    """
    if bond_pairs.numel() == 0:
        return 1.0, 0
    cand = pair_keys(i, j, num_nodes)
    bonds = pair_keys(bond_pairs[:, 0], bond_pairs[:, 1], num_nodes)
    hit = torch.isin(bonds, cand)
    misses = int((~hit).sum())
    return float(hit.float().mean()), misses
