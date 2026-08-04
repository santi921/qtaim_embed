"""Neighbor and candidate-pair construction for 3D encoders.

Pure-torch replacements for torch_cluster.radius_graph and PyG's
dimenet.triplets(), neither of which is installable on the torch 2.11/cu130
stack. Distance computation is chunked over rows so peak memory is
O(chunk * n) rather than O(n^2), and all pair enumeration is batch-aware so
batched graphs never get cross-molecule edges.
"""

import torch
from rdkit.Chem import GetPeriodicTable

_CHUNK = 256

# Covalent radii indexed by atomic number (RCOV[z], Angstrom). Same source as
# qtaim_generator's bond_agreement.py, so learned bond results stay comparable
# to the published geometric reference numbers.
_PT = GetPeriodicTable()
RCOV = torch.tensor(
    [0.0] + [_PT.GetRcovalent(z) for z in range(1, 119)], dtype=torch.float32
)


def radius_neighbors(pos, batch=None, cutoff=5.0, chunk=_CHUNK):
    """Bidirectional radius graph.

    Args:
        pos: (N, 3) positions.
        batch: (N,) graph assignment; None treats all atoms as one molecule.
        cutoff: scalar distance cutoff.
        chunk: rows per distance block.

    Returns:
        edge_index: (2, E) with edge_index[0] = source j, edge_index[1] = target i.
        d_ij: (E,) distances.
    """
    n = pos.shape[0]
    if batch is None:
        batch = pos.new_zeros(n, dtype=torch.long)
    ar = torch.arange(n, device=pos.device)
    src, dst = [], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = torch.cdist(pos[a:b], pos)
        m = (d <= cutoff) & (batch[a:b, None] == batch[None, :])
        m &= ar[None, :] != ar[a:b, None]
        li, lj = m.nonzero(as_tuple=True)
        dst.append(li + a)
        src.append(lj)
    edge_index = torch.stack([torch.cat(src), torch.cat(dst)])
    # cdist's mm-based path carries ~1e-7 cancellation error; recompute the
    # kept distances exactly so downstream bases see rotation-stable values
    d_ij = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)
    return edge_index, d_ij


def candidate_pairs(pos, z, batch=None, rcov=RCOV, pool_multiplier=2.0, chunk=_CHUNK):
    """Element-aware candidate pairs for bond classification (T3).

    Returns (i, j, d_ij) with i < j and
    d_ij <= pool_multiplier * (rcov_i + rcov_j), same-molecule pairs only.
    """
    n = pos.shape[0]
    if batch is None:
        batch = pos.new_zeros(n, dtype=torch.long)
    r = rcov.to(pos.device)[z]
    ar = torch.arange(n, device=pos.device)
    ii, jj = [], []
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = torch.cdist(pos[a:b], pos)
        thr = pool_multiplier * (r[a:b, None] + r[None, :])
        m = (d <= thr) & (batch[a:b, None] == batch[None, :])
        m &= ar[None, :] > ar[a:b, None]
        li, lj = m.nonzero(as_tuple=True)
        ii.append(li + a)
        jj.append(lj)
    i, j = torch.cat(ii), torch.cat(jj)
    return i, j, (pos[i] - pos[j]).norm(dim=-1)


def build_triplets(edge_index, num_nodes):
    """For each directed edge j->i, enumerate incoming edges k->j with k != i.

    Returns idx_i, idx_j, idx_k, idx_kj, idx_ji as DimeNet's
    InteractionPPBlock expects. Pure torch, no torch-sparse.
    """
    row, col = edge_index  # row = source (j), col = target (i)
    order = torch.argsort(col)
    row_s = row[order]
    counts = torch.bincount(col, minlength=num_nodes)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    n_per_edge = counts[row]  # in-degree of j, per edge
    idx_ji = torch.repeat_interleave(
        torch.arange(row.numel(), device=row.device), n_per_edge
    )
    starts = ptr[row]
    within = torch.arange(int(n_per_edge.sum()), device=row.device) - \
        torch.repeat_interleave(
            torch.cat([n_per_edge.new_zeros(1), n_per_edge.cumsum(0)])[:-1],
            n_per_edge,
        )
    sel = torch.repeat_interleave(starts, n_per_edge) + within
    idx_kj = order[sel]

    idx_i, idx_j, idx_k = col[idx_ji], row[idx_ji], row_s[sel]
    mask = idx_k != idx_i
    return idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]
