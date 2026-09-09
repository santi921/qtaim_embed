"""Neighbor and candidate-pair construction for 3D encoders.

Pure-torch replacements for torch_cluster.radius_graph and PyG's
dimenet.triplets(), neither of which is installable on the torch 2.11/cu130
stack. Batched radius graphs use one per-molecule dense distance block
([G, N_b, N_b] via to_dense_batch); single molecules, very large blocks, and
candidate pairs use a row-chunked cdist so peak memory is O(chunk * n). All
pair enumeration is batch-aware so batched graphs never get cross-molecule
edges.
"""

import functools

import torch
from rdkit.Chem import GetPeriodicTable
from torch.profiler import record_function

_CHUNK = 256


def _profiled(name):
    """Wrap a function body in a torch.profiler range so benchmarks can
    attribute device time to the neighbor build (performance plan A0)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with record_function(name):
                return fn(*args, **kwargs)

        return wrapper

    return deco

# Covalent radii indexed by atomic number (RCOV[z], Angstrom). Same source as
# qtaim_generator's bond_agreement.py, so learned bond results stay comparable
# to the published geometric reference numbers.
_PT = GetPeriodicTable()
RCOV = torch.tensor(
    [0.0] + [_PT.GetRcovalent(z) for z in range(1, 119)], dtype=torch.float32
)


# Largest per-molecule dense distance block (G * N_b^2 elements) before the
# dense path falls back to the chunked builder: 2^28 floats is 1 GB in fp32.
_DENSE_MAX_ELEMENTS = 1 << 28


def _radius_neighbors_chunked(pos, batch, cutoff, chunk):
    n = pos.shape[0]
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
    return torch.stack([torch.cat(src), torch.cat(dst)])


def _radius_neighbors_dense(pos, batch, cutoff):
    """One [G, N_b, N_b] distance block over to_dense_batch positions.

    Same-molecule pairs only by construction, so the batch-wide cdist and its
    per-chunk nonzero syncs disappear: one kernel, one mask, one nonzero.
    Requires batch to be sorted (PyG Batch guarantees this).
    """
    from torch_geometric.utils import to_dense_batch

    dense, mask = to_dense_batch(pos, batch)
    d = torch.cdist(dense, dense)
    keep = (d <= cutoff) & mask[:, :, None] & mask[:, None, :]
    keep &= ~torch.eye(dense.shape[1], dtype=torch.bool, device=pos.device)[None]
    g, i, j = keep.nonzero(as_tuple=True)
    counts = mask.sum(1)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    base = ptr[g]
    return torch.stack([base + j, base + i])


@_profiled("neighbors.radius")
def radius_neighbors(pos, batch=None, cutoff=5.0, chunk=_CHUNK):
    """Bidirectional radius graph.

    Args:
        pos: (N, 3) positions.
        batch: (N,) sorted graph assignment; None treats all atoms as one molecule.
        cutoff: scalar distance cutoff.
        chunk: rows per distance block (chunked fallback only).

    Returns:
        edge_index: (2, E) with edge_index[0] = source j, edge_index[1] = target i.
        d_ij: (E,) distances.

    Batched inputs use a per-molecule dense distance block (E6 benchmark:
    65x faster than the batch-wide chunked cdist at batch 512, identical edge
    set); single molecules and blocks over _DENSE_MAX_ELEMENTS use the
    chunked builder.
    """
    n = pos.shape[0]
    if batch is None:
        batch = pos.new_zeros(n, dtype=torch.long)
        use_dense = False
    else:
        n_b = int(torch.bincount(batch).max()) if n > 0 else 0
        n_graphs = int(batch.max()) + 1 if n > 0 else 0
        use_dense = 0 < n_graphs * n_b * n_b <= _DENSE_MAX_ELEMENTS
    if use_dense:
        edge_index = _radius_neighbors_dense(pos, batch, cutoff)
    else:
        edge_index = _radius_neighbors_chunked(pos, batch, cutoff, chunk)
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


@_profiled("neighbors.triplets")
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
