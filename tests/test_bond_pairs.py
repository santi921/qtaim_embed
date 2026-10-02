"""Bond-pair extraction, candidate labels, candidate recall, and pair head."""

from functools import partial
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.data.bonds import (
    bond_pairs_from_heterograph,
    candidate_labels,
    candidate_recall,
)
from qtaim_embed.data.lmdb import TransformMol
from qtaim_embed.models.encoders.neighbors import RCOV, candidate_pairs
from qtaim_embed.models.link_pred.pair_head import SymmetricPairHead

DATA_DIR = Path(__file__).parent / "data"
LINK_LMDB = DATA_DIR / "lmdb_link" / "train" / "molecule.lmdb"


def _fixture_dataset():
    return LMDBMoleculeDataset(
        {"src": str(LINK_LMDB)}, transform=partial(TransformMol, dtype="float32")
    )


def _brute_force_pairs(graph):
    ei = graph["atom", "a2b", "bond"].edge_index
    by_bond = {}
    for a, b in ei.t().tolist():
        by_bond.setdefault(b, []).append(a)
    pairs = set()
    for atoms in by_bond.values():
        assert len(atoms) == 2
        lo, hi = sorted(atoms)
        if lo != hi:
            pairs.add((lo, hi))
    return pairs


def _toy_graph(num_atoms, bonds):
    g = HeteroData()
    g["atom"].num_nodes = num_atoms
    g["atom"].pos = torch.randn(num_atoms, 3)
    g["atom"].z = torch.full((num_atoms,), 6, dtype=torch.long)
    g["bond"].num_nodes = len(bonds)
    src, dst = [], []
    for k, (a, b) in enumerate(bonds):
        src += [a, b]
        dst += [k, k]
    g["atom", "a2b", "bond"].edge_index = torch.tensor([src, dst], dtype=torch.long)
    return g


def test_bond_pairs_match_brute_force_on_fixture():
    ds = _fixture_dataset()
    for idx in range(min(len(ds), 20)):
        g = ds[idx]
        pairs = bond_pairs_from_heterograph(g)
        assert pairs.shape[1] == 2
        assert bool((pairs[:, 0] < pairs[:, 1]).all())
        assert set(map(tuple, pairs.tolist())) == _brute_force_pairs(g)


def test_bond_pairs_batched_offsets():
    ds = _fixture_dataset()
    graphs = [ds[i] for i in range(4)]
    batch = Batch.from_data_list(graphs)
    pairs = bond_pairs_from_heterograph(batch)
    expected = set()
    offset = 0
    for g in graphs:
        expected |= {(a + offset, b + offset) for a, b in _brute_force_pairs(g)}
        offset += int(g["atom"].num_nodes)
    assert set(map(tuple, pairs.tolist())) == expected


def test_bond_pairs_zero_bonds_and_self_bonds():
    g = _toy_graph(3, [])
    assert bond_pairs_from_heterograph(g).shape == (0, 2)
    g = _toy_graph(3, [(0, 0), (2, 1)])
    pairs = bond_pairs_from_heterograph(g)
    assert pairs.tolist() == [[1, 2]]


def test_bond_pairs_skips_zero_bond_placeholder():
    from qtaim_embed.data.grapher import build_hetero_graph_skeleton

    bonded = build_hetero_graph_skeleton(3, [(0, 1), (1, 2)])
    empty = build_hetero_graph_skeleton(1, [])
    assert bond_pairs_from_heterograph(empty).shape == (0, 2)
    batch = Batch.from_data_list([bonded, empty, bonded])
    pairs = bond_pairs_from_heterograph(batch)
    assert pairs.tolist() == [[0, 1], [1, 2], [4, 5], [5, 6]]


def test_bond_pairs_rejects_malformed_a2b():
    g = _toy_graph(3, [(0, 1)])
    g["atom", "a2b", "bond"].edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    with pytest.raises(ValueError):
        bond_pairs_from_heterograph(g)


def test_candidate_labels_match_python_set():
    g = _toy_graph(6, [(0, 1), (1, 2), (3, 5)])
    pairs = bond_pairs_from_heterograph(g)
    i, j = torch.triu_indices(6, 6, offset=1)
    y = candidate_labels(i, j, pairs, 6)
    truth = {(0, 1), (1, 2), (3, 5)}
    for a, b, lab in zip(i.tolist(), j.tolist(), y.tolist()):
        assert lab == float((a, b) in truth)
    assert candidate_labels(i, j, pairs.new_zeros((0, 2)), 6).sum() == 0


def test_candidate_recall_on_fixture_at_two_x():
    ds = _fixture_dataset()
    batch = Batch.from_data_list([ds[i] for i in range(len(ds))])
    atom = batch["atom"]
    i, j, d = candidate_pairs(atom.pos, atom.z, atom.batch, RCOV, pool_multiplier=2.0)
    pairs = bond_pairs_from_heterograph(batch)
    recall, misses = candidate_recall(i, j, pairs, int(atom.num_nodes))
    # Measured 2026-09-08 on the full 75-graph fixture: 8 of 1230 QTAIM bond
    # paths are O-H / C-H contacts at 2.0-2.8x covalent (hydrogen bonds) and
    # fall outside the 2.0x pool. Recall is an upper bound for any model.
    assert recall >= 0.99
    y = candidate_labels(i, j, pairs, int(atom.num_nodes))
    assert 0.0 < float(y.mean()) < 1.0
    i3, j3, _ = candidate_pairs(atom.pos, atom.z, atom.batch, RCOV, pool_multiplier=3.0)
    recall3, misses3 = candidate_recall(i3, j3, pairs, int(atom.num_nodes))
    assert misses3 == 0 and recall3 == 1.0


def test_pair_head_symmetric_and_shapes():
    torch.manual_seed(0)
    head = SymmetricPairHead(in_dim=8, hidden=[16], rbf="bessel", rbf_n=10, dropout=0.0)
    head.eval()
    h = torch.randn(5, 8)
    i = torch.tensor([0, 1, 2])
    j = torch.tensor([3, 4, 4])
    d = torch.tensor([1.2, 2.5, 3.1])
    r = torch.tensor([1.5, 1.5, 1.6])
    out = head(h, i, j, d, r)
    assert out.shape == (3,)
    swapped = head(h, j, i, d, r)
    assert torch.allclose(out, swapped, atol=1e-6)


def test_pair_head_gaussian_and_no_ratio():
    head = SymmetricPairHead(in_dim=4, hidden=[8], rbf="gaussian", rbf_n=6, use_ratio=False, dropout=0.0)
    h = torch.randn(3, 4)
    out = head(h, torch.tensor([0]), torch.tensor([2]), torch.tensor([1.0]))
    assert out.shape == (1,)
    head_ratio = SymmetricPairHead(in_dim=4, hidden=[8], rbf_n=6, dropout=0.0)
    with pytest.raises(ValueError):
        head_ratio(h, torch.tensor([0]), torch.tensor([2]), torch.tensor([1.0]))
