"""collate_hetero_direct must batch exactly like Batch.from_data_list for the
attributes the models read (feat, labels, pos, z, batch, edge_index)."""
from functools import partial

import torch
from torch_geometric.data import Batch

from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.data.dataloader import DataLoaderLMDB, collate_hetero_direct
from qtaim_embed.data.lmdb import TransformMol
from qtaim_embed.utils.tests import make_hetero, make_hetero_graph

KEYS = ("feat", "labels", "pos", "z")


def _assert_same(direct, ref):
    assert direct.node_types == ref.node_types
    assert set(direct.edge_types) == set(ref.edge_types)
    for nt in ref.node_types:
        assert direct[nt].num_nodes == ref[nt].num_nodes
        assert torch.equal(direct[nt].batch, ref[nt].batch)
        for key in KEYS:
            assert (key in direct[nt]) == (key in ref[nt])
            if key in ref[nt]:
                assert torch.equal(direct[nt][key], ref[nt][key]), (nt, key)
    for et in ref.edge_types:
        assert torch.equal(direct[et].edge_index, ref[et].edge_index), et
    assert direct.num_graphs == ref.num_graphs


def test_matches_pyg_on_lmdb_fixture_with_pos_z():
    ds = LMDBMoleculeDataset(
        config={"src": "tests/data/lmdb/train/molecule.lmdb"},
        transform=partial(TransformMol, dtype="float32"),
    )
    graphs = [ds[i] for i in range(min(len(ds), 12))]
    assert "pos" in graphs[0]["atom"] and "z" in graphs[0]["atom"]
    _assert_same(collate_hetero_direct(graphs), Batch.from_data_list(graphs))


def test_matches_pyg_on_synthetic_graphs_with_labels_and_empty_edges():
    g1, _ = make_hetero_graph()
    g2, _ = make_hetero(num_atoms=2, num_bonds=1, a2b=[(0, 0), (1, 0)], b2a=[(0, 0), (0, 1)])
    g3, _ = make_hetero(num_atoms=3, num_bonds=0, a2b=[], b2a=[])  # fake-bond path
    graphs = [g1, g2, g3]
    for g in graphs:
        g["atom"].labels = torch.randn(g["atom"].num_nodes, 2)
        g["bond"].labels = torch.randn(g["bond"].num_nodes, 1)
    # one graph with an empty edge type exercises the zero-length offset path
    g2["atom", "a2a", "atom"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    _assert_same(collate_hetero_direct(graphs), Batch.from_data_list(graphs))


def test_dataloader_lmdb_uses_direct_collate_and_returns_labels():
    ds = LMDBMoleculeDataset(
        config={"src": "tests/data/lmdb_node/train/molecule.lmdb"},
        transform=partial(TransformMol, dtype="float32"),
    )
    loader = DataLoaderLMDB(dataset=ds, batch_size=4, shuffle=False, num_workers=0)
    graph, labels = next(iter(loader))
    ref = Batch.from_data_list([ds[i] for i in range(4)])
    _assert_same(graph, ref)
    for nt, lab in labels.items():
        assert torch.equal(lab, ref[nt].labels)


def test_schema_mismatch_falls_back_to_pyg():
    g1, _ = make_hetero_graph()
    g2, _ = make_hetero_graph()
    g2["atom"].pos = torch.zeros(g2["atom"].num_nodes, 3)  # only the second graph has pos
    out = collate_hetero_direct([g1, g2])
    assert isinstance(out, Batch)
