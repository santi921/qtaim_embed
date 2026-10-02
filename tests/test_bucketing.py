"""BucketBatchSampler: static shape classes, complete epochs, waste bound; the
LMDBDataModule hook yields grid-aligned dense shapes."""
from functools import partial

import numpy as np
import pytest
import torch

from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.core.datamodule import LMDBDataModule
from qtaim_embed.data.bucketing import BucketBatchSampler, graph_sizes, shape_class
from qtaim_embed.data.lmdb import TransformMol
from qtaim_embed.models.layers_dense import to_dense_hetero

LMDB = "tests/data/lmdb"


def _sizes(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    atoms = rng.integers(10, 100, n)
    bonds = (atoms * rng.uniform(0.9, 1.3, n)).astype(int)
    return atoms, bonds


class TestBucketBatchSampler:
    def test_epoch_covers_every_index_once(self):
        atoms, bonds = _sizes()
        s = BucketBatchSampler(atoms, bonds, batch_size=64, grid=16)
        batches = list(iter(s))
        flat = sorted(i for b in batches for i in b)
        assert flat == list(range(len(atoms)))
        assert len(batches) == len(s)
        assert all(len(b) <= 64 for b in batches)

    def test_batches_share_a_static_shape(self):
        atoms, bonds = _sizes()
        s = BucketBatchSampler(atoms, bonds, batch_size=32, grid=16, min_class_fraction=0.0)
        ca, cb = shape_class(atoms, bonds, 16)
        for b in iter(s):
            assert len(set(zip(ca[b].tolist(), cb[b].tolist()))) == 1
        assert all(na % 16 == 0 and nb % 16 == 0 for na, nb in s.shapes)

    def test_merged_classes_pad_to_class_maximum(self):
        atoms, bonds = _sizes(300)
        s = BucketBatchSampler(atoms, bonds, batch_size=128, grid=8, min_class_fraction=0.5)
        for (na, nb), idx in s.classes:
            assert (atoms[idx] <= na).all() and (bonds[idx] <= nb).all()
            assert len(idx) >= 64 or (na, nb) == s.shapes[-1]

    def test_waste_below_random_padding(self):
        atoms, bonds = _sizes()
        s = BucketBatchSampler(atoms, bonds, batch_size=64, grid=16)
        random_waste = 1 - (atoms.sum() + bonds.sum()) / (len(atoms) * (atoms.max() + bonds.max()))
        assert s.padding_waste() < 0.25 < random_waste

    def test_shuffle_is_seeded_and_changes_per_epoch(self):
        atoms, bonds = _sizes(500)
        a = list(iter(BucketBatchSampler(atoms, bonds, 32, seed=1)))
        b = list(iter(BucketBatchSampler(atoms, bonds, 32, seed=1)))
        assert a == b
        s = BucketBatchSampler(atoms, bonds, 32, seed=1)
        e1, e2 = list(iter(s)), list(iter(s))
        assert e1 != e2  # batch composition reshuffles within each class
        assert sorted(i for b in e1 for i in b) == sorted(i for b in e2 for i in b)
        fixed = BucketBatchSampler(atoms, bonds, 32, shuffle=False)
        assert list(iter(fixed)) == list(iter(fixed))

    def test_rank_sharding_partitions_batches(self):
        atoms, bonds = _sizes()
        full = BucketBatchSampler(atoms, bonds, batch_size=64, grid=16, seed=3)
        shards = [BucketBatchSampler(atoms, bonds, batch_size=64, grid=16, seed=3,
                                     rank=r, world_size=3) for r in range(3)]
        assert len({len(s) for s in shards}) == 1
        seen = [tuple(b) for s in shards for b in s]
        assert len(seen) == len(set(seen))
        assert all(len(s._batches()) == len(s) for s in shards)
        full_batches = {tuple(b) for b in full}
        assert set(seen) <= full_batches
        assert len(full_batches) - len(seen) < 3

    def test_unshuffled_sharding_keeps_every_sample(self):
        atoms, bonds = _sizes(n=1000)
        shards = [BucketBatchSampler(atoms, bonds, batch_size=64, grid=16, shuffle=False,
                                     rank=r, world_size=4) for r in range(4)]
        assert len({len(s) for s in shards}) == 1
        covered = {i for s in shards for b in s for i in b}
        assert covered == set(range(1000))

    def test_indices_match_iteration_order(self):
        atoms, bonds = _sizes(n=500)
        s = BucketBatchSampler(atoms, bonds, batch_size=32, grid=16, shuffle=False)
        assert s.indices() == [i for b in s for i in b]
        assert sorted(s.indices()) == list(range(500))

    def test_fewer_batches_than_ranks_keeps_every_rank_busy(self):
        atoms, bonds = _sizes(n=7)
        for ws in (3, 4):
            for shuffle, drop_last in ((False, False), (True, True), (True, False)):
                shards = [BucketBatchSampler(atoms, bonds, batch_size=8, grid=16, shuffle=shuffle,
                                             drop_last=drop_last, rank=r, world_size=ws) for r in range(ws)]
                counts = [len(list(iter(s))) for s in shards]
                assert counts == [len(shards[0])] * ws, (ws, shuffle, drop_last, counts)

    def test_rank_out_of_range(self):
        atoms, bonds = _sizes(n=100)
        with pytest.raises(ValueError):
            BucketBatchSampler(atoms, bonds, batch_size=8, rank=2, world_size=2)

    def test_batch_shape_is_constant_within_a_class(self):
        atoms, bonds = _sizes()
        s = BucketBatchSampler(atoms, bonds, batch_size=64, grid=16, min_class_fraction=0.5)
        for (na, nb), idx in s.classes:
            for b in (idx[:5], idx[-5:], idx[::7]):
                assert s.batch_shape(atoms[b], bonds[b]) == (na, nb)
        # unknown shapes fall back to their own rounding
        assert s.batch_shape([1000], [1000]) == (1008, 1008)

    def test_drop_last(self):
        atoms, bonds = _sizes(333)
        s = BucketBatchSampler(atoms, bonds, batch_size=50, drop_last=True, min_class_fraction=0.0)
        assert all(len(b) == 50 for b in iter(s)) and len(list(iter(s))) == len(s)


def test_graph_sizes_cache_write_is_atomic(tmp_path):
    ds = LMDBMoleculeDataset(config={"src": f"{LMDB}/train"}, transform=partial(TransformMol, dtype="float32"))
    cache = tmp_path / "sizes.npz"
    graph_sizes(ds, cache_path=str(cache))
    assert cache.exists() and not list(tmp_path.glob("*.tmp.*"))


def test_graph_sizes_matches_dataset_and_caches(tmp_path):
    ds = LMDBMoleculeDataset(config={"src": f"{LMDB}/train/molecule.lmdb"},
                             transform=partial(TransformMol, dtype="float32"))
    cache = tmp_path / "sizes.npz"
    atoms, bonds = graph_sizes(ds, cache_path=str(cache), num_workers=0, batch_size=7)
    assert len(atoms) == len(ds) and cache.exists()
    for i in (0, 3, len(ds) - 1):
        assert atoms[i] == ds[i]["atom"].num_nodes and bonds[i] == ds[i]["bond"].num_nodes
    atoms2, _ = graph_sizes(ds, cache_path=str(cache))
    assert np.array_equal(atoms, atoms2)


def test_lmdb_datamodule_bucketing_yields_static_dense_shapes(tmp_path):
    cfg = {
        "dataset": {"train_lmdb": f"{LMDB}/train", "val_lmdb": f"{LMDB}/val", "bucketing": True,
                    "bucket_grid": 8, "bucket_cache_dir": str(tmp_path), "seed": 3},
        "model": {"dense_grid": 8},
        "optim": {"train_batch_size": 8, "num_workers": 0, "pin_memory": False, "persistent_workers": False},
    }
    dm = LMDBDataModule(cfg)
    dm.setup("fit")
    loader = dm.train_dataloader()
    sampler = loader.batch_sampler
    seen = 0
    shapes = set()
    for graph, labels in loader:
        seen += int(graph.num_graphs)
        assert int(graph.num_graphs) == 8  # drop_last: static batch dimension
        feats = {nt: graph[nt].feat for nt in graph.node_types}
        db = to_dense_hetero(graph, feats, grid=8, shape=graph.dense_shape)
        _, n_b, b_b = db.shape
        assert (n_b, b_b) == tuple(graph.dense_shape)
        assert n_b % 8 == 0 and b_b % 8 == 0
        na, nb = torch.bincount(graph["atom"].batch), torch.bincount(graph["bond"].batch)
        assert int(na.max()) <= n_b and int(nb.max()) <= b_b
        shapes.add((n_b, b_b))
    assert seen == len(sampler) * 8
    assert len(dm.train_dataset) - seen < 8 * len(sampler.classes)
    assert shapes <= set(sampler.shapes)  # one static shape per class, never more
    # eval keeps every graph and is still stamped
    val_seen = 0
    for graph, _ in dm.val_dataloader():
        val_seen += int(graph.num_graphs)
        assert tuple(graph.dense_shape) in set(dm.val_dataloader().batch_sampler.shapes)
    assert val_seen == len(dm.val_dataset)
    val = dm.val_dataloader()
    assert sum(int(g.num_graphs) for g, _ in val) == len(dm.val_dataset)
    assert sorted(val.batch_sampler.indices()) == list(range(len(dm.val_dataset)))
    cfg["dataset"]["bucketing_eval"] = False
    dm_plain = LMDBDataModule(cfg)
    dm_plain.setup("fit")
    assert dm_plain.val_dataloader().batch_sampler.__class__.__name__ != "BucketBatchSampler"
    assert dm_plain.train_dataloader().batch_sampler.__class__.__name__ == "BucketBatchSampler"
