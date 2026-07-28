"""LMDBMoleculeDataset must refuse shard directories that mix graphs built
before and after the atom.pos/atom.z schema change: mixed batches either
crash Batch.from_data_list (KeyError: 'pos') or silently drop pos/z
depending on which graph leads the batch.
"""

import os
import pickle

import lmdb
import pytest
import torch
from torch_geometric.data import HeteroData

from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.data.lmdb import serialize_graph


def _make_graph(with_pos: bool) -> HeteroData:
    data = HeteroData()
    data["atom"].num_nodes = 2
    data["atom"].feat = torch.zeros(2, 4)
    data["bond"].num_nodes = 1
    data["bond"].feat = torch.zeros(1, 3)
    data["global"].num_nodes = 1
    data["global"].feat = torch.zeros(1, 2)
    if with_pos:
        data["atom"].pos = torch.zeros(2, 3)
        data["atom"].z = torch.tensor([1, 8], dtype=torch.long)
    return data


def _write_shard(path: str, with_pos: bool) -> None:
    env = lmdb.open(str(path), subdir=False, map_size=1024**3)
    with env.begin(write=True) as txn:
        entry = pickle.dumps(
            {"molecule_graph": serialize_graph(_make_graph(with_pos), ret=True)},
            protocol=-1,
        )
        txn.put(b"0", entry)
        txn.put(b"length", pickle.dumps(1, protocol=-1))
    env.sync()
    env.close()


def _make_dir(tmp_path, name: str, shard_pos_flags) -> str:
    d = tmp_path / name
    d.mkdir()
    for i, with_pos in enumerate(shard_pos_flags):
        _write_shard(os.path.join(str(d), f"shard_{i}.lmdb"), with_pos)
    return str(d)


def test_mixed_schema_directory_raises(tmp_path):
    src = _make_dir(tmp_path, "mixed", [True, False])
    with pytest.raises(RuntimeError, match="Mixed graph schemas"):
        LMDBMoleculeDataset(config={"src": src})


def test_homogeneous_directories_load(tmp_path):
    for name, flags in (("all_new", [True, True]), ("all_old", [False, False])):
        src = _make_dir(tmp_path, name, flags)
        ds = LMDBMoleculeDataset(config={"src": src})
        assert len(ds) == 2
        obj = ds[0]
        assert "molecule_graph" in obj
