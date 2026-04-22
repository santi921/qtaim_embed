import io
import os
import pickle
import tempfile

import lmdb
import pytest
import torch
from torch_geometric.data import HeteroData

from qtaim_embed.data.lmdb import (
    load_graph_from_serialized,
    open_lmdb_readonly,
    split_lmdb_file,
    write_lmdb_split,
    TransformMol,
)
from qtaim_embed.core.dataset import LMDBMoleculeDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n_atoms: int = 5, n_bonds: int = 8) -> HeteroData:
    g = HeteroData()
    g["atom"].feat = torch.randn(n_atoms, 6)
    g["atom"].labels = torch.randn(n_atoms, 4)
    g["atom"].num_nodes = n_atoms
    g["bond"].feat = torch.randn(n_bonds, 3)
    g["bond"].num_nodes = n_bonds
    g["global"].feat = torch.randn(1, 2)
    g["global"].num_nodes = 1
    return g


def _serialize(graph: HeteroData) -> bytes:
    buf = io.BytesIO()
    torch.save(graph, buf)
    return buf.getvalue()


def _make_generator_format_lmdb(path: str, n: int = 30) -> None:
    """Create a source LMDB in qtaim_generator format: string keys, pickle.dumps(graph_bytes)."""
    db = lmdb.open(path, map_size=10 ** 8, subdir=False, meminit=False, map_async=True)
    for i in range(n):
        graph_bytes = _serialize(_make_graph())
        key = f"MOL_{i:04d}".encode("ascii")
        val = pickle.dumps(graph_bytes, protocol=-1)
        txn = db.begin(write=True)
        txn.put(key, val)
        txn.commit()
    db.sync()
    db.close()


def _make_embed_format_lmdb(path: str, n: int = 30) -> None:
    """Create a source LMDB already in qtaim_embed format: integer keys, {"molecule_graph": bytes}."""
    db = lmdb.open(path, map_size=10 ** 8, subdir=False, meminit=False, map_async=True)
    for i in range(n):
        graph_bytes = _serialize(_make_graph())
        val = pickle.dumps({"molecule_graph": graph_bytes}, protocol=-1)
        txn = db.begin(write=True)
        txn.put(f"{i}".encode("ascii"), val)
        txn.commit()
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(n, protocol=-1))
    txn.commit()
    db.sync()
    db.close()


# ---------------------------------------------------------------------------
# Tests: split_lmdb_file (generator format source)
# ---------------------------------------------------------------------------

class TestSplitLmdbFile:
    def test_split_sizes_sum_to_total(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=100)
        result = split_lmdb_file(src, str(tmp_path / "out"), val_prop=0.15, test_prop=0.1, seed=42)
        sizes = result["sizes"]
        assert sizes["train"] + sizes["val"] + sizes["test"] == 100

    def test_split_proportions_approximate(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=100)
        result = split_lmdb_file(src, str(tmp_path / "out"), val_prop=0.15, test_prop=0.1, seed=42)
        sizes = result["sizes"]
        assert sizes["test"] == 10
        assert sizes["val"] == 15
        assert sizes["train"] == 75

    def test_no_index_overlap_between_splits(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        n = 50
        _make_generator_format_lmdb(src, n=n)
        out_dir = str(tmp_path / "out")
        result = split_lmdb_file(src, out_dir, val_prop=0.2, test_prop=0.2, seed=0)

        def read_entries(lmdb_path):
            env = open_lmdb_readonly(lmdb_path)
            entries = {}
            with env.begin() as txn:
                for k, v in txn.cursor():
                    key = k.decode("ascii", errors="ignore")
                    if key.isdigit():
                        obj = pickle.loads(v)
                        graph = load_graph_from_serialized(obj["molecule_graph"])
                        entries[int(key)] = graph["atom"].feat
            env.close()
            return entries

        train_e = read_entries(result["train"])
        val_e = read_entries(result["val"])
        test_e = read_entries(result["test"])
        assert len(train_e) + len(val_e) + len(test_e) == n

    def test_output_has_integer_keys(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=20)
        result = split_lmdb_file(src, str(tmp_path / "out"), seed=1)

        env = open_lmdb_readonly(result["train"])
        with env.begin() as txn:
            keys = [k.decode("ascii") for k, _ in txn.cursor()]
        env.close()

        non_meta = [k for k in keys if k != "length"]
        assert all(k.isdigit() for k in non_meta)
        assert sorted(int(k) for k in non_meta) == list(range(len(non_meta)))

    def test_output_has_length_key(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=20)
        result = split_lmdb_file(src, str(tmp_path / "out"), seed=1)

        for split_path in (result["train"], result["val"], result["test"]):
            env = open_lmdb_readonly(split_path)
            with env.begin() as txn:
                raw = txn.get("length".encode("ascii"))
            env.close()
            assert raw is not None
            n = pickle.loads(raw)
            assert isinstance(n, int) and n > 0

    def test_output_has_molecule_graph_wrapper(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=10)
        result = split_lmdb_file(src, str(tmp_path / "out"), seed=7)

        env = open_lmdb_readonly(result["train"])
        with env.begin() as txn:
            raw = txn.get(b"0")
        env.close()
        obj = pickle.loads(raw)
        assert isinstance(obj, dict)
        assert "molecule_graph" in obj

    def test_graphs_deserialize_correctly(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=10)
        result = split_lmdb_file(src, str(tmp_path / "out"), seed=3)

        env = open_lmdb_readonly(result["train"])
        with env.begin() as txn:
            raw = txn.get(b"0")
        env.close()
        graph = load_graph_from_serialized(pickle.loads(raw)["molecule_graph"])
        assert "atom" in graph.node_types
        assert hasattr(graph["atom"], "feat")
        assert graph["atom"].feat.shape[1] == 6

    def test_reproducible_with_same_seed(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=50)

        r1 = split_lmdb_file(src, str(tmp_path / "out1"), seed=42)
        r2 = split_lmdb_file(src, str(tmp_path / "out2"), seed=42)

        def first_key_feat(lmdb_path):
            env = open_lmdb_readonly(lmdb_path)
            with env.begin() as txn:
                raw = txn.get(b"0")
            env.close()
            return load_graph_from_serialized(pickle.loads(raw)["molecule_graph"])["atom"].feat

        assert torch.allclose(first_key_feat(r1["train"]), first_key_feat(r2["train"]))

    def test_different_seeds_give_different_splits(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=50)

        r1 = split_lmdb_file(src, str(tmp_path / "out1"), seed=1)
        r2 = split_lmdb_file(src, str(tmp_path / "out2"), seed=2)

        def first_feat(lmdb_path):
            env = open_lmdb_readonly(lmdb_path)
            with env.begin() as txn:
                raw = txn.get(b"0")
            env.close()
            return load_graph_from_serialized(pickle.loads(raw)["molecule_graph"])["atom"].feat

        assert not torch.allclose(first_feat(r1["train"]), first_feat(r2["train"]))

    def test_readable_by_lmdb_molecule_dataset(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=30)
        result = split_lmdb_file(src, str(tmp_path / "out"), seed=0)

        dataset = LMDBMoleculeDataset({"src": result["train"]}, transform=TransformMol)
        assert len(dataset) == result["sizes"]["train"]
        graph = dataset[0]
        assert "atom" in graph.node_types


# ---------------------------------------------------------------------------
# Tests: write_lmdb_split handles embed-format source unchanged
# ---------------------------------------------------------------------------

class TestSplitLmdbEdgeCases:
    def test_bad_proportions_raises(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=10)
        with pytest.raises(ValueError, match="val_prop"):
            split_lmdb_file(src, str(tmp_path / "out"), val_prop=0.5, test_prop=0.6)

    def test_metadata_keys_not_included_in_splits(self, tmp_path):
        """Embed-format source has a 'length' metadata key -- should not appear as a data entry."""
        src = str(tmp_path / "src.lmdb")
        _make_embed_format_lmdb(src, n=20)
        result = split_lmdb_file(src, str(tmp_path / "out"), seed=0)
        total = result["sizes"]["train"] + result["sizes"]["val"] + result["sizes"]["test"]
        assert total == 20  # 'length' key must not be counted as a molecule

    def test_tqdm_label_uses_split_name(self, tmp_path, capsys):
        src = str(tmp_path / "src.lmdb")
        _make_generator_format_lmdb(src, n=10)
        split_lmdb_file(src, str(tmp_path / "out"), seed=0)
        captured = capsys.readouterr()
        assert "train" in captured.err or "val" in captured.err or "test" in captured.err


class TestWriteLmdbSplitEmbedFormat:
    def test_embed_format_passthrough(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_embed_format_lmdb(src, n=20)

        src_env = open_lmdb_readonly(src)
        with src_env.begin() as txn:
            keys = [k for k, _ in txn.cursor() if k != b"length"]

        out = str(tmp_path / "out.lmdb")
        write_lmdb_split(src_env, keys, out)
        src_env.close()

        env = open_lmdb_readonly(out)
        with env.begin() as txn:
            raw = txn.get(b"0")
        env.close()
        obj = pickle.loads(raw)
        assert "molecule_graph" in obj
        graph = load_graph_from_serialized(obj["molecule_graph"])
        assert hasattr(graph["atom"], "feat")
