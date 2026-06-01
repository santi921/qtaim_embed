import io
import os
import pickle
import tempfile

import lmdb
import pytest
import torch
from torch_geometric.data import HeteroData

from qtaim_embed.data.lmdb import (
    _LMDB_METADATA_KEYS,
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
            keys = [k for k, _ in txn.cursor()]
        env.close()

        non_meta = [k.decode("ascii") for k in keys if k not in _LMDB_METADATA_KEYS]
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

    def test_metadata_propagated_to_splits(self, tmp_path):
        """Metadata keys written to source LMDB are copied to every split."""
        src = str(tmp_path / "src.lmdb")
        n = 30
        feature_names = {"atom": ["feat_a"], "bond": ["feat_b"], "global": []}
        feature_size = {"atom": 1, "bond": 1, "global": 0}
        target_dict = {"atom": ["label_x"], "bond": [], "global": []}
        element_set = {"C", "H", "O"}

        db = lmdb.open(src, map_size=10 ** 8, subdir=False, meminit=False, map_async=True)
        for i in range(n):
            graph_bytes = _serialize(_make_graph())
            val = pickle.dumps({"molecule_graph": graph_bytes}, protocol=-1)
            txn = db.begin(write=True)
            txn.put(f"{i}".encode("ascii"), val)
            txn.commit()
        txn = db.begin(write=True)
        txn.put(b"length", pickle.dumps(n, protocol=-1))
        txn.put(b"feature_names", pickle.dumps(feature_names, protocol=-1))
        txn.put(b"feature_size", pickle.dumps(feature_size, protocol=-1))
        txn.put(b"target_dict", pickle.dumps(target_dict, protocol=-1))
        txn.put(b"element_set", pickle.dumps(element_set, protocol=-1))
        txn.commit()
        db.sync()
        db.close()

        result = split_lmdb_file(src, str(tmp_path / "out"), val_prop=0.2, test_prop=0.1, seed=0)

        for split_path in (result["train"], result["val"], result["test"]):
            env = open_lmdb_readonly(split_path)
            with env.begin() as txn:
                assert pickle.loads(txn.get(b"feature_names")) == feature_names
                assert pickle.loads(txn.get(b"feature_size")) == feature_size
                assert pickle.loads(txn.get(b"target_dict")) == target_dict
                assert pickle.loads(txn.get(b"element_set")) == element_set
            env.close()


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


# ---------------------------------------------------------------------------
# Tests: composition-based splitting
# ---------------------------------------------------------------------------

def _make_composition_graph(counts: dict, elem_order: list) -> HeteroData:
    """Build a graph whose atom.feat element one-hots encode `counts`."""
    n_elem = len(elem_order)
    rows = []
    for sym, c in counts.items():
        onehot = [1.0 if e == sym else 0.0 for e in elem_order]
        rows.extend([[0.0, 0.0] + onehot for _ in range(c)])
    feat = torch.tensor(rows, dtype=torch.float32)
    n_atoms = feat.shape[0]
    g = HeteroData()
    g["atom"].feat = feat
    g["atom"].num_nodes = n_atoms
    g["bond"].feat = torch.zeros(max(n_atoms - 1, 1), 3)
    g["bond"].num_nodes = max(n_atoms - 1, 1)
    g["global"].feat = torch.zeros(1, 2)
    g["global"].num_nodes = 1
    return g


def _make_composition_lmdb(path: str, formulas: list, elem_order: list) -> None:
    """Embed-format LMDB with element one-hot atom feats + feature_names metadata.

    `formulas` is one dict of element counts per molecule.
    """
    atom_feature_names = ["total_degree", "total_H"] + [
        f"chemical_symbol_{e}" for e in elem_order
    ]
    feature_names = {"atom": atom_feature_names, "bond": ["b0", "b1", "b2"], "global": ["g0", "g1"]}

    db = lmdb.open(path, map_size=10 ** 8, subdir=False, meminit=False, map_async=True)
    for i, counts in enumerate(formulas):
        graph_bytes = _serialize(_make_composition_graph(counts, elem_order))
        val = pickle.dumps({"molecule_graph": graph_bytes}, protocol=-1)
        txn = db.begin(write=True)
        txn.put(f"{i}".encode("ascii"), val)
        txn.commit()
    txn = db.begin(write=True)
    txn.put(b"length", pickle.dumps(len(formulas), protocol=-1))
    txn.put(b"feature_names", pickle.dumps(feature_names, protocol=-1))
    txn.commit()
    db.sync()
    db.close()


class TestCompositionSplit:
    ELEMS = ["C", "H", "O", "N"]
    # five distinct formulas, several molecules each
    FORMULAS = (
        [{"C": 1, "H": 4}] * 6
        + [{"C": 2, "H": 6}] * 6
        + [{"C": 1, "H": 2, "O": 1}] * 6
        + [{"N": 2}] * 6
        + [{"O": 2, "H": 2}] * 6
    )

    def _split_membership(self, result):
        """Return {key: split_name} across all three split LMDBs."""
        membership = {}
        for split_name in ("train", "val", "test"):
            env = open_lmdb_readonly(result[split_name])
            with env.begin() as txn:
                for k, v in txn.cursor():
                    if k in _LMDB_METADATA_KEYS:
                        continue
                    g = load_graph_from_serialized(pickle.loads(v)["molecule_graph"])
                    # recover formula signature from element one-hot sums
                    feat = g["atom"].feat
                    sig = tuple(int(feat[:, 2 + j].sum().item()) for j in range(len(self.ELEMS)))
                    membership[(split_name, k)] = sig
            env.close()
        return membership

    def test_same_formula_lands_in_one_split(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_composition_lmdb(src, self.FORMULAS, self.ELEMS)
        result = split_lmdb_file(
            src, str(tmp_path / "out"), val_prop=0.2, test_prop=0.2,
            seed=7, method="composition",
        )
        # map each formula signature -> set of splits it appears in
        sig_to_splits = {}
        for (split_name, _key), sig in self._split_membership(result).items():
            sig_to_splits.setdefault(sig, set()).add(split_name)
        # every formula must appear in exactly one split
        for sig, splits in sig_to_splits.items():
            assert len(splits) == 1, f"formula {sig} leaked across splits {splits}"

    def _split_sig_multisets(self, result):
        """Return {split_name: sorted([formula_sig, ...])} for membership compare."""
        out = {"train": [], "val": [], "test": []}
        for (split_name, _key), sig in self._split_membership(result).items():
            out[split_name].append(sig)
        return {s: sorted(v) for s, v in out.items()}

    def test_composition_deterministic(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_composition_lmdb(src, self.FORMULAS, self.ELEMS)
        r1 = split_lmdb_file(src, str(tmp_path / "o1"), val_prop=0.2, test_prop=0.2,
                             seed=7, method="composition")
        r2 = split_lmdb_file(src, str(tmp_path / "o2"), val_prop=0.2, test_prop=0.2,
                             seed=7, method="composition")
        # per-molecule membership (by formula), not just sizes
        assert self._split_sig_multisets(r1) == self._split_sig_multisets(r2)

    def test_total_preserved(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_composition_lmdb(src, self.FORMULAS, self.ELEMS)
        result = split_lmdb_file(src, str(tmp_path / "out"), val_prop=0.2, test_prop=0.2,
                                 seed=7, method="composition")
        sizes = result["sizes"]
        assert sizes["train"] + sizes["val"] + sizes["test"] == len(self.FORMULAS)

    def test_requires_feature_names(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_embed_format_lmdb(src, n=10)  # no feature_names metadata
        with pytest.raises(ValueError, match="feature_names"):
            split_lmdb_file(src, str(tmp_path / "out"), method="composition")

    def test_hash_assignment_pinned(self):
        """Pin the exact split assignment so it can't silently drift from the
        converter path. Reference values generated from
        qtaim_gen.source.utils.splits.assign_formula_to_split(formula, (0.6,0.2,0.2), 42).
        """
        from qtaim_embed.data.lmdb import _assign_formula_to_split
        ratios = (0.6, 0.2, 0.2)
        names = ("train", "val", "test")
        expected = {
            "H4C1": "train", "H6C2": "train", "H2C1O1": "train", "N2": "val",
            "H2O2": "train", "Fe1O3": "train", "Ag1": "train",
        }
        for formula, exp in expected.items():
            got = names[_assign_formula_to_split(formula, ratios, seed=42)]
            assert got == exp, f"{formula}: got={got} expected={exp}"

    def test_hash_parity_with_qtaim_gen(self):
        """Cross-check the local hash against qtaim_gen's directly. Skips
        visibly when qtaim_gen is not importable (it isn't, inside the
        qtaim_embed-only env), so this never silently passes."""
        qg_splits = pytest.importorskip("qtaim_gen.source.utils.splits")
        from qtaim_embed.data.lmdb import _assign_formula_to_split
        ratios = (0.6, 0.2, 0.2)
        names = ("train", "val", "test")
        for formula in ["H4C1", "H6C2", "H2C1O1", "N2", "H2O2", "Fe1O3", "Ag1"]:
            assert names[_assign_formula_to_split(formula, ratios, 42)] == \
                qg_splits.assign_formula_to_split(formula, ratios, 42)

    def test_empty_formula_routes_to_train(self, tmp_path):
        """Graphs with no element columns set (formula derives to '') must all
        go to train, matching the converter's missing-formula convention."""
        src = str(tmp_path / "src.lmdb")
        # "X" is absent from ELEMS, so its one-hot rows are all zeros ->
        # element-column sums are 0 -> formula derives to "" for every molecule
        _make_composition_lmdb(src, [{"X": 2}] * 20, self.ELEMS)
        result = split_lmdb_file(src, str(tmp_path / "out"), val_prop=0.2, test_prop=0.2,
                                 seed=7, method="composition")
        assert result["sizes"] == {"train": 20, "val": 0, "test": 0}

    def test_rejects_scaled_source(self, tmp_path):
        """Composition split must refuse a scaled LMDB (element one-hots would
        no longer be 0/1, silently corrupting formulas)."""
        src = str(tmp_path / "src.lmdb")
        _make_composition_lmdb(src, self.FORMULAS, self.ELEMS)
        db = lmdb.open(src, map_size=10 ** 8, subdir=False, meminit=False, map_async=True)
        txn = db.begin(write=True)
        txn.put(b"scaled", pickle.dumps(True, protocol=-1))
        txn.commit()
        db.sync()
        db.close()
        with pytest.raises(ValueError, match="scaled"):
            split_lmdb_file(src, str(tmp_path / "out"), method="composition")

    def test_invalid_method_raises(self, tmp_path):
        src = str(tmp_path / "src.lmdb")
        _make_embed_format_lmdb(src, n=5)
        with pytest.raises(ValueError, match="method must be"):
            split_lmdb_file(src, str(tmp_path / "out"), method="bogus")
