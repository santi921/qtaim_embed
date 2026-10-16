"""Per-vertical test metrics: name parsing, streaming stats, and one real pass."""

from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch

from qtaim_embed.core.datamodule import LMDBDataModule
from qtaim_embed.models.utils import load_node_level_model_from_config
from qtaim_embed.scripts.eval.eval_stratified import (
    _Accum,
    _atom_targets,
    evaluate,
    group_key,
)
from qtaim_embed.utils.data import get_default_node_level_config

FIXTURE = "./data/lmdb_node/test/molecule.lmdb"


class TestGroupKey:
    """OMol4M names are "__"-delimited; splitting on the first non-alpha run
    silently mangles them (ani1xbb -> ani, 5A_elytes -> nothing)."""

    @pytest.mark.parametrize(
        "name,depth,expected",
        [
            ("5A_elytes__1002_Hg+2_5_group_1_shell_15_0_1", 1, "5A_elytes"),
            ("ani1xbb__aniBB_010_416098_0_1", 1, "ani1xbb"),
            ("electrolytes_reactivity__73856_6_2_2", 1, "electrolytes_reactivity"),
            ("omol__metal_organics__restart5to6__job_174", 1, "omol"),
            ("omol__metal_organics__restart5to6__job_174", 2, "omol__metal_organics"),
            ("91_12490.xyz", 1, "91_12490.xyz"),  # no separator: whole name
        ],
    )
    def test_splits_on_double_underscore(self, name, depth, expected):
        assert group_key(name, depth) == expected

    def test_depth_beyond_the_name_is_harmless(self):
        assert group_key("ani1xbb__x", 9) == "ani1xbb__x"


class TestAccum:
    def test_matches_closed_form(self):
        y = torch.tensor([[1.0, 10.0], [2.0, 20.0], [4.0, 30.0], [8.0, 65.0]])
        p = torch.tensor([[1.5, 11.0], [2.5, 18.0], [3.0, 33.0], [9.0, 60.0]])
        a = _Accum(2)
        a.update(p[:2], y[:2])  # two chunks: the point of streaming
        a.update(p[2:], y[2:])

        expected_mae = (p - y).abs().mean(0).double()
        ss_res = ((p - y) ** 2).sum(0).double()
        ss_tot = ((y - y.mean(0)) ** 2).sum(0).double()
        assert torch.allclose(a.mae(), expected_mae, atol=1e-10)
        assert torch.allclose(a.r2(), 1 - ss_res / ss_tot, atol=1e-8)

    def test_constant_target_gives_nan_not_inf(self):
        """Zero within-group variance makes R2 undefined, not minus infinity."""
        y = torch.full((5, 1), 3.0)
        a = _Accum(1)
        a.update(y + 0.1, y)
        assert torch.isnan(a.r2()).all()
        assert float(a.mae()) == pytest.approx(0.1, abs=1e-6)

    def test_empty_is_nan(self):
        assert torch.isnan(_Accum(3).r2()).all()


def test_rejects_bond_and_global_targets():
    with pytest.raises(SystemExit, match="only atom targets"):
        _atom_targets({"atom": ["a"], "bond": ["b"], "global": []})


def test_atom_targets_ignores_none_padding():
    assert _atom_targets({"atom": ["a", "b"], "bond": [None], "global": [None]}) == ["a", "b"]


def _atom_only_checkpoint(tmp_path):
    config = get_default_node_level_config()
    config["dataset"] = {
        "train_lmdb": "./data/lmdb_node/train/",
        "val_lmdb": "./data/lmdb_node/train/",
        "test_lmdb": "./data/lmdb_node/test/",
    }
    config["optim"]["num_workers"] = 0
    atom_only = {"atom": ["extra_feat_atom_esp_total"], "bond": [], "global": []}
    config["model"] = deepcopy(config["model"])
    config["model"]["target_dict"] = atom_only

    dm = LMDBDataModule(config=config)
    dm.setup("fit")
    fs = dm.train_dataset.feature_size
    config["model"]["atom_feature_size"] = fs["atom"]
    config["model"]["bond_feature_size"] = fs["bond"]
    config["model"]["global_feature_size"] = fs["global"]
    model = load_node_level_model_from_config(config["model"])

    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False, enable_progress_bar=False,
        enable_checkpointing=False, limit_train_batches=1, limit_val_batches=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dm)
    path = str(tmp_path / "atom_only.ckpt")
    trainer.save_checkpoint(path)
    return path


def test_end_to_end_on_the_fixture(tmp_path):
    """Every molecule lands in exactly one group and the stats are finite."""
    ckpt = _atom_only_checkpoint(tmp_path)
    acc, targets = evaluate(
        ckpt=ckpt, lmdb_src=FIXTURE, group_depth=1, batch_size=4,
        device="cpu", limit=8,
    )
    assert targets == ["extra_feat_atom_esp_total"]
    assert sum(a.n_mols for a in acc.values()) == 8
    assert sum(a.n for a in acc.values()) > 8  # atoms, not molecules
    for key, a in acc.items():
        assert key  # fixture names carry no "__", so one group per molecule
        assert torch.isfinite(a.mae()).all()
        assert a.n > 0
