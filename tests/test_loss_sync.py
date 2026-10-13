"""The epoch loss the monitors read, and the checkpoint the test stage scores.

`{mode}_loss` is logged twice: rank-local per step (so the progress bar never
sits on a collective) and again from the epoch-end hook, all-reduced, which
overrides it. These tests pin the override and the weighting, since a silent
regression there puts EarlyStopping and every sweep objective back on rank 0's
shard of the validation set.
"""

import argparse
import importlib
from copy import deepcopy

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from qtaim_embed.core.datamodule import QTAIMNodeTaskDataModule
from qtaim_embed.models.utils import load_node_level_model_from_config
from qtaim_embed.utils.training import best_ckpt_path

MODES = ("m_train", "m_val", "m_test")


def _node_model_and_dm(batch_size=4):
    # imported here, not at module scope: a top-level import makes pytest
    # collect TestNodePred's own (slow) tests a second time
    from test_node_models import TestNodePred

    cfg = deepcopy(TestNodePred.config_base)
    cfg["dataset"]["num_workers"] = 0
    # small batches so the validation split spans several of them: with one
    # batch the epoch mean and the last batch are indistinguishable
    cfg["dataset"]["train_batch_size"] = batch_size
    dm = QTAIMNodeTaskDataModule(config=cfg)
    dm.setup(stage="fit")
    fs = dm.train_dataset.feature_size
    cfg["model"]["atom_feature_size"] = fs["atom"]
    cfg["model"]["bond_feature_size"] = fs["bond"]
    cfg["model"]["global_feature_size"] = fs["global"]
    pl.seed_everything(7, workers=True)
    return load_node_level_model_from_config(cfg["model"]), dm


class TestLossMetricWiring:
    def test_keys_are_prefixed(self):
        """nn.ModuleDict rejects the key "train": it shadows Module.train()."""
        model, _ = _node_model_and_dm()
        assert set(model.loss_metrics.keys()) == set(MODES)

    def test_metric_state_stays_out_of_state_dict(self):
        """Old checkpoints must still load: torchmetrics states are not persistent."""
        model, _ = _node_model_and_dm()
        assert not [k for k in model.state_dict() if "loss_metrics" in k]

    def test_epoch_loss_is_the_batch_weighted_mean(self):
        """The logged val_loss must be the whole-set mean, not the last batch."""
        model, dm = _node_model_and_dm()
        seen = []
        original = model.validation_step

        def spy(batch, batch_idx):
            out = original(batch, batch_idx)
            # shared_step pads every target type up to max_nodes before
            # concatenating, so the weight it logs is that padded row count
            n = max(
                batch[1][t].shape[0]
                for t, names in model.hparams.target_dict.items()
                if names != [None] and len(names) > 0
            )
            seen.append((float(out), n))
            return out

        model.validation_step = spy
        trainer = pl.Trainer(
            accelerator="cpu", logger=False, enable_checkpointing=False,
            enable_progress_bar=False, max_epochs=1,
        )
        trainer.validate(model, dm, verbose=False)
        logged = float(trainer.callback_metrics["val_loss"])

        assert len(seen) > 1, "need several batches for the mean to be meaningful"
        expected = sum(l * n for l, n in seen) / sum(n for _, n in seen)
        assert logged == pytest.approx(expected, rel=1e-4)
        # the bug this replaces: the bare key held only the final batch
        if len(set(l for l, _ in seen)) > 1:
            assert logged != pytest.approx(seen[-1][0], rel=1e-6)


class TestBestCkptPath:
    def test_returns_the_saved_path(self, tmp_path):
        cb = ModelCheckpoint(dirpath=str(tmp_path))
        cb.best_model_path = str(tmp_path / "best.ckpt")
        trainer = pl.Trainer(callbacks=[cb], logger=False)
        assert best_ckpt_path(trainer) == str(tmp_path / "best.ckpt")

    def test_none_when_nothing_saved(self, tmp_path):
        """Falls back to current weights rather than raising, unlike ckpt_path="best"."""
        cb = ModelCheckpoint(dirpath=str(tmp_path))
        cb.best_model_path = ""
        trainer = pl.Trainer(callbacks=[cb], logger=False)
        assert best_ckpt_path(trainer) is None

    def test_none_without_a_checkpoint_callback(self):
        trainer = pl.Trainer(logger=False, enable_checkpointing=False)
        assert best_ckpt_path(trainer) is None


@pytest.mark.parametrize(
    "module",
    [
        "qtaim_embed.scripts.train.train_qtaim_node",
        "qtaim_embed.scripts.train.train_qtaim_graph",
        "qtaim_embed.scripts.train.train_qtaim_graph_classifier",
        "qtaim_embed.scripts.train.train_qtaim_link",
    ],
)
def test_test_only_flag_and_best_ckpt_are_wired(module):
    """Every entry point takes --test_only and scores the selected checkpoint."""
    src = importlib.import_module(module)
    path = src.__file__
    text = open(path).read()
    assert '"--test_only"' in text
    assert "ckpt_path=args.test_only" in text
    assert "trainer.test(model, dm)" not in text  # must pass a ckpt_path
    assert "best_ckpt_path(trainer)" in text
    # torch >= 2.6 refuses these checkpoints under weights_only=True
    assert text.count("weights_only=False") >= 2
    # SLURM rejects devices=N against a mismatched --ntasks-per-node, so the
    # config value has to be overridable from the command line
    assert '"--num_devices"' in text
    assert 'config["optim"]["num_devices"] = args.num_devices' in text


class TestCheckpointLoading:
    """torch >= 2.6 defaults torch.load to weights_only=True.

    Every checkpoint this repo writes holds an instantiated nn.ReLU in
    hyper_parameters["activation"], which that mode refuses, so each load site
    passes weights_only=False (these are checkpoints the pipeline itself
    wrote). Without it, passing any ckpt_path to trainer.test raises
    UnpicklingError, which broke --test_only and every post-fit test stage.
    """

    @staticmethod
    def _one_epoch_checkpoint(tmp_path):
        model, dm = _node_model_and_dm()
        cb = ModelCheckpoint(dirpath=str(tmp_path), save_last=True)
        trainer = pl.Trainer(
            max_epochs=1, accelerator="cpu", logger=False, callbacks=[cb],
            enable_progress_bar=False, limit_train_batches=2, limit_val_batches=2,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, dm)
        path = cb.last_model_path or cb.best_model_path
        assert path, "no checkpoint written"
        return model, dm, path

    def test_hparams_hold_a_module_so_weights_only_true_fails(self, tmp_path):
        """If this ever passes, hparams became primitives and the kwargs can go."""
        _, _, path = self._one_epoch_checkpoint(tmp_path)
        with pytest.raises(Exception):
            torch.load(path, map_location="cpu", weights_only=True)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert isinstance(payload["hyper_parameters"]["activation"], torch.nn.Module)

    def test_test_stage_accepts_a_ckpt_path(self, tmp_path):
        """The regression: ckpt_path + the torch 2.6 default raised UnpicklingError."""
        model, dm, path = self._one_epoch_checkpoint(tmp_path)
        trainer = pl.Trainer(
            accelerator="cpu", logger=False, enable_checkpointing=False,
            enable_progress_bar=False, limit_test_batches=2,
        )
        trainer.test(model, dm, ckpt_path=path, weights_only=False, verbose=False)
        assert "test_loss" in trainer.callback_metrics
