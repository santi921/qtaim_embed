"""The node sweep builds each trial's model config from a fixed key list;
encoder_* sweep parameters must reach the model instead of silently
falling back to encoder_fn="none"."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import qtaim_embed.scripts.train.bayes_opt_node as bon
from qtaim_embed.models.encoders import ENCODER_DEFAULTS

SWEEP_DIR = Path(__file__).resolve().parents[1] / "profiling" / "train_configs"


class _Stop(Exception):
    pass


def _trial_model_config(monkeypatch, sweep_path):
    params = json.loads(sweep_path.read_text())
    trial = {
        k: (v["values"][0] if "values" in v else v["min"]) for k, v in params.items()
    }

    class FakeWandb:
        config = trial

        @staticmethod
        @contextmanager
        def init(**kwargs):
            yield None

    monkeypatch.setattr(bon, "wandb", FakeWandb)
    captured = {}

    def make_model(config):
        captured.update(config["model"])
        raise _Stop

    obj = object.__new__(bon.TrainingObject)
    obj.lmdbs = True
    obj.wandb_name, obj.wandb_entity = "p", "e"
    obj.make_model = make_model
    with pytest.raises(_Stop):
        obj.train()
    return trial, captured


@pytest.mark.parametrize(
    "name", ["schnet", "dimenetpp", "equivariant"]
)
def test_encoder_keys_reach_trial_model(monkeypatch, name):
    trial, model_cfg = _trial_model_config(
        monkeypatch, SWEEP_DIR / f"sweep_omol4m_node_{name}.json"
    )
    assert model_cfg["encoder_fn"] == name
    for k in ENCODER_DEFAULTS:
        if k in trial:
            assert model_cfg[k] == trial[k], k


def test_sweep_without_encoder_keys_has_none(monkeypatch):
    _, model_cfg = _trial_model_config(monkeypatch, SWEEP_DIR / "sweep_hpc_node.json")
    assert not any(k in model_cfg for k in ENCODER_DEFAULTS)
