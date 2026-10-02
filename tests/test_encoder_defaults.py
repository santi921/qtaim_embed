"""ENCODER_DEFAULTS is the single source of encoder knobs (todo 036)."""

import inspect
from types import SimpleNamespace

import pytest

from qtaim_embed.models.encoders import (
    ENCODER_DEFAULTS,
    attach_encoder,
    build_encoder,
    check_encoder_hparams,
    encoder_kwargs_from_config,
)
from qtaim_embed.models.graph_level.base_gcn import GCNGraphPred
from qtaim_embed.models.graph_level.base_gcn_classifier import GCNGraphPredClassifier
from qtaim_embed.models.link_pred.bond_model import GCNBondPred
from qtaim_embed.models.node_level.base_gcn import GCNNodePred
from qtaim_embed.utils.data import (
    get_default_bond_level_config,
    get_default_graph_level_config,
    get_default_graph_level_config_classif,
    get_default_node_level_config,
)

MODELS = [GCNGraphPred, GCNGraphPredClassifier, GCNNodePred, GCNBondPred]
CONFIGS = [
    get_default_graph_level_config,
    get_default_graph_level_config_classif,
    get_default_node_level_config,
    get_default_bond_level_config,
]


@pytest.mark.parametrize("cls", MODELS, ids=lambda c: c.__name__)
def test_every_model_accepts_every_encoder_knob(cls):
    params = inspect.signature(cls.__init__).parameters
    missing = set(ENCODER_DEFAULTS) - set(params)
    assert not missing, f"{cls.__name__} lacks {sorted(missing)}"


@pytest.mark.parametrize("make", CONFIGS, ids=lambda f: f.__name__)
def test_every_default_config_carries_every_encoder_knob(make):
    model_cfg = make()["model"]
    missing = set(ENCODER_DEFAULTS) - set(model_cfg)
    assert not missing, sorted(missing)
    kwargs = encoder_kwargs_from_config(model_cfg)
    assert set(kwargs) == set(ENCODER_DEFAULTS)


def test_build_encoder_falls_back_for_hparams_saved_before_a_knob_existed():
    old = SimpleNamespace(encoder_fn="schnet", encoder_hidden=8, encoder_cutoff=4.0,
                          encoder_n_interactions=1, encoder_num_gaussians=5)
    enc = build_encoder(old)
    assert enc.embedding.num_embeddings == ENCODER_DEFAULTS["encoder_max_z"] == 119
    small = SimpleNamespace(**{**ENCODER_DEFAULTS, "encoder_fn": "schnet", "encoder_hidden": 8, "encoder_max_z": 40})
    assert build_encoder(small).embedding.num_embeddings == 40
    assert build_encoder(SimpleNamespace()) is None


def test_attach_encoder_width_and_guards():
    none = SimpleNamespace(**ENCODER_DEFAULTS)
    assert attach_encoder(none) == (None, 0)
    sch = SimpleNamespace(**{**ENCODER_DEFAULTS, "encoder_fn": "schnet", "encoder_hidden": 12, "encoder_n_interactions": 1})
    enc, width = attach_encoder(sch)
    assert enc is not None and width == 12
    with pytest.raises(AssertionError):
        check_encoder_hparams("nope")
    with pytest.raises(AssertionError):
        check_encoder_hparams("schnet", compiled=True, conv_fn="ResidualBlock")
    check_encoder_hparams("schnet", compiled=True, conv_fn="ResidualBlockDense")
