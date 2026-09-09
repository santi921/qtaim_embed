"""conv_fn="ResidualBlockDense" models must match their ResidualBlock twins."""
import torch
import pytest
from torch_geometric.data import Batch

from qtaim_embed.models.node_level.base_gcn import GCNNodePred
from qtaim_embed.models.graph_level.base_gcn import GCNGraphPred
from qtaim_embed.models.graph_level.base_gcn_classifier import GCNGraphPredClassifier
from qtaim_embed.models.utils import convert_model_to_dense
from qtaim_embed.utils.tests import make_hetero

torch.manual_seed(0)


def _batch():
    g1, _ = make_hetero(4, 3, a2b=[(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)],
                        b2a=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)])
    g2, _ = make_hetero(2, 1, a2b=[(0, 0), (1, 0)], b2a=[(0, 0), (0, 1)])
    g3, _ = make_hetero(6, 5, a2b=[(i, i) for i in range(5)] + [(i + 1, i) for i in range(5)],
                        b2a=[(i, i) for i in range(5)] + [(i, i + 1) for i in range(5)])
    b = Batch.from_data_list([g1, g2, g3])
    feats = {nt: b[nt].feat for nt in b.node_types}
    return b, feats


COMMON = dict(atom_input_size=2, bond_input_size=3, global_input_size=4, n_conv_layers=4,
              resid_n_graph_convs=2, dropout=0.0, activation="ReLU", hidden_size=8,
              embedding_size=8, lr=1e-3)


@pytest.mark.parametrize("batch_norm", [False, True])
def test_node_model_dense_matches_reference(batch_norm):
    ref = GCNNodePred(conv_fn="ResidualBlock", batch_norm=batch_norm,
                      target_dict={"atom": ["a"], "bond": ["b1", "b2"]}, **COMMON)
    dense = convert_model_to_dense(ref)
    assert dense.hparams.conv_fn == "ResidualBlockDense"
    b, feats = _batch()
    ref.train(); dense.train()
    with torch.no_grad():
        for _ in range(2):  # BN running stats through the padded path
            o_ref, o_dense = ref(b, dict(feats)), dense(b, dict(feats))
    ref.eval(); dense.eval()
    with torch.no_grad():
        o_ref, o_dense = ref(b, dict(feats)), dense(b, dict(feats))
    assert set(o_ref) == set(o_dense) == {"atom", "bond"}
    for k in o_ref:
        assert torch.allclose(o_dense[k], o_ref[k], atol=1e-5), k


def test_graph_model_dense_matches_reference():
    ref = GCNGraphPred(conv_fn="ResidualBlock", batch_norm=False, target_dict={"global": ["e"]},
                       global_pooling="SumPoolingThenCat", fc_layer_size=[8], fc_batch_norm=False,
                       pooling_ntypes=["atom", "bond", "global"], **COMMON)
    dense = convert_model_to_dense(ref)
    b, feats = _batch()
    ref.eval(); dense.eval()
    with torch.no_grad():
        assert torch.allclose(dense(b, dict(feats)), ref(b, dict(feats)), atol=1e-5)


def test_classifier_dense_constructs_and_runs():
    m = GCNGraphPredClassifier(conv_fn="ResidualBlockDense", batch_norm=False, target_dict={"global": ["c"]},
                               global_pooling="SumPoolingThenCat", fc_layer_size=[8], fc_batch_norm=False,
                               pooling_ntypes=["atom", "bond", "global"], loss_fn="cross_entropy", **COMMON)
    b, feats = _batch()
    m.eval()
    with torch.no_grad():
        out = m(b, dict(feats))
    assert out.shape[0] == 3


def test_dense_model_trains_one_step_and_saves_hparams():
    m = GCNNodePred(conv_fn="ResidualBlockDense", batch_norm=True, target_dict={"atom": ["a"]},
                    dense_grid=4, **COMMON)
    assert m.hparams.dense_grid == 4
    b, feats = _batch()
    labels = torch.randn(b["atom"].num_nodes, 1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    out = m(b, dict(feats))["atom"]
    loss = torch.nn.functional.mse_loss(out, labels)
    loss.backward()
    # the last block's bond/global heads feed no target, so only the first
    # block's parameters are guaranteed a gradient
    assert all(p.grad is not None for n, p in m.named_parameters() if n.startswith("conv_layers.0."))
    opt.step()


def test_compiled_with_encoder_allowed_only_for_dense():
    with pytest.raises(AssertionError):
        GCNNodePred(conv_fn="ResidualBlock", compiled=True, encoder_fn="schnet", target_dict={"atom": ["a"]}, **COMMON)
    m = GCNNodePred(conv_fn="ResidualBlockDense", compiled=True, encoder_fn="schnet", target_dict={"atom": ["a"]}, **COMMON)
    assert m.forward_fn == m.compiled_forward


def test_loader_passes_bn_first_and_global_aggr_flags():
    from qtaim_embed.models.utils import load_node_level_model_from_config
    from qtaim_embed.utils.data import get_default_node_level_config
    cfg = get_default_node_level_config()["model"]
    cfg.update({"atom_feature_size": 5, "bond_feature_size": 4, "global_feature_size": 3,
                "target_dict": {"atom": ["a"]}, "bn_before_activation": True, "global_aggr": "mean",
                "n_conv_layers": 2})
    m = load_node_level_model_from_config(cfg)
    convs = m.conv_layers[0].layers[0].convs
    assert all(mod.bn_before_activation for mod in convs.values())
    assert {et[1]: mod.graph_conv.aggr for et, mod in convs.items()}["a2g"] == "mean"
    assert {et[1]: mod.graph_conv.aggr for et, mod in convs.items()}["a2b"] == "add"


def test_bn_before_activation_defaults():
    from qtaim_embed.models.utils import load_node_level_model_from_config
    from qtaim_embed.utils.data import get_default_node_level_config
    from qtaim_embed.models.node_level.base_gcn import GCNNodePred
    cfg = get_default_node_level_config()["model"]
    assert cfg["bn_before_activation"] is True
    cfg.update({"atom_feature_size": 5, "bond_feature_size": 4, "global_feature_size": 3,
                "target_dict": {"atom": ["a"]}, "n_conv_layers": 2})
    cfg.pop("bn_before_activation")  # configs written before the key existed
    m = load_node_level_model_from_config(cfg)
    assert m.hparams.bn_before_activation is True
    assert all(mod.bn_before_activation for mod in m.conv_layers[0].layers[0].convs.values())
    # constructors keep the old order so pre-2026-09-09 checkpoints load unchanged
    direct = GCNNodePred(target_dict={"atom": ["a"]}, **COMMON)
    assert direct.hparams.bn_before_activation is False
