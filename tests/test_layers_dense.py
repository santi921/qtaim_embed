"""DenseResidualBlock must equal ResidualBlock (HeteroConv of GraphConvDropoutBatch)
after copy_residual_block_weights, on padded batches of unequal molecules."""
import torch
import pytest
from torch_geometric.data import Batch

from qtaim_embed.models.layers import ResidualBlock, EDGE_TYPE_MAP
from qtaim_embed.models.layers_dense import (
    DenseResidualBlock,
    MaskedBatchNorm,
    copy_residual_block_weights,
    to_dense_hetero,
)
from qtaim_embed.utils.models import get_layer_args
from qtaim_embed.utils.tests import hyperparams, make_hetero

H = 12


def _graphs():
    g1, _ = make_hetero(4, 3, a2b=[(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)],
                        b2a=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)])
    g2, _ = make_hetero(2, 1, a2b=[(0, 0), (1, 0)], b2a=[(0, 0), (0, 1)])
    g3, _ = make_hetero(6, 5, a2b=[(i, i) for i in range(5)] + [(i + 1, i) for i in range(5)],
                        b2a=[(i, i) for i in range(5)] + [(i, i + 1) for i in range(5)])
    return [g1, g2, g3]


def _batch_and_feats(seed=0):
    torch.manual_seed(seed)
    batch = Batch.from_data_list(_graphs())
    feats = {nt: torch.randn(batch[nt].num_nodes, H) for nt in ("atom", "bond", "global")}
    return batch, feats


def _hparams(batch_norm=False, dropout=0.0, target_dict=None, **extra):
    return hyperparams(config={
        "atom_input_size": H, "bond_input_size": H, "global_input_size": H, "n_conv_layers": 4,
        "norm": "both", "bias": True, "batch_norm_tf": batch_norm, "dropout": dropout,
        "activation": torch.nn.ReLU(), "embedding_size": H, "hidden_size": H,
        "conv_fn": "ResidualBlock", "allow_zero_in_degree": True,
        "target_dict": target_dict or {"atom": ["a"], "bond": ["b1", "b2"], "global": ["g", "g2", "g3"]},
        **extra,
    })


def _pair(layer_ind=0, output_block=False, batch_norm=False, dropout=0.0, **extra):
    hp = _hparams(batch_norm=batch_norm, dropout=dropout, **extra)
    args = get_layer_args(hp, layer_ind=layer_ind, embedding_in=True, activation=hp.activation)
    ref = ResidualBlock(args, aggregate="sum", resid_n_graph_convs=2, output_block=output_block)
    dense = DenseResidualBlock(args, aggregate="sum", resid_n_graph_convs=2, output_block=output_block)
    copy_residual_block_weights(ref, dense)
    return ref, dense


def _run_both(ref, dense, batch, feats, grid=4, index_bn=True):
    ei = {et: batch[et].edge_index for et in batch.edge_types}
    out_ref = ref({k: v.clone() for k, v in feats.items()}, ei)
    db = to_dense_hetero(batch, feats, grid=grid)
    if not index_bn:
        db.valid = None  # compiled-path semantics: masked arithmetic batch norm
    out_dense = db.to_flat(dense(db, db.x))
    return out_ref, out_dense, db


class TestDenseResidualBlock:
    def test_parity_hidden_block(self):
        ref, dense = _pair()
        ref.eval(); dense.eval()
        batch, feats = _batch_and_feats()
        out_ref, out_dense, _ = _run_both(ref, dense, batch, feats)
        for nt in out_ref:
            assert out_dense[nt].shape == out_ref[nt].shape
            assert torch.allclose(out_dense[nt], out_ref[nt], atol=1e-5), nt

    def test_parity_output_block_with_target_dims(self):
        ref, dense = _pair(layer_ind=-1, output_block=True)
        ref.eval(); dense.eval()
        batch, feats = _batch_and_feats(1)
        out_ref, out_dense, _ = _run_both(ref, dense, batch, feats)
        assert out_ref["atom"].shape[1] == 1 and out_ref["bond"].shape[1] == 2 and out_ref["global"].shape[1] == 3
        for nt in out_ref:
            assert torch.allclose(out_dense[nt], out_ref[nt], atol=1e-5), nt
        assert dense.out_feats == ref.out_feats

    @pytest.mark.parametrize("index_bn", [True, False])
    def test_parity_with_batch_norm_train_and_eval(self, index_bn):
        ref, dense = _pair(batch_norm=True)
        ref.train(); dense.train()
        batch, feats = _batch_and_feats(2)
        for _ in range(3):  # running stats must track the unpadded reference
            out_ref, out_dense, _ = _run_both(ref, dense, batch, feats, index_bn=index_bn)
        for nt in out_ref:
            assert torch.allclose(out_dense[nt], out_ref[nt], atol=1e-4), nt
        for hetero, dl in zip(ref.layers, dense.layers):
            for t, nt in enumerate(("atom", "bond", "global")):
                out = dl.out_feats[nt]
                for s, e in enumerate(("b2a g2a a2a".split(), "a2b g2b b2b".split(), "a2g b2g g2g".split())[t]):
                    bn = hetero.convs[EDGE_TYPE_MAP[e]].batch_norm
                    sl = slice(s * out, (s + 1) * out)
                    assert torch.allclose(dl.norms[t].running_mean[sl], bn.running_mean, atol=1e-5)
                    assert torch.allclose(dl.norms[t].running_var[sl], bn.running_var, atol=1e-4)
        ref.eval(); dense.eval()
        out_ref, out_dense, _ = _run_both(ref, dense, batch, feats, index_bn=index_bn)
        for nt in out_ref:
            assert torch.allclose(out_dense[nt], out_ref[nt], atol=1e-4), nt

    def test_padded_rows_stay_zero_and_shapes_are_grid_multiples(self):
        _, dense = _pair()
        dense.eval()
        batch, feats = _batch_and_feats(3)
        db = to_dense_hetero(batch, feats, grid=8)
        assert db.shape == (3, 8, 8)
        out = dense(db, db.x)
        for nt in ("atom", "bond"):
            pad = (db.mask[nt] == 0).squeeze(-1)
            assert out[nt][pad].abs().sum() == 0
        # fixed shape override for bucketed loaders
        db2 = to_dense_hetero(batch, feats, shape=(16, 24))
        assert db2.shape == (3, 16, 24)

    def test_incidence_matches_edge_index_and_flat_round_trip(self):
        batch, feats = _batch_and_feats(4)
        db = to_dense_hetero(batch, feats, grid=4)
        flat = db.to_flat()
        for nt in feats:
            assert torch.equal(flat[nt], feats[nt])
        ei = batch[EDGE_TYPE_MAP["a2b"]].edge_index
        ptr_a, ptr_b = batch["atom"].ptr, batch["bond"].ptr
        g = batch["atom"].batch[ei[0]]
        assert db.inc_a2b.sum() == ei.shape[1]
        assert (db.inc_a2b[g, ei[1] - ptr_b[g], ei[0] - ptr_a[g]] == 1).all()
        assert torch.equal(db.inc_b2a.transpose(1, 2), db.inc_a2b)

    def test_dropout_active_in_train_mode(self):
        _, dense = _pair(dropout=0.5)
        dense.train()
        batch, feats = _batch_and_feats(5)
        db = to_dense_hetero(batch, feats, grid=4)
        a, b = dense(db, db.x)["atom"], dense(db, db.x)["atom"]
        assert not torch.allclose(a, b)


def test_masked_batch_norm_matches_batchnorm1d_on_valid_rows():
    torch.manual_seed(0)
    x = torch.randn(10, 5)
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=torch.float32).unsqueeze(-1)
    ref = torch.nn.BatchNorm1d(5)
    mbn = MaskedBatchNorm(5)
    valid = mask.squeeze(-1).bool()
    y_ref = ref(x[valid])
    y = mbn(x, mask)
    assert torch.allclose(y[valid], y_ref, atol=1e-5)
    assert torch.allclose(mbn.running_mean, ref.running_mean, atol=1e-6)
    assert torch.allclose(mbn.running_var, ref.running_var, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast path")
def test_masked_batch_norm_bf16_masked_path_tracks_fp32_stats():
    torch.manual_seed(0)
    ref = torch.nn.BatchNorm1d(16).cuda()
    mbn = MaskedBatchNorm(16).cuda()
    x = (torch.randn(4096, 16, device="cuda") * 3 + 1).to(torch.bfloat16)
    mask = torch.ones(4096, 1, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out_ref = ref(x)
        out = mbn(x, mask)
    assert out.dtype == out_ref.dtype
    assert torch.allclose(mbn.running_mean, ref.running_mean, atol=1e-5, rtol=1e-4)
    assert torch.allclose(mbn.running_var, ref.running_var, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out.float(), out_ref.float(), atol=2e-2, rtol=1e-2)
    # the index (eager) path agrees too
    valid = torch.arange(4096, device="cuda")
    mbn2 = MaskedBatchNorm(16).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out2 = mbn2(x, mask, valid)
    assert torch.allclose(mbn2.running_mean, ref.running_mean, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out2.float(), out_ref.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("index_bn", [True, False])
@pytest.mark.parametrize("flags", [
    {"bn_before_activation": True},
    {"global_aggr": "mean"},
    {"bn_before_activation": True, "global_aggr": "mean"},
])
def test_parity_with_bn_first_and_mean_global_aggregation(flags, index_bn):
    batch, feats = _batch_and_feats()
    ref, dense = _pair(batch_norm=True, **flags)
    for mode in (True, False):
        ref.train(mode); dense.train(mode)
        out_ref, out_dense, _ = _run_both(ref, dense, batch, feats, index_bn=index_bn)
        for nt in out_ref:
            assert torch.allclose(out_ref[nt], out_dense[nt], atol=1e-5), (nt, mode, flags)
    convs = ref.layers[0].convs
    for et, mod in convs.items():
        assert mod.bn_before_activation == flags.get("bn_before_activation", False)
        want = "mean" if flags.get("global_aggr") == "mean" and et[1] in ("a2g", "b2g") else "add"
        assert mod.graph_conv.aggr == want, (et, mod.graph_conv.aggr)


def test_bn_before_activation_output_is_nonnegative_with_relu():
    batch, feats = _batch_and_feats()
    ref, _ = _pair(batch_norm=True, bn_before_activation=True)
    ref.eval()
    ei = {et: batch[et].edge_index for et in batch.edge_types}
    # ResidualBlock adds the input back, so check the per-layer conv output instead
    out = ref.layers[0]({k: v.clone() for k, v in feats.items()}, ei)
    assert all((v >= 0).all() for v in out.values())


@pytest.mark.parametrize("index_bn", [True, False])
def test_output_block_parity_and_linear_head_with_bn_first(index_bn):
    batch, feats = _batch_and_feats()
    ref, dense = _pair(layer_ind=-1, output_block=True, batch_norm=True,
                       bn_before_activation=True, global_aggr="mean")
    final = ref.layers[-1].convs
    assert all(mod.activation is None and mod.batch_norm is None for mod in final.values())
    inner = ref.layers[0].convs
    assert all(mod.activation is not None and mod.batch_norm is not None and mod.bn_before_activation
               for mod in inner.values())
    for mode in (True, False):
        ref.train(mode); dense.train(mode)
        out_ref, out_dense, _ = _run_both(ref, dense, batch, feats, index_bn=index_bn)
        for nt in out_ref:
            assert torch.allclose(out_ref[nt], out_dense[nt], atol=1e-5), (nt, mode)
    ref.eval()
    ei = {et: batch[et].edge_index for et in batch.edge_types}
    out = ref({k: v.clone() for k, v in feats.items()}, ei)
    assert any((v < 0).any() for v in out.values())  # predictions are not clamped
