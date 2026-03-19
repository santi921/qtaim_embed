"""
Numerical correctness tests for PyG layer implementations.

These tests verify that the layers compute mathematically correct outputs
by using known inputs and manually-computed expected values:

1. GraphConvDropoutBatch: out_i = lin_root(x_i) + sum(lin_rel(x_j) for j -> i)
2. ResidualBlock: zero weights -> residual pass-through (out = input)
3. UnifySize: exact W*x linear transform
4. SumPoolingThenCat / MeanPoolingThenCat: exact aggregated values
5. Set2Set: LSTM attention pooling value verification and batch independence
6. End-to-end convergence: loss decreases over training steps
7. Diverse molecule numerical stability across all pooling functions
"""

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax

from qtaim_embed.models.layers import (
    GraphConvDropoutBatch,
    MeanPoolingThenCat,
    ResidualBlock,
    Set2Set,
    Set2SetThenCat,
    SumPoolingThenCat,
    UnifySize,
)
from qtaim_embed.utils.tests import (
    make_hetero_graph,
    make_test_model,
    get_dataset_graph_level,
    get_hyperparams_resid,
)
from qtaim_embed.utils.data import get_default_graph_level_config
from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask

ATOL = 1e-5


def _zero_resid_weights(resid_block: ResidualBlock) -> None:
    """Set all conv weights in a ResidualBlock to zero."""
    for hetero_conv in resid_block.layers:
        for conv_module in hetero_conv.convs.values():
            conv_module.graph_conv.lin_root.weight.data.zero_()
            conv_module.graph_conv.lin_rel.weight.data.zero_()
            conv_module.graph_conv.lin_rel.bias.data.zero_()


class TestGraphConvDropoutBatch:
    """Verify the core conv formula: out_i = lin_root(x_i) + sum(lin_rel(x_j) for j->i)."""

    def _identity_conv(self, in_feats: int) -> GraphConvDropoutBatch:
        """Return a conv with identity root/rel weights and zero bias."""
        conv = GraphConvDropoutBatch(
            in_feats=in_feats,
            out_feats=in_feats,
            dropout=0.0,
            batch_norm_tf=False,
            activation=None,
        )
        torch.nn.init.eye_(conv.graph_conv.lin_root.weight)
        torch.nn.init.eye_(conv.graph_conv.lin_rel.weight)
        conv.graph_conv.lin_rel.bias.data.zero_()
        conv.eval()
        return conv

    def test_chain_graph(self):
        """Linear chain 0->1->2: out_i = x_i + sum of incoming neighbor features."""
        conv = self._identity_conv(2)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]])  # 0->1, 1->2

        with torch.no_grad():
            out = conv(x, edge_index)

        # node 0: x_0 + 0 (no in-edges)       = [1, 0]
        # node 1: x_1 + x_0                    = [1, 1]
        # node 2: x_2 + x_1                    = [1, 2]
        expected = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        assert torch.allclose(out, expected, atol=ATOL), (
            f"Chain graph:\nExpected:\n{expected}\nGot:\n{out}"
        )

    def test_no_edges_returns_root_transform(self):
        """With no edges, out_i = lin_root(x_i). With identity, out_i = x_i."""
        conv = self._identity_conv(4)
        x = torch.randn(5, 4)
        edge_index = torch.zeros(2, 0, dtype=torch.long)

        with torch.no_grad():
            out = conv(x, edge_index)

        assert torch.allclose(out, x, atol=ATOL), (
            f"No-edge graph should return x unchanged, diff max: {(out - x).abs().max()}"
        )

    def test_self_loop_doubles_features(self):
        """Self-loop edge i->i: out_i = lin_root(x_i) + lin_rel(x_i) = 2*x_i."""
        n = 4
        conv = self._identity_conv(3)
        x = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
        edge_index = torch.tensor([[i for i in range(n)], [i for i in range(n)]])

        with torch.no_grad():
            out = conv(x, edge_index)

        assert torch.allclose(out, 2.0 * x, atol=ATOL), (
            f"Self-loop should double features, diff max: {(out - 2*x).abs().max()}"
        )

    def test_fan_in_aggregation(self):
        """Node 3 receives from nodes 0,1,2: out_3 = x_3 + x_0 + x_1 + x_2."""
        conv = self._identity_conv(2)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])

        with torch.no_grad():
            out = conv(x, edge_index)

        # nodes 0,1,2: no in-edges -> out = x_i
        # node 3: [0,0] + [1,0] + [0,1] + [1,1] = [2, 2]
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
        assert torch.allclose(out, expected, atol=ATOL), (
            f"Fan-in:\nExpected:\n{expected}\nGot:\n{out}"
        )

    def test_weighted_edges_scale_messages(self):
        """Edge weights w_ij scale the neighbor message: out_i = x_i + sum(w_ij * x_j)."""
        conv = self._identity_conv(2)
        x = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        edge_index = torch.tensor([[0, 1], [2, 2]])  # 0->2 and 1->2
        edge_weight = torch.tensor([0.5, 3.0])  # scale messages

        with torch.no_grad():
            out = conv(x, edge_index, edge_weight)

        # node 2: x_2 + 0.5*x_0 + 3.0*x_1 = [0,0] + [1,0] + [0,6] = [1, 6]
        expected = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 6.0]])
        assert torch.allclose(out, expected, atol=ATOL), (
            f"Weighted edges:\nExpected:\n{expected}\nGot:\n{out}"
        )


class TestUnifySize:
    """Verify UnifySize applies an exact linear transform per node type."""

    def test_known_weights_and_features(self):
        """Set explicit weights; verify output = W*x for each node type."""
        input_dim = {"atom": 2, "bond": 3, "global": 4}
        output_dim = 4
        uni = UnifySize(input_dim, output_dim)

        # atom: pad 2->4 (first 2 rows are identity, rest zero)
        uni.linears[0].weight.data = torch.zeros(4, 2)
        uni.linears[0].weight.data[:2, :2] = torch.eye(2)
        # bond: pad 3->4
        uni.linears[1].weight.data = torch.zeros(4, 3)
        uni.linears[1].weight.data[:3, :3] = torch.eye(3)
        # global: identity 4->4
        uni.linears[2].weight.data = torch.eye(4)

        feats = {
            "atom": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bond": torch.tensor([[1.0, 0.0, -1.0]]),
            "global": torch.tensor([[0.5, -0.5, 1.0, -1.0]]),
        }

        with torch.no_grad():
            out = uni(feats)

        assert torch.allclose(
            out["atom"], torch.tensor([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]]), atol=ATOL
        )
        assert torch.allclose(out["bond"], torch.tensor([[1.0, 0.0, -1.0, 0.0]]), atol=ATOL)
        assert torch.allclose(
            out["global"], torch.tensor([[0.5, -0.5, 1.0, -1.0]]), atol=ATOL
        )

    def test_output_shape(self):
        """All node types are mapped to the same output dimension."""
        input_dim = {"atom": 5, "bond": 7, "global": 3}
        output_dim = 16
        uni = UnifySize(input_dim, output_dim)
        feats = {nt: torch.randn(4, sz) for nt, sz in input_dim.items()}

        with torch.no_grad():
            out = uni(feats)

        for nt in input_dim:
            assert out[nt].shape == (4, output_dim), (
                f"Expected ({4}, {output_dim}), got {out[nt].shape} for {nt}"
            )


class TestResidualBlock:
    """Verify residual skip connection behavior."""

    def test_zero_weights_is_identity(self):
        """
        With all conv weights zeroed, each layer outputs zeros for all nodes.
        The skip connection then returns exactly the original input:
            out = layer2(layer1(x)) + x = 0 + x = x
        """
        input_dim = {"atom": 2, "bond": 3, "global": 4}
        hidden = 10
        uni = UnifySize(input_dim, hidden)
        layer_args = get_hyperparams_resid()  # hidden=10, batch_norm_tf=False, dropout=0.0

        resid = ResidualBlock(
            layer_args=layer_args,
            resid_n_graph_convs=2,
            aggregate="sum",
            output_block=False,
        )
        _zero_resid_weights(resid)
        resid.eval()

        graph, feats = make_hetero_graph()
        with torch.no_grad():
            x_unified = uni(feats)
            out = resid(x_unified, graph.edge_index_dict)

        for ntype in x_unified:
            assert torch.allclose(out[ntype], x_unified[ntype], atol=ATOL), (
                f"ResidualBlock zero-weights should be identity for node type '{ntype}'"
            )

    def test_output_block_no_residual(self):
        """
        With output_block=True the skip connection is disabled.
        All-zero weights should produce all-zero output.
        """
        input_dim = {"atom": 2, "bond": 3, "global": 4}
        hidden = 10
        uni = UnifySize(input_dim, hidden)
        layer_args = get_hyperparams_resid()
        # output_block also needs '*_inner' keys for the penultimate layers
        for et in ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]:
            layer_args[et + "_inner"] = dict(layer_args[et])

        resid = ResidualBlock(
            layer_args=layer_args,
            resid_n_graph_convs=2,
            aggregate="sum",
            output_block=True,
        )
        _zero_resid_weights(resid)
        resid.eval()

        graph, feats = make_hetero_graph()
        with torch.no_grad():
            x_unified = uni(feats)
            out = resid(x_unified, graph.edge_index_dict)

        for ntype in out:
            assert torch.allclose(out[ntype], torch.zeros_like(out[ntype]), atol=ATOL), (
                f"output_block=True with zero weights should give zeros for '{ntype}'"
            )


class TestPoolingLayers:
    """Verify SumPoolingThenCat and MeanPoolingThenCat with known exact values."""

    # 3 atoms (2-dim), 2 bonds (3-dim), 1 global (2-dim, direct_cat)
    atom_feats = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
    bond_feats = torch.tensor([[1.0, -1.0, 0.0], [2.0, 1.0, -1.0]])
    global_feats = torch.tensor([[5.0, -5.0]])
    feats = {"atom": atom_feats, "bond": bond_feats, "global": global_feats}
    batch_dict = {
        "atom": torch.zeros(3, dtype=torch.long),
        "bond": torch.zeros(2, dtype=torch.long),
        "global": torch.zeros(1, dtype=torch.long),
    }
    ntypes = ["atom", "bond", "global"]
    ntypes_direct_cat = ["global"]
    in_feats = [2, 3, 2]

    def test_sum_pooling_exact(self):
        """
        Sum atoms = [4, 1], sum bonds = [3, 0, -1], global direct = [5, -5].
        Expected output = [4, 1, 3, 0, -1, 5, -5].
        """
        pool = SumPoolingThenCat(
            ntypes=self.ntypes,
            in_feats=self.in_feats,
            ntypes_direct_cat=self.ntypes_direct_cat,
        )
        with torch.no_grad():
            out = pool(self.feats, self.batch_dict)

        expected = torch.tensor([[4.0, 1.0, 3.0, 0.0, -1.0, 5.0, -5.0]])
        assert torch.allclose(out, expected, atol=ATOL), (
            f"Sum pooling:\nExpected: {expected}\nGot:      {out}"
        )

    def test_mean_pooling_exact(self):
        """
        Mean atoms = [4/3, 1/3], mean bonds = [1.5, 0, -0.5], global direct = [5, -5].
        """
        pool = MeanPoolingThenCat(
            ntypes=self.ntypes,
            in_feats=self.in_feats,
            ntypes_direct_cat=self.ntypes_direct_cat,
        )
        with torch.no_grad():
            out = pool(self.feats, self.batch_dict)

        expected = torch.tensor([[4 / 3, 1 / 3, 1.5, 0.0, -0.5, 5.0, -5.0]])
        assert torch.allclose(out, expected, atol=1e-4), (
            f"Mean pooling:\nExpected: {expected}\nGot:      {out}"
        )

    def test_sum_pooling_batched_graphs(self):
        """Two graphs: each pooled independently and stacked into batch dimension."""
        pool = SumPoolingThenCat(ntypes=["atom"], in_feats=[2], ntypes_direct_cat=[])
        feats = {"atom": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])}
        batch = {"atom": torch.tensor([0, 0, 1, 1])}

        with torch.no_grad():
            out = pool(feats, batch)

        expected = torch.tensor([[4.0, 6.0], [12.0, 14.0]])
        assert out.shape == (2, 2)
        assert torch.allclose(out, expected, atol=ATOL), (
            f"Batched sum:\nExpected {expected}\nGot {out}"
        )

    def test_mean_pooling_batched_graphs(self):
        """Mean over each graph's atoms independently."""
        pool = MeanPoolingThenCat(ntypes=["atom"], in_feats=[2], ntypes_direct_cat=[])
        feats = {"atom": torch.tensor([[0.0, 4.0], [2.0, 0.0], [1.0, 1.0], [3.0, 3.0]])}
        batch = {"atom": torch.tensor([0, 0, 1, 1])}

        with torch.no_grad():
            out = pool(feats, batch)

        expected = torch.tensor([[1.0, 2.0], [2.0, 2.0]])
        assert out.shape == (2, 2)
        assert torch.allclose(out, expected, atol=ATOL), (
            f"Batched mean:\nExpected {expected}\nGot {out}"
        )


class TestConvergence:
    """Verify full model converges on the standard test dataset."""

    def test_graph_level_loss_decreases(self):
        """
        After 100 gradient steps with Adam, training loss must be strictly lower
        than the initial loss (before any weight updates).
        """
        torch.manual_seed(42)

        dataset = get_dataset_graph_level(
            log_scale_features=True,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )
        dl = DataLoaderMoleculeGraphTask(
            dataset, batch_size=len(dataset.graphs), shuffle=False
        )
        batch_graph, batch_label = next(iter(dl))

        config = get_default_graph_level_config()
        config["model"]["atom_feature_size"] = dataset.feature_size["atom"]
        config["model"]["bond_feature_size"] = dataset.feature_size["bond"]
        config["model"]["global_feature_size"] = dataset.feature_size["global"]
        config["model"]["target_dict"]["global"] = dataset.target_dict["global"]
        config["model"]["initializer"] = None
        config["model"]["fc_batch_norm"] = False  # avoid BN instability on small batches
        config["model"]["hidden_size"] = 32
        config["model"]["n_conv_layers"] = 2

        model = load_graph_level_model_from_config(config["model"])

        def _loss(model):
            feats = {
                nt: batch_graph[nt].feat
                for nt in batch_graph.node_types
                if hasattr(batch_graph[nt], "feat")
            }
            logits = model(batch_graph, feats)
            return F.mse_loss(logits, batch_label["global"])

        model.eval()
        with torch.no_grad():
            initial_loss = _loss(model).item()

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(100):
            model.train()
            loss = _loss(model)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            final_loss = _loss(model).item()

        assert final_loss < initial_loss, (
            f"Model did not converge: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# TorchMetrics MultioutputWrapper DDP sync verification
# ---------------------------------------------------------------------------

def test_multioutput_wrapper_dist_reduce_states():
    """Verify that MultioutputWrapper metrics have dist_reduce_fx on all states.

    In DDP, TorchMetrics .compute() performs an all-reduce on internal states
    before computing the final metric value. This test checks that
    MultioutputWrapper preserves the dist_reduce_fx on all metric states,
    ensuring sync_dist=False is safe when logging .compute() results.
    """
    import torchmetrics
    from torchmetrics.wrappers import MultioutputWrapper

    ntasks = 3
    metrics = {
        "r2": MultioutputWrapper(torchmetrics.R2Score(), num_outputs=ntasks),
        "mae": MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=ntasks),
        "mse": MultioutputWrapper(torchmetrics.MeanSquaredError(), num_outputs=ntasks),
    }

    for name, metric in metrics.items():
        # Check that the wrapper has output metrics with dist_reduce_fx
        assert hasattr(metric, "metrics"), (
            f"{name}: MultioutputWrapper missing metrics"
        )
        for i, sub_metric in enumerate(metric.metrics):
            for state_name in sub_metric._defaults:
                reduce_fx = sub_metric._reductions.get(state_name)
                assert reduce_fx is not None, (
                    f"{name}[{i}].{state_name} has no dist_reduce_fx -- "
                    f"DDP all-reduce will not work"
                )


def test_multioutput_wrapper_split_vs_full():
    """Verify MultioutputWrapper gives same result on split data vs full data.

    Simulates what happens in DDP: each rank updates metrics with its shard,
    then states are summed (all-reduced). The result should match computing
    on the full dataset in one shot.
    """
    import torchmetrics
    from torchmetrics.wrappers import MultioutputWrapper

    torch.manual_seed(42)
    ntasks = 2
    n_samples = 100
    preds = torch.randn(n_samples, ntasks)
    targets = preds + 0.1 * torch.randn(n_samples, ntasks)

    # Full dataset computation
    full_mae = MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=ntasks)
    full_mae.update(preds, targets)
    full_result = full_mae.compute()

    # Simulated 2-rank split: each rank updates independently, then states are summed
    split = n_samples // 2
    rank0_mae = MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=ntasks)
    rank1_mae = MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=ntasks)
    rank0_mae.update(preds[:split], targets[:split])
    rank1_mae.update(preds[split:], targets[split:])

    # Simulate DDP all-reduce: sum the internal states across ranks
    for sub0, sub1 in zip(rank0_mae.metrics, rank1_mae.metrics):
        for state_name in sub0._defaults:
            state0 = getattr(sub0, state_name)
            state1 = getattr(sub1, state_name)
            setattr(sub0, state_name, state0 + state1)

    split_result = rank0_mae.compute()

    torch.testing.assert_close(
        split_result, full_result, atol=1e-5, rtol=1e-5,
        msg=f"Split computation ({split_result}) != full ({full_result})"
    )


# ---------------------------------------------------------------------------
# Set2Set numerical correctness
# ---------------------------------------------------------------------------


class TestSet2SetNumericalCorrectness:
    """Verify Set2Set produces correct values, not just correct shapes."""

    def test_single_graph_single_iter_values(self):
        """Manual step-by-step LSTM+attention math matches module output."""
        torch.manual_seed(0)
        s2s = Set2Set(input_dim=2, n_iters=1, n_layers=1, ntype="atom")
        s2s.eval()

        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        batch = torch.tensor([0, 0, 0])

        with torch.no_grad():
            actual = s2s(x, batch)

        # Manual computation replicating Set2Set.forward
        batch_size = 1
        h = (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))
        q_star = torch.zeros(1, 4)

        with torch.no_grad():
            q, _ = s2s.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, 2)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            alpha = softmax(e, batch, dim=0)
            readout = global_add_pool(x * alpha, batch, size=batch_size)
            expected = torch.cat([q, readout], dim=-1)

        assert actual.shape == (1, 4)
        assert torch.allclose(actual, expected, atol=ATOL), (
            f"Set2Set values:\nExpected: {expected}\nGot:      {actual}"
        )

    def test_multi_graph_batch_independence(self):
        """Batched output[i] == individual graph output for each graph."""
        torch.manual_seed(42)
        s2s = Set2Set(input_dim=2, n_iters=2, n_layers=1, ntype="atom")
        s2s.eval()

        x_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        x_b = torch.tensor([[2.0, 2.0]])

        with torch.no_grad():
            out_a = s2s(x_a, torch.tensor([0, 0]))
            out_b = s2s(x_b, torch.tensor([0]))

        x_cat = torch.cat([x_a, x_b], dim=0)
        batch = torch.tensor([0, 0, 1])
        with torch.no_grad():
            out_batched = s2s(x_cat, batch)

        assert out_batched.shape == (2, 4)
        assert torch.allclose(out_batched[0], out_a[0], atol=ATOL), (
            f"Graph 0 mismatch:\nbatched={out_batched[0]}\nindiv={out_a[0]}"
        )
        assert torch.allclose(out_batched[1], out_b[0], atol=ATOL), (
            f"Graph 1 mismatch:\nbatched={out_batched[1]}\nindiv={out_b[0]}"
        )

    def test_n_iters_changes_output(self):
        """More iterations produce different (refined) output."""
        torch.manual_seed(7)
        s2s_1 = Set2Set(input_dim=3, n_iters=1, n_layers=1, ntype="atom")
        s2s_3 = Set2Set(input_dim=3, n_iters=3, n_layers=1, ntype="atom")
        s2s_3.load_state_dict(s2s_1.state_dict())
        s2s_1.eval()
        s2s_3.eval()

        x = torch.randn(5, 3)
        batch = torch.tensor([0, 0, 0, 1, 1])

        with torch.no_grad():
            out_1 = s2s_1(x, batch)
            out_3 = s2s_3(x, batch)

        assert not torch.allclose(out_1, out_3, atol=1e-3), (
            "n_iters=1 and n_iters=3 should give different outputs"
        )

    def test_set2set_then_cat_matches_parts(self):
        """Set2SetThenCat output matches individual Set2Set per ntype."""
        torch.manual_seed(99)
        ntypes = ["atom", "bond"]
        ntypes_direct_cat = []
        in_feats = [2, 3]

        pool = Set2SetThenCat(
            ntypes=ntypes,
            ntypes_direct_cat=ntypes_direct_cat,
            in_feats=in_feats,
            n_iters=2,
            n_layers=1,
        )
        pool.eval()

        feats = {
            "atom": torch.randn(4, 2),
            "bond": torch.randn(3, 3),
        }
        batch_dict = {
            "atom": torch.tensor([0, 0, 1, 1]),
            "bond": torch.tensor([0, 0, 1]),
        }

        with torch.no_grad():
            out = pool(feats, batch_dict)

        assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"

        with torch.no_grad():
            atom_out = pool.layers["atom"](feats["atom"], batch_dict["atom"])
            bond_out = pool.layers["bond"](feats["bond"], batch_dict["bond"])

        expected = torch.cat([atom_out, bond_out], dim=-1)
        assert torch.allclose(out, expected, atol=ATOL), (
            f"Concatenation mismatch:\nExpected: {expected}\nGot:      {out}"
        )

    @pytest.mark.parametrize("dim", [1, 4, 16])
    def test_output_dim_2x_input(self, dim):
        """Output is always (B, 2*input_dim)."""
        s2s = Set2Set(input_dim=dim, n_iters=1, n_layers=1, ntype="atom")
        s2s.eval()
        x = torch.randn(5, dim)
        batch = torch.tensor([0, 0, 0, 1, 1])
        with torch.no_grad():
            out = s2s(x, batch)
        assert out.shape == (2, 2 * dim)


# ---------------------------------------------------------------------------
# Diverse molecule numerical stability
# ---------------------------------------------------------------------------


def _make_diverse_model(dataset, pooling_fn="SumPoolingThenCat"):
    """Build a small model matching the given dataset's feature sizes."""
    return make_test_model(
        atom_feat_size=dataset.feature_size["atom"],
        bond_feat_size=dataset.feature_size["bond"],
        global_feat_size=dataset.feature_size["global"],
        target_dict={"global": dataset.target_dict["global"]},
        pooling_fn=pooling_fn,
    )


class TestDiverseMoleculeNumerical:
    """Run real test dataset through the model, check for NaN/Inf."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.dataset = get_dataset_graph_level(
            log_scale_features=True,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

    def _full_batch(self):
        dl = DataLoaderMoleculeGraphTask(
            self.dataset, batch_size=len(self.dataset.graphs), shuffle=False
        )
        batch_graph, batch_label = next(iter(dl))
        return batch_graph, batch_label

    def _forward(self, model, graph):
        """Extract features and run model forward."""
        feats = {
            nt: graph[nt].feat
            for nt in graph.node_types
            if hasattr(graph[nt], "feat")
        }
        with torch.no_grad():
            return model(graph, feats)

    def test_output_shape_matches_targets(self):
        """output.shape == (num_molecules, num_targets)."""
        model = _make_diverse_model(self.dataset)
        batch_graph, _ = self._full_batch()
        out = self._forward(model, batch_graph)

        num_molecules = len(self.dataset.graphs)
        num_targets = len(self.dataset.target_dict["global"])
        assert out.shape == (num_molecules, num_targets), (
            f"Expected ({num_molecules}, {num_targets}), got {out.shape}"
        )

    def test_per_molecule_forward_consistency(self):
        """Batch output[i] == individual forward on molecule i (first 5)."""
        torch.manual_seed(42)
        model = _make_diverse_model(self.dataset)
        batch_graph, _ = self._full_batch()
        batch_out = self._forward(model, batch_graph)

        n_check = min(5, len(self.dataset.graphs))
        for i in range(n_check):
            single_graph = Batch.from_data_list([self.dataset.graphs[i]])
            single_out = self._forward(model, single_graph)

            assert torch.allclose(batch_out[i], single_out[0], atol=1e-4), (
                f"Molecule {i}: batch={batch_out[i]}, single={single_out[0]}"
            )

    @pytest.mark.parametrize(
        "pooling_fn",
        [
            "SumPoolingThenCat",
            "MeanPoolingThenCat",
            "WeightAndSumThenCat",
            "WeightAndMeanThenCat",
            "GlobalAttentionPoolingThenCat",
            "Set2SetThenCat",
        ],
    )
    def test_all_pooling_fns_real_data(self, pooling_fn):
        """Every pooling function produces finite output on real data."""
        model = _make_diverse_model(self.dataset, pooling_fn=pooling_fn)
        batch_graph, _ = self._full_batch()
        out = self._forward(model, batch_graph)
        assert torch.isfinite(out).all(), (
            f"{pooling_fn}: NaN/Inf in output"
        )
