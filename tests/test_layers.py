from dgl.nn import HeteroGraphConv
import numpy as np
import torch
from qtaim_embed.utils.tests import (
    make_hetero_graph,
    get_hyperparams_gcn,
    get_hyperparams_resid,
)
from qtaim_embed.models.layers import (
    ResidualBlock,
    GraphConvDropoutBatch,
    GlobalAttentionPoolingThenCat,
    Set2SetThenCat,
    SumPoolingThenCat,
    WeightAndSumThenCat,
    UnifySize,
    MeanPoolingThenCat,
    WeightAndMeanThenCat,
)
from qtaim_embed.utils.models import get_layer_args


class TestLayers:
    input_dim = {"atom": 2, "bond": 3, "global": 4}
    output_dim = 10
    in_feats = [10, 10, 10]
    uni = UnifySize(input_dim, output_dim)
    graph, feats = make_hetero_graph()

    def test_unify_size(self):
        # uni = UnifySize(self.input_dim, self.output_dim)
        # graph, feats = make_hetero_graph()
        out = self.uni(self.feats)
        for ntype, size in self.input_dim.items():
            assert out[ntype].shape == (
                self.graph.number_of_nodes(ntype),
                self.output_dim,
            )

    def test_gcn(self):
        # uni = UnifySize(self.input_dim, self.output_dim)
        layer_args = get_hyperparams_gcn()

        gcn = HeteroGraphConv(
            {
                "a2b": GraphConvDropoutBatch(**layer_args["a2b"]),
                "b2a": GraphConvDropoutBatch(**layer_args["b2a"]),
                "a2g": GraphConvDropoutBatch(**layer_args["a2g"]),
                "g2a": GraphConvDropoutBatch(**layer_args["g2a"]),
                "b2g": GraphConvDropoutBatch(**layer_args["b2g"]),
                "g2b": GraphConvDropoutBatch(**layer_args["g2b"]),
                "a2a": GraphConvDropoutBatch(**layer_args["a2a"]),
                "b2b": GraphConvDropoutBatch(**layer_args["b2b"]),
                "g2g": GraphConvDropoutBatch(**layer_args["g2g"]),
            },
            aggregate="sum",
        )

        graph, feats = make_hetero_graph()
        out = self.uni(feats)
        out = gcn(graph, out)

        for ntype, size in self.input_dim.items():
            assert out[ntype].shape == (graph.number_of_nodes(ntype), self.output_dim)

    def test_residual(self):
        uni = UnifySize(self.input_dim, self.output_dim)
        layer_args = get_hyperparams_resid()

        resid = ResidualBlock(
            layer_args=layer_args,
            resid_n_graph_convs=10,
            aggregate="sum",
            output_block=False,
        )

        graph, feats = make_hetero_graph()
        out = uni(feats)
        out = resid(graph, out)

        for ntype, size in self.input_dim.items():
            assert out[ntype].shape == (graph.number_of_nodes(ntype), self.output_dim)

    def test_sum(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = ["global"]

        sum_pool = SumPoolingThenCat(
            ntypes=ntypes, ntypes_direct_cat=ntypes_direct_cat, in_feats=self.in_feats
        )
        out = sum_pool(self.graph, out)

        assert out.shape == (1, np.sum(self.in_feats))

    def test_weight_sum(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = []

        weight_sum_pool = WeightAndSumThenCat(
            ntypes=ntypes, ntypes_direct_cat=ntypes_direct_cat, in_feats=self.in_feats
        )
        # reset weights to 0 for testing
        for ntype in ["atom", "bond", "global"]:
            weight_sum_pool.atom_weighting[ntype].weight.data = torch.zeros_like(
                weight_sum_pool.atom_weighting[ntype].weight.data
            )
            weight_sum_pool.atom_weighting[ntype].bias.data = torch.zeros_like(
                weight_sum_pool.atom_weighting[ntype].bias.data
            )

        out = weight_sum_pool(self.graph, out)
        assert out.shape == (1, np.sum(self.in_feats))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_mean(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = ["global"]

        sum_pool = MeanPoolingThenCat(
            ntypes=ntypes, ntypes_direct_cat=ntypes_direct_cat, in_feats=self.in_feats
        )
        out = sum_pool(self.graph, out)

        assert out.shape == (1, np.sum(self.in_feats))

    def test_weight_sum(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = []

        weight_sum_pool = WeightAndMeanThenCat(
            ntypes=ntypes, ntypes_direct_cat=ntypes_direct_cat, in_feats=self.in_feats
        )
        # reset weights to 0 for testing
        for ntype in ["atom", "bond", "global"]:
            weight_sum_pool.atom_weighting[ntype].weight.data = torch.zeros_like(
                weight_sum_pool.atom_weighting[ntype].weight.data
            )
            weight_sum_pool.atom_weighting[ntype].bias.data = torch.zeros_like(
                weight_sum_pool.atom_weighting[ntype].bias.data
            )

        out = weight_sum_pool(self.graph, out)
        assert out.shape == (1, np.sum(self.in_feats))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_set2set(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = ["global"]

        set2set_pool = Set2SetThenCat(
            ntypes=ntypes,
            ntypes_direct_cat=ntypes_direct_cat,
            in_feats=self.in_feats,
            n_iters=1,
            n_layers=1,
        )
        out = set2set_pool(self.graph, out)

        test_shape = 0
        for ind, i in enumerate(ntypes):
            test_shape += self.in_feats[ind] * 2
        for ntype in ntypes_direct_cat:
            test_shape -= self.in_feats[ntypes.index(ntype)]

        assert out.shape == (1, test_shape)

    def test_gap(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = ["global"]

        gap_pool = GlobalAttentionPoolingThenCat(
            ntypes=ntypes, ntypes_direct_cat=ntypes_direct_cat, in_feats=self.in_feats
        )

        out = gap_pool(self.graph, out)
        assert out.shape == (1, np.sum(self.in_feats))
