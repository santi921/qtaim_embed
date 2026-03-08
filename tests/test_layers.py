import numpy as np
import torch
from torch_geometric.nn import HeteroConv
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
    EDGE_TYPE_MAP,
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
                self.graph[ntype].num_nodes,
                self.output_dim,
            )

    def test_gcn(self):
        layer_args = get_hyperparams_gcn()

        gcn = HeteroConv(
            {
                EDGE_TYPE_MAP["a2b"]: GraphConvDropoutBatch(**layer_args["a2b"]),
                EDGE_TYPE_MAP["b2a"]: GraphConvDropoutBatch(**layer_args["b2a"]),
                EDGE_TYPE_MAP["a2g"]: GraphConvDropoutBatch(**layer_args["a2g"]),
                EDGE_TYPE_MAP["g2a"]: GraphConvDropoutBatch(**layer_args["g2a"]),
                EDGE_TYPE_MAP["b2g"]: GraphConvDropoutBatch(**layer_args["b2g"]),
                EDGE_TYPE_MAP["g2b"]: GraphConvDropoutBatch(**layer_args["g2b"]),
                EDGE_TYPE_MAP["a2a"]: GraphConvDropoutBatch(**layer_args["a2a"]),
                EDGE_TYPE_MAP["b2b"]: GraphConvDropoutBatch(**layer_args["b2b"]),
                EDGE_TYPE_MAP["g2g"]: GraphConvDropoutBatch(**layer_args["g2g"]),
            },
            aggr="sum",
        )

        graph, feats = make_hetero_graph()
        out = self.uni(feats)
        edge_index_dict = {etype: graph[etype].edge_index for etype in graph.edge_types}
        out = gcn(out, edge_index_dict)

        for ntype, size in self.input_dim.items():
            assert out[ntype].shape == (graph[ntype].num_nodes, self.output_dim)

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
        edge_index_dict = {etype: graph[etype].edge_index for etype in graph.edge_types}
        out = resid(out, edge_index_dict)

        for ntype, size in self.input_dim.items():
            assert out[ntype].shape == (graph[ntype].num_nodes, self.output_dim)

    def test_sum(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = ["global"]

        sum_pool = SumPoolingThenCat(
            ntypes=ntypes, ntypes_direct_cat=ntypes_direct_cat, in_feats=self.in_feats
        )
        batch_dict = {nt: torch.zeros(self.graph[nt].num_nodes, dtype=torch.long) for nt in ntypes}
        out = sum_pool(out, batch_dict)

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

        batch_dict = {nt: torch.zeros(self.graph[nt].num_nodes, dtype=torch.long) for nt in ntypes}
        out = weight_sum_pool(out, batch_dict)
        assert out.shape == (1, np.sum(self.in_feats))
        assert torch.allclose(out, torch.zeros_like(out))

    def test_mean(self):
        out = self.uni(self.feats)
        ntypes = ["atom", "bond", "global"]
        ntypes_direct_cat = ["global"]

        sum_pool = MeanPoolingThenCat(
            ntypes=ntypes, ntypes_direct_cat=ntypes_direct_cat, in_feats=self.in_feats
        )
        batch_dict = {nt: torch.zeros(self.graph[nt].num_nodes, dtype=torch.long) for nt in ntypes}
        out = sum_pool(out, batch_dict)

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

        batch_dict = {nt: torch.zeros(self.graph[nt].num_nodes, dtype=torch.long) for nt in ntypes}
        out = weight_sum_pool(out, batch_dict)
        assert out.shape == (1, np.sum(self.in_feats))

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
        batch_dict = {nt: torch.zeros(self.graph[nt].num_nodes, dtype=torch.long) for nt in ntypes}
        out = set2set_pool(out, batch_dict)

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

        batch_dict = {nt: torch.zeros(self.graph[nt].num_nodes, dtype=torch.long) for nt in ntypes}
        out = gap_pool(out, batch_dict)
        assert out.shape == (1, np.sum(self.in_feats))
