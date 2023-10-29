import torch
from torch import nn
import torch.nn.functional as F

from copy import deepcopy
from typing import List, Tuple, Dict, Optional


import torchmetrics
import dgl.nn.pytorch as dglnn
import pytorch_lightning as pl
import dgl
from dgl import function as fn
from dgl.readout import sum_nodes, max_nodes, mean_nodes, softmax_nodes


class UnifySize(nn.Module):
    """
    A layer to unify the feature size of nodes of different types.
    Each feature uses a linear fc layer to map the size.

    NOTE, after this transformation, each data point is just a linear combination of its
    feature in the original feature space (x_new_ij = x_ik w_kj), there is not mixing of
    feature between data points.

    Args:
        input_dim (dict): feature sizes of nodes with node type as key and size as value
        output_dim (int): output feature size, i.e. the size we will turn all the
            features to
    """

    def __init__(self, input_dim, output_dim):
        super(UnifySize, self).__init__()

        self.linears = nn.ModuleDict(
            {
                k: nn.Linear(size, output_dim, bias=False)
                for k, size in input_dim.items()
            }
        )

    def forward(self, feats):
        """
        Args:
            feats (dict): features dict with node type as key and feature as value

        Returns:
            dict: size adjusted features
        """
        return {k: self.linears[k](x) for k, x in feats.items()}


class GraphConvDropoutBatch(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        dropout=0.5,
        batch_norm_tf=True,
    ):
        super(GraphConvDropoutBatch, self).__init__()
        # create graph convolutional layer
        self.graph_conv = dglnn.GraphConv(
            in_feats=in_feats,
            out_feats=out_feats,
            norm=norm,
            weight=weight,
            bias=bias,
            activation=activation,
            allow_zero_in_degree=allow_zero_in_degree,
        )
        self.dropout = None
        self.batch_norm = None
        self.out_feats = out_feats
        # create dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

        # create batch norm layer
        if batch_norm_tf:
            self.batch_norm = nn.BatchNorm1d(out_feats)

    def forward(self, graph, feat, weight=None, edge_weight=None):
        # apply graph convolutional layer
        feat = self.graph_conv(graph, feat, weight, edge_weight)
        # apply dropout to output features
        if self.dropout is not None:
            feat = self.dropout(feat)
        # apply batch norm
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)

        return feat


class ResidualBlock(nn.Module):
    def __init__(
        self,
        layer_args,
        aggregate="sum",
        resid_n_graph_convs=2,
        output_block=False,
    ):
        super(ResidualBlock, self).__init__()
        # create graph convolutional layer
        self.layers = []
        self.output_block = output_block

        for i in range(resid_n_graph_convs):
            if output_block == True:
                if i == resid_n_graph_convs - 1:
                    # print("triggered separate outer layer")
                    self.layers.append(
                        dglnn.HeteroGraphConv(
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
                            aggregate=aggregate,
                        )
                    )
                else:
                    # print("triggered separate intermediate layer")
                    self.layers.append(
                        dglnn.HeteroGraphConv(
                            {
                                "a2b": GraphConvDropoutBatch(**layer_args["a2b_inner"]),
                                "b2a": GraphConvDropoutBatch(**layer_args["b2a_inner"]),
                                "a2g": GraphConvDropoutBatch(**layer_args["a2g_inner"]),
                                "g2a": GraphConvDropoutBatch(**layer_args["g2a_inner"]),
                                "b2g": GraphConvDropoutBatch(**layer_args["b2g_inner"]),
                                "g2b": GraphConvDropoutBatch(**layer_args["g2b_inner"]),
                                "a2a": GraphConvDropoutBatch(**layer_args["a2a_inner"]),
                                "b2b": GraphConvDropoutBatch(**layer_args["b2b_inner"]),
                                "g2g": GraphConvDropoutBatch(**layer_args["g2g_inner"]),
                            },
                            aggregate=aggregate,
                        )
                    )

            else:
                # print("triggered normal outer layer")
                self.layers.append(
                    dglnn.HeteroGraphConv(
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
                        aggregate=aggregate,
                    )
                )

        self.layers = nn.ModuleList(self.layers)
        self.out_feats = {}
        for k, v in self.layers[-1].mods.items():
            self.out_feats[k] = v.out_feats

    def forward(self, graph, feat, weight=None, edge_weight=None):
        input_feats = feat
        for layer in self.layers:
            feat = layer(graph, feat, weight, edge_weight)
        if self.output_block == True:
            return feat
        else:
            feat = {k: feat[k] + input_feats[k] for k in feat.keys()}
        return feat


class Set2Set(nn.Module):
    r"""
    Compared to the Official dgl implementation, we allowed node type.

    For each individual graph in the batch, set2set computes

    .. math::
        q_t &= \mathrm{LSTM} (q^*_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(x_i \cdot q_t)

        r_t &= \sum_{i=1}^N \alpha_{i,t} x_i

        q^*_t &= q_t \Vert r_t

    for this graph.

    Args:
        input_dim: The size of each input sample.
        n_iters: The number of iterations.
        n_layers: The number of recurrent layers.
        ntype: Type of the node to apply Set2Set.
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int, ntype: str):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.ntype = ntype
        self.lstm = torch.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, graph: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """
        Compute set2set pooling.

        Args:
            graph: the input graph
            feat: The input feature with shape :math:`(N, D)` where  :math:`N` is the
                number of nodes in the graph, and :math:`D` means the size of features.

        Returns:
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to
            the batch size, and :math:`D` means the size of features.
        """
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * dgl.broadcast_nodes(graph, q, ntype=self.ntype)).sum(
                    dim=-1, keepdim=True
                )
                graph.nodes[self.ntype].data["e"] = e
                alpha = dgl.softmax_nodes(graph, "e", ntype=self.ntype)
                graph.nodes[self.ntype].data["r"] = feat * alpha
                readout = dgl.sum_nodes(graph, "r", ntype=self.ntype)
                q_star = torch.cat([q, readout], dim=-1)

            return q_star

    def extra_repr(self):
        """
        Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = "n_iters={n_iters}"
        return summary.format(**self.__dict__)


class Set2SetThenCat(nn.Module):
    """
    Set2Set for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        n_iter: number of LSTM iteration
        n_layers: number of LSTM layers
        ntypes: node types to perform Set2Set, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform Set2Set, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        n_iters: int,
        n_layers: int,
        ntypes: List[str],
        in_feats: List[int],
        ntypes_direct_cat: Optional[List[str]] = None,
    ):
        super(Set2SetThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat

        self.layers = nn.ModuleDict()
        for nt, sz in zip(ntypes, in_feats):
            if nt not in ntypes_direct_cat:
                self.layers[nt] = Set2Set(
                    input_dim=sz, n_iters=n_iters, n_layers=n_layers, ntype=nt
                )

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            graph: the graph
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size.
        Returns:
            update features. Each tensor is of shape (B, D), where B is the batch size
                and D is the feature size. Note D could be different for different
                node type.

        """
        rst = []
        for nt in self.ntypes:
            if nt not in self.ntypes_direct_cat:
                ft = self.layers[nt](graph, feats[nt])
                rst.append(ft)

        if self.ntypes_direct_cat is not None:
            for nt in self.ntypes_direct_cat:
                rst.append(feats[nt])

        res = torch.cat(rst, dim=-1)  # dim=-1 to deal with batched graph

        return res


class SumPoolingThenCat(nn.Module):
    """
    SumPooling for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform SumPooling, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform SumPooling, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(SumPoolingThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        # self.layers = nn.ModuleDict()
        # for nt, sz in zip(ntypes, in_feats):
        #    if nt not in ntypes_direct_cat:
        #        self.layers[nt] = dgl.SumPooling(ntype=nt)

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sumpooling of each node type and each graph in the batch.
        """
        rst = []
        with graph.local_scope():
            graph.ndata["h"] = feats

            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    rst.append(dgl.readout_nodes(graph, "h", ntype=ntype, op="sum"))

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        return torch.cat(rst, dim=-1)


class WeightAndSumThenCat(nn.Module):
    """
    WeightAndSum for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform WeightAndSum, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform WeightAndSum, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(WeightAndSumThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        self.atom_weighting = nn.ModuleDict()
        for ntype, size in zip(ntypes, in_feats):
            if ntype not in ntypes_direct_cat:
                self.atom_weighting[ntype] = nn.Linear(size, 1)

        # for ntype, size in zip(ntypes, in_feats):
        #    self.layers[ntype] = WeightAndSum(in_feats=size)

    def forward(
        self, graph: dgl.DGLGraph, feats: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the sumpooling of each node type and each graph in the batch.
        """
        rst = []
        with graph.local_scope():
            weight_dict = {}
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    weight_dict[ntype] = self.atom_weighting[ntype](feats[ntype])

            graph.ndata["h"] = feats
            graph.ndata["w"] = weight_dict

            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    rst.append(
                        dgl.readout_nodes(graph, "h", "w", ntype=ntype, op="sum")
                    )

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        return torch.cat(rst, dim=-1)


class GlobalAttentionPoolingThenCat(nn.Module):
    """
    GlobalAttentionPooling for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform GlobalAttentionPooling, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform GlobalAttentionPooling, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(GlobalAttentionPoolingThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        self.gate_nn = nn.ModuleDict()
        for ntype, in_feat in zip(ntypes, in_feats):
            self.gate_nn[ntype] = nn.Linear(in_feat, 1)

    def forward(self, graph, feats, get_attention=False):
        rst = []
        readout_dict = {}
        gate_dict = {}
        gated_feats = {}
        with graph.local_scope():
            # gather, assign gate to graph
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    gate_dict[ntype] = F.leaky_relu(self.gate_nn[ntype](feats[ntype]))

            graph.ndata["gate"] = gate_dict
            graph.nodes["atom"].data["gate"]
            graph.nodes["bond"].data["gate"]

            # gather, assign gated features to graph
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    gate = softmax_nodes(graph=graph, feat="gate", ntype=ntype)
                    gated_feats[ntype] = feats[ntype] * gate
            graph.ndata.pop("gate")

            # gather, assign readout features to graph
            graph.ndata["r"] = gated_feats
            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    readout_dict[ntype] = sum_nodes(graph, "r", ntype=ntype)
                    rst.append(readout_dict[ntype])
            graph.ndata.pop("r")

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        rst = torch.cat(rst, dim=-1)

        if get_attention:
            return rst, gate
        else:
            return rst


class MultitaskLinearSoftmax(nn.Module):
    """
    Multihead attention with softmax.

     Args:
        n_tasks: number of tasks
    """

    def __init__(self, n_tasks, in_feats, out_feats):
        super(MultitaskLinearSoftmax, self).__init__()
        self.n_tasks = n_tasks
        self.layers_dict = nn.ModuleDict()
        for i in range(n_tasks):
            self.layers_dict[str(i)] = nn.ModuleList()
            self.layers_dict[str(i)].append(nn.Linear(in_feats, out_feats))
            self.layers_dict[str(i)].append(nn.Softmax(dim=1))

    def forward(self, x):
        ret_dict = {}
        for i in range(self.n_tasks):
            x_temp = x
            for layer in self.layers_dict[str(i)]:
                x_temp = layer(x_temp)
            ret_dict[str(i)] = x_temp
        out_dict_as_tensor = torch.stack([ret_dict[k] for k in ret_dict.keys()], dim=1)
        return out_dict_as_tensor
