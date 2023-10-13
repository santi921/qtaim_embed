import torch
import pytorch_lightning as pl
import torchmetrics
import dgl.nn.pytorch as dglnn
from torch import nn
from copy import deepcopy


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
        for i in range(resid_n_graph_convs):
            if output_block == True:
                if i == resid_n_graph_convs - 1:
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

    def forward(self, graph, feat, weight=None, edge_weight=None):
        input_feats = feat
        for layer in self.layers:
            feat = layer(graph, feat, weight, edge_weight)
        feat = {k: feat[k] + input_feats[k] for k in feat.keys()}
        return feat
