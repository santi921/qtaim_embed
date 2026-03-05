from typing import List, Tuple, Dict, Optional
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from qtaim_embed.models.layers import GraphConvDropoutBatch


class DotPredictor(nn.Module):
    """
    Computes a scalar score for each edge using a dot product between
    source and destination node features.
    """

    def forward(self, edge_index, h):
        """
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges]. Row 0 is source nodes,
            row 1 is destination nodes.
        h : torch.Tensor
            Node feature tensor of shape [num_nodes, feat_dim].

        Returns
        -------
        torch.Tensor
            Scalar score for each edge of shape [num_edges].
        """
        src, dst = edge_index
        score = torch.sum(h[src] * h[dst], dim=1)
        return score


class MLPPredictor(nn.Module):
    """
    MLP-based edge scorer. Concatenates source and destination node features
    and passes them through an MLP to produce a scalar score per edge.
    """

    def __init__(
        self,
        h_feats,
        h_dims=[100, 100],
        dropout=0.5,
        activation=None,
        batch_norm=False,
        **kwargs
    ):
        super().__init__()

        if activation is not None:
            self.activation = getattr(torch.nn, activation)()
        else:
            self.activation = None

        self.layers = nn.ModuleList()
        input_size = h_feats * 2
        output_size = h_dims[0]
        self.layers.append(nn.Linear(input_size, output_size))

        for i in range(1, len(h_dims)):
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dims[i - 1]))
            if activation is not None:
                self.layers.append(self.activation)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

            input_size = output_size

            self.layers.append(nn.Linear(h_dims[i - 1], h_dims[i]))

        if batch_norm:
            self.layers.append(nn.BatchNorm1d(h_dims[-1]))
        if activation is not None:
            self.layers.append(self.activation)
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(h_dims[-1], 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, edge_index, h):
        """
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges].
        h : torch.Tensor
            Node feature tensor of shape [num_nodes, feat_dim].

        Returns
        -------
        torch.Tensor
            Scalar score for each edge of shape [num_edges].
        """
        src, dst = edge_index
        h_concat = torch.cat([h[src], h[dst]], dim=1)
        for layer in self.layers[:-1]:
            h_concat = layer(h_concat)
        # Final sigmoid layer and squeeze to get scalar per edge
        h_concat = self.layers[-1](h_concat)
        return h_concat.squeeze(1)


class AttentionPredictor(nn.Module):
    """
    Attention-based edge scorer. Uses a gate mechanism to compute a weighted
    score for each edge based on source and destination node features.
    """

    def __init__(self, in_feats: int, **kwargs):
        super(AttentionPredictor, self).__init__()
        self.in_feats = in_feats
        self.gate_nn = nn.Linear(2 * in_feats, 1)

    def forward(self, edge_index, h, get_attention=False):
        """
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges].
        h : torch.Tensor
            Node feature tensor of shape [num_nodes, feat_dim].
        get_attention : bool, optional
            If True, also return the attention gate values. Default: False.

        Returns
        -------
        torch.Tensor
            Scalar score for each edge of shape [num_edges].
        torch.Tensor (optional)
            Gate attention values, returned only if get_attention=True.
        """
        src, dst = edge_index
        h_concat = torch.cat([h[src], h[dst]], dim=1)
        gate = F.leaky_relu(self.gate_nn(h_concat))
        gate = F.softmax(gate, dim=1)
        score = torch.sum(gate * h[src], dim=1)
        if get_attention:
            return score, gate
        return score


class UnifySize(nn.Module):
    """
    A layer to unify the feature size of nodes of different types.
    Each feature uses a linear fc layer to map the size.

    NOTE, after this transformation, each data point is just a linear combination of its
    feature in the original feature space (x_new_ij = x_ik w_kj), there is not mixing of
    feature between data points.

    Args:
        input_dim (int): input feature size
        output_dim (int): output feature size, i.e. the size we will turn all the
            features to
    """

    def __init__(self, input_dim, output_dim, **kwargs):
        super(UnifySize, self).__init__()

        self.linears = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, feats):
        """
        Args:
            feats (torch.Tensor): input features

        Returns:
            torch.Tensor: size adjusted features
        """
        return self.linears(feats)


class ResidualBlockHomo(nn.Module):
    """
    Homogeneous graph residual block using GraphConvDropoutBatch layers.
    Supports an input block mode where the first layer output is used
    as a skip connection target (with detached clone for gradient isolation).
    """

    def __init__(self, layer_args, resid_n_graph_convs=2, input_block=False, **kwargs):
        super(ResidualBlockHomo, self).__init__()
        # create graph convolutional layer
        self.layers = []
        self.input_block = input_block

        for i in range(resid_n_graph_convs):
            layer_arg_copy = deepcopy(layer_args)
            layer_arg_copy = self.pad_args_graph_conv(layer_arg_copy)

            if input_block:
                if i == 0:
                    layer_arg_copy["in_feats"] = layer_args["embedding_size"]

                    self.layers.append(
                        GraphConvDropoutBatch(
                            in_feats=layer_arg_copy["in_feats"],
                            out_feats=layer_arg_copy["out_feats"],
                            norm=layer_arg_copy["norm"],
                            weight=layer_arg_copy["weight"],
                            bias=layer_arg_copy["bias"],
                            activation=layer_arg_copy["activation"],
                            allow_zero_in_degree=layer_arg_copy["allow_zero_in_degree"],
                            dropout=layer_arg_copy["dropout"],
                            batch_norm_tf=layer_arg_copy["batch_norm_tf"],
                        ),
                    )

                else:
                    layer_arg_copy["in_feats"] = layer_args["out_feats"]
                    self.layers.append(
                        GraphConvDropoutBatch(
                            in_feats=layer_arg_copy["in_feats"],
                            out_feats=layer_arg_copy["out_feats"],
                            norm=layer_arg_copy["norm"],
                            weight=layer_arg_copy["weight"],
                            bias=layer_arg_copy["bias"],
                            activation=layer_arg_copy["activation"],
                            allow_zero_in_degree=layer_arg_copy["allow_zero_in_degree"],
                            dropout=layer_arg_copy["dropout"],
                            batch_norm_tf=layer_arg_copy["batch_norm_tf"],
                        ),
                    )
            else:
                self.layers.append(
                    GraphConvDropoutBatch(
                        in_feats=layer_arg_copy["in_feats"],
                        out_feats=layer_arg_copy["out_feats"],
                        norm=layer_arg_copy["norm"],
                        weight=layer_arg_copy["weight"],
                        bias=layer_arg_copy["bias"],
                        activation=layer_arg_copy["activation"],
                        allow_zero_in_degree=layer_arg_copy["allow_zero_in_degree"],
                        dropout=layer_arg_copy["dropout"],
                        batch_norm_tf=layer_arg_copy["batch_norm_tf"],
                    ),
                )

        self.layers = nn.ModuleList(self.layers)
        self.out_feats = self.layers[-1].out_feats

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass for the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor of shape [num_nodes, feat_dim].
        edge_index : torch.Tensor
            Edge index tensor of shape [2, num_edges].
        edge_weight : torch.Tensor, optional
            Edge weight tensor of shape [num_edges].

        Returns
        -------
        torch.Tensor
            Output node features after residual connection.
        """
        input_feats = x

        if self.input_block:
            feats_rectified = self.layers[0](x, edge_index, edge_weight)
            feats = feats_rectified.detach().clone()
            for layer in self.layers[1:]:
                feats = layer(feats, edge_index, edge_weight)
            x = feats_rectified + feats
        else:
            for layer in self.layers:
                x = layer(x, edge_index, edge_weight)
            x = x + input_feats

        return x

    def pad_args_graph_conv(self, layer_args):
        """
        Pad the arguments for the graph convolutional layer.

        Args:
            layer_args (dict): The arguments for the graph convolutional layer.

        Returns:
            dict: The padded arguments.
        """
        if "activation" not in layer_args:
            layer_args["activation"] = None
        if "allow_zero_in_degree" not in layer_args:
            layer_args["allow_zero_in_degree"] = False
        if "batch_norm_tf" not in layer_args:
            layer_args["batch_norm_tf"] = True
        if "weight" not in layer_args:
            layer_args["weight"] = True
        if "bias" not in layer_args:
            layer_args["bias"] = True
        if "norm" not in layer_args:
            layer_args["norm"] = "both"
        if "dropout" not in layer_args:
            layer_args["dropout"] = 0.1

        return layer_args
