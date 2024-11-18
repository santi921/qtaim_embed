from typing import List, Tuple, Dict, Optional
from copy import deepcopy

import dgl.nn.pytorch as dglnn
import dgl
from dgl.readout import sum_nodes, softmax_nodes

import torch
from torch import nn
import torch.nn.functional as F

from qtaim_embed.models.layers import GraphConvDropoutBatch


class DotPredictor(nn.Module):
    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        ret_val = torch.sum(edges.src["h"] * edges.dst["h"], dim=1)
        return {"score": ret_val}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            # g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            g.apply_edges(self.apply_edges)
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            # return g.edata['score'][:, 0]
            return g.edata["score"]


class MLPPredictor(nn.Module):
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
        # self.layers.append(nn.Linear(h_feats * 2, h_dims[0]))
        input_size = h_feats * 2
        output_size = h_dims[0]
        self.layers.append(nn.Linear(input_size, output_size))

        for i in range(1, len(h_dims)):
            #print("mlp layer: ", i)
            
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

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        for layer in self.layers[:-1]:
            h = layer(h)
        return {"score": h.squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


class AttentionPredictor(nn.Module):
    def __init__(self, in_feats: int, **kwargs):
        super(AttentionPredictor, self).__init__()
        self.in_feats = in_feats
        self.gate_nn = nn.Linear(2 * in_feats, 1)

    def apply_edges(self, edges, get_attention=False):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)

        gate = F.leaky_relu(self.gate_nn(h))
        gate = F.softmax(gate, dim=1)
        rst = torch.sum(gate * edges.src["h"], dim=1)
        if get_attention:
            return {"score": rst}, gate
        return {"score": rst}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


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

    def __init__(self, input_dim, output_dim, **kwargs):
        super(UnifySize, self).__init__()

        self.linears = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, feats):
        """
        Args:
            feats (dict): features dict with node type as key and feature as value

        Returns:
            dict: size adjusted features
        """
        # return {k: self.linears[k](x) for k, x in feats.items()}
        return self.linears(feats)


class ResidualBlockHomo(nn.Module):
    def __init__(self, layer_args, resid_n_graph_convs=2, input_block=False, **kwargs):
        super(ResidualBlockHomo, self).__init__()
        # create graph convolutional layer
        self.layers = []
        self.input_block = input_block

        for i in range(resid_n_graph_convs):
            if input_block:
                layer_arg_copy = deepcopy(layer_args)

                if i == 0:
                    layer_arg_copy["in_feats"] = layer_args["embedding_size"]
                    print("layer arg copy:", layer_arg_copy)
                    self.layers.append(
                        GraphConvDropoutBatch(**layer_arg_copy),
                    )

                else:
                    layer_arg_copy["in_feats"] = layer_args["out_feats"]
                    #print("layer arg copy:", layer_arg_copy)
                    self.layers.append(
                        GraphConvDropoutBatch(**layer_arg_copy),
                    )
            else:
                #print("layer args:", layer_args)
                self.layers.append(
                    GraphConvDropoutBatch(**layer_args),
                )

        self.layers = nn.ModuleList(self.layers)
        self.out_feats = self.layers[-1].out_feats

    def forward(self, graph, feat, weight=None, edge_weight=None):
        input_feats = feat

        if self.input_block == True:
            # print("input block")
            feats_rectified = self.layers[0](graph, input_feats, weight, edge_weight)
            feats = feats_rectified.detach().clone()
            for layer in self.layers[1:]:
                feats = layer(graph, feats, weight, edge_weight)

            feat = feats_rectified + feats

        else:
            # print("no input block")
            for layer in self.layers:
                feat = layer(graph, feat, weight, edge_weight)
            feat = feat + input_feats

        return feat
