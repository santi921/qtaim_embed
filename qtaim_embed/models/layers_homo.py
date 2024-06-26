
from typing import List, Tuple, Dict, Optional


import dgl.nn.pytorch as dglnn
import dgl
from dgl.readout import sum_nodes, softmax_nodes

import torch
from torch import nn
import torch.nn.functional as F


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
                        GraphConvDropoutBatch(**layer_args["conv"])
                    )
                else:
                    # print("triggered separate intermediate layer")
                    self.layers.append(
                        GraphConvDropoutBatch(**layer_args["conv_inner"])
                    )

            else:
                # print("triggered normal outer layer")
                self.layers.append(
                    GraphConvDropoutBatch(**layer_args["conv"])
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
        ret_val = torch.sum(edges.src['h'] * edges.dst['h'], dim=1)
        return {'score': ret_val}


    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            #g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            g.apply_edges(self.apply_edges)
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats, h_dims=[100, 50], dropout=0.5, activation=None, batch_norm=False):
        super().__init__()

        if activation is not None:
            self.activation = getattr(torch.nn, activation)()
        else: 
            self.activation = None


        self.layers = nn.ModuleList()
        #self.layers.append(nn.Linear(h_feats * 2, h_dims[0]))
        input_size = h_feats * 2

        for i in range(1, len(h_dims)):
            
            output_size = h_dims[i]
            self.layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

            if batch_norm:
                self.layers.append(nn.BatchNorm1d(h_dims[i-1]))
            
            if activation is not None:
                self.layers.append(self.activation)
            
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))

            self.layers.append(nn.Linear(h_dims[i-1], h_dims[i]))
            
        
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
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        for layer in self.layers[:-1]:
            h = layer(h)
        return {'score': h.squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
    
class AttentionPredictor(nn.Module):
    def __init__(
        self,
        in_feats: int
    ):
        super(AttentionPredictor, self).__init__()
        self.in_feats = in_feats
        self.gate_nn = nn.Linear(2*in_feats, 1)


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
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)

        gate = F.leaky_relu(self.gate_nn(h))
        gate = F.softmax(gate, dim=1)
        rst = torch.sum(gate * edges.src['h'], dim=1)
        if get_attention:
            return {'score': rst}, gate
        return {'score': rst}
        

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        