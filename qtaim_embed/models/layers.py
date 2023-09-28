
import torch
import pytorch_lightning as pl
import torchmetrics
import dgl.nn.pytorch as dglnn
from torch import nn

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