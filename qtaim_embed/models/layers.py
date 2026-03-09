from typing import List, Tuple, Dict, Optional


from torch_geometric.nn import GraphConv, HeteroConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler


# Mapping from short edge type names (used in layer_args) to PyG full triplets
EDGE_TYPE_MAP = {
    "a2b": ("atom", "a2b", "bond"),
    "b2a": ("bond", "b2a", "atom"),
    "a2g": ("atom", "a2g", "global"),
    "g2a": ("global", "g2a", "atom"),
    "b2g": ("bond", "b2g", "global"),
    "g2b": ("global", "g2b", "bond"),
    "a2a": ("atom", "a2a", "atom"),
    "b2b": ("bond", "b2b", "bond"),
    "g2g": ("global", "g2g", "global"),
}


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

    def __init__(self, input_dim: Dict[str, int], output_dim: int):
        super(UnifySize, self).__init__()
        self.node_types = list(input_dim.keys())
        self.linears = nn.ModuleList(
            [nn.Linear(input_dim[k], output_dim, bias=False) for k in self.node_types]
        )

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            feats (dict): features dict with node type as key and feature as value

        Returns:
            dict: size adjusted features
        """
        output = {}
        with profiler.record_function("Unify"):
            for i, node_type in enumerate(self.node_types):
                output[node_type] = self.linears[i](feats[node_type])

            return output


class GraphConvDropoutBatch(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        norm: str = "both",
        activation: Optional[nn.Module] = None,
        dropout: float = 0.1,
        batch_norm_tf: bool = True,
        **kwargs,
    ):
        super(GraphConvDropoutBatch, self).__init__()
        # Create graph convolutional layer using PyG's GraphConv
        # GraphConv supports bipartite message passing needed for HeteroConv
        # Uses additive aggregation (equivalent to sum of neighbor messages)
        self.graph_conv = GraphConv(
            in_channels=in_feats,
            out_channels=out_feats,
            aggr="add",
        )
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_feats) if batch_norm_tf else None
        self.out_feats = out_feats

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (N, in_feats).
            edge_index: Edge index tensor of shape (2, E).
            edge_weight: Optional edge weight tensor of shape (E,).

        Returns:
            torch.Tensor: The output features after applying graph convolution,
                activation, dropout, and batch normalization.
        """
        with profiler.record_function("GCN Conv"):

            # Apply graph convolutional layer
            x = self.graph_conv(x, edge_index, edge_weight)

            # Apply activation
            if self.activation is not None:
                x = self.activation(x)

            # Apply dropout to output features
            if self.dropout is not None:
                x = self.dropout(x)

            # Apply batch normalization
            if self.batch_norm is not None:
                x = self.batch_norm(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        layer_args: Dict[str, Dict[str, Dict[str, int]]],
        aggregate: str = "sum",
        resid_n_graph_convs: int = 2,
        output_block: bool = False,
    ):
        """
        Args:
            layer_args: A dictionary containing arguments for each graph convolution layer.
                Keys are short edge type names (e.g., "a2b", "b2a") which are internally
                mapped to PyG triplet format (e.g., ("atom", "a2b", "bond")).
            aggregate: Aggregation type for HeteroConv (e.g., "sum", "mean").
            resid_n_graph_convs: Number of graph convolution layers in the residual block.
            output_block: Whether this is the output block (affects layer configuration).
        """
        super(ResidualBlock, self).__init__()
        self.output_block = output_block
        self.layers = nn.ModuleList()

        # All short edge type names used for building HeteroConv dicts
        edge_types = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]

        for i in range(resid_n_graph_convs):
            if output_block == True:
                if i == resid_n_graph_convs - 1:
                    # Outer layer of output block
                    conv_dict = {
                        EDGE_TYPE_MAP[et]: GraphConvDropoutBatch(**layer_args[et])
                        for et in edge_types
                    }
                else:
                    # Inner layer of output block
                    conv_dict = {
                        EDGE_TYPE_MAP[et]: GraphConvDropoutBatch(
                            **layer_args[et + "_inner"]
                        )
                        for et in edge_types
                    }
            else:
                # Normal (non-output) block
                conv_dict = {
                    EDGE_TYPE_MAP[et]: GraphConvDropoutBatch(**layer_args[et])
                    for et in edge_types
                }

            self.layers.append(HeteroConv(conv_dict, aggr=aggregate))

        self.layers = nn.ModuleList(self.layers)

        # Extract out_feats from the last layer's modules
        # HeteroConv stores sub-modules in self.convs dict with triplet keys
        self.out_feats = {}
        for triplet_key, conv_module in self.layers[-1].convs.items():
            # Map back from triplet to short name for compatibility
            short_name = triplet_key[1]  # e.g., ("atom", "a2b", "bond") -> "a2b"
            self.out_feats[short_name] = conv_module.out_feats

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_dict: Node features as a dictionary with node type as key and features as value.
            edge_index_dict: Edge indices as a dictionary with edge type triplet as key
                and edge index tensor as value.

        Returns:
            Updated node features as a dictionary.
        """
        with profiler.record_function("ResidualBlock"):
            input_feats = x_dict
            for layer in self.layers:
                x_dict = layer(x_dict, edge_index_dict)

            if not self.output_block:
                # Add residual connections
                for k in x_dict.keys():
                    x_dict[k].add_(input_feats[k])
            return x_dict


# class AttentionBlock(nn.Module):
# TODO


class Set2Set(nn.Module):
    r"""
    Set2Set pooling for PyG heterogeneous graphs.

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

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute set2set pooling.

        Args:
            x: The input feature with shape (N, D) where N is the number of nodes
                of this type in the batch, and D means the size of features.
            batch: Batch assignment vector of shape (N,) mapping each node to its
                graph index in the batch.

        Returns:
            The output feature with shape (B, 2*D), where B refers to the batch size,
            and D means the size of input features.
        """
        batch_size = batch.max().item() + 1

        h = (
            x.new_zeros((self.n_layers, batch_size, self.input_dim)),
            x.new_zeros((self.n_layers, batch_size, self.input_dim)),
        )

        q_star = x.new_zeros(batch_size, self.output_dim)

        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)

            # broadcast_nodes equivalent: index into batch vector
            e = (x * q[batch]).sum(dim=-1, keepdim=True)

            # softmax_nodes equivalent: scatter softmax over batch
            alpha = softmax(e, batch, dim=0)

            # sum_nodes equivalent: scatter sum of weighted features
            readout = global_add_pool(x * alpha, batch, size=batch_size)

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
        self, feats: Dict[str, torch.Tensor], batch_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            feats: node features with node type as key and the corresponding
                features as value. Each tensor is of shape (N, D) where N is the number
                of nodes of the corresponding node type, and D is the feature size.
            batch_dict: batch assignment vectors with node type as key and batch
                vector as value. Each tensor is of shape (N,).
        Returns:
            Concatenated pooled features. Tensor of shape (B, D_total), where B is
            the batch size and D_total is the sum of feature sizes across node types.
        """
        rst = []
        for nt in self.ntypes:
            if nt not in self.ntypes_direct_cat:
                ft = self.layers[nt](feats[nt], batch_dict[nt])
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

    def forward(
        self, feats: Dict[str, torch.Tensor], batch_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the sum pooling of each node type and each graph in the batch.

        Args:
            feats: node features with node type as key and features as value.
            batch_dict: batch assignment vectors with node type as key.
        """
        rst = []

        for ntype in self.ntypes:
            if ntype not in self.ntypes_direct_cat:
                rst.append(global_add_pool(feats[ntype], batch_dict[ntype]))

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

    def forward(
        self, feats: Dict[str, torch.Tensor], batch_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the weighted sum pooling of each node type and each graph in the batch.

        Args:
            feats: node features with node type as key and features as value.
            batch_dict: batch assignment vectors with node type as key.
        """
        rst = []
        for ntype in self.ntypes:
            if ntype not in self.ntypes_direct_cat:
                w = torch.sigmoid(self.atom_weighting[ntype](feats[ntype]))
                weighted = feats[ntype] * w
                rst.append(global_add_pool(weighted, batch_dict[ntype]))

        if self.ntypes_direct_cat:
            rst.extend(feats[ntype] for ntype in self.ntypes_direct_cat)
        return torch.cat(rst, dim=-1)


class MeanPoolingThenCat(nn.Module):
    """
    MeanPooling for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform MeanPooling, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform MeanPooling, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(MeanPoolingThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats

    def forward(
        self, feats: Dict[str, torch.Tensor], batch_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the mean pooling of each node type and each graph in the batch.

        Args:
            feats: node features with node type as key and features as value.
            batch_dict: batch assignment vectors with node type as key.
        """
        rst = []

        for ntype in self.ntypes:
            if ntype not in self.ntypes_direct_cat:
                rst.append(global_mean_pool(feats[ntype], batch_dict[ntype]))

        if self.ntypes_direct_cat is not None:
            for ntype in self.ntypes_direct_cat:
                rst.append(feats[ntype])

        return torch.cat(rst, dim=-1)


class WeightAndMeanThenCat(nn.Module):
    """
    WeightAndMean for nodes (separate for different node type) and then concatenate the
    features of different node types to create a representation of the graph.

     Args:
        ntypes: node types to perform WeightAndMean, e.g. ['atom', 'bond']
        in_feats: node feature sizes. The order should be the same as `ntypes`.
        ntypes_direct_cat: node types to which not perform WeightAndMean, whose feature is
            directly concatenated. e.g. ['global']
    """

    def __init__(
        self,
        ntypes: list,
        in_feats: list,
        ntypes_direct_cat: list,
    ):
        super(WeightAndMeanThenCat, self).__init__()
        self.ntypes = ntypes
        self.ntypes_direct_cat = ntypes_direct_cat
        self.in_feats = in_feats
        self.atom_weighting = nn.ModuleDict()
        for ntype, size in zip(ntypes, in_feats):
            if ntype not in ntypes_direct_cat:
                self.atom_weighting[ntype] = nn.Linear(size, 1)

    def forward(
        self, feats: Dict[str, torch.Tensor], batch_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the weighted mean pooling of each node type and each graph in the batch.

        Args:
            feats: node features with node type as key and features as value.
            batch_dict: batch assignment vectors with node type as key.
        """
        rst = []
        for ntype in self.ntypes:
            if ntype not in self.ntypes_direct_cat:
                w = torch.sigmoid(self.atom_weighting[ntype](feats[ntype]))
                weighted = feats[ntype] * w
                rst.append(global_mean_pool(weighted, batch_dict[ntype]))

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

    def forward(self, feats, batch_dict, get_attention=False):
        """
        Compute global attention pooling of each node type and each graph in the batch.

        Args:
            feats: node features with node type as key and features as value.
            batch_dict: batch assignment vectors with node type as key.
            get_attention: if True, also return the last attention weights.
        """
        with profiler.record_function("GAT Global"):

            rst = []
            gate = None

            for ntype in self.ntypes:
                if ntype not in self.ntypes_direct_cat:
                    gate = F.leaky_relu(self.gate_nn[ntype](feats[ntype]))
                    # softmax_nodes equivalent: scatter softmax over batch
                    gate = softmax(gate, batch_dict[ntype], dim=0)
                    gated = feats[ntype] * gate
                    # sum_nodes equivalent: scatter sum
                    rst.append(global_add_pool(gated, batch_dict[ntype]))

            if self.ntypes_direct_cat is not None:
                for ntype in self.ntypes_direct_cat:
                    rst.append(feats[ntype])

            rst = torch.cat(rst, dim=-1)

            if get_attention:
                return rst, gate
            else:
                return rst


# class GlobalTransformerPoolingThenCat(nn.Module):
# TODO


class MultitaskLinearSoftmax(nn.Module):
    """
    Multihead attention with softmax.

     Args:
        n_tasks: number of tasks
        in_feats: input feature size
        out_feats: output feature size
    """

    def __init__(self, n_tasks, in_feats, out_feats):
        super(MultitaskLinearSoftmax, self).__init__()
        self.n_tasks = n_tasks

        # Create a single linear layer for each task
        self.linear_layers = nn.Linear(in_feats, out_feats * n_tasks, bias=True)
        self.out_feats = out_feats

    @torch.jit.export  # Decorate the forward method for TorchScript
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the linear layer to produce (batch_size, n_tasks * out_feats)
        linear_output = self.linear_layers(x)

        # Reshape to (batch_size, n_tasks, out_feats)
        reshaped_output = linear_output.view(x.size(0), self.n_tasks, self.out_feats)

        # Apply softmax along the last dimension (out_feats)
        softmax_output = torch.softmax(reshaped_output, dim=-1)

        return softmax_output
