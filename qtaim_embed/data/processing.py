import torch
from collections import defaultdict
from typing import Optional, Dict, List
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
import numpy as np
import dgl


def _transform(X, copy, with_mean=True, with_std=True, threshold=1.0e-3, eta=1.0e-3):
    """
    Args:
        X: a list of 1D tensor or a 2D tensor
    Returns:
        rst: 2D array
        mean: 1D array
        std: 1D array
    """
    if isinstance(X, list):
        X = torch.stack(X)
    scaler = sk_StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
    rst = scaler.fit(X)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    for i, v in enumerate(std):
        if v <= threshold:
            print(
                "Standard deviation for feature {} is {}, smaller than {}. "
                "You may want to exclude this feature.".format(i, v, threshold)
            )

    rst = scaler.transform(X)
    # make all values < eta in std to be eta
    std[std < eta] = eta
    # manually scale the data
    rst = (rst - mean) / std

    return rst, mean, std


class HeteroGraphStandardScaler:
    """
    Standardize hetero graph features by centering and normalization.
    Only node features are standardized.

    The mean and std can be provided for standardization. If `None` they are computed
    from the features of the graphs.

    Args:
        copy: whether to copy the values as used by sklearn.preprocessing.StandardScaler
        mean: with node type as key and the mean value as the value
        std: with node type as key and the std value as the value

    Returns:
        Graphs with their node features standardized. Note, these are the input graphs.
    """

    def __init__(
        self,
        copy: bool = True,
        features_tf: bool = True,
        mean: Optional[Dict[str, torch.Tensor]] = None,
        std: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.copy = copy
        self._mean = mean
        self._std = std
        self.features_tf = features_tf

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def __call__(self, graphs) -> List[dgl.DGLGraph]:
        g = graphs[0]
        node_types = g.ntypes
        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)
        if self.features_tf:
            graph_key = "feat"
        else:
            graph_key = "labels"

        # obtain feats from ALL graphs
        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data[graph_key]
                node_feats[nt].append(data)
                node_feats_size[nt].append(len(data))

        dtype = node_feats[node_types[0]][0].dtype

        # standardize
        if self._mean is not None and self._std is not None:
            for nt in node_types:
                feats = (torch.cat(node_feats[nt]) - self._mean[nt]) / self._std[nt]
                node_feats[nt] = feats

        else:
            self._std = {}
            self._mean = {}

            for nt in node_types:
                # if node_feats is empty, skip
                # cat_feats = torch.cat(node_feats[nt])
                # check that cat_feats isn't just (n, )
                if torch.cat(node_feats[nt]).shape[1] > 0:
                    feats, mean, std = _transform(torch.cat(node_feats[nt]), self.copy)
                    node_feats[nt] = torch.tensor(feats, dtype=dtype)
                    mean = torch.tensor(mean, dtype=dtype)
                    std = torch.tensor(std, dtype=dtype)
                    self._mean[nt] = mean
                    self._std[nt] = std
                else:
                    node_feats[nt] = torch.cat(node_feats[nt])

        # assign data back
        for nt in node_types:
            feats = torch.split(node_feats[nt], node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data[graph_key] = ft

        return graphs


class HeteroGraphLogMagnitudeScaler:
    """
    Standardize hetero graph features or labels by log scalling their magnitude.


    Args:
        copy: whether to copy the values as used by sklearn.preprocessing.StandardScaler

    Returns:
        Graphs with their node features standardized. Note, these are the input graphs.
    """

    def __init__(
        self,
        copy: bool = True,
        features_tf: bool = True,
        shift: float = None,
    ):
        self.copy = copy
        self.features_tf = features_tf
        self.shift = shift

    def __call__(self, graphs) -> List[dgl.DGLGraph]:
        g = graphs[0]
        node_types = g.ntypes
        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)
        if self.features_tf:
            graph_key = "feat"
        else:
            graph_key = "labels"

        # obtain feats from ALL graphs
        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data[graph_key]
                node_feats[nt].append(data)
                node_feats_size[nt].append(len(data))

        # dtype = node_feats[node_types[0]][0].dtype

        # log scale
        for nt in node_types:
            if len(node_feats[nt]) != 0:
                # log scale
                feats = torch.cat(node_feats[nt])
                # get the sign of the data
                sign = torch.sign(feats)
                # get the magnitude of the data
                feats = torch.abs(feats)
                # if shift is not None:
                feats = feats + self.shift
                # log scale the magnitude
                feats = torch.log(feats)
                # put the sign back
                feats = feats * sign
                # assign
                node_feats[nt] = feats

        # assign data back
        for nt in node_types:
            feats = torch.split(node_feats[nt], node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data[graph_key] = ft

        return graphs
