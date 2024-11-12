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
    # print("mean", mean)
    # print("std", std)
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
    # rst = (rst - mean) / std

    return rst, mean, std


def log_scale_from_dict(dict_params):
    """
    Helper function to create a log scale from a dictionary of parameters
    Takes:
        dict_params: dictionary with the following keys:
            copy: whether to copy the values as used by sklearn.preprocessing.StandardScaler
            features_tf: whether the features are in the "feat" key of the node data
            shift: shift to apply to the data before log scaling
    Returns:
        log_scale: HeteroGraphLogMagnitudeScaler object
    """
    return HeteroGraphLogMagnitudeScaler(
        copy=dict_params["copy"],
        features_tf=dict_params["features_tf"],
        shift=dict_params["shift"],
    )


def standard_scale_from_dict(dict_params):
    """
    Helper function to create a standard scale from a dictionary of parameters
    Takes:
        dict_params: dictionary with the following keys:
            copy: whether to copy the values as used by sklearn.preprocessing.StandardScaler
            features_tf: whether the features are in the "feat" key of the node data
            mean: with node type as key and the mean value as the value
            std: with node type as key and the std value as the value
    Returns:
        standard_scale: HeteroGraphStandardScaler object
    """
    if type(dict_params["mean"]) != torch.Tensor:
        dict_params["mean"] = torch.tensor(dict_params["mean"])
    if type(dict_params["std"]) != torch.Tensor:
        dict_params["std"] = torch.tensor(dict_params["std"])

    return HeteroGraphStandardScaler(
        copy=dict_params["copy"],
        features_tf=dict_params["features_tf"],
        mean=dict_params["mean"],
        std=dict_params["std"],
    )


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
        # node_types = g.ntypes
        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)
        if self.features_tf:
            graph_key = "feat"
        else:
            graph_key = "labels"
        node_types = list(g.ndata[graph_key].keys())
        # obtain feats from ALL graphs
        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data[graph_key]
                node_feats[nt].append(data)
                node_feats_size[nt].append(len(data))

        # standardize
        if self._mean is not None and self._std is not None:
            for nt in node_types:
                feats = (torch.cat(node_feats[nt]) - self._mean[nt]) / self._std[nt]
                node_feats[nt] = feats

        else:
            self._std = {}
            self._mean = {}
            dtype = node_feats[node_types[0]][0].dtype
            for nt in node_types:
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

    def inverse(self, graphs):
        """
        Perform inverse standardization on the given features.
        Takes:
            graphs: list of dgl graphs
        Returns:
            graphs: list of dgl graphs with inverse standardized features
        """
        # check that mean and std are not None
        assert (
            self._mean is not None and self._std is not None
        ), "must set up scaler first before inverting data"

        g = graphs[0]

        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)
        if self.features_tf:
            graph_key = "feat"
        else:
            graph_key = "labels"
        # print("graph key", graph_key)
        node_types = list(g.ndata[graph_key].keys())
        # print("node types", node_types)
        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data[graph_key]
                node_feats[nt].append(data)
                node_feats_size[nt].append(len(data))

        for nt in node_types:
            if len(node_feats[nt]) != 0:
                node_feats_flat = torch.cat(node_feats[nt])
                feats = node_feats_flat * self._std[nt] + self._mean[nt]
                node_feats[nt] = feats

        for nt in node_types:
            # node_feats[nt]
            # node_feats_size[nt]
            feats = torch.split(node_feats[nt], node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data[graph_key] = ft
        print("... > standard scaler - inverse done")
        return graphs

    def inverse_feats(self, feats):
        """
        Perform inverse standardization on the given features.
        Takes:
            feats: list of dgl graphs
        Returns:
            feats: list of dgl graphs with inverse standardized features
        """
        # node_feats = defaultdict(list)
        feats_ret = {}
        for nt in feats.keys():
            if len(feats[nt]) != 0:
                # node_feats_flat = torch.cat(feats[nt])
                node_feats_flat = feats[nt]
                feats_temp = node_feats_flat * self._std[nt] + self._mean[nt]
                feats_ret[nt] = feats_temp
        return feats_ret


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
        # node_types = g.ntypes

        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)
        if self.features_tf:
            graph_key = "feat"
        else:
            graph_key = "labels"
        node_types = list(g.ndata[graph_key].keys())
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
                # error here
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

    def inverse(self, graphs):
        """
        Perform inverse standardization on the given features.
        Takes:
            graphs: list of dgl graphs
        Returns:
            graphs: list of dgl graphs with inverse standardized features
        """
        g = graphs[0]
        # node_types = g.ntypes
        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)
        if self.features_tf:
            graph_key = "feat"
        else:
            graph_key = "labels"
        node_types = list(g.ndata[graph_key].keys())
        # node_types = g[graph_key].ntypes
        # obtain feats from ALL graphs
        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data[graph_key]
                node_feats[nt].append(data)
                node_feats_size[nt].append(len(data))

        # log scale
        for nt in node_types:
            if len(node_feats[nt]) != 0:
                try:  # deals with batched graphs
                    feats = torch.cat(node_feats[nt])
                except:
                    feats = node_feats[nt]
                # get the sign of the data
                sign = torch.sign(feats)
                # get the magnitude of the data
                feats = torch.abs(feats)
                # if shift is not None:
                # feats = feats + self.shift
                # exp scale the magnitude
                feats = torch.exp(feats)
                # undo the shift
                feats = feats - self.shift
                # put the sign back
                feats = feats * sign
                # assign
                node_feats[nt] = feats

        # assign data back
        for nt in node_types:
            feats = torch.split(node_feats[nt], node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data[graph_key] = ft
        print("... > log scaler - inverse done")
        return graphs

    def inverse_feats(self, feats):
        """
        Perform inverse standardization on the given features.
        Takes:
            feats: list of dgl graphs
        Returns:
            feats: list of dgl graphs with inverse standardized features
        """
        # node_feats = defaultdict(list)
        feats_ret = {}
        # log scale
        for nt in feats.keys():
            if len(feats[nt]) != 0:
                try:  # deals with batched graphs
                    # log scale
                    # feats_temp = torch.cat(feats[nt])
                    feats_temp = feats[nt]
                except:
                    feats_temp = torch.cat(feats[nt])

                sign = torch.sign(feats_temp)
                # get the magnitude of the data
                feats_temp = torch.abs(feats_temp)
                # if shift is not None:
                # feats = feats + self.shift
                # exp scale the magnitude
                feats_temp = torch.exp(feats_temp)
                # undo the shift
                feats_temp = feats_temp - self.shift
                # put the sign back
                feats_temp = feats_temp * sign
                # assign
                # feats[nt] = feats
                feats_ret[nt] = feats_temp

        return feats_ret
