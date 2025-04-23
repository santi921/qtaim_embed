import dgl
import torch
from collections import defaultdict
from typing import Optional, Dict, List

from qtaim_embed.utils.scalers import _transform


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
        # print("SCALLING CALL ON STANDARD CALLED")
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


class HeteroGraphStandardScalerIterative:
    """
    Standardize hetero graph features by centering and normalization.
    Only node features are standardized. This variant differs because it
    computes the mean and std iteratively. This is useful for large datasets
    where the mean and std cannot be computed in one go.

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
        load: Optional[bool] = False,
        load_path: Optional[str] = None,
        dict_node_sizes: Optional[Dict[str, int]] = None,
        finalized: Optional[bool] = False,
    ):
        if load:
            self._load_scaler(load_path=load_path)
        else:
            self.copy = copy
            self._sum_x2 = {}
            self.features_tf = features_tf
            
            if mean is None:
                self._mean = {}
            else: 
                self._mean = mean
            
            if std is None:
                self._std = {}
            else:
                self._std = std
            
            if dict_node_sizes is None:
                self.dict_node_sizes = {}
            else:
                self.dict_node_sizes = dict_node_sizes
        
        self.finalized = finalized
            
            

    def update(self, graphs):
        """
        Update the class mean and std values from the given graphs.
        Don't standardize the graphs in this pass
        Takes:
            graphs: list of dgl graphs
        """
        g = graphs[0]
        # node_types = g.ntypes
        node_feats = defaultdict(list)

        if self.features_tf:  # separate track for features and labels
            graph_key = "feat"
        else:
            graph_key = "labels"

        node_types = list(g.ndata[graph_key].keys())
        # obtain feats from ALL graphs

        for g in graphs:
            for nt in node_types:
                data = g.nodes[nt].data[graph_key]
                node_feats[nt].append(data)
                # node_feats_size[nt].append(len(data))

        # standardize
        # print(node_feats)
        dtype = node_feats[node_types[0]][0].dtype

        for nt in node_types:
            # Update running statistics for new node types
            if nt not in self._mean:
                self._mean[nt] = torch.zeros(node_feats[nt][0].shape[1], dtype=torch.float64)
                self._sum_x2[nt] = torch.zeros(node_feats[nt][0].shape[1], dtype=torch.float64)
                self._std[nt] = torch.zeros(node_feats[nt][0].shape[1], dtype=torch.float64)
                self.dict_node_sizes[nt] = 0

            if torch.cat(node_feats[nt]).shape[1] > 0:
                feats = torch.cat(node_feats[nt]).to(torch.float64)  # Use higher precision
                # clean nans 
                feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                #print("feats shape: ", feats.shape)
                mean = torch.mean(feats, dim=0)
                #print("mean shape: ", mean.shape)
                mean = torch.as_tensor(mean, dtype=torch.float64, device=feats.device)
                #if nt == "global": 
                #    print("mean", self._mean[nt])
                #    print("mean", mean)
                #    print("n", self.dict_node_sizes[nt])
                #    print("n_new", feats.shape[0])
                self._mean[nt] = compute_running_average(
                    old_avg=self._mean[nt],
                    new_value=mean,
                    n=self.dict_node_sizes[nt],
                    n_new=feats.shape[0]
                )
                #print("nan in sum_x2", torch.sum(feats**2, dim=0))
                self._sum_x2[nt] += torch.sum(feats**2, dim=0)
                self.dict_node_sizes[nt] += feats.shape[0]

    
    def finalize(self):
        """
        Finalize the scaler by computing the mean and std from the given graphs.
        This is done by iterating over the graphs and computing the mean and std
        for each node type. The mean and std are stored in the class.
        """
        # compute std from mean and sum_x2
        print("...> finalizing scaler")
        for nt in self._mean.keys():
            if self.dict_node_sizes[nt] > 0:
                #print("mean", self._mean[nt], "sum_x2", self._sum_x2[nt])
                # nan check 
                #print("mean", torch.isnan(self._mean[nt]).any(), "sum_x2", torch.isnan(self._sum_x2[nt]).any())
                #print("node size", self.dict_node_sizes[nt])
                # compute std from mean and sum_x2
                self._std[nt] = torch.sqrt(
                    self._sum_x2[nt] / self.dict_node_sizes[nt] - self._mean[nt] ** 2
                )
                
                #print("sum_x2", self._sum_x2[nt])
                #print("self._std[nt]", self._std[nt])
                #print("std", self._std[nt])
            else:
                self._std[nt] = torch.zeros_like(self._mean[nt])
        self.finalized = True
        

    def __call__(self, graphs) -> List[dgl.DGLGraph]:

        # assert that the scaler is finalized
        assert self.finalized, "must finalize the scaler before using it"
        g = graphs[0]
        # node_types = g.ntypes
        node_feats = defaultdict(list)
        node_feats_size = defaultdict(list)
        if self.features_tf:  # separate track for features and labels
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
        if self._mean is not {} and self._std is not {}:
            for nt in node_types:
                feats = (torch.cat(node_feats[nt]) - self._mean[nt]) / self._std[nt]
                node_feats[nt] = feats

        # assign data back
        for nt in node_types:
            feats = torch.split(node_feats[nt], node_feats_size[nt])
            for g, ft in zip(graphs, feats):
                g.nodes[nt].data[graph_key] = ft

        return graphs
        

    @property
    def mean(self):
        """
        Returns the mean of the scaler.
        """
        return self._mean

    @property
    def std(self):
        """
        Returns the std of the scaler.
        """
        return self._std

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

        assert self.finalized, "must finalize the scaler before using it"

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
        assert self.finalized, "must finalize the scaler before using it"

        feats_ret = {}
        for nt in feats.keys():
            if len(feats[nt]) != 0:
                # node_feats_flat = torch.cat(feats[nt])
                node_feats_flat = feats[nt]
                feats_temp = node_feats_flat * self._std[nt] + self._mean[nt]
                feats_ret[nt] = feats_temp
        return feats_ret

    def save_scaler(self, path):
        """
        Save the scaler to a file.
        Takes:
            path: the path to the file where the scaler will be saved
        Returns:
            None
        """
        torch.save({
            'mean': self._mean,
            'std': self._std, 
            'sum_x2': self._sum_x2,
            'dict_node_sizes': self.dict_node_sizes,
            'finalized': self.finalized, 
            'features_tf': self.features_tf,
        }, path)
        
    def _load_scaler(self, load_path):
        """
        Load the scaler from a file.
        """
        # Load the scaler from the file
        data = torch.load(load_path)
        self._mean = data['mean']
        self._std = data['std']
        self._sum_x2 = data['sum_x2']
        self.dict_node_sizes = data['dict_node_sizes']
        self.finalized = data['finalized']
        self.features_tf = data['features_tf']


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


def merge_scalers(
        list_scalers: List[HeteroGraphStandardScalerIterative],
        features_tf: bool = True,
        finalize_merged: bool = False
    ):
    """
    Merge a list of scalers into one scaler.
    Takes:
        list_scalers: list of scalers
    Returns:
        merged_scaler: merged scaler
    """

    dict_node_sizes_merged = {}
    mean_merged = {}
    std_merged = {}
    #x2_merged = {}
    finalized_list = []

    for scaler in list_scalers:
        finalized_list.append(scaler.finalized)

        for nt in scaler._mean.keys():
            if nt not in mean_merged:
                mean_merged[nt] = torch.zeros_like(scaler._mean[nt])
                std_merged[nt] = torch.zeros_like(scaler._std[nt])
                dict_node_sizes_merged[nt] = 0
            
            # update the mean and std
            mean_merged[nt] += scaler._mean[nt] * scaler.dict_node_sizes[nt]
            std_merged[nt] += scaler._std[nt]**2 * scaler.dict_node_sizes[nt]
            dict_node_sizes_merged[nt] += scaler.dict_node_sizes[nt]

    # finalize the mean and std
    for nt in mean_merged.keys():
        if dict_node_sizes_merged[nt] > 0:
            mean_merged[nt] = mean_merged[nt] / dict_node_sizes_merged[nt]
            std_merged[nt] = torch.sqrt(std_merged[nt] / dict_node_sizes_merged[nt])
    finalized = False            
    
    if finalize_merged:
        finalized=True
    
    if all(finalized_list):
        finalized = True
    
    #print(std_merged)

    merged_scaler = HeteroGraphStandardScalerIterative(
        features_tf=features_tf, 
        mean=mean_merged, 
        std=std_merged,
        dict_node_sizes=dict_node_sizes_merged, 
        finalized=finalized
    )

    return merged_scaler


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


def compute_running_average(
    old_avg: float, 
    new_value: float, 
    n: int, 
    n_new: Optional[int] = 0
) -> float:
    """simple running average
    Args:
        old_avg (float): old average
        new_value (float): new value
        n (int): number of samples
        n_new (Optional[int]): number of new samples
    """
    # assert shapes 
    assert old_avg.shape == new_value.shape, "old_avg and new_value must have the same shape"
    #print("new_value", new_value)
    #print("old_avg", old_avg)
    #print("n", n)
    #print("n_new", n_new)
    if n == 0:
        return new_value
    if n_new == 0:
        return old_avg
    
    return old_avg + (new_value - old_avg) * n_new / (n + n_new)
    

