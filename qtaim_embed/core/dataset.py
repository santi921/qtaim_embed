import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from copy import deepcopy

from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.molwrapper import mol_wrappers_from_df
from qtaim_embed.data.processing import (
    HeteroGraphStandardScaler,
    HeteroGraphLogMagnitudeScaler,
)

# from qtaim_embed.utils.data import train_validation_test_split


class HeteroGraphNodeLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file,
        standard_scale_features=True,
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_targets=True,
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        debug=False,
        extra_keys={
            "atom": [
                "extra_feat_atom_esp_total",
            ],
            "bond": [
                "extra_feat_bond_esp_total",
                "bond_length",
            ],
            "global": [],
        },
        target_dict={
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["extra_feat_bond_esp_total"],
        },
        extra_dataset_info={},
    ):
        """
        Baseline dataset for hetero graph node label prediction. Includes global feautures.
        TODO: add support for no global features
        Args:
            file (string): path to data file
            standard_scale_features (bool): whether to scale features
            log_scale_features (bool): whether to log scale features
            allowed_ring_size (list): list of allowed ring sizes
            allowed_charges (list): list of allowed charges
            self_loop (bool): whether to add self loops to the graph
            debug (bool): whether to run in debug mode
            extra_keys (dict): dictionary of keys to grab from the data file
            target_dict (dict): dictionary of keys to use as labels
            extra_dataset_info (dict): dictionary of extra info to be stored in the dataset
        """
        # check if file ends in pkl
        if file[-3:] == "pkl":
            try:
                df = pd.read_pickle(file)
            except:
                import pickle5 as pickle

                with open(file, "rb") as fh:
                    df = pickle.load(fh)

        elif file[-3:] == "json":
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file)

        if debug:
            print("... > running in debug mode")
            df = df.head(100)
        for key_check in ["atom", "bond", "global"]:
            if key_check not in extra_keys.keys():
                extra_keys[key_check] = []

        mol_wrappers, element_set = mol_wrappers_from_df(
            df=df,
            bond_key="bonds",
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
        )

        grapher = get_grapher(
            element_set,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            allowed_ring_size=allowed_ring_size,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
            self_loop=self_loop,
        )

        graph_list = []
        print("... > Building graphs and featurizing")
        for mol in tqdm(mol_wrappers):
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)

        self.standard_scale_features = standard_scale_features
        self.log_scale_features = log_scale_features
        self.log_scale_labels = log_scale_targets
        self.standard_scale_labels = standard_scale_targets
        self.feature_scalers = []
        self.label_scalers = []
        self.data = mol_wrappers
        self.element_set = element_set
        self.feature_names = names
        self.graphs = graph_list
        self.allowed_spins = allowed_spins
        self.allowed_charges = allowed_charges
        self.target_dict = target_dict
        self.extra_dataset_info = extra_dataset_info

        self.load()
        print("... > loaded dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        return self.graphs[idx]  # , self.labels[idx]

    def get_include_exclude_indices(self):
        target_locs = {}
        # get locations of target features
        for node_type, value_list in self.target_dict.items():
            if node_type not in target_locs:
                target_locs[node_type] = []

            for value in value_list:
                target_locs[node_type].append(
                    self.feature_names[node_type].index(value)
                )

        # now partition features into feats in target_locs and feats not in target_locs
        include_locs = {}
        exclude_locs = {}
        include_names = {}
        exclude_names = {}

        for node_type, value_list in self.feature_names.items():
            # if node_type not in include_locs:
            #    include_locs[node_type] = []
            #    exclude_locs[node_type] = []
            #    include_names[node_type] = []
            #    exclude_names[node_type] = []

            for i, value in enumerate(value_list):
                if node_type in target_locs.keys():
                    if i in target_locs[node_type]:
                        if node_type not in include_names:
                            include_names[node_type] = []
                            include_locs[node_type] = []

                        include_locs[node_type].append(i)
                        include_names[node_type].append(value)
                    else:
                        if node_type not in exclude_names:
                            exclude_names[node_type] = []
                            exclude_locs[node_type] = []
                        exclude_locs[node_type].append(i)
                        exclude_names[node_type].append(value)
                else:
                    if node_type not in exclude_names:
                        exclude_names[node_type] = []
                        exclude_locs[node_type] = []
                    exclude_locs[node_type].append(i)
                    exclude_names[node_type].append(value)

        self.include_locs = include_locs
        self.exclude_locs = exclude_locs
        self.include_names = include_names
        self.exclude_names = exclude_names
        print("included in labels")
        # print(self.include_locs)
        print(self.include_names)
        print("included in graph features")
        # print(self.exclude_locs)
        print(self.exclude_names)

    def load(self):
        self.get_include_exclude_indices()
        print("original loader node types:", self.graphs[0].ndata["feat"].keys())
        print("original loader label types:", self.graphs[0].ndata["labels"].keys())
        print("include names: ", self.include_names.keys())
        print("... > parsing labels and features in graphs")
        for graph in tqdm(self.graphs):
            labels = {}
            features_new = {}
            for key, value in graph.ndata["feat"].items():
                if key in self.include_names.keys():
                    graph_features = {}

                    graph_features[key] = graph.ndata["feat"][key][
                        :, self.exclude_locs[key]
                    ]

                    features_new.update(graph_features)
                    # if key == "global":
                    labels[key] = graph.ndata["feat"][key][:, self.include_locs[key]]
                    # else:
                    #    labels[key] = graph.ndata["feat"][key][
                    #        :, self.include_locs[key]
                    #    ]
                graph.ndata["feat"] = features_new
                graph.ndata["labels"] = labels

            # label_list.append(labels)
        print("original loader node types:", graph.ndata["feat"].keys())
        print("original loader label types:", graph.ndata["labels"].keys())

        if self.log_scale_features:
            print("... > Log scaling features")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=True, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.feature_scalers.append(log_scaler)
            print("... > Log scaling features complete")

        if self.standard_scale_features:
            print("... > Scaling features")
            scaler = HeteroGraphStandardScaler()
            self.graphs = scaler(self.graphs)
            # self.scaler_feat_mean = scaler.mean
            # self.scaler_feat_std = scaler.std
            self.feature_scalers.append(scaler)
            print("... > Scaling features complete")
            print("... > feature mean(s): \n", scaler.mean)
            print("... > feature std(s):  \n", scaler.std)

        if self.log_scale_labels:
            print("... > Log scaling targets")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=False, shift=1)
            self.graphs = log_scaler(self.graphs)
            # self.log_label_scaler = log_scaler
            self.label_scalers.append(log_scaler)
            print("... > Log scaling targets complete")

        if self.standard_scale_labels:
            print("... > Scaling targets")
            scaler = HeteroGraphStandardScaler(features_tf=False)
            self.graphs = scaler(self.graphs)
            # self.scaler_label_mean = scaler.mean
            # self.scaler_label_std = scaler.std
            # self.standard_label_scaler = scaler
            self.label_scalers.append(scaler)
            print("... > Scaling targets complete")
            print("... > feature mean(s): \n", scaler.mean)
            print("... > feature std(s):  \n", scaler.std)

        # self.labels = label_list

    def feature_names(self):
        return self.exclude_names

    def label_names(self):
        return self.include_names

    def feature_size(self):
        len_dict = {}
        for key, value in self.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict

    def unscale_features(self, graphs):
        """
        Perform inverse standardization on the given graphs.
        Takes:
            graphs: list of dgl graphs
        Returns:
            graphs: list of dgl graphs with inverse standardized features
        """
        # assert that one of standard_scale_targets or log_scale_targets is true
        assert (
            self.log_scale_features or self.standard_scale_features
        ), "a scaler must be used to unscale features"
        graphs_ret = deepcopy(graphs)
        if self.standard_scale_features and self.log_scale_features:
            print("unscaling feats with log and standard scalers")
            graphs_ret = self.feature_scalers[1].inverse(graphs_ret)
            graphs_ret = self.feature_scalers[0].inverse(graphs_ret)

        elif self.log_scale_features or self.standard_scale_features:
            graphs_ret = self.feature_scalers[0].inverse(graphs_ret)

        return graphs_ret

    def unscale_targets(self, graphs):
        """
        Perform inverse standardization of targets on the given graphs.
        Takes:
            graphs: list of dgl graphs
        Returns:
            graphs: list of dgl graphs with inverse standardized targets

        """
        # assert that one of standard_scale_targets or log_scale_targets is true
        assert (
            self.log_scale_labels or self.standard_scale_labels
        ), "a scaler must be used to unscale targets"

        graphs_ret = deepcopy(graphs)

        if self.standard_scale_labels and self.log_scale_labels:
            print("unscaling feats with log and standard scalers")
            graphs_ret = self.label_scalers[1].inverse(graphs_ret)
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        elif self.log_scale_labels or self.standard_scale_labels:
            if self.log_scale_labels:
                print("unscaling feats with log scaler")
            if self.standard_scale_labels:
                print("unscaling feats with standard scaler")
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        return graphs_ret


class HeteroGraphGraphLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file,
        standard_scale_features=True,
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_targets=True,
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        debug=False,
        extra_keys={
            "atom": [
                "extra_feat_atom_esp_total",
            ],
            "bond": [
                "extra_feat_bond_esp_total",
                "bond_length",
            ],
            "global": ["extra_feat_global_E1_CAM"],
        },
        target_list=["extra_feat_global_E1_CAM"],
        extra_dataset_info={},
    ):
        """
        Baseline dataset for hetero graph node label prediction. Includes global feautures.
        TODO: add support for no global features
        Args:
            file (string): path to data file
            standard_scale_features (bool): whether to scale features
            log_scale_features (bool): whether to log scale features
            allowed_ring_size (list): list of allowed ring sizes
            allowed_charges (list): list of allowed charges
            self_loop (bool): whether to add self loops to the graph
            debug (bool): whether to run in debug mode
            extra_keys (dict): dictionary of keys to grab from the data file
            target_list (list of strings): dictionary of global keys to use as labels
            extra_dataset_info (dict): dictionary of extra info to be stored in the dataset
        """
        # check if file ends in pkl
        if file[-3:] == "pkl":
            try:
                df = pd.read_pickle(file)
            except:
                import pickle5 as pickle

                with open(file, "rb") as fh:
                    df = pickle.load(fh)

        elif file[-3:] == "json":
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file)

        if debug:
            print("... > running in debug mode")
            df = df.head(100)
        for key_check in ["atom", "bond", "global"]:
            if key_check not in extra_keys.keys():
                extra_keys[key_check] = []

        mol_wrappers, element_set = mol_wrappers_from_df(
            df=df,
            bond_key="bonds",
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
        )

        grapher = get_grapher(
            element_set,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            allowed_ring_size=allowed_ring_size,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
            self_loop=self_loop,
        )

        graph_list = []
        print("... > Building graphs and featurizing")
        for mol in tqdm(mol_wrappers):
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)

        self.standard_scale_features = standard_scale_features
        self.log_scale_features = log_scale_features
        self.log_scale_labels = log_scale_targets
        self.standard_scale_labels = standard_scale_targets
        self.feature_scalers = []
        self.label_scalers = []
        self.data = mol_wrappers
        self.element_set = element_set
        self.feature_names = names
        self.allowed_spins = allowed_spins
        self.allowed_spins = allowed_charges
        self.graphs = graph_list
        target_dict = {"global": target_list}
        self.target_dict = target_dict
        self.extra_dataset_info = extra_dataset_info

        self.load()
        print("... > loaded dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        return self.graphs[idx]  # , self.labels[idx]

    def get_include_exclude_indices(self):
        target_locs = {}
        # get locations of target features
        for node_type, value_list in self.target_dict.items():
            if node_type not in target_locs:
                target_locs[node_type] = []

            for value in value_list:
                target_locs[node_type].append(
                    self.feature_names[node_type].index(value)
                )

        # now partition features into feats in target_locs and feats not in target_locs
        include_locs = {}
        exclude_locs = {}
        include_names = {}
        exclude_names = {}

        for node_type, value_list in self.feature_names.items():
            # if node_type not in include_locs:
            #    include_locs[node_type] = []
            #    exclude_locs[node_type] = []
            #    include_names[node_type] = []
            #    exclude_names[node_type] = []

            for i, value in enumerate(value_list):
                if node_type in target_locs.keys():
                    if i in target_locs[node_type]:
                        if node_type not in include_names:
                            include_names[node_type] = []
                            include_locs[node_type] = []

                        include_locs[node_type].append(i)
                        include_names[node_type].append(value)
                    else:
                        if node_type not in exclude_names:
                            exclude_names[node_type] = []
                            exclude_locs[node_type] = []
                        exclude_locs[node_type].append(i)
                        exclude_names[node_type].append(value)
                else:
                    if node_type not in exclude_names:
                        exclude_names[node_type] = []
                        exclude_locs[node_type] = []
                    exclude_locs[node_type].append(i)
                    exclude_names[node_type].append(value)

        self.include_locs = include_locs
        self.exclude_locs = exclude_locs
        self.include_names = include_names
        self.exclude_names = exclude_names
        print("included in labels")
        # print(self.include_locs)
        print(self.include_names)
        print("included in graph features")
        # print(self.exclude_locs)
        print(self.exclude_names)

    def load(self):
        self.get_include_exclude_indices()
        print("original loader node types:", self.graphs[0].ndata["feat"].keys())
        print("original loader label types:", self.graphs[0].ndata["labels"].keys())
        print("include names: ", self.include_names.keys())
        print("... > parsing labels and features in graphs")
        for graph in tqdm(self.graphs):
            labels = {}
            features_new = {}
            for key, value in graph.ndata["feat"].items():
                if key in self.include_names.keys():
                    graph_features = {}

                    graph_features[key] = graph.ndata["feat"][key][
                        :, self.exclude_locs[key]
                    ]

                    features_new.update(graph_features)
                    # if key == "global":
                    labels[key] = graph.ndata["feat"][key][:, self.include_locs[key]]
                    # else:
                    #    labels[key] = graph.ndata["feat"][key][
                    #        :, self.include_locs[key]
                    #    ]
                graph.ndata["feat"] = features_new
                graph.ndata["labels"] = labels

            # label_list.append(labels)
        print("original loader node types:", graph.ndata["feat"].keys())
        print("original loader label types:", graph.ndata["labels"].keys())

        if self.log_scale_features:
            print("... > Log scaling features")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=True, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.feature_scalers.append(log_scaler)
            print("... > Log scaling features complete")

        if self.standard_scale_features:
            print("... > Scaling features")
            scaler = HeteroGraphStandardScaler()
            self.graphs = scaler(self.graphs)
            # self.scaler_feat_mean = scaler.mean
            # self.scaler_feat_std = scaler.std
            self.feature_scalers.append(scaler)
            print("... > Scaling features complete")
            print("... > feature mean(s): \n", scaler.mean)
            print("... > feature std(s):  \n", scaler.std)

        if self.log_scale_labels:
            print("... > Log scaling targets")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=False, shift=1)
            self.graphs = log_scaler(self.graphs)
            # self.log_label_scaler = log_scaler
            self.label_scalers.append(log_scaler)
            print("... > Log scaling targets complete")

        if self.standard_scale_labels:
            print("... > Scaling targets")
            scaler = HeteroGraphStandardScaler(features_tf=False)
            self.graphs = scaler(self.graphs)
            # self.scaler_label_mean = scaler.mean
            # self.scaler_label_std = scaler.std
            # self.standard_label_scaler = scaler
            self.label_scalers.append(scaler)
            print("... > Scaling targets complete")
            print("... > feature mean(s): \n", scaler.mean)
            print("... > feature std(s):  \n", scaler.std)

        # self.labels = label_list

    def feature_names(self):
        return self.exclude_names

    def label_names(self):
        return self.include_names

    def feature_size(self):
        len_dict = {}
        for key, value in self.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict

    def unscale_features(self, graphs):
        """
        Perform inverse standardization on the given graphs.
        Takes:
            graphs: list of dgl graphs
        Returns:
            graphs: list of dgl graphs with inverse standardized features
        """
        # assert that one of standard_scale_targets or log_scale_targets is true
        assert (
            self.log_scale_features or self.standard_scale_features
        ), "a scaler must be used to unscale features"
        graph_ret = deepcopy(graphs)
        if self.standard_scale_features and self.log_scale_features:
            print("unscaling feats with log and standard scalers")
            graph_ret = self.feature_scalers[1].inverse(graph_ret)
            graph_ret = self.feature_scalers[0].inverse(graph_ret)

        elif self.log_scale_features or self.standard_scale_features:
            graph_ret = self.feature_scalers[0].inverse(graph_ret)

        return graph_ret

    def unscale_targets(self, graphs):
        """
        Perform inverse standardization of targets on the given graphs.
        Takes:
            graphs: list of dgl graphs
        Returns:
            graphs: list of dgl graphs with inverse standardized targets

        """
        # assert that one of standard_scale_targets or log_scale_targets is true
        assert (
            self.log_scale_labels or self.standard_scale_labels
        ), "a scaler must be used to unscale targets"
        graphs_ret = deepcopy(graphs)
        if self.standard_scale_labels and self.log_scale_labels:
            print("unscaling feats with log and standard scalers")
            graphs_ret = self.label_scalers[1].inverse(graphs_ret)
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        elif self.log_scale_labels or self.standard_scale_labels:
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        return graphs_ret


class HeteroGraphGraphLabelClassifierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file,
        standard_scale_features=True,
        log_scale_features=True,
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        debug=False,
        extra_keys={
            "atom": [
                "extra_feat_atom_esp_total",
            ],
            "bond": [
                "extra_feat_bond_esp_total",
                "bond_length",
            ],
            "global": ["NR-AR"],
        },
        target_list=["NR-AR"],
        extra_dataset_info={},
        impute=False,
    ):
        """
        Baseline dataset for hetero graph node label prediction. Includes global feautures.

        Args:
            file (string): path to data file
            standard_scale_features (bool): whether to scale features
            log_scale_features (bool): whether to log scale features
            allowed_ring_size (list): list of allowed ring sizes
            allowed_charges (list): list of allowed charges
            self_loop (bool): whether to add self loops to the graph
            debug (bool): whether to run in debug mode
            extra_keys (dict): dictionary of keys to grab from the data file
            target_list (list of strings): dictionary of global keys to use as labels
            extra_dataset_info (dict): dictionary of extra info to be stored in the dataset
        """
        # check if file ends in pkl
        if file[-3:] == "pkl":
            try:
                df = pd.read_pickle(file)
            except:
                import pickle5 as pickle

                with open(file, "rb") as fh:
                    df = pickle.load(fh)

        elif file[-3:] == "json":
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file)

        if debug:
            print("... > running in debug mode")
            df = df.head(100)
        for key_check in ["atom", "bond", "global"]:
            if key_check not in extra_keys.keys():
                extra_keys[key_check] = []

        mol_wrappers, element_set = mol_wrappers_from_df(
            df=df,
            bond_key="bonds",
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
        )

        grapher = get_grapher(
            element_set,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            allowed_ring_size=allowed_ring_size,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
            self_loop=self_loop,
        )

        graph_list = []
        print("... > Building graphs and featurizing")
        for mol in tqdm(mol_wrappers):
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)

        self.standard_scale_features = standard_scale_features
        self.log_scale_features = log_scale_features
        self.feature_scalers = []
        self.data = mol_wrappers  # to be filtered
        self.element_set = element_set
        self.feature_names = names
        self.graphs = graph_list  # to be filtered
        target_dict = {"global": target_list}
        self.allowed_spins = allowed_spins
        self.target_dict = target_dict
        self.extra_dataset_info = extra_dataset_info
        self.impute = impute

        self.load()
        print("... > loaded dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        return self.graphs[idx]  # , self.labels[idx]

    def get_include_exclude_indices(self):
        target_locs = {}
        # get locations of target features
        for node_type, value_list in self.target_dict.items():
            if node_type not in target_locs:
                target_locs[node_type] = []

            for value in value_list:
                target_locs[node_type].append(
                    self.feature_names[node_type].index(value)
                )

        # now partition features into feats in target_locs and feats not in target_locs
        include_locs = {}
        exclude_locs = {}
        include_names = {}
        exclude_names = {}

        for node_type, value_list in self.feature_names.items():
            for i, value in enumerate(value_list):
                if node_type in target_locs.keys():
                    if i in target_locs[node_type]:
                        if node_type not in include_names:
                            include_names[node_type] = []
                            include_locs[node_type] = []

                        include_locs[node_type].append(i)
                        include_names[node_type].append(value)
                    else:
                        if node_type not in exclude_names:
                            exclude_names[node_type] = []
                            exclude_locs[node_type] = []
                        exclude_locs[node_type].append(i)
                        exclude_names[node_type].append(value)
                else:
                    if node_type not in exclude_names:
                        exclude_names[node_type] = []
                        exclude_locs[node_type] = []
                    exclude_locs[node_type].append(i)
                    exclude_names[node_type].append(value)

        self.include_locs = include_locs
        self.exclude_locs = exclude_locs
        self.include_names = include_names
        self.exclude_names = exclude_names
        print("included in labels")
        # print(self.include_locs)
        print(self.include_names)
        print("included in graph features")
        # print(self.exclude_locs)
        print(self.exclude_names)

    def load(self):
        self.get_include_exclude_indices()
        print("original loader node types:", self.graphs[0].ndata["feat"].keys())
        print("original loader label types:", self.graphs[0].ndata["labels"].keys())
        print("include names: ", self.include_names.keys())
        print("... > parsing labels and features in graphs")
        filtered_graph_count = 0
        filter_ind = []
        categories_set_list = [set([]) for i in range(len(self.include_locs["global"]))]
        if self.impute:
            self.mode_dict = {}

        # print("categories set list: ", categories_set_list)
        for ind, graph in tqdm(enumerate(self.graphs)):
            labels = {}
            features_new = {}
            for key, value in graph.ndata["feat"].items():
                if key in self.include_names.keys():
                    graph_features = {}

                    graph_features[key] = graph.ndata["feat"][key][
                        :, self.exclude_locs[key]
                    ]

                    features_new.update(graph_features)
                    labels_temp = graph.ndata["feat"][key][:, self.include_locs[key]]
                    # print("labels temp: ", labels_temp)
                    # check if any of the labels are nan
                    if self.impute:
                        for ind_task, val in enumerate(labels_temp[0]):
                            if not torch.isnan(val):
                                if ind_task not in self.mode_dict:
                                    self.mode_dict[ind_task] = [0, 0]
                                self.mode_dict[ind_task][int(val.item())] += 1

                    if torch.isnan(labels_temp).any():
                        filtered_graph_count += 1
                        filter_ind.append(ind)
                        list_labels = labels_temp.tolist()[0]
                        for i, label in enumerate(list_labels):
                            if not np.isnan(label):
                                categories_set_list[i].add(int(label))

                    else:
                        list_labels = labels_temp.tolist()[0]

                        [
                            categories_set_list[i].add(int(list_labels[i]))
                            for i in range(len(self.include_locs["global"]))
                        ]  # update number of categories
                    # print("labels temp: ", labels_temp[0])
                    labels[key] = labels_temp

            graph.ndata["feat"] = features_new
            # print(labels)
            graph.ndata["labels"] = labels
        if self.impute:
            # print("mode dict: ", self.mode_dict)
            # convert mode dict to mode
            for key, value in self.mode_dict.items():
                self.mode_dict[key] = value.index(max(value))

        self.categories_set_list = categories_set_list
        # print("categories set list: ", categories_set_list)
        dict_distro = {}
        graphs_filtered = []
        mol_wrappers_filtered = []
        for ind, graph in enumerate(self.graphs):
            if self.impute == False:
                if ind not in filter_ind:
                    graphs_filtered.append(graph)
                    mol_wrappers_filtered.append(self.data[ind])
                    # convert labels to one hot
                    labels = []
                    labels_raw = graph.ndata["labels"]["global"]

                    if len(categories_set_list) == 1:
                        labels = np.zeros((1, len(categories_set_list[0])))
                    else:
                        labels = np.zeros(
                            (len(categories_set_list), len(categories_set_list[0]))
                        )
                        labels_raw = labels_raw[0]

                    # print("labels raw: ", labels_raw)
                    # for i, label in enumerate(labels_raw):
                    for j, category_set in enumerate(categories_set_list):
                        ind_hot = list(category_set).index(int(labels_raw[j]))
                        if j not in dict_distro:
                            dict_distro[j] = [0 for i in range(len(category_set))]
                        dict_distro[j][ind_hot] += 1
                        if len(categories_set_list) == 1:
                            labels[0, ind_hot] = 1
                        else:
                            labels[j, ind_hot] = 1

                    labels = torch.tensor(np.array([labels]), dtype=torch.int64)
                    # labels = torch.tensor(labels.flatten())
                    # print(labels)
                    graph.ndata["labels"] = {"global": labels}
            else:
                graphs_filtered.append(graph)
                mol_wrappers_filtered.append(self.data[ind])
                labels = []
                labels_raw = graph.ndata["labels"]["global"]
                if len(categories_set_list) == 1:
                    labels = np.zeros((1, len(categories_set_list[0])))
                else:
                    labels = np.zeros(
                        (len(categories_set_list), len(categories_set_list[0]))
                    )
                    labels_raw = labels_raw[0]
                for j, category_set in enumerate(categories_set_list):
                    # print(category_set)
                    # print("labels raw: ", labels_raw[j])
                    if labels_raw[j] == np.NaN or torch.isnan(labels_raw[j]):
                        ind_hot = self.mode_dict[j]
                    else:
                        ind_hot = list(category_set).index(int(labels_raw[j]))

                    if j not in dict_distro:
                        dict_distro[j] = [0 for i in range(len(category_set))]
                    dict_distro[j][ind_hot] += 1
                    if len(categories_set_list) == 1:
                        labels[0, ind_hot] = 1
                    else:
                        labels[j, ind_hot] = 1
                labels = torch.tensor(np.array([labels]), dtype=torch.int64)
                graph.ndata["labels"] = {"global": labels}

        self.graphs = graphs_filtered
        self.data = mol_wrappers_filtered
        self.dict_distro = dict_distro
        # label_list.append(labels)
        print("... > number of categories for each label: ")
        for i, category_set in enumerate(categories_set_list):
            print(
                "...... > label ",
                self.include_names["global"][i],
                ": ",
                len(category_set),
                "with distribution: ",
                dict_distro[i],
            )
        print("original loader node types:", graph.ndata["feat"].keys())
        print("original loader label types:", graph.ndata["labels"].keys())

        if self.impute:
            print("number of graphs imputed: ", filtered_graph_count)
        else:
            print("number of graphs filtered: ", filtered_graph_count)

        if self.log_scale_features:
            print("... > Log scaling features")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=True, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.feature_scalers.append(log_scaler)
            print("... > Log scaling features complete")

        if self.standard_scale_features:
            print("... > Scaling features")
            scaler = HeteroGraphStandardScaler()
            self.graphs = scaler(self.graphs)
            # self.scaler_feat_mean = scaler.mean
            # self.scaler_feat_std = scaler.std
            self.feature_scalers.append(scaler)
            print("... > Scaling features complete")
            print("... > feature mean(s): \n", scaler.mean)
            print("... > feature std(s):  \n", scaler.std)

        # self.labels = label_list

    def feature_names(self):
        return self.exclude_names

    def label_names(self):
        return self.include_names

    def feature_size(self):
        len_dict = {}
        for key, value in self.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict

    def unscale_features(self, graphs):
        """
        Perform inverse standardization on the given graphs.
        Takes:
            graphs: list of dgl graphs
        Returns:
            graphs: list of dgl graphs with inverse standardized features
        """
        # assert that one of standard_scale_targets or log_scale_targets is true
        assert (
            self.log_scale_features or self.standard_scale_features
        ), "a scaler must be used to unscale features"
        graph_ret = deepcopy(graphs)
        if self.standard_scale_features and self.log_scale_features:
            print("unscaling feats with log and standard scalers")
            graph_ret = self.feature_scalers[1].inverse(graph_ret)
            graph_ret = self.feature_scalers[0].inverse(graph_ret)

        elif self.log_scale_features or self.standard_scale_features:
            graph_ret = self.feature_scalers[0].inverse(graph_ret)

        return graph_ret


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def feature_names(self):
        return self.dataset.exclude_names

    def label_names(self):
        return self.dataset.include_names

    def feature_size(self):
        len_dict = {}
        for key, value in self.dataset.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict


