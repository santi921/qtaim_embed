import torch
from tqdm import tqdm
import pandas as pd
from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.molwrapper import mol_wrappers_from_df


class HeteroGraphNodeLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file,
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        self_loop=True,
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
            allowed_ring_size (list): list of allowed ring sizes
            allowed_charges (list): list of allowed charges
            self_loop (bool): whether to add self loops to the graph
            extra_keys (dict): dictionary of keys to grab from the data file
            target_dict (dict): dictionary of keys to use as labels
            extra_dataset_info (dict): dictionary of extra info to be stored in the dataset
        """
        # check if file ends in pkl
        if file[-3:] == "pkl":
            df = pd.read_pickle(file)
        elif file[-3:] == "json":
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file)

        mol_wrappers, element_set = mol_wrappers_from_df(
            df, extra_keys["atom"], extra_keys["bond"], extra_keys["global"]
        )

        grapher = get_grapher(
            element_set,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            allowed_ring_size=allowed_ring_size,
            allowed_charges=allowed_charges,
            self_loop=self_loop,
        )

        graph_list = []
        print("... > Building graphs and featurizing")
        for mol in tqdm(mol_wrappers):
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)

        self.data = mol_wrappers
        self.element_set = element_set
        self.feature_names = names
        self.graphs = graph_list
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
            if node_type not in include_locs:
                include_locs[node_type] = []
                exclude_locs[node_type] = []
                include_names[node_type] = []
                exclude_names[node_type] = []

            for i, value in enumerate(value_list):
                if node_type in target_locs.keys():
                    if i in target_locs[node_type]:
                        include_locs[node_type].append(i)
                        include_names[node_type].append(value)
                    else:
                        exclude_locs[node_type].append(i)
                        exclude_names[node_type].append(value)
                else:
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

        label_list = []
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
                    if key == "global":
                        labels[key] = graph.ndata["feat"][key][
                            :, self.include_locs[key]
                        ]
                    else:
                        labels[key] = graph.ndata["feat"][key][
                            :, self.include_locs[key]
                        ]
                graph.ndata["feat"] = features_new
                graph.ndata["labels"] = labels
            # label_list.append(labels)

        self.labels = label_list


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

    def featuze_size(self):
        len_ret = 0
        for _, value in self.dataset.include_locs.items():
            len_ret += len(value)

        return len_ret
