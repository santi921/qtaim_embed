import torch
from tqdm import tqdm


class HeteroGraphNodeLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        molecule_wrappers,
        graphs,
        feature_names,
        target_dict,
        extra_dataset_info={},
    ):
        """
        Args:
            molecule_wrappers (list): list of MoleculeWrapper objects
            feature_names (list): list of feature names
            graphs (list): list of dgl graphs
            target_dict (dict): dict with node type as keys and target names as value
        """
        self.data = molecule_wrappers
        self.feature_names = feature_names
        self.graphs = graphs
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
