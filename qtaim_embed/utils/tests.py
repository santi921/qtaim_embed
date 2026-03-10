import torch
import numpy as np
from pathlib import Path
import pandas as pd
from torch_geometric.data import HeteroData

from qtaim_embed.utils.models import (
    get_layer_args,
)
from qtaim_embed.utils.data import get_default_graph_level_config
from qtaim_embed.models.utils import load_graph_level_model_from_config

from qtaim_embed.core.dataset import (
    HeteroGraphNodeLabelDataset,
    HeteroGraphGraphLabelDataset,
    HeteroGraphGraphLabelClassifierDataset,
)

# Project root directory (qtaim_embed/../ from this file's location)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "tests" / "data"


class hyperparams:
    def __init__(self, config):
        self.config = config
        # set every key in config as an attribute
        for k, v in config.items():
            setattr(self, k, v)


def get_data():
    return pd.read_pickle(DATA_DIR / "qm8_test.pkl")


def get_data_spin_charge():
    return pd.read_pickle(DATA_DIR / "labelled_spin_charge.pkl")


def get_dataset(
    log_scale_features,
    log_scale_targets,
    standard_scale_features,
    standard_scale_targets,
):
    # get the root directory for this package
    # (i.e. the directory where setup.py is located)
    dataset = HeteroGraphNodeLabelDataset(
        file=str(DATA_DIR / "labelled_data.pkl"),
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        extra_keys={
            "atom": ["extra_feat_atom_esp_total"],
            "bond": [
                "bond_length",
                "extra_feat_bond_esp_total",
            ],
            "global": ["extra_feat_global_E1_CAM"],
        },
        target_dict={"global": ["extra_feat_global_E1_CAM"]},
        extra_dataset_info={},
        debug=True,
        element_set=[],
        log_scale_features=log_scale_features,
        log_scale_targets=log_scale_targets,
        standard_scale_features=standard_scale_features,
        standard_scale_targets=standard_scale_targets,
        bond_key="bonds",
        map_key="extra_feat_bond_indices_qtaim",
    )

    return dataset


def get_dataset_graph_level(
    log_scale_features,
    log_scale_targets,
    standard_scale_features,
    standard_scale_targets,
):
    # get the root directory for this package
    # (i.e. the directory where setup.py is located)
    root_dir = Path(__file__).parent.parent
    dataset = HeteroGraphGraphLabelDataset(
        file=str(DATA_DIR / "labelled_data.pkl"),
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        element_set=[],
        extra_keys={
            "atom": ["extra_feat_atom_esp_total"],
            "bond": [
                "bond_length",
                "extra_feat_bond_esp_total",
            ],
            "global": ["extra_feat_global_E1_CAM"],
        },
        target_list=["extra_feat_global_E1_CAM"],
        extra_dataset_info={},
        debug=True,
        log_scale_features=log_scale_features,
        log_scale_targets=log_scale_targets,
        standard_scale_features=standard_scale_features,
        standard_scale_targets=standard_scale_targets,
        bond_key="bonds",
        map_key="extra_feat_bond_indices_qtaim",
    )

    return dataset


def get_dataset_graph_level_multitask(
    log_scale_features,
    log_scale_targets,
    standard_scale_features,
    standard_scale_targets,
):
    # get the root directory for this package
    # (i.e. the directory where setup.py is located)
    root_dir = Path(__file__).parent.parent
    dataset = HeteroGraphGraphLabelDataset(
        file=str(DATA_DIR / "labelled_data.pkl"),
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        element_set=[],
        extra_keys={
            "atom": ["extra_feat_atom_esp_total"],
            "bond": [
                "bond_length",
                "extra_feat_bond_esp_total",
            ],
            "global": ["extra_feat_global_E1_CAM", "extra_feat_global_E2_CAM"],
        },
        target_list=["extra_feat_global_E1_CAM", "extra_feat_global_E2_CAM"],
        extra_dataset_info={},
        debug=True,
        log_scale_features=log_scale_features,
        log_scale_targets=log_scale_targets,
        standard_scale_features=standard_scale_features,
        standard_scale_targets=standard_scale_targets,
        bond_key="bonds",
        map_key="extra_feat_bond_indices_qtaim",
    )

    return dataset


def get_datasets_graph_level_classifier(log_scale_features, standard_scale_features):
    root_dir = Path(__file__).parent.parent
    dataset_single = HeteroGraphGraphLabelClassifierDataset(
        file=str(DATA_DIR / "test_classifier_labelled.pkl"),
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        extra_keys={
            "atom": ["extra_feat_atom_esp_total"],
            "bond": [
                "bond_length",
                "extra_feat_bond_esp_total",
            ],
            "global": ["NR-AR"],
        },
        target_list=["NR-AR"],
        extra_dataset_info={},
        debug=True,
        element_set=[],
        log_scale_features=log_scale_features,
        standard_scale_features=standard_scale_features,
        bond_key="bonds",
        map_key="extra_feat_bond_indices_qtaim",
    )
    dataset_multi = HeteroGraphGraphLabelClassifierDataset(
        file=str(DATA_DIR / "test_classifier_labelled.pkl"),
        standard_scale_features=True,
        log_scale_features=True,
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        debug=True,
        extra_keys={
            "atom": [
                "extra_feat_atom_esp_total",
            ],
            "bond": [
                "extra_feat_bond_esp_total",
                "bond_length",
            ],
            "global": ["NR-AR", "SR-p53"],
        },
        target_list=["NR-AR", "SR-p53"],
        extra_dataset_info={},
        element_set=[],
        bond_key="bonds",
        map_key="extra_feat_bond_indices_qtaim",
    )
    return dataset_single, dataset_multi


def get_invalid_data():
    return pd.read_pickle(DATA_DIR / "qm8_invalid.pkl")


def make_hetero(num_atoms, num_bonds, a2b, b2a, self_loop=True):
    """
    Create a hetero graph and create features.
    A global node is connected to all atoms and bonds.

    Atom features are:
    [[0,1],
     [2,3],
     .....]

    Bond features are:
    [[0,1,2],
     [3,4,5],
     .....]

    Global features are:
    [[0,1,2,3]]
    """
    if num_bonds == 0:
        # create a fake bond and create an edge atom->bond
        num_bonds = 1
        a2b = [(0, 0)]
        b2a = [(0, 0)]

    data = HeteroData()

    # Set number of nodes
    data["atom"].num_nodes = num_atoms
    data["bond"].num_nodes = num_bonds
    data["global"].num_nodes = 1

    # Convert edge lists from list of tuples to [2, E] tensors
    def _edges_to_tensor(edge_list):
        if len(edge_list) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        src = [e[0] for e in edge_list]
        dst = [e[1] for e in edge_list]
        return torch.tensor([src, dst], dtype=torch.long)

    data["atom", "a2b", "bond"].edge_index = _edges_to_tensor(a2b)
    data["bond", "b2a", "atom"].edge_index = _edges_to_tensor(b2a)
    data["atom", "a2g", "global"].edge_index = _edges_to_tensor(
        [(i, 0) for i in range(num_atoms)]
    )
    data["global", "g2a", "atom"].edge_index = _edges_to_tensor(
        [(0, i) for i in range(num_atoms)]
    )
    data["bond", "b2g", "global"].edge_index = _edges_to_tensor(
        [(i, 0) for i in range(num_bonds)]
    )
    data["global", "g2b", "bond"].edge_index = _edges_to_tensor(
        [(0, i) for i in range(num_bonds)]
    )

    if self_loop:
        data["atom", "a2a", "atom"].edge_index = _edges_to_tensor(
            [(i, i) for i in range(num_atoms)]
        )
        data["bond", "b2b", "bond"].edge_index = _edges_to_tensor(
            [(i, i) for i in range(num_bonds)]
        )
        data["global", "g2g", "global"].edge_index = _edges_to_tensor([(0, 0)])

    feats_size = {"atom": 2, "bond": 3, "global": 4}
    num_nodes_map = {"atom": num_atoms, "bond": num_bonds, "global": 1}
    feats = {}
    for ntype, size in feats_size.items():
        num_node = num_nodes_map[ntype]
        ft = torch.tensor(
            np.arange(num_node * size).reshape(num_node, size), dtype=torch.float32
        )
        data[ntype].feat = ft
        feats[ntype] = ft

    return data, feats


def make_hetero_graph():
    return make_hetero(
        num_atoms=4,
        num_bonds=3,
        a2b=[(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)],
        b2a=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)],
        self_loop=True,
    )


def make_test_model(
    atom_feat_size: int,
    bond_feat_size: int,
    global_feat_size: int,
    target_dict: dict = None,
    pooling_fn: str = "SumPoolingThenCat",
):
    """Build a small GCNGraphPred for testing.

    Shared helper used by edge-case, numerical, and logger tests.
    """
    if target_dict is None:
        target_dict = {"global": ["target_1"]}
    config = get_default_graph_level_config()["model"]
    config["atom_feature_size"] = atom_feat_size
    config["bond_feature_size"] = bond_feat_size
    config["global_feature_size"] = global_feat_size
    config["target_dict"] = target_dict
    config["n_conv_layers"] = 2
    config["hidden_size"] = 16
    config["embedding_size"] = 16
    config["fc_hidden_size_1"] = 16
    config["fc_num_layers"] = 1
    config["shape_fc"] = "flat"
    config["fc_batch_norm"] = False
    config["fc_dropout"] = 0.0
    config["dropout"] = 0.0
    config["batch_norm"] = False
    config["global_pooling_fn"] = pooling_fn
    config["conv_fn"] = "GraphConvDropoutBatch"
    config["initializer"] = None
    config["restore"] = False
    config["classifier"] = False
    config["compiled"] = False
    model = load_graph_level_model_from_config(config)
    model.eval()
    return model


def get_hyperparams_resid():
    parms = hyperparams(
        config={
            "atom_input_size": 10,
            "bond_input_size": 10,
            "global_input_size": 10,
            "n_conv_layers": 2,
            "in_feats": 10,
            "out_feats": 10,
            "norm": "both",
            "weight": True,
            "bias": True,
            "batch_norm_tf": False,
            "dropout": 0.0,
            "activation": None,
            "embedding_size": 10,
            "conv_fn": "ResidualBlock",
            "allow_zero_in_degree": True,
            "hidden_size": 10,
        },
    )
    return get_layer_args(
        hparams=parms,
        layer_ind=0,
        embedding_in=True,
    )


def get_hyperparams_gcn():
    parms = hyperparams(
        config={
            "atom_input_size": 10,
            "bond_input_size": 10,
            "global_input_size": 10,
            "n_conv_layers": 2,
            "in_feats": 10,
            "out_feats": 10,
            "norm": "both",
            "weight": True,
            "bias": True,
            "batch_norm_tf": False,
            "dropout": 0.0,
            "activation": None,
            "embedding_size": 10,
            "conv_fn": "GraphConvDropoutBatch",
            "allow_zero_in_degree": True,
            "hidden_size": 10,
        },
    )
    return get_layer_args(
        hparams=parms,
        layer_ind=0,
        embedding_in=True,
    )
