import torch
import numpy as np
from pathlib import Path
import pandas as pd
import dgl

from qtaim_embed.utils.models import (
    get_layer_args,
)

from qtaim_embed.core.dataset import (
    HeteroGraphNodeLabelDataset,
    HeteroGraphGraphLabelDataset,
    HeteroGraphGraphLabelClassifierDataset,
)


class hyperparams:
    def __init__(self, config):
        self.config = config
        # set every key in config as an attribute
        for k, v in config.items():
            setattr(self, k, v)


def get_data():
    return pd.read_pickle("./data/qm8_test.pkl")


def get_data_spin_charge():
    return pd.read_pickle("./data/labelled_spin_charge.pkl")


def get_dataset(
    log_scale_features,
    log_scale_targets,
    standard_scale_features,
    standard_scale_targets,
):
    # get the root directory for this package
    # (i.e. the directory where setup.py is located)
    root_dir = Path(__file__).parent.parent
    dataset = HeteroGraphNodeLabelDataset(
        file="./data/labelled_data.pkl",
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
        log_scale_features=log_scale_features,
        log_scale_targets=log_scale_targets,
        standard_scale_features=standard_scale_features,
        standard_scale_targets=standard_scale_targets,
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
        file="./data/labelled_data.pkl",
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
        target_list=["extra_feat_global_E1_CAM"],
        extra_dataset_info={},
        debug=True,
        log_scale_features=log_scale_features,
        log_scale_targets=log_scale_targets,
        standard_scale_features=standard_scale_features,
        standard_scale_targets=standard_scale_targets,
    )

    return dataset


def get_datasets_graph_level_classifier(log_scale_features, standard_scale_features):
    root_dir = Path(__file__).parent.parent
    dataset_single = HeteroGraphGraphLabelClassifierDataset(
        file="./data/test_classifier_labelled.pkl",
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
        log_scale_features=log_scale_features,
        standard_scale_features=standard_scale_features,
    )
    dataset_multi = HeteroGraphGraphLabelClassifierDataset(
        file="./data/test_classifier_labelled.pkl",
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
    )
    return dataset_single, dataset_multi


def get_invalid_data():
    return pd.read_pickle("./data/qm8_invalid.pkl")


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

    edge_dict = {
        ("atom", "a2b", "bond"): a2b,
        ("bond", "b2a", "atom"): b2a,
        ("atom", "a2g", "global"): [(i, 0) for i in range(num_atoms)],
        ("global", "g2a", "atom"): [(0, i) for i in range(num_atoms)],
        ("bond", "b2g", "global"): [(i, 0) for i in range(num_bonds)],
        ("global", "g2b", "bond"): [(0, i) for i in range(num_bonds)],
    }

    if self_loop:
        a2a = [(i, i) for i in range(num_atoms)]
        b2b = [(i, i) for i in range(num_bonds)]
        g2g = [(0, 0)]
        edge_dict.update(
            {
                ("atom", "a2a", "atom"): a2a,
                ("bond", "b2b", "bond"): b2b,
                ("global", "g2g", "global"): g2g,
            }
        )
    g = dgl.heterograph(edge_dict)

    feats_size = {"atom": 2, "bond": 3, "global": 4}
    feats = {}
    for ntype, size in feats_size.items():
        num_node = g.number_of_nodes(ntype)
        ft = torch.tensor(
            np.arange(num_node * size).reshape(num_node, size), dtype=torch.float32
        )
        g.nodes[ntype].data.update({"feat": ft})
        feats[ntype] = ft

    return g, feats


def make_hetero_graph():
    return make_hetero(
        num_atoms=4,
        num_bonds=3,
        a2b=[(0, 0), (1, 0), (1, 1), (1, 2), (2, 1), (3, 2)],
        b2a=[(0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 3)],
        self_loop=True,
    )


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
        },
    )
    return get_layer_args(
        hparams=parms,
        layer_ind=-1,
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
        },
    )
    return get_layer_args(
        hparams=parms,
        layer_ind=0,
        embedding_in=True,
    )
