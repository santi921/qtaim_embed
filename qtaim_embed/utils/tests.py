import pandas as pd
from qtaim_embed.core.dataset import HeteroGraphNodeLabelDataset


def get_data():
    return pd.read_pickle("./data/qm8_test.pkl")


def get_dataset(
    log_scale_features,
    log_scale_targets,
    standard_scale_features,
    standard_scale_targets,
):
    dataset = HeteroGraphNodeLabelDataset(
        file="./data/labelled_data.pkl",
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
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


def get_invalid_data():
    return pd.read_pickle("./data/qm8_invalid.pkl")
