from pathlib import Path
import numpy as np
from qtaim_embed.core.dataset import Subset


def get_default_node_level_config():
    root = Path(__file__).parent.parent.parent
    # to string
    root = str(root)
    return {
        "dataset": {
            "allowed_ring_size": [3, 4, 5, 6, 7],
            "allowed_charges": None,
            "self_loop": True,
            "extra_keys": {
                "atom": ["extra_feat_atom_esp_total"],
                "bond": [
                    "extra_feat_bond_esp_total",
                    "extra_feat_bond_ellip_e_dens",
                    "extra_feat_bond_eta",
                    "bond_length",
                ],
            },
            "target_dict": {
                "atom": ["extra_feat_atom_esp_total"],
                "bond": [
                    "extra_feat_bond_esp_total",
                    "extra_feat_bond_ellip_e_dens",
                    "extra_feat_bond_eta",
                ],
            },
            "extra_dataset_info": {},
            "debug": False,
            "log_scale_features": False,
            "log_scale_targets": False,
            "standard_scale_features": True,
            "standard_scale_targets": True,
            "val_prop": 0.15,
            "test_prop": 0.1,
            "seed": 42,
            "train_batch_size": 128,
            "test_dataset_loc": None,
            "train_dataset_loc": root + "/tests/data/labelled_data.pkl",
        },
        "model": {},
    }


def get_default_graph_level_config():
    root = Path(__file__).parent.parent.parent
    # to string
    root = str(root)

    return {
        "dataset": {
            "allowed_ring_size": [3, 4, 5, 6, 7],
            "allowed_charges": None,
            "self_loop": True,
            "extra_keys": {
                "atom": ["extra_feat_atom_esp_total"],
                "bond": [
                    "extra_feat_bond_esp_total",
                    "bond_length",
                ],
                "global": ["extra_feat_global_E1_CAM"],
            },
            "target_list": ["extra_feat_global_E1_CAM"],
            "extra_dataset_info": {},
            "debug": False,
            "log_scale_features": False,
            "log_scale_targets": False,
            "standard_scale_features": True,
            "standard_scale_targets": True,
            "val_prop": 0.15,
            "test_prop": 0.1,
            "seed": 42,
            "train_batch_size": 128,
            "test_dataset_loc": None,
            "train_dataset_loc": root + "/tests/data/labelled_data.pkl",
            "num_workers": 1,
        },
        "model": {
            "classifier": False,
            "n_conv_layers": 8,
            "resid_n_graph_convs": 2,
            "target_dict": {"global": "extra_feat_global_E1_CAM"},
            "conv_fn": "ResidualBlock",
            "global_pooling_fn": "SumPoolingThenCat",
            "dropout": 0.2,
            "batch_norm": False,
            "activation": "ReLU",
            "bias": True,
            "norm": "both",
            "aggregate": "sum",
            "lr": 0.01,
            "scheduler_name": "reduce_on_plateau",
            "weight_decay": 0.00001,
            "lr_plateau_patience": 25,
            "lr_scale_factor": 0.6,
            "loss_fn": "mse",
            "embedding_size": 20,
            # "fc_layer_size": [256, 128],
            "shape_fc": "cone",
            "fc_hidden_size_1": 256,
            "fc_num_layers": 3,
            "fc_dropout": 0.2,
            "fc_batch_norm": True,
            "lstm_iters": 3,
            "lstm_layers": 2,
            "output_dims": 1,
            "pooling_ntypes": ["atom", "bond", "global"],
            "pooling_ntypes_direct": ["global"],
            "restore": False,
            "max_epochs": 1000,
        },
        "optim": {
            "num_devices": 1,
            "num_nodes": 1,
            "gradient_clip_val": 5.0,
            "strategy": "auto",
            "precision": "bf16",
            "accumulate_grad_batches": 3,
        },
    }


def train_validation_test_split(dataset, validation=0.1, test=0.1, random_seed=None):
    """
    Split a dataset into training, validation, and test set.

    The training set will be automatically determined based on `validation` and `test`,
    i.e. train = 1 - validation - test.

    Args:
        dataset: the dataset
        validation (float, optional): The amount of data (fraction) to be assigned to
            validation set. Defaults to 0.1.
        test (float, optional): The amount of data (fraction) to be assigned to test
            set. Defaults to 0.1.
        random_seed (int, optional): random seed that determines the permutation of the
            dataset. Defaults to 35.

    Returns:
        [train set, validation set, test_set]
    """
    assert validation + test < 1.0, "validation + test >= 1"
    size = len(dataset)
    num_val = int(size * validation)
    num_test = int(size * test)
    num_train = size - num_val - num_test

    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.random.permutation(size)
    train_idx = idx[:num_train]
    val_idx = idx[num_train : num_train + num_val]
    test_idx = idx[num_train + num_val :]
    return [
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    ]
