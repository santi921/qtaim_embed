import argparse
import torch
import json
import numpy as np
import argparse
import dgl
import torch
import tempfile
from copy import deepcopy

from qtaim_embed.data.lmdb import construct_lmdb_and_save_dataset
from qtaim_embed.utils.data import train_validation_test_split
from qtaim_embed.core.dataset import HeteroGraphNodeLabelDataset


torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
# seed_torch()
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="convert dataset to lmdb for instant training"
    )
    parser.add_argument(
        "-dataset_loc",
        type=str,
        help="location of json file containing dataset",
    )

    parser.add_argument(
        "-config",
        type=str,
        help="location of json file containing config",
    )

    parser.add_argument(
        "-lmdb_dir",
        type=str,
        help="location of lmdb directory",
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="whether or not to presplit the dataset into train, val, and test sets",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="debug mode",
    )

    args = parser.parse_args()
    config_loc = args.config
    dataset_loc = args.dataset_loc
    lmdb_dir = args.lmdb_dir

    debug = bool(args.debug)
    split = bool(args.split)

    # read json
    with open(config_loc) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HeteroGraphNodeLabelDataset(
        file=dataset_loc,
        allowed_ring_size=config["dataset"]["allowed_ring_size"],
        allowed_charges=config["dataset"]["allowed_charges"],
        allowed_spins=config["dataset"]["allowed_spins"],
        self_loop=config["dataset"]["self_loop"],
        extra_keys=config["dataset"]["extra_keys"],
        element_set=config["dataset"]["element_set"],
        target_dict=config["dataset"]["target_dict"],
        extra_dataset_info=config["dataset"]["extra_dataset_info"],
        debug=debug,
        log_scale_features=config["dataset"]["log_scale_features"],
        log_scale_targets=config["dataset"]["log_scale_targets"],
        standard_scale_features=config["dataset"]["standard_scale_features"],
        standard_scale_targets=config["dataset"]["standard_scale_targets"],
        bond_key=config["dataset"]["bond_key"],
        map_key=config["dataset"]["map_key"],
        
    )

    if split == True:
        if config["dataset"]["test_prop"] > 0.0:
            train_dataset, val_dataset, test_dataset = train_validation_test_split(
                dataset,
                test=config["dataset"]["test_prop"],
                validation=config["dataset"]["val_prop"],
                random_seed=config["dataset"]["seed"],
            )

            print("training set size: ", len(train_dataset))
            print("validation set size: ", len(val_dataset))
            print("test set size: ", len(test_dataset))
            print(dataset.feature_names)
            construct_lmdb_and_save_dataset(dataset, lmdb_dir)
            construct_lmdb_and_save_dataset(val_dataset, lmdb_dir + "/val/")
            construct_lmdb_and_save_dataset(train_dataset, lmdb_dir + "/train/")
            construct_lmdb_and_save_dataset(test_dataset, lmdb_dir + "/test/")

        else:

            train_dataset, val_dataset = train_validation_test_split(
                dataset,
                test=0.0,
                validation=config["dataset"]["val_prop"],
                random_seed=config["dataset"]["seed"],
            )
            print("training set size: ", len(train_dataset))
            print("validation set size: ", len(val_dataset))

            construct_lmdb_and_save_dataset(dataset, lmdb_dir)
            construct_lmdb_and_save_dataset(val_dataset, lmdb_dir + "/val/")
            construct_lmdb_and_save_dataset(train_dataset, lmdb_dir + "/train/")

    else:
        construct_lmdb_and_save_dataset(dataset, lmdb_dir)
