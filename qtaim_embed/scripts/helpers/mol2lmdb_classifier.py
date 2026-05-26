#!/usr/bin/env python3
"""
Convert a classifier pickle dataset to LMDB for fast training.

Usage:
    python mol2lmdb_classifier.py \
        -dataset_loc data/oact_classifier/actinides_classifier.pkl \
        -config qtaim_embed/scripts/helpers/settings_classifier_actinides.json \
        -lmdb_dir data/oact_classifier/actinides_lmdb \
        --split --num_workers 4
"""
import argparse
import gc
import json

import torch

from qtaim_embed.core.dataset import HeteroGraphGraphLabelClassifierDataset
from qtaim_embed.data.lmdb import construct_lmdb_and_save_dataset
from qtaim_embed.utils.data import train_validation_test_split

torch.set_float32_matmul_precision("high")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert classifier dataset to LMDB for instant training"
    )
    parser.add_argument(
        "-dataset_loc", type=str, help="Path to pickle file with classifier data"
    )
    parser.add_argument(
        "-config", type=str, help="Path to JSON config file"
    )
    parser.add_argument(
        "-lmdb_dir", type=str, help="Output LMDB directory"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Pre-split into train/val/test sets"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (small subset)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers for preprocessing"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    dataset = HeteroGraphGraphLabelClassifierDataset(
        file=args.dataset_loc,
        allowed_ring_size=config["dataset"]["allowed_ring_size"],
        allowed_charges=config["dataset"]["allowed_charges"],
        allowed_spins=config["dataset"]["allowed_spins"],
        self_loop=config["dataset"]["self_loop"],
        extra_keys=config["dataset"]["extra_keys"],
        element_set=config["dataset"]["element_set"],
        target_list=config["dataset"]["target_list"],
        extra_dataset_info=config["dataset"]["extra_dataset_info"],
        debug=bool(args.debug),
        log_scale_features=config["dataset"]["log_scale_features"],
        standard_scale_features=config["dataset"]["standard_scale_features"],
        bond_key=config["dataset"]["bond_key"],
        map_key=config["dataset"]["map_key"],
        num_workers=args.num_workers,
        rbf_cutoff=config["dataset"].get("rbf_cutoff", 5.0),
    )

    # Free MoleculeWrapper objects
    dataset.data = None
    gc.collect()

    if args.split:
        if config["dataset"]["test_prop"] > 0.0:
            train_dataset, val_dataset, test_dataset = train_validation_test_split(
                dataset,
                test=config["dataset"]["test_prop"],
                validation=config["dataset"]["val_prop"],
                random_seed=config["dataset"]["seed"],
            )
            print(f"Training set: {len(train_dataset)}")
            print(f"Validation set: {len(val_dataset)}")
            print(f"Test set: {len(test_dataset)}")

            construct_lmdb_and_save_dataset(dataset, args.lmdb_dir, num_workers=args.num_workers)
            construct_lmdb_and_save_dataset(val_dataset, args.lmdb_dir + "/val/", num_workers=args.num_workers)
            construct_lmdb_and_save_dataset(train_dataset, args.lmdb_dir + "/train/", num_workers=args.num_workers)
            construct_lmdb_and_save_dataset(test_dataset, args.lmdb_dir + "/test/", num_workers=args.num_workers)
        else:
            train_dataset, val_dataset = train_validation_test_split(
                dataset,
                test=0.0,
                validation=config["dataset"]["val_prop"],
                random_seed=config["dataset"]["seed"],
            )
            print(f"Training set: {len(train_dataset)}")
            print(f"Validation set: {len(val_dataset)}")

            construct_lmdb_and_save_dataset(dataset, args.lmdb_dir, num_workers=args.num_workers)
            construct_lmdb_and_save_dataset(val_dataset, args.lmdb_dir + "/val/", num_workers=args.num_workers)
            construct_lmdb_and_save_dataset(train_dataset, args.lmdb_dir + "/train/", num_workers=args.num_workers)
    else:
        construct_lmdb_and_save_dataset(dataset, args.lmdb_dir, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
