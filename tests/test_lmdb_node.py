import pytorch_lightning as pl
import torch

from qtaim_embed.core.datamodule import (
    QTAIMNodeTaskDataModule,
    LMDBMoleculeDataset,
    LMDBDataModule,
)

from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.utils.data import get_default_node_level_config
from qtaim_embed.data.lmdb import construct_lmdb_and_save_dataset
from qtaim_embed.models.utils import load_node_level_model_from_config


def test_write():
    config_w_test = get_default_node_level_config()

    dm = QTAIMNodeTaskDataModule(config=config_w_test)
    dm.setup("fit")

    train_dl_size = len(dm.train_dataset)
    val_dl_size = len(dm.val_dataset)
    test_dl_size = len(dm.test_dataset)

    construct_lmdb_and_save_dataset(dm.train_dataset, "./data/lmdb_node/train/")
    construct_lmdb_and_save_dataset(dm.val_dataset, "./data/lmdb_node/val/")
    construct_lmdb_and_save_dataset(dm.test_dataset, "./data/lmdb_node/test/")

    config = {
        "dataset": {
            "train_lmdb": "./data/lmdb_node/train/molecule.lmdb",
            "val_lmdb": "./data/lmdb_node/val/molecule.lmdb",
            "test_lmdb": "./data/lmdb_node/test/molecule.lmdb",
        }
    }

    train_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["train_lmdb"]})
    val_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["val_lmdb"]})
    test_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["test_lmdb"]})

    assert train_lmdb.__len__() == train_dl_size
    assert val_lmdb.__len__() == val_dl_size
    assert test_lmdb.__len__() == test_dl_size


def test_multi_out():
    config_w_test = get_default_node_level_config()
    config_w_test["dataset"]["verbose"] = False
    dm = QTAIMNodeTaskDataModule(config=config_w_test)
    dm.setup("fit")
    construct_lmdb_and_save_dataset(dm.train_dataset, "./data/lmdb_node/train/")

    config = {
        "dataset": {
            "train_lmdb": "./data/lmdb_node/train/",
            "val_lmdb": "./data/lmdb_node/val/",
            "test_lmdb": "./data/lmdb_node/test/",
        },
        "optim": {
            "train_batch_size": 1,
            "num_devices": 1,
            "num_nodes": 1,
            "gradient_clip_val": 5.0,
            "strategy": "auto",
            "precision": "bf16",
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": False,
            "accumulate_grad_batches": 3,
        },
    }

    # check that the folders have been created with the correct number of files
    dm_lmdb = LMDBDataModule(config=config)
    dm_lmdb.setup("fit")
    dl = dm_lmdb.train_dataloader()

    for batched_graphs, batched_labels in dl:
        assert batched_graphs["atom"].feat.shape[1] == 12
        assert batched_graphs["bond"].feat.shape[1] == 8
        assert batched_graphs["global"].feat.shape[1] == 3
        assert batched_labels["atom"].shape[1] == 1
        assert batched_labels["bond"].shape[1] == 3
        break


def test_model_lmdb():
    config = get_default_node_level_config()

    config["dataset"] = {
        "train_lmdb": "./data/lmdb_node/train/",
        "val_lmdb": "./data/lmdb_node/train/",
        "test_lmdb": "./data/lmdb_node/train/",
    }

    # check that the folders have been created with the correct number of files
    dm_lmdb = LMDBDataModule(config=config)
    dm_lmdb.setup("fit")
    feature_size = dm_lmdb.train_dataset.feature_size

    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]
    model = load_node_level_model_from_config(config["model"])
    dl = dm_lmdb.train_dataloader()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=16,
        log_every_n_steps=1,
    )

    trainer.fit(model, dl)
