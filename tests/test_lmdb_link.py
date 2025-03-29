import pytorch_lightning as pl
import torch

from qtaim_embed.core.datamodule import QTAIMLinkTaskDataModule, LMDBLinkDataModule

from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.utils.data import get_default_link_level_config
from qtaim_embed.data.lmdb import construct_lmdb_and_save_dataset
from qtaim_embed.models.utils import load_link_model_from_config


def test_write():
    config_w_test = get_default_link_level_config()

    dm = QTAIMLinkTaskDataModule(config=config_w_test)
    feature_size, feat_name = dm.prepare_data("fit")

    dm.setup("fit")

    train_dl_size = len(dm.train_dataset)
    val_dl_size = len(dm.val_dataset)
    test_dl_size = len(dm.test_dataset)

    construct_lmdb_and_save_dataset(dm.train_dataset, "./data/lmdb_link/train/")
    construct_lmdb_and_save_dataset(dm.val_dataset, "./data/lmdb_link/val/")
    construct_lmdb_and_save_dataset(dm.test_dataset, "./data/lmdb_link/test/")

    config = {
        "dataset": {
            "train_lmdb": "./data/lmdb_link/train/molecule.lmdb",
            "val_lmdb": "./data/lmdb_link/val/molecule.lmdb",
            "test_lmdb": "./data/lmdb_link/test/molecule.lmdb",
        }
    }

    train_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["train_lmdb"]})
    val_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["val_lmdb"]})
    test_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["test_lmdb"]})

    assert train_lmdb.__len__() == train_dl_size
    assert val_lmdb.__len__() == val_dl_size
    assert test_lmdb.__len__() == test_dl_size


def test_multi_out():
    config_def = get_default_link_level_config()

    config_def["dataset"]["verbose"] = False
    dm = QTAIMLinkTaskDataModule(config=config_def)
    feature_size, feat_name = dm.prepare_data()
    construct_lmdb_and_save_dataset(dm.train_dataset, "./data/lmdb_link/train/")

    config_w_test = {
        "dataset": {
            "train_lmdb": "./data/lmdb_link/train/",
            "val_lmdb": "./data/lmdb_link/val/",
            "test_lmdb": "./data/lmdb_link/test/",
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
    config_w_test["model"] = config_def["model"]

    # check that the folders have been created with the correct number of files
    dm_lmdb = LMDBLinkDataModule(config=config_w_test)
    dm_lmdb.prepare_data()
    dm_lmdb.setup("fit")
    config_w_test["model"]["input_size"] = dm.node_len

    dl = dm_lmdb.train_dataloader()

    for batched_graphs, batched_neg_graphs, feats in dl:
        assert feats.shape[1] == config_w_test["model"]["input_size"]
        break


def test_model_lmdb():
    config = get_default_link_level_config()

    config["dataset"] = {
        "train_lmdb": "./data/lmdb_link/train/",
        "val_lmdb": "./data/lmdb_link/train/",
        "test_lmdb": "./data/lmdb_link/train/",
    }

    # check that the folders have been created with the correct number of files
    dm_lmdb = LMDBLinkDataModule(config=config)
    feat_name, feature_size = dm_lmdb.prepare_data()
    dm_lmdb.setup("fit")
    #print(config["model"])
    config["model"]["input_size"] = dm_lmdb.node_len
    model = load_link_model_from_config(config["model"]) # error here

    dl = dm_lmdb.train_dataloader()

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        enable_progress_bar=True,
        devices=1,
        strategy="auto",
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=16,
    )

    trainer.fit(model, dl)


# test_write()
# test_multi_out()
# test_model_lmdb()
