
import pytorch_lightning as pl
import torch

from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule, LMDBMoleculeDataset, LMDBDataModule
from qtaim_embed.utils.data import (
    get_default_node_level_config,
    get_default_graph_level_config,
)
from qtaim_embed.data.lmdb import construct_lmdb_and_save_dataset
from qtaim_embed.data.dataloader import DataLoaderLMDB
from qtaim_embed.models.utils import load_graph_level_model_from_config

def test_write():
    config_w_test = get_default_graph_level_config()
    
    dm = QTAIMGraphTaskDataModule(config= config_w_test)
    feature_size, feat_name = dm.prepare_data("fit")
    dm.setup("fit")

    train_dl_size = len(dm.train_dataset)
    val_dl_size = len(dm.val_dataset)
    test_dl_size = len(dm.test_dataset)
    
    construct_lmdb_and_save_dataset(dm.train_dataset, "./data/lmdb/train/")
    construct_lmdb_and_save_dataset(dm.val_dataset, "./data/lmdb/val/")
    construct_lmdb_and_save_dataset(dm.test_dataset, "./data/lmdb/test/")
    
    config = {
        "dataset":{
            "train_lmdb": "./data/lmdb/train/molecule.lmdb",
            "val_lmdb": "./data/lmdb/val/molecule.lmdb",
            "test_lmdb": "./data/lmdb/test/molecule.lmdb",
        }
    }
    
    train_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["train_lmdb"]})
    val_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["val_lmdb"]})
    test_lmdb = LMDBMoleculeDataset({"src": config["dataset"]["test_lmdb"]})
    
    assert train_lmdb.__len__() == train_dl_size
    assert val_lmdb.__len__() == val_dl_size
    assert test_lmdb.__len__() == test_dl_size


def test_multi_out():

    config_w_test = get_default_graph_level_config()

    config_w_test["dataset"]["verbose"] = False
    config_w_test["dataset"]["extra_keys"] = {
        "atom": ["extra_feat_atom_esp_total"],
        "bond": [
            "extra_feat_bond_esp_total",
            "bond_length",
        ],
        "global": ["extra_feat_global_E1_CAM", "extra_feat_global_E2_CAM"],
    }
    config_w_test["dataset"]["target_list"] = ["extra_feat_global_E1_CAM", "extra_feat_global_E2_CAM"]

    dm = QTAIMGraphTaskDataModule(config = config_w_test)
    feature_size, feat_name = dm.prepare_data("fit")
    dm.setup("fit")
    construct_lmdb_and_save_dataset(dm.train_dataset, "./data/lmdb/train/")

    
    config = {
        "dataset":{
            "train_lmdb": "./data/lmdb/train/",
            "val_lmdb": "./data/lmdb/val/",
            "test_lmdb": "./data/lmdb/test/",
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
            }
        }
    
    # check that the folders have been created with the correct number of files
    dm_lmdb = LMDBDataModule(config=config)
    
    dm_lmdb.prepare_data()
    dm_lmdb.setup("fit")
    dl = dm_lmdb.train_dataloader()
    
    for batched_graphs, batched_labels in dl:
        assert batched_labels["global"].reshape(-1).shape == torch.Size([2])
        break




def test_model_lmdb(): 
    config = get_default_graph_level_config()

    config["dataset"] = {
        "train_lmdb": "./data/lmdb/train/",
        "val_lmdb": "./data/lmdb/train/",
        "test_lmdb": "./data/lmdb/train/",
        "target_dict": {
            "global":["extra_feat_global_E1_CAM", "extra_feat_global_E2_CAM"]
            }
    }

    # check that the folders have been created with the correct number of files
    dm_lmdb = LMDBDataModule(config=config)
    feat_name, feature_size = dm_lmdb.prepare_data()
    print(feature_size)
    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]
    config["model"]["target_dict"] = config["dataset"]["target_dict"]
    print(config["model"])
    model = load_graph_level_model_from_config(config["model"])
    dm_lmdb.setup("fit")
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

test_model_lmdb()
