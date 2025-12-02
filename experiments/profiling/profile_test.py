import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from qtaim_embed.utils.data import get_default_node_level_config
from qtaim_embed.models.utils import load_node_level_model_from_config
import time
import pytorch_lightning as pl

from qtaim_embed.core.datamodule import (
    LMDBDataModule,
)

config = get_default_node_level_config()

config["dataset"] = {
    "train_lmdb": "/home/santiagovargas/dev/qtaim_embed/tests/data/lmdb_node/train/",
    "val_lmdb": "/home/santiagovargas/dev/qtaim_embed/tests/data/lmdb_node/train/",
    "test_lmdb": "/home/santiagovargas/dev/qtaim_embed/tests/data/lmdb_node/train/",
}

config["dataset"] = {
    "train_lmdb": "/home/santiagovargas/dev/qtaim_embed/qtaim_embed/scripts/train/node_lmdb_qm9/train/",
    "val_lmdb": "/home/santiagovargas/dev/qtaim_embed/qtaim_embed/scripts/train/node_lmdb_qm9/train/",
    "test_lmdb": "/home/santiagovargas/dev/qtaim_embed/qtaim_embed/scripts/train/node_lmdb_qm9/train/",
}

config["optim"]["train_batch_size"] = 256


# check that the folders have been created with the correct number of files
dm_lmdb = LMDBDataModule(config=config)
feat_name, feature_size = dm_lmdb.prepare_data()


config["model"]["conv_fn"] = "GraphConvDropoutBatch"

# config["model"]['compiled'] = True
config["model"]["atom_feature_size"] = feature_size["atom"]
config["model"]["bond_feature_size"] = feature_size["bond"]
config["model"]["global_feature_size"] = feature_size["global"]
model = load_node_level_model_from_config(config["model"])
dm_lmdb.setup("fit")
dl = dm_lmdb.train_dataloader()

trainer = pl.Trainer(
    max_epochs=2,
    accelerator="gpu",
    enable_progress_bar=True,
    devices=1,
    strategy="auto",
    enable_checkpointing=True,
    default_root_dir="./test_save_load/",
    precision="bf16",
)

start_time = time.time()
trainer.fit(model, dl)
print("Training time: ", time.time() - start_time)
