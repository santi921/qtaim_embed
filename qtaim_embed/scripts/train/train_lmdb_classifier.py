#!/usr/bin/env python3
"""
Train a graph-level binary classifier from pre-built LMDB datasets.

Usage:
    python train_lmdb_classifier.py \
        -config qtaim_embed/scripts/helpers/settings_classifier_actinides.json \
        -train_lmdb data/oact_classifier/actinides_lmdb/train/ \
        -val_lmdb data/oact_classifier/actinides_lmdb/val/ \
        -test_lmdb data/oact_classifier/actinides_lmdb/test/ \
        -project_name oact_actinides_classifier \
        -log_save_dir ./logs_actinides/
"""

import argparse
import json
import logging

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from qtaim_embed.core.datamodule import LMDBDataModule
from qtaim_embed.data.lmdb import TransformMol
from qtaim_embed.models.utils import LogParameters, load_graph_level_model_from_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train graph-level classifier from LMDB"
    )
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-train_lmdb", type=str, required=True)
    parser.add_argument("-val_lmdb", type=str, required=True)
    parser.add_argument("-test_lmdb", type=str, default=None)
    parser.add_argument("-project_name", type=str, default="oact_classifier")
    parser.add_argument("-log_save_dir", type=str, default="./logs_classifier/")
    parser.add_argument("--on_gpu", default=False, action="store_true")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if config["optim"]["precision"] == "16":
        config["optim"]["precision"] = "16-mixed"
    elif config["optim"]["precision"] == "32":
        config["optim"]["precision"] = int(config["optim"]["precision"])

    # Set LMDB paths
    config["dataset"]["train_lmdb"] = args.train_lmdb
    config["dataset"]["val_lmdb"] = args.val_lmdb
    if args.test_lmdb:
        config["dataset"]["test_lmdb"] = args.test_lmdb
    config["dataset"]["log_save_dir"] = args.log_save_dir

    # Create LMDB data module and get feature sizes from a sample
    dm = LMDBDataModule(config=config)
    dm.setup(stage="fit")

    # Get feature sizes from a sample graph
    sample = dm.train_dataset[0]
    atom_feat_size = sample["atom"].feat.shape[1]
    bond_feat_size = sample["bond"].feat.shape[1]
    global_feat_size = sample["global"].feat.shape[1]

    logger.info(
        "Feature sizes: atom=%d, bond=%d, global=%d",
        atom_feat_size, bond_feat_size, global_feat_size,
    )
    logger.info("Train set: %d, Val set: %d", len(dm.train_dataset), len(dm.val_dataset))
    if hasattr(dm, "test_dataset"):
        logger.info("Test set: %d", len(dm.test_dataset))

    # Configure model
    config["model"]["classifier"] = True
    config["model"]["atom_feature_size"] = atom_feat_size
    config["model"]["bond_feature_size"] = bond_feat_size
    config["model"]["global_feature_size"] = global_feat_size
    config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]

    logger.info("Config:")
    for k, v in config.items():
        logger.info("  %s: %s", k, v)

    model = load_graph_level_model_from_config(config["model"])
    logger.info("Model constructed")

    with wandb.init(project=args.project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(args.log_save_dir, name="tb_logs")
        logger_wb = WandbLogger(
            project=args.project_name, name="classifier", entity="santi"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            dirpath=args.log_save_dir,
            filename="model_{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}",
            monitor="val_f1",
            mode="max",
            auto_insert_metric_name=True,
            save_last=True,
            save_top_k=3,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_f1",
            min_delta=0.001,
            patience=100,
            verbose=True,
            mode="max",
        )

        trainer = pl.Trainer(
            max_epochs=config["model"]["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=config["optim"]["num_devices"],
            gradient_clip_val=config["optim"]["gradient_clip_val"],
            accumulate_grad_batches=config["optim"]["accumulate_grad_batches"],
            enable_progress_bar=True,
            callbacks=[
                early_stopping_callback,
                lr_monitor,
                log_parameters,
                checkpoint_callback,
            ],
            enable_checkpointing=True,
            strategy=(
                DDPStrategy(find_unused_parameters=True)
                if config["optim"]["strategy"] == "ddp"
                else config["optim"]["strategy"]
            ),
            default_root_dir=args.log_save_dir,
            logger=[logger_tb, logger_wb],
            precision=config["optim"]["precision"],
        )

        trainer.fit(model, dm)

        if args.test_lmdb:
            trainer.test(model, dm)

    run.finish()


if __name__ == "__main__":
    main()
