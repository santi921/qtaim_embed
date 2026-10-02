#!/usr/bin/env python3
"""Train the T3 bond classifier (GCNBondPred) from LMDB heterographs.

Example:
    qtaim-embed-train-bond -config settings_bond.json --log_save_dir ./bond_run/
"""

import argparse
import json
import logging
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from qtaim_embed.core.datamodule import LMDBBondDataModule
from qtaim_embed.models.utils import LogParameters, load_bond_model_from_config
from qtaim_embed.utils.data import get_default_bond_level_config
from qtaim_embed.utils.training import build_trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def bond_callbacks(config: dict) -> list:
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["dataset"]["log_save_dir"],
        filename="bond_{epoch:03d}-{val_macro_f1:.4f}",
        monitor="val_macro_f1",
        mode="max",
        auto_insert_metric_name=True,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=config["model"]["extra_stop_patience"], mode="min"
    )
    return [early_stopping, LearningRateMonitor(logging_interval="step"), LogParameters(), checkpoint_callback]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default=None)
    parser.add_argument("--log_save_dir", type=str, default="./bond_logs/")
    parser.add_argument("--project_name", type=str, default="qtaim_embed_bond")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default="auto")
    args = parser.parse_args(argv)

    if args.config is None:
        logger.info("Using default bond config")
        config = get_default_bond_level_config()
    else:
        with open(args.config, "r") as f:
            config = json.load(f)

    config["dataset"]["log_save_dir"] = args.log_save_dir
    if args.num_workers is not None:
        config["optim"]["num_workers"] = args.num_workers
    if args.max_epochs is not None:
        config["model"]["max_epochs"] = args.max_epochs

    for k, v in config.items():
        logger.info("%s\t%s", str(k).ljust(10), v)

    dm = LMDBBondDataModule(config=config)
    dm.setup(stage="fit")
    if config["model"].get("use_atom_feat", False):
        config["model"]["atom_input_size"] = dm.train_dataset.feature_size["atom"]
        logger.warning(
            "use_atom_feat=True: atom features must not derive from the bond list "
            "(see docs/plans/2026-09-08-feat-t3-bond-classifier-plan.md, leakage rules)"
        )
    model = load_bond_model_from_config(config["model"])

    loggers = [TensorBoardLogger(config["dataset"]["log_save_dir"], name="bond_logs")]
    run = None
    if not args.no_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        run = wandb.init(project=args.project_name, entity=args.wandb_entity)
        loggers.append(WandbLogger(project=args.project_name, name="bond", entity=args.wandb_entity))
        run.config.update(config["dataset"], allow_val_change=True)
        run.config.update(config["model"], allow_val_change=True)
        run.config.update(config["optim"], allow_val_change=True)

    trainer = build_trainer(config, loggers, bond_callbacks(config), accelerator=args.accelerator)
    trainer.fit(model, dm)
    if config["dataset"].get("test_lmdb") is not None:
        trainer.test(model, dm)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
