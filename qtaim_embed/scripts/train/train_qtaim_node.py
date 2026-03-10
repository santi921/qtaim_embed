#!/usr/bin/env python3

import wandb, argparse, torch, json
import numpy as np
from copy import deepcopy
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)


from qtaim_embed.utils.data import get_default_node_level_config
from qtaim_embed.core.datamodule import QTAIMNodeTaskDataModule, LMDBDataModule
from qtaim_embed.models.utils import LogParameters, load_node_level_model_from_config

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")


def main(argv=None):
    # print("here")
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, default=None)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_lmdb", default=False, action="store_true")
    parser.add_argument("--log_save_dir", type=str, default="./test_logs/")
    parser.add_argument("--wandb_entity", type=str, default="santi")
    parser.add_argument("--project_name", type=str, default="qtaim_embed_test")
    parser.add_argument("--dataset_loc", type=str, default=None)
    parser.add_argument("--dataset_test_loc", type=str, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of parallel workers for dataset preprocessing (default: 1)",
    )

    args = parser.parse_args()

    debug = bool(args.debug)
    use_lmdb = bool(args.use_lmdb)
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    dataset_test_loc = args.dataset_test_loc
    log_save_dir = args.log_save_dir
    wandb_entity = args.wandb_entity
    config = args.config

    # print options
    print("debug: ", debug)
    print("use_lmdb: ", use_lmdb)
    print("project_name: ", project_name)
    print("dataset_loc: ", dataset_loc)
    print("dataset_test_loc: ", dataset_test_loc)
    print("log_save_dir: ", log_save_dir)
    print("wandb_entity: ", wandb_entity)
    print("config: ", config)

    if config is None:
        print("...using default config!")
        config = get_default_node_level_config()
    else:
        with open(config, "r") as f:
            config = json.load(f)

    # set log save dir
    config["dataset"]["log_save_dir"] = log_save_dir
    # set num_workers from CLI (overrides config file)
    config["dataset"]["num_workers"] = args.num_workers

    print(">" * 40 + "config_settings" + "<" * 40)

    # for k, v in config.items():
    #    print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    if "target_dict" not in config["dataset"]:
        config["model"]["target_dict"] = config["dataset"]["target_dict"]
    if "target_dict" not in config["model"]:
        config["model"]["target_dict"] = config["dataset"]["target_dict"]

    if use_lmdb:
        print("...using lmdbs!")
        dm = LMDBDataModule(config=config)

    else:
        assert dataset_loc is not None, "dataset_loc must be provided if not using lmdb"
        # dataset
        config["dataset"]["train_dataset_loc"] = dataset_loc
        extra_keys = config["dataset"]["extra_keys"]

        if debug:
            config["dataset"]["debug"] = debug

        if config["optim"]["precision"] == "16" or config["optim"]["precision"] == "32":
            config["optim"]["precision"] = int(config["optim"]["precision"])

        dm = QTAIMNodeTaskDataModule(config=config)
        # config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]

        if dataset_test_loc is not None:
            test_config = deepcopy(config)
            test_config["dataset"]["test_dataset_loc"] = dataset_test_loc
            dm_test = QTAIMNodeTaskDataModule(
                config=test_config,
            )
            dm_test.setup(stage="test")

    # setup() runs on all ranks (DDP-safe); prepare_data() is a no-op
    dm.setup(stage="fit")
    feature_names = dm.train_dataset.feature_names
    feature_size = dm.train_dataset.feature_size
    print(feature_names, feature_size)
    print("feature size dict: ", feature_size)
    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))

    print(">" * 40 + "config_settings" + "<" * 40)
    model = load_node_level_model_from_config(config["model"])
    print("model constructed!")

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(
            config["dataset"]["log_save_dir"], name="test_logs"
        )
        logger_wb = WandbLogger(
            project=project_name, name="test_logs", entity=wandb_entity
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            dirpath=config["dataset"]["log_save_dir"],
            filename="model_lightning_{epoch:03d}-{val_loss:.4f}",
            monitor="val_mae",
            mode="min",
            auto_insert_metric_name=True,
            save_last=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=config["model"]["extra_stop_patience"],
            verbose=False,
            mode="min",
        )

        dm.setup(stage="fit")
        val_dl = dm.train_dataloader()
        _, _ = next(iter(val_dl))

        # DDP Strategy Note:
        # This project requires strategy="ddp" (not "ddp_spawn") for multi-GPU
        # training. LMDB datasets use lazy per-worker env init that is compatible
        # with fork-based DDP but not spawn-based ddp_spawn (LMDB environments
        # are not picklable). For PCIe GPUs without NVLink (e.g. A5000), set
        # NCCL_P2P_DISABLE=1 before launching.
        trainer = pl.Trainer(
            max_epochs=config["model"]["max_epochs"],
            accelerator="gpu",
            devices=config["optim"]["num_devices"],
            num_nodes=config["optim"]["num_nodes"],
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
            strategy=config["optim"]["strategy"],
            default_root_dir=config["dataset"]["log_save_dir"],
            logger=[logger_tb, logger_wb],
            precision=config["optim"]["precision"],
        )

        # log dataset and optim settings from config
        run.config.update(config["dataset"])
        run.config.update(config["optim"])

        print("dataset and optim settings logged!")
        print("fitting model!")
        trainer.fit(model, dm)

        print("model fitted, testing!")
        if use_lmdb:
            if "test_lmdb" in config["dataset"]:
                trainer.test(model, dm)

        else:
            if config["dataset"]["test_prop"] > 0.0:
                trainer.test(model, dm)

    run.finish()
