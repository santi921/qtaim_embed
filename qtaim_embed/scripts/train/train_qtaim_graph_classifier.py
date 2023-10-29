import wandb, argparse, torch, json
import numpy as np
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from qtaim_embed.core.datamodule import QTAIMGraphTaskClassifyDataModule
from qtaim_embed.models.utils import LogParameters, load_graph_level_model_from_config
from qtaim_embed.utils.data import get_default_graph_level_config_classif


torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_gpu", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("-project_name", type=str, default="qtaim_embed_test")
    parser.add_argument("-dataset_loc", type=str, default=None)
    parser.add_argument("-log_save_dir", type=str, default="./test_logs/")
    parser.add_argument("-config", type=str, default=None)

    args = parser.parse_args()

    on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    config = args.config

    if config is None:
        config = get_default_graph_level_config_classif()
    else:
        config = json.load(open(config, "r"))

    if config["optim"]["precision"] == "16" or config["optim"]["precision"] == "32":
        config["optim"]["precision"] = int(config["optim"]["precision"])

    # set log save dir
    config["dataset"]["log_save_dir"] = log_save_dir

    # dataset
    if dataset_loc is not None:
        config["dataset"]["train_dataset_loc"] = dataset_loc
    extra_keys = config["dataset"]["extra_keys"]

    if debug:
        config["dataset"]["debug"] = debug
    print(">" * 40 + "config_settings" + "<" * 40)

    # for k, v in config.items():
    #    print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    dm = QTAIMGraphTaskClassifyDataModule(config=config)

    feature_names, feature_size = dm.prepare_data(stage="fit")
    config["model"]["classifier"] = True
    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]
    config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]
    # config["dataset"]["feature_names"] = feature_names

    print(">" * 40 + "config_settings" + "<" * 40)
    for k, v in config.items():
        print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))

    print(">" * 40 + "config_settings" + "<" * 40)

    model = load_graph_level_model_from_config(config["model"])
    print("model constructed!")

    with wandb.init(project=project_name) as run:
        log_parameters = LogParameters()
        logger_tb = TensorBoardLogger(
            config["dataset"]["log_save_dir"], name="test_logs"
        )
        logger_wb = WandbLogger(project=project_name, name="test_logs")
        lr_monitor = LearningRateMonitor(logging_interval="step")

        checkpoint_callback = ModelCheckpoint(
            dirpath=config["dataset"]["log_save_dir"],
            filename="model_lightning_{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            auto_insert_metric_name=True,
            save_last=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=200, verbose=False, mode="min"
        )

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

        trainer.fit(model, dm)
        trainer.test(model, dm)
    run.finish()
