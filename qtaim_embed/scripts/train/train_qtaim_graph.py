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
from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule, LMDBDataModule
from qtaim_embed.models.utils import LogParameters, load_graph_level_model_from_config
from qtaim_embed.utils.data import get_default_graph_level_config


torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_gpu", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("-project_name", type=str, default="qtaim_embed_test")
    parser.add_argument("-dataset_loc", type=str, default=None)
    parser.add_argument("-dataset_test_loc", type=str, default=None)
    parser.add_argument("-log_save_dir", type=str, default="./test_logs/")
    parser.add_argument("-config", type=str, default=None)
    parser.add_argument("-wandb_entity", type=str, default="santi")
    parser.add_argument("--use_lmdb", default=False, action="store_true")


    args = parser.parse_args()

    on_gpu = bool(args.on_gpu)
    debug = bool(args.debug)
    use_lmdb = bool(args.use_lmdb)
    project_name = args.project_name
    dataset_loc = args.dataset_loc
    dataset_test_loc = args.dataset_test_loc
    log_save_dir = args.log_save_dir
    wandb_entity = args.wandb_entity
    config = args.config

    if config is None:
        config = get_default_graph_level_config()
    else:
        config = json.load(open(config, "r"))


    # set log save dir
    config["dataset"]["log_save_dir"] = log_save_dir

    print(">" * 40 + "config_settings" + "<" * 40)

    # for k, v in config.items():
    #    print("{}\t\t\t{}".format(str(k).ljust(20), str(v).ljust(20)))
    if use_lmdb:
        print("using lmdbs!")
        dm = LMDBDataModule(config=config)
        config["model"]["target_dict"]["global"] = {"global": ["value"]}

    else:
        # dataset
        if dataset_loc is not None:
            config["dataset"]["train_dataset_loc"] = dataset_loc
        extra_keys = config["dataset"]["extra_keys"]

        if debug:
            config["dataset"]["debug"] = debug

        if config["optim"]["precision"] == "16" or config["optim"]["precision"] == "32":
            config["optim"]["precision"] = int(config["optim"]["precision"])

        dm = QTAIMGraphTaskDataModule(config=config)
        config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]

        
        if dataset_test_loc is not None:
            test_config = deepcopy(config)
            test_config["dataset"]["test_dataset_loc"] = dataset_test_loc
            dm_test = QTAIMGraphTaskDataModule(
                config=test_config, 
            )
            dm_test.prepare_data(stage="test")

    feature_names, feature_size = dm.prepare_data(stage="fit")  
    print(feature_names, feature_size)  
    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]
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
        logger_wb = WandbLogger(project=project_name, name="test_logs", entity=wandb_entity)
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
            monitor="val_loss", min_delta=0.00, patience=config["model"]["extra_stop_patience"], verbose=False, mode="min"
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

        # log dataset and optim settings from config 
        run.config.update(config["dataset"])
        run.config.update(config["optim"])

        trainer.fit(model, dm)
        
        if use_lmdb:
            if "test_lmdb" in config["dataset"]:
                trainer.test(model, dm)
        
        else:
            if config["dataset"]["test_prop"] > 0.0:
                trainer.test(model, dm)
        
        if dataset_test_loc is not None:
            
            batch_graph, batch_labels = next(iter(dm_test.test_dataloader()))
            scalers = dm.full_dataset.label_scalers
            
            if config["dataset"]["per_atom"] == True:
        
                (
                    mean_mae_test,
                    mean_rmse_test,
                    ewt_prop_test,
                    preds_unscaled, 
                    labels_unscaled
                ) = model.evaluate_manually(
                    batch_graph=batch_graph,
                    batch_label=batch_labels,
                    scaler_list=scalers,
                    per_atom=True,
                )
                # make a table of the results
                print(">" * 40 + "test_results" + "<" * 40)
                print("mean_mae_test: ", mean_mae_test.numpy())
                print("mean_rmse_test: ", mean_rmse_test.numpy())
                print("ewt_prop_test: ", ewt_prop_test.numpy())
                # save results to pkl
                results = {
                    "mean_mae_test": mean_mae_test.numpy(),
                    "mean_rmse_test": mean_rmse_test.numpy(),
                    "ewt_prop_test": ewt_prop_test.numpy(),
                    "preds_unscaled": preds_unscaled.numpy(),
                    "labels_unscaled": labels_unscaled.numpy(),
                }
            else:
                r2_val, mae_val, mse_val, preds_unscaled, labels_unscaled = model.evaluate_manually(
                    batch_graph, batch_labels, scalers, per_atom=False
                )
                # make a table of the results
                print(">" * 40 + "test_results" + "<" * 40)
                print("r2_test: ", r2_val.numpy())
                print("mae_test: ", mae_val.numpy())
                print("mse_test: ", mse_val.numpy())
                # save results to pkl 
                results = {
                    "r2_val": r2_val.numpy(),
                    "mae_val": mae_val.numpy(),
                    "mse_val": mse_val.numpy(),
                    "preds_unscaled": preds_unscaled.numpy(),
                    "labels_unscaled": labels_unscaled.numpy(),
                }
            pd.to_pickle(results, config["dataset"]["log_save_dir"] + "test_results.pkl")


    run.finish()
