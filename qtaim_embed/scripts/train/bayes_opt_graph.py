import wandb, argparse, torch, json
import numpy as np
from copy import deepcopy

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
from qtaim_embed.core.datamodule import QTAIMGraphTaskDataModule
from qtaim_embed.models.utils import load_graph_level_model_from_config

# from qtaim_embed.utils.data import get_default_graph_level_config

torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")


class TrainingObject:
    def __init__(
        self,
        sweep_config,
        log_save_dir,
        project_name,
        dataset_loc,
    ):
        self.sweep_config = sweep_config
        self.log_save_dir = log_save_dir
        self.wandb_name = project_name
        self.wandb_entity = "santi"
        self.dataset_loc = dataset_loc
        self.extra_keys = self.sweep_config["parameters"]["extra_keys"]["values"][0]

        print("extra keys: ", self.extra_keys)
        print("debug value: ", self.sweep_config["parameters"]["debug"]["values"])

        dm_config = {
            "dataset": {
                "allowed_ring_size": self.sweep_config["parameters"][
                    "allowed_ring_size"
                ]["values"][0],
                "allowed_charges": self.sweep_config["parameters"]["allowed_charges"][
                    "values"
                ][0],
                "allowed_spins": self.sweep_config["parameters"]["allowed_spins"][
                    "values"
                ][0],
                "self_loop": self.sweep_config["parameters"]["self_loop"]["values"][0],
                "extra_keys": self.extra_keys,
                "target_list": self.sweep_config["parameters"]["target_list"]["values"][
                    0
                ],
                "extra_dataset_info": self.sweep_config["parameters"][
                    "extra_dataset_info"
                ]["values"][0],
                "debug": self.sweep_config["parameters"]["debug"]["values"][0],
                "log_scale_features": self.sweep_config["parameters"][
                    "log_scale_features"
                ]["values"][0],
                "log_scale_targets": self.sweep_config["parameters"][
                    "log_scale_targets"
                ]["values"][0],
                "standard_scale_features": self.sweep_config["parameters"][
                    "standard_scale_features"
                ]["values"][0],
                "standard_scale_targets": self.sweep_config["parameters"][
                    "standard_scale_targets"
                ]["values"][0],
                "val_prop": self.sweep_config["parameters"]["val_prop"]["values"][0],
                "test_prop": self.sweep_config["parameters"]["test_prop"]["values"][0],
                "seed": self.sweep_config["parameters"]["seed"]["values"][0],
                "train_batch_size": self.sweep_config["parameters"]["train_batch_size"][
                    "values"
                ][0],
                "train_dataset_loc": self.dataset_loc,
                "num_workers": self.sweep_config["parameters"]["num_workers"]["values"][
                    0
                ],
            },
        }
        print("config settings:")
        for k, v in dm_config.items():
            print("--> Level - {}".format(k))
            for kk, vv in v.items():
                print("{}\t\t{}".format(str(kk).ljust(20), str(vv).ljust(20)))
        self.dm = QTAIMGraphTaskDataModule(config=dm_config)
        feature_names, feature_size = self.dm.prepare_data(stage="fit")
        self.feature_names = feature_names
        self.feature_size = feature_size

    def make_model(self, config):
        # config["model"]["in_feats"] = self.in_feats
        config["model"]["atom_feature_size"] = self.feature_size["atom"]
        config["model"]["bond_feature_size"] = self.feature_size["bond"]
        config["model"]["global_feature_size"] = self.feature_size["global"]
        config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]
        print("config settings:")
        print("atom_feature_size: {}".format(config["model"]["atom_feature_size"]))
        print("bond_feature_size: {}".format(config["model"]["bond_feature_size"]))
        print("global_feature_size: {}".format(config["model"]["global_feature_size"]))
        print("target_dict: {}".format(config["model"]["target_dict"]))
        model = load_graph_level_model_from_config(config["model"])
        return model

    def train(self):
        with wandb.init(project=self.wandb_name, entity=self.wandb_entity) as run:
            init_config = wandb.config
            # init_config["target_dict"]["global"] = init_config["target_list"]
            target_dict = {"global": init_config["target_list"]}
            config = {
                "model": {
                    "n_conv_layers": init_config["n_conv_layers"],
                    "resid_n_graph_convs": init_config["resid_n_graph_convs"],
                    "target_dict": target_dict,  # check
                    "conv_fn": init_config["conv_fn"],
                    "global_pooling_fn": init_config["global_pooling_fn"],
                    "dropout": init_config["dropout"],
                    "batch_norm": init_config["batch_norm"],
                    "activation": init_config["activation"],
                    "classifier": False,
                    "bias": init_config["bias"],
                    "norm": init_config["norm"],
                    "aggregate": init_config["aggregate"],
                    "lr": init_config["lr"],
                    "scheduler_name": init_config["scheduler_name"],
                    "weight_decay": init_config["weight_decay"],
                    "lr_plateau_patience": init_config["lr_plateau_patience"],
                    "lr_scale_factor": init_config["lr_scale_factor"],
                    "loss_fn": init_config["loss_fn"],
                    "embedding_size": init_config["embedding_size"],
                    "fc_hidden_size_1": init_config["fc_hidden_size_1"],
                    "fc_dropout": init_config["fc_dropout"],
                    "shape_fc": init_config["shape_fc"],
                    "fc_num_layers": init_config["fc_num_layers"],
                    "fc_batch_norm": init_config["fc_batch_norm"],
                    "lstm_iters": init_config["lstm_iters"],
                    "lstm_layers": init_config["lstm_layers"],
                    "output_dims": init_config["output_dims"],  # check
                    "pooling_ntypes": init_config["pooling_ntypes"],  # check
                    "pooling_ntypes_direct": init_config[
                        "pooling_ntypes_direct"
                    ],  # check
                    "num_heads_gat": init_config["num_heads_gat"],
                    "dropout_feat_gat": init_config["dropout_feat_gat"],
                    "dropout_attn_gat": init_config["dropout_attn_gat"],
                    "hidden_size_gat": init_config["hidden_size_gat"],
                    "residual_gat": init_config["residual_gat"],
                    "restore": init_config["restore"],
                    "max_epochs": init_config["max_epochs"],
                },
                "dataset": {
                    "allowed_ring_size": init_config["allowed_ring_size"],  # check
                    "allowed_charges": init_config["allowed_charges"],  # check
                    "self_loop": init_config["self_loop"],
                    "extra_keys": init_config["extra_keys"],  # check
                    "extra_dataset_info": init_config["extra_dataset_info"],
                    "debug": init_config["debug"],
                    "target_list": init_config["target_list"],  # check
                    "log_scale_features": init_config["log_scale_features"],
                    "log_scale_targets": init_config["log_scale_targets"],
                    "standard_scale_features": init_config["standard_scale_features"],
                    "standard_scale_targets": init_config["standard_scale_targets"],
                    "val_prop": init_config["val_prop"],
                    "test_prop": init_config["test_prop"],
                    "seed": init_config["seed"],
                    "train_batch_size": init_config["train_batch_size"],
                    "train_dataset_loc": self.dataset_loc,
                    "log_save_dir": self.log_save_dir,
                    "num_workers": init_config["num_workers"],
                },
                "optim": {
                    "num_devices": init_config["num_devices"],
                    "num_nodes": init_config["num_nodes"],
                    "accumulate_grad_batches": init_config["accumulate_grad_batches"],
                    "gradient_clip_val": init_config["gradient_clip_val"],
                    "precision": init_config["precision"],
                    "strategy": init_config["strategy"],
                },
            }

            # make helper to convert from old config to new config

            model = self.make_model(config)
            # log dataset
            wandb.log({"dataset": self.dataset_loc})

            checkpoint_callback = ModelCheckpoint(
                dirpath=self.log_save_dir,
                filename="model_lightning_{epoch:02d}-{val_mae:.2f}",
                monitor="val_mae",
                mode="min",
                auto_insert_metric_name=True,
                save_last=True,
            )

            early_stopping_callback = EarlyStopping(
                monitor="val_mae",
                min_delta=0.00,
                patience=200,
                verbose=False,
                mode="min",
            )
            lr_monitor = LearningRateMonitor(logging_interval="step")
            logger_wb = WandbLogger(name="test_logs")

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
                    checkpoint_callback,
                ],
                enable_checkpointing=True,
                strategy=config["optim"]["strategy"],
                default_root_dir=self.log_save_dir,
                logger=[logger_wb],
                precision=config["optim"]["precision"],
            )

            trainer.fit(model, self.dm)
            if config["dataset"]["test_prop"] > 0.0:
                trainer.test(model, self.dm)
            
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default="bayes")
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument(
        "-dataset_loc", type=str, default="../../dataset/qm_9_merge_3_qtaim.json"
    )
    parser.add_argument("-log_save_dir", type=str, default="./logs_lightning/")
    parser.add_argument("-project_name", type=str, default="qtaim_embed_lightning")
    parser.add_argument("-sweep_config", type=str, default="./sweep_config.json")

    args = parser.parse_args()
    method = str(args.method)
    debug = bool(args.debug)

    dataset_loc = args.dataset_loc
    log_save_dir = args.log_save_dir
    wandb_project_name = args.project_name
    sweep_config_loc = args.sweep_config
    sweep_config = {}
    sweep_params = json.load(open(sweep_config_loc, "r"))
    sweep_params["debug"] = {"values": [debug]}
    sweep_config["parameters"] = sweep_params
    # sweep_config["log_save_dir"] = log_save_dir
    if method == "bayes":
        sweep_config["method"] = method
        sweep_config["metric"] = {"name": "val_mae", "goal": "minimize"}

    # wandb loop
    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name, entity="santi")
    training_obj = TrainingObject(
        sweep_config,
        log_save_dir,
        dataset_loc=dataset_loc,
        project_name=wandb_project_name,
    )

    print("method: {}".format(method))
    # print("on_gpu: {}".format(on_gpu))
    print("debug: {}".format(debug))
    print("dataset_loc: {}".format(dataset_loc))
    print("log_save_dir: {}".format(log_save_dir))
    print("wandb_project_name: {}".format(wandb_project_name))
    print("sweep_config_loc: {}".format(sweep_config_loc))
    wandb.agent(sweep_id, function=training_obj.train, count=3000, entity="santi")
