import torch
import pytorch_lightning as pl
from qtaim_embed.core.datamodule import QTAIMNodeTaskDataModule
from qtaim_embed.models.utils import load_node_level_model_from_config
from copy import deepcopy

torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")


class TestNodePred:
    target_dict = {
        "atom": ["extra_feat_atom_esp_total"],
        "bond": [
            "extra_feat_bond_esp_total",
            "extra_feat_bond_ellip_e_dens",
            "extra_feat_bond_eta",
        ],
        "global": [],
    }

    config_base = {
        "dataset": {
            "allowed_ring_size": [3, 4, 5, 6],
            "allowed_charges": [],
            "allowed_spins": [],
            "self_loop": True,
            "per_atom": False,
            "element_set": [],
            "val_prop": 0.1,
            "test_prop": 0.1,
            "debug": False,
            "seed": 42,
            "bond_key": "bonds",
            "map_key": "extra_feat_bond_indices_qtaim",
            "num_workers": 4,
            "train_batch_size": 32,
            "extra_dataset_info": {},
            "log_scale_features": False,
            "log_scale_targets": False,
            "standard_scale_features": True,
            "standard_scale_targets": True,
            "extra_keys": {
                "atom": ["extra_feat_atom_esp_total"],
                "bond": [
                    "bond_length",
                    "extra_feat_bond_esp_total",
                    "extra_feat_bond_ellip_e_dens",
                    "extra_feat_bond_eta",
                ],
            },
            "target_dict": target_dict,
            "train_dataset_loc": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/high/high_train_50.pkl",
            "verbose": False,
        },
        "model": {
            "conv_fn": "ResidualBlock",
            "dropout": 0.2,
            "initializer": "kaiming",
            "batch_norm_tf": True,
            "activation": "ReLU",
            "bias": True,
            "norm": "both",
            "aggregate": "sum",
            "n_conv_layers": 6,
            "lr": 0.01,
            "weight_decay": 5e-05,
            "lr_plateau_patience": 25,
            "lr_scale_factor": 0.5,
            "scheduler_name": "reduce_on_plateau",
            "loss_fn": "mse",
            "resid_n_graph_convs": 3,
            "embedding_size": 25,
            "num_heads": 2,
            "feat_drop": 0.1,
            "attn_drop": 0.1,
            "residual": False,
            "num_heads_gat": 2,
            "compiled": False,
            "dropout_feat_gat": 0.1,
            "dropout_attn_gat": 0.1,
            "hidden_size": 100,
            "residual_gat": True,
            "batch_norm": True,
            "pooling_ntypes": ["atom", "bond", "global"],
            "pooling_ntypes_direct": ["global"],
            "restore": False,
            "max_epochs": 1000,
            "extra_stop_patience": 10,
            "target_dict": target_dict,
        },
    }

    dm = QTAIMNodeTaskDataModule(config=config_base)
    names, feature_size = dm.prepare_data(stage="fit")

    val_dataloader = dm.val_dataloader()
    scalers = dm.full_dataset.label_scalers

    def main_train(self, model="ResidualBlock"):
        config = deepcopy(self.config_base)

        if model == "ResidualBlock":
            config["model"]["conv_fn"] = "ResidualBlock"
            config["model"]["resid_n_graph_convs"] = 2
            config["model"]["n_conv_layers"] = 8

        elif model == "GATConv":
            config["model"]["conv_fn"] = "GATConv"
            config["model"]["n_conv_layers"] = 5

        elif model == "GraphConvDropoutBatch":
            config["model"]["conv_fn"] = "GraphConvDropoutBatch"

        else:
            raise ValueError(f"Model {model} not recognized.")

        config["model"]["atom_feature_size"] = self.feature_size["atom"]
        config["model"]["bond_feature_size"] = self.feature_size["bond"]
        config["model"]["global_feature_size"] = self.feature_size["global"]

        model = load_node_level_model_from_config(config["model"])

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="gpu",
            devices=[0],
            gradient_clip_val=10.0,
            accumulate_grad_batches=1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            strategy="auto",
            precision="bf16",
        )

        trainer.fit(model, self.dm)

        model.cpu()

        (r2_dict, mae_dict, pred_dict, label_dict) = model.evaluate_manually(
            test_dataloader=self.val_dataloader, scaler_list=self.scalers
        )
        assert type(r2_dict) == dict, "r2_dict is not a dict."
        assert len(r2_dict["atom"]) == 1, "r2 computed wrong"
        assert len(r2_dict["bond"]) == 3, "r2 computed wrong"

    def test_resid(self):
        self.main_train("ResidualBlock")

    def test_gat(self):
        self.main_train("GATConv")

    def test_gcn(self):
        self.main_train("GraphConvDropoutBatch")
