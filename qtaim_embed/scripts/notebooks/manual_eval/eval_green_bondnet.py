import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import matplotlib.pyplot as plt


import torch, json
import numpy as np
import pytorch_lightning as pl

from bondnet.data.datamodule import BondNetLightningDataModule
from bondnet.utils import seed_torch
from bondnet.model.training_utils import load_model_lightning

seed_torch()
torch.set_float32_matmul_precision("high")  # might have to disable on older GPUs
torch.multiprocessing.set_sharing_strategy("file_system")


def evaluate(model, nodes, data_loader, device=None, name="test"):
    """
    basic loop for training a classifier. Gets mae

    Args:
        model(pytorch model): pytorch model
        nodes(dict): node feature dictionary
        data_loader(loader obj): loader object with data to eval
        device(str): cpu/gpu
    Returns:
        mae(float): mae
    """

    model.eval()

    dict_result_raw = {}

    with torch.no_grad():
        count, mae, mae_no_std = 0.0, 0.0, 0.0
        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            norm_atom = label["norm_atom"]
            # norm_atom = None
            norm_bond = label["norm_bond"]
            # norm_bond = None
            stdev = label["scaler_stdev"]
            reaction_types = label["reaction_types"]
            # print device batched_graph is on
            # move batched_graph to device
            batched_graph = batched_graph.to(device)
            if device is not None:
                feats = {k: v.to(device) for k, v in feats.items()}
                target = target.to(device)
                norm_atom = norm_atom.to(device)
                norm_bond = norm_bond.to(device)
                stdev = stdev.to(device)

            pred = model(
                graph=batched_graph,
                feats=feats,
                reactions=label["reaction"],
                norm_atom=norm_atom,
                norm_bond=norm_bond,
                reverse=False,
            )

            pred = pred.view(-1)

            if device is None:
                pred_np = pred.detach().numpy()
                target_np = target.detach().numpy()
                stdev_np = stdev.detach().numpy()
            else:
                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                stdev_np = stdev.detach().cpu().numpy()

            t1 = torch.tensor(pred_np)
            t2 = torch.tensor(target_np)
            t2 = t2.squeeze()
            # r2score = R2Score()
            # sc = r2score(t1, t2)
            target_np = target_np.reshape(-1)
            x = pred_np * stdev_np
            y = target_np * stdev_np
            # plt.scatter(pred_np, target_np)
            # plt.scatter(x, y)
            x_as_list = [float(i) for i in list(x)]
            y_as_list = [float(i) for i in list(y)]
            dict_res = {"pred": x_as_list, "target": y_as_list}

            # plt.title("Predicted vs. True")
            # plt.xlabel("Predicted")
            # plt.ylabel("True")
            # plt.grid(False)
            # plt.savefig("./{}.png".format(name))
            mae = mean_absolute_error(x, y)
            mse = mean_squared_error(x, y) ** 0.5
            sc = r2_score(x, y)
            # mae += metric_fn(pred, target, stdev)
            # mae_no_std += metric_fn(pred_np, target_np, None)
            # count += len(target)

    return mae, mse, sc, dict_res


def main():
    config = {
        "model": {
            "augment": False,
            "classifier": False,
            "classif_categories": 3,
            "cat_weights": [1.0, 1.0, 1.0],
            "extra_features": {"global": ["global_dHrxn298"]},
            "extra_info": [],
            "feature_filter": False,
            "filter_species": [3, 6],
            "precision": "bf16",
            "filter_outliers": False,
        },
        "optim": {
            "batch_size": 100,
            "num_devices": 1,
            "num_nodes": 1,
            "num_workers": 4,
            "val_size": 0.0,
            "test_size": 1.0,
            "strategy": "auto",
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "dataset": {
            "log_save_dir": "./100/",
            "lmdb_dir": "./lmdb_data/",
            "target_var": "dE0",
            "overwrite": True,
        },
        "dataset_transfer": {
            "log_save_dir": "./model_log_transfer/",
            "lmdb_dir": "./lmdb_data_transfer/",
            "target_var": "dHrxn298",
            "overwrite": True,
        },
    }

    config_qtaim = {
        "model": {
            "augment": False,
            "classifier": False,
            "classif_categories": 3,
            "cat_weights": [1.0, 1.0, 1.0],
            "extra_features": {
                # "atom": ["grad_norm", "esp_total", "Hamiltonian_K"],
                # "bond": ["grad_norm", "ellip_e_dens", "ave_loc_ion_E",
                #        "e_loc_func", "esp_total",  "Hamiltonian_K"],
                "atom": [
                    "grad_norm",
                    "esp_total",
                    "Hamiltonian_K",
                    "eta",
                    "eig_hess",
                    "lap_norm",
                    "esp_e",
                    "esp_nuc",
                    "delta_g_hirsh",
                    "delta_g_promolecular",
                    "ave_loc_ion_E",
                    "lap_e_density",
                    "energy_density",
                    "Lagrangian_K",
                ],
                "bond": [
                    "grad_norm",
                    "ellip_e_dens",
                    "ave_loc_ion_E",
                    "esp_nuc",
                    "energy_density",
                    "esp_total",
                    "Hamiltonian_K",
                    "lap_norm",
                    "esp_e",
                    "esp_nuc",
                    "delta_g_hirsh",
                    "delta_g_promolecular",
                    "lap_e_density",
                    "Lagrangian_K",
                ],
                "global": ["global_dHrxn298"],
                # "global": [],
                "mappings": ["indices_qtaim"],
            },
            "extra_info": [],
            "feature_filter": False,
            "filter_species": [3, 6],
            "precision": "bf16",
            "filter_outliers": False,
        },
        "optim": {
            "batch_size": 100,
            "num_devices": 1,
            "num_nodes": 1,
            "num_workers": 4,
            "val_size": 0.0,
            "test_size": 1.0,
            "strategy": "auto",
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "dataset": {
            "log_save_dir": "./100/",
            "lmdb_dir": "./lmdb_data/",
            "target_var": "dE0",
            # "target_var": "dHrxn298",
            "overwrite": True,
        },
        "dataset_transfer": {
            "log_save_dir": "./model_log_transfer/",
            "lmdb_dir": "./lmdb_data_transfer/",
            "target_var": "dHrxn298",
            "overwrite": True,
        },
    }

    test_path = "/home/santiagovargas/dev/qtaim_embed/data/green_splits/ccsdtf_121423_qtaim_learning_test.json"
    if config["model"]["precision"] == "16" or config["model"]["precision"] == "32":
        config["model"]["precision"] = int(config["model"]["precision"])

    config["dataset"]["data_dir"] = test_path
    config_qtaim["dataset"]["data_dir"] = test_path
    # extra_keys = config["model"]["extra_features"]
    config["model"]["filter_sparse_rxns"] = False
    config["model"]["debug"] = False
    config_qtaim["model"]["debug"] = False

    log_save_dir = "./test/"

    model_100 = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0126/non/model_lightning_epoch=26-val_l1=14.55.ckpt",
        }
    }

    model_1000 = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0126/non/model_lightning_epoch=38-val_l1=11.38.ckpt",
        }
    }

    model_10000 = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0126/non/model_lightning_epoch=427-val_l1=8.97.ckpt",
        }
    }

    model = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0126/non/model_lightning_epoch=78-val_l1=7.85.ckpt",
        }
    }

    model_qtaim_100 = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0208_3/model_lightning_epoch=07-val_l1=16.47.ckpt",
        }
    }

    model_qtaim_1000 = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0208_3/model_lightning_epoch=47-val_l1=10.76.ckpt",
        }
    }

    model_qtaim_10000 = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0208_3/model_lightning_epoch=187-val_l1=7.75.ckpt",
        }
    }

    model_qtaim = {
        "model": {
            "restore": True,
            "restore_path": "/home/santiagovargas/dev/qtaim_embed/data/saved_models/final_set/bondnet/0208_3/model_lightning_epoch=250-val_l1=7.38.ckpt",
        }
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # No QTAIM
    dm = BondNetLightningDataModule(config)
    feature_size, feature_names = dm.prepare_data()
    dm.setup(stage="predict")
    model_100 = load_model_lightning(model_100["model"], load_dir=log_save_dir)
    model_1000 = load_model_lightning(model_1000["model"], load_dir=log_save_dir)
    model_10000 = load_model_lightning(model_10000["model"], load_dir=log_save_dir)
    model = load_model_lightning(model["model"], load_dir=log_save_dir)

    model_100.to(device)
    model_1000.to(device)
    model_10000.to(device)
    model.to(device)

    data_loader = dm.test_dataloader()
    l1_acc_100, _, sc_100, df_pred_100 = evaluate(
        model_100, feature_names, data_loader, device
    )
    l1_acc_1000, _, sc_1000, df_pred_1000 = evaluate(
        model_1000, feature_names, data_loader, device
    )
    l1_acc_10000, _, sc_10000, df_pred_10000 = evaluate(
        model_10000, feature_names, data_loader, device
    )
    l1_acc, _, sc, df_pred = evaluate(model, feature_names, data_loader, device)

    print("100")
    print("MAE: ", l1_acc_100)
    # print("MSE: ", mse_100)
    print("R2: ", sc_100)

    print("1000")
    print("MAE: ", l1_acc_1000)
    # print("MSE: ", mse_1000)
    print("R2: ", sc_1000)

    print("10000")
    print("MAE: ", l1_acc_10000)
    # print("MSE: ", mse_10000)
    print("R2: ", sc_10000)

    print("full")
    print("MAE: ", l1_acc)
    # print("MSE: ", mse_10000)
    print("R2: ", sc)

    # QTAIM
    dm_qtaim = BondNetLightningDataModule(config_qtaim)
    feature_size, feature_names = dm_qtaim.prepare_data()
    dm_qtaim.setup(stage="predict")
    model_qtaim_100 = load_model_lightning(
        model_qtaim_100["model"], load_dir=log_save_dir
    )
    model_qtaim_1000 = load_model_lightning(
        model_qtaim_1000["model"], load_dir=log_save_dir
    )
    model_qtaim_10000 = load_model_lightning(
        model_qtaim_10000["model"], load_dir=log_save_dir
    )
    model_qtaim = load_model_lightning(model_qtaim["model"], load_dir=log_save_dir)
    model_qtaim_100.to(device)
    model_qtaim_1000.to(device)
    model_qtaim_10000.to(device)
    model_qtaim.to(device)
    data_loader_qtaim = dm_qtaim.test_dataloader()
    l1_acc_qtaim_100, _, sc_qtaim_100, df_pred_qtaim_100 = evaluate(
        model_qtaim_100, feature_names, data_loader_qtaim, device
    )
    l1_acc_qtaim_1000, _, sc_qtaim_1000, df_pred_qtaim_1000 = evaluate(
        model_qtaim_1000, feature_names, data_loader_qtaim, device
    )
    l1_acc_qtaim_10000, _, sc_qtaim_10000, df_pred_qtaim_10000 = evaluate(
        model_qtaim_10000, feature_names, data_loader_qtaim, device
    )
    l1_acc_qtaim, _, sc_qtaim, df_pred_qtaim = evaluate(
        model_qtaim, feature_names, data_loader_qtaim, device
    )

    print("\n\n")

    print("Qtaim 100")
    print("MAE: ", l1_acc_qtaim_100)
    # print("MSE: ", mse_qtaim_100)
    print("R2: ", sc_qtaim_100)

    print("Qtaim 1000")
    print("MAE: ", l1_acc_qtaim_1000)
    # print("MSE: ", mse_qtaim_1000)
    print("R2: ", sc_qtaim_1000)

    print("Qtaim 10000")
    print("MAE: ", l1_acc_qtaim_10000)
    # print("MSE: ", mse_qtaim_10000)
    print("R2: ", sc_qtaim_10000)

    print("Qtaim")
    print("MAE: ", l1_acc_qtaim)
    # print("MSE: ", mse_qtaim_10000)
    print("R2: ", sc_qtaim)

    dict_results_full = {
        "100": df_pred_100,
        "1000": df_pred_1000,
        "10000": df_pred_10000,
        "full": df_pred,
        "qtaim_100": df_pred_qtaim_100,
        "qtaim_1000": df_pred_qtaim_1000,
        "qtaim_10000": df_pred_qtaim_10000,
        "qtaim": df_pred_qtaim,
    }

    json.dump(
        dict_results_full, open("./green_bondnet_results_revamp.json", "w"), indent=4
    )


main()
