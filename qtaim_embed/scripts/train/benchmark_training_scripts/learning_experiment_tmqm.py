import json
import torch
from sklearn.metrics import r2_score
import argparse
import numpy as np
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import torchmetrics

from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask


from qtaim_embed.scripts.train.learning_utils import (
    get_datasets_tmqm_low,
    get_datasets_tmqm_mid,
    get_datasets_tmqm_high,
)

torch.set_float32_matmul_precision("high")


def manual_statistics(model, dataset_test, scaler_list):

    data_loader = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=500, shuffle=False
    )
    preds_list_raw = []
    labels_list_raw = []
    n_atom_list = []
    model.eval()

    for batch_graph, batched_labels in data_loader:
        preds = model.forward(batch_graph, batch_graph.ndata["feat"])
        preds_raw = deepcopy(preds.detach())
        labels_raw = deepcopy(batched_labels)["global"]
        n_atoms = batch_graph.batch_num_nodes("atom")

        n_atom_list.append(n_atoms)
        preds_list_raw.append(preds_raw)
        labels_list_raw.append(labels_raw)

    preds_raw = torch.cat(preds_list_raw, dim=0)
    labels_raw = torch.cat(labels_list_raw, dim=0)
    n_atom_list = torch.cat(n_atom_list, dim=0)

    for scaler in scaler_list[::-1]:
        labels_unscaled = scaler.inverse_feats({"global": labels_raw})["global"].view(
            -1, model.hparams.ntasks
        )
        preds_unscaled = scaler.inverse_feats({"global": preds_raw})["global"].view(
            -1, model.hparams.ntasks
        )

    abs_diff = np.abs(preds_unscaled - labels_unscaled)
    y = labels_unscaled
    y_pred = preds_unscaled

    r2_manual = torchmetrics.functional.r2_score(y_pred, y)
    mae_per_atom = torch.mean(abs_diff / n_atoms)
    mae_per_molecule = torch.mean(abs_diff)
    ewt_prop = torch.sum(abs_diff < 0.043) / len(abs_diff)
    rmse_per_molecule = torch.mean(torch.sqrt(torch.mean(abs_diff**2)))
    mse_per_atom = abs_diff**2 / n_atoms
    mean_rmse_per_atom = torch.sqrt(torch.mean(mse_per_atom))

    return (
        mae_per_atom.numpy(),
        mean_rmse_per_atom.numpy(),
        r2_manual.numpy(),
        ewt_prop.numpy(),
    )


def main():

    parser = argparse.ArgumentParser(
        description="select what level of dataset to run experiment on"
    )
    parser.add_argument(
        "--level",
        type=str,
        help="select what level of dataset to run experiment on",
        required=True,
    )
    args = parser.parse_args()
    level = str(args.level)
    print("level: ", level)

    results_dict = {}

    if level == "low":
        loc_dict = {
            "50": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/low/low_train_50.pkl",
            # "500": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/low/low_train_500.pkl",
            # "5000":
            # "10000":
            # "all":
            # "test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/low/new_parse_tmQM_wB97MV_TPSS_QTAIM_corrected_test.pkl"
            "test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/low/low_train_50.pkl",
        }
        model_dict, dict_keys, dict_datasets = get_datasets_tmqm_low(loc_dict)

    elif level == "high":
        loc_dict = {
            "50": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/high/high_train_50.pkl",
            # "500": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/high/high_train_500.pkl",
            # "5000":
            # "10000":
            # "all":
            # "test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/high/new_parse_test_tmQMg_qtaim_best_corrected.pkl"
            "test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/high/high_train_50.pkl",
        }
        model_dict, dict_keys, dict_datasets = get_datasets_tmqm_high(loc_dict)
    else:
        print("level not found")
        return

    for keys in dict_datasets.keys():

        model_temp = load_graph_level_model_from_config(model_dict[keys])
        test_dataset = dict_datasets[keys]["test"]

        for name in dict_datasets[keys].keys():
            if name != "test":
                dataloader_train = DataLoaderMoleculeGraphTask(
                    dict_datasets[keys][name],
                    batch_size=256,
                    shuffle=True,
                    num_workers=0,
                )
                dataloader_test = DataLoaderMoleculeGraphTask(
                    test_dataset,
                    batch_size=len(test_dataset.graphs),
                    shuffle=False,
                    num_workers=0,
                )
                early_stopping_callback = EarlyStopping(
                    monitor="val_mae",
                    min_delta=0.00,
                    patience=100,
                    verbose=False,
                    mode="min",
                )
                lr_monitor = LearningRateMonitor(logging_interval="step")

                trainer = pl.Trainer(
                    max_epochs=1000,
                    accelerator="gpu",
                    gradient_clip_val=10.0,
                    devices=1,
                    accumulate_grad_batches=3,
                    enable_progress_bar=True,
                    callbacks=[
                        early_stopping_callback,
                        lr_monitor,
                    ],
                    enable_checkpointing=True,
                    strategy="auto",
                    # default_root_dir=model_save_string,
                    default_root_dir="./test/",
                    precision="bf16-mixed",
                )

                trainer.fit(model_temp, dataloader_train)
                trainer.save_checkpoint(f"./libe_learning_test/{keys}_{name}.ckpt")

                # batch_graph, batched_labels = next(iter(dataloader_test))
                mae_per_atom, rmse_per_atom, r2_manual, ewt_prop = manual_statistics(
                    model_temp, test_dataset, dict_datasets[keys][name].label_scalers
                )

                results_dict[f"{keys}_{name}"] = {
                    "mae_per_atom": float(mae_per_atom),
                    "rmse_per_atom": float(rmse_per_atom),
                    "r2_manual": float(r2_manual),
                    "ewt_prop": float(ewt_prop),
                }

    print(results_dict)
    # save results dict
    json.dump(
        results_dict,
        open("./tmqm_{}_learning_results_dict.json".format(level), "w"),
        indent=4,
    )


main()
