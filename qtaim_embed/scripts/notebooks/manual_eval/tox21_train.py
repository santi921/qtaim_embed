import os
import numpy as np

from sklearn.metrics import f1_score, log_loss

import torch
from torchmetrics.wrappers import MultioutputWrapper
import torchmetrics

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
import pytorch_lightning as pl


from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask
from qtaim_embed.core.dataset import HeteroGraphGraphLabelClassifierDataset


def manual_eval_separate_tasks(model, dataset_list):
    n_tasks = len(dataset_list)

    temp_data_loader = []
    for i in dataset_list:
        temp_data_loader.append(
            DataLoaderMoleculeGraphTask(i, batch_size=len(i.graphs), shuffle=False)
        )
    task_ind = 0
    statistics_dict = {"f1": [], "acc": [], "auroc": [], "cross_entropy": []}
    for task_ind in range(n_tasks):
        batch_graph, batch_label = next(iter(temp_data_loader[task_ind]))

        labels = batch_label["global"]
        labels_one_hot = torch.argmax(labels, axis=2)
        # print("labels one hot: ", labels_one_hot.shape)
        logits = model.forward(batch_graph, batch_graph.ndata["feat"])
        logits_one_hot = torch.argmax(logits, axis=-1)

        # take logits from first task only
        labels = labels.reshape(labels.shape[0], labels.shape[-1])

        logits_pred = logits[:, task_ind]
        logits_one_hot = logits_one_hot[:, task_ind].reshape(-1)
        # print("logits one hot: ", logits_one_hot.shape)
        # print("logits pred shape: ", logits_pred.shape)
        # print("labels shape: ", labels.shape)

        # compute acc, auroc, f1 manually
        acc_manual = torch.sum(logits_one_hot == labels_one_hot) / len(labels_one_hot)
        f1 = f1_score(labels_one_hot, logits_one_hot, pos_label=0, average="binary")

        # this is wrong
        auroc_manual = torchmetrics.functional.classification.binary_auroc(
            preds=logits_pred,
            target=labels,
        )
        # this is wrong
        cross_ent = torch.nn.functional.cross_entropy(
            target=labels.float(), input=logits_pred
        )

        statistics_dict["f1"].append(f1)
        statistics_dict["acc"].append(acc_manual.numpy())
        statistics_dict["auroc"].append(auroc_manual.numpy())
        statistics_dict["cross_entropy"].append(cross_ent.detach().numpy())
    return statistics_dict


def train_model(model, dataloader_train, dataloader_val):

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=100,
        verbose=False,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=[0],
        num_nodes=1,
        gradient_clip_val=100.0,
        accumulate_grad_batches=2,
        enable_progress_bar=True,
        callbacks=[
            early_stopping_callback,
            lr_monitor,
        ],
        enable_checkpointing=True,
        strategy="auto",
        default_root_dir="./qtaim/",
        precision="32",
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dataloader_train, val_dataloaders=dataloader_val)


def main():
    tox21_test_loc = "../../../../data/splits_1205/test_tox21_qtaim_1205_labelled.pkl"
    tox21_train_loc = "../../../../data/splits_1205/train_tox21_qtaim_1205_labelled.pkl"

    bl_keys = {
        "atom": [],
        "bond": ["bond_length"],
        "global": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    }

    qtaim_dict = {
        "atom": [
            "extra_feat_atom_Lagrangian_K",
            "extra_feat_atom_Hamiltonian_K",
            "extra_feat_atom_e_density",
            "extra_feat_atom_lap_e_density",
            "extra_feat_atom_e_loc_func",
            "extra_feat_atom_ave_loc_ion_E",
            "extra_feat_atom_delta_g_promolecular",
            "extra_feat_atom_delta_g_hirsh",
            "extra_feat_atom_esp_nuc",
            "extra_feat_atom_esp_e",
            "extra_feat_atom_esp_total",
            "extra_feat_atom_grad_norm",
            "extra_feat_atom_lap_norm",
            "extra_feat_atom_eig_hess",
            "extra_feat_atom_det_hessian",
            "extra_feat_atom_ellip_e_dens",
            "extra_feat_atom_eta",
        ],
        "bond": [
            "bond_length",
            "extra_feat_bond_Lagrangian_K",
            "extra_feat_bond_Hamiltonian_K",
            "extra_feat_bond_e_density",
            "extra_feat_bond_lap_e_density",
            "extra_feat_bond_e_loc_func",
            "extra_feat_bond_ave_loc_ion_E",
            "extra_feat_bond_delta_g_promolecular",
            "extra_feat_bond_delta_g_hirsh",
            "extra_feat_bond_esp_nuc",
            "extra_feat_bond_esp_e",
            "extra_feat_bond_esp_total",
            "extra_feat_bond_grad_norm",
            "extra_feat_bond_lap_norm",
            "extra_feat_bond_eig_hess",
            "extra_feat_bond_det_hessian",
            "extra_feat_bond_ellip_e_dens",
            "extra_feat_bond_eta",
        ],
        "global": [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
    }

    dataset_bl_train = HeteroGraphGraphLabelClassifierDataset(
        file=tox21_train_loc,
        allowed_ring_size=[3, 4, 5, 6],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        extra_keys=bl_keys,
        target_list=[
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
        extra_dataset_info={},
        debug=False,
        impute=True,
        element_set={
            "Fe",
            "H",
            "Cu",
            "Cr",
            "Ge",
            "Na",
            "P",
            "N",
            "C",
            "Br",
            "S",
            "V",
            "F",
            "Se",
            "B",
            "Cl",
            "Zn",
            "Ti",
            "O",
            "Si",
            "Ni",
            "Ca",
            "Al",
            "As",
        },
        log_scale_features=False,
        standard_scale_features=True,
    )

    dataset_bl_test = HeteroGraphGraphLabelClassifierDataset(
        file=tox21_test_loc,
        allowed_ring_size=[3, 4, 5, 6],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        extra_keys=bl_keys,
        target_list=[
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
        extra_dataset_info={},
        debug=False,
        impute=False,
        element_set={
            "Fe",
            "H",
            "Cu",
            "Cr",
            "Ge",
            "Na",
            "P",
            "N",
            "C",
            "Br",
            "S",
            "V",
            "F",
            "Se",
            "B",
            "Cl",
            "Zn",
            "Ti",
            "O",
            "Si",
            "Ni",
            "Ca",
            "Al",
            "As",
        },
        log_scale_features=False,
        standard_scale_features=True,
    )

    dataset_train_qtaim = HeteroGraphGraphLabelClassifierDataset(
        file=tox21_train_loc,
        allowed_ring_size=[3, 4, 5, 6],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        extra_keys=qtaim_dict,
        target_list=[
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
        extra_dataset_info={},
        debug=False,
        impute=False,
        element_set={
            "Fe",
            "H",
            "Cu",
            "Cr",
            "Ge",
            "Na",
            "P",
            "N",
            "C",
            "Br",
            "S",
            "V",
            "F",
            "Se",
            "B",
            "Cl",
            "Zn",
            "Ti",
            "O",
            "Si",
            "Ni",
            "Ca",
            "Al",
            "As",
        },
        log_scale_features=False,
        standard_scale_features=True,
    )

    dataset_test_qtaim = HeteroGraphGraphLabelClassifierDataset(
        file=tox21_test_loc,
        allowed_ring_size=[3, 4, 5, 6],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        extra_keys=qtaim_dict,
        target_list=[
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ],
        extra_dataset_info={},
        debug=False,
        impute=False,
        element_set={
            "Fe",
            "H",
            "Cu",
            "Cr",
            "Ge",
            "Na",
            "P",
            "N",
            "C",
            "Br",
            "S",
            "V",
            "F",
            "Se",
            "B",
            "Cl",
            "Zn",
            "Ti",
            "O",
            "Si",
            "Ni",
            "Ca",
            "Al",
            "As",
        },
        log_scale_features=False,
        standard_scale_features=True,
    )

    # make a dataset for each task
    dict_datasets = {}
    dict_datasets["qtaim"] = {}
    dict_datasets["bl"] = {}
    dict_datasets["qtaim"]["test"] = dataset_test_qtaim
    dict_datasets["bl"]["test"] = dataset_bl_test
    dict_datasets["qtaim"]["single_list"] = []
    dict_datasets["bl"]["single_list"] = []

    for task in qtaim_dict["global"]:
        qtaim_dict_temp = qtaim_dict.copy()
        qtaim_dict_temp["global"] = [task]
        base_dict_bl_temp = bl_keys.copy()
        base_dict_bl_temp["global"] = [task]

        dict_datasets["qtaim"]["single_list"].append(
            HeteroGraphGraphLabelClassifierDataset(
                file=tox21_test_loc,
                allowed_ring_size=[3, 4, 5, 6],
                allowed_charges=None,
                allowed_spins=None,
                self_loop=True,
                extra_keys=qtaim_dict_temp,
                target_list=[task],
                extra_dataset_info={},
                debug=False,
                impute=False,
                element_set={
                    "Fe",
                    "H",
                    "Cu",
                    "Cr",
                    "Ge",
                    "Na",
                    "P",
                    "N",
                    "C",
                    "Br",
                    "S",
                    "V",
                    "F",
                    "Se",
                    "B",
                    "Cl",
                    "Zn",
                    "Ti",
                    "O",
                    "Si",
                    "Ni",
                    "Ca",
                    "Al",
                    "As",
                },
                log_scale_features=False,
                standard_scale_features=True,
            )
        )

        dict_datasets["bl"]["single_list"].append(
            HeteroGraphGraphLabelClassifierDataset(
                file=tox21_test_loc,
                allowed_ring_size=[3, 4, 5, 6],
                allowed_charges=None,
                allowed_spins=None,
                self_loop=True,
                extra_keys=base_dict_bl_temp,
                target_list=[task],
                extra_dataset_info={},
                debug=False,
                impute=False,
                element_set={
                    "Fe",
                    "H",
                    "Cu",
                    "Cr",
                    "Ge",
                    "Na",
                    "P",
                    "N",
                    "C",
                    "Br",
                    "S",
                    "V",
                    "F",
                    "Se",
                    "B",
                    "Cl",
                    "Zn",
                    "Ti",
                    "O",
                    "Si",
                    "Ni",
                    "Ca",
                    "Al",
                    "As",
                },
                log_scale_features=False,
                standard_scale_features=True,
            )
        )

    dataloader_qtaim_train = DataLoaderMoleculeGraphTask(
        dataset=dataset_train_qtaim, batch_size=1024, shuffle=True
    )

    dataloader_qtaim_test = DataLoaderMoleculeGraphTask(
        dataset=dataset_test_qtaim, batch_size=1024, shuffle=False
    )

    dataloader_bl_train = DataLoaderMoleculeGraphTask(
        dataset=dataset_bl_train, batch_size=1024, shuffle=True
    )

    dataloader_bl_test = DataLoaderMoleculeGraphTask(
        dataset=dataset_bl_test, batch_size=1024, shuffle=False
    )

    qtaim_model_bl_dict = {
        "atom_feature_size": 48,
        "bond_feature_size": 24,
        "global_feature_size": 3,
        "conv_fn": "ResidualBlock",
        "target_dict": {
            "global": [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ]
        },
        "dropout": 0.2,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "fc_num_layers": 2,
        "aggregate": "sum",
        "n_conv_layers": 4,
        "lr": 0.02,
        "weight_decay": 0,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.1,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 2,
        "embedding_size": 50,
        "fc_layer_size": [1024, 512, 256],
        "fc_dropout": 0.1,
        "fc_batch_norm": True,
        "n_fc_layers": 3,
        "global_pooling_fn": "MeanPoolingThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 9,
        "lstm_layers": 2,
        "num_heads": 3,
        "feat_drop": 0.1,
        "attn_drop": 0.1,
        "residual": False,
        "hidden_size": 10,
        "ntasks": 12,
        "shape_fc": "cone",
        "num_heads_gat": 1,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "hidden_size": 10,
        "residual_gat": False,
        "batch_norm": True,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 1024,
        "restore": False,
        "classifier": True,
    }

    non_qtaim_model_bl_dict = {
        "atom_feature_size": 31,
        "bond_feature_size": 7,
        "global_feature_size": 3,
        "conv_fn": "GraphConvDropoutBatch",
        "target_dict": {
            "global": [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ]
        },
        "dropout": 0.5,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "fc_num_layers": 2,
        "aggregate": "sum",
        "n_conv_layers": 2,
        "lr": 0.02,
        "weight_decay": 0,
        "lr_plateau_patience": 10,
        "lr_scale_factor": 0.1,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 2,
        "embedding_size": 1,
        "fc_layer_size": [1024, 512, 256],
        "fc_dropout": 0.1,
        "fc_batch_norm": True,
        "n_fc_layers": 3,
        "global_pooling_fn": "MeanPoolingThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 9,
        "lstm_layers": 2,
        "num_heads": 3,
        "feat_drop": 0.1,
        "attn_drop": 0.1,
        "residual": False,
        "hidden_size": 10,
        "ntasks": 12,
        "shape_fc": "cone",
        "num_heads_gat": 1,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "hidden_size": 10,
        "residual_gat": False,
        "batch_norm": True,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 512,
        "restore": False,
        "classifier": True,
    }

    model_temp_qtaim = load_graph_level_model_from_config(qtaim_model_bl_dict)
    model_temp_noqtaim = load_graph_level_model_from_config(non_qtaim_model_bl_dict)

    train_model(model_temp_qtaim, dataloader_qtaim_train, dataloader_qtaim_test)
    train_model(model_temp_noqtaim, dataloader_bl_train, dataloader_bl_test)

    model_temp_qtaim.cpu()
    model_temp_noqtaim.cpu()

    stats_dict = manual_eval_separate_tasks(
        model_temp_qtaim, dataset_list=dict_datasets["qtaim"]["single_list"]
    )
    print("-" * 50)
    print(
        "qtaim model:auroc: {:.4f} cross_entropy {:.4f}".format(
            float(np.array(stats_dict["auroc"]).mean()),
            float(np.array(stats_dict["cross_entropy"]).mean()),
        )
    )
    # print("f1 breakdown: ", np.array(np.array(stats_dict["f1"])))
    print("auroc breakdown: ", np.array(np.array(stats_dict["auroc"])))

    stats_dict_no_qtaim = manual_eval_separate_tasks(
        model_temp_noqtaim, dataset_list=dict_datasets["bl"]["single_list"]
    )
    print(
        "no qtaim model:  auroc: {:.4f} cross_entropy {:.4f}".format(
            float(np.array(stats_dict_no_qtaim["auroc"]).mean()),
            float(np.array(stats_dict_no_qtaim["cross_entropy"]).mean()),
        )
    )
    # print("f1 breakdown: ", np.array(np.array(stats_dict_no_qtaim["f1"])))
    print("auroc breakdown: ", np.array(np.array(stats_dict_no_qtaim["auroc"])))


main()
