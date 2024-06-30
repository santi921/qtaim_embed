import torch
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import torchmetrics
import argparse
from copy import deepcopy
from torchmetrics.wrappers import MultioutputWrapper
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask
from qtaim_embed.models.utils import get_test_train_preds_as_df, test_and_predict_libe
from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset, LMDBBaseDataset, LMDBMoleculeDataset
from qtaim_embed.models.initializers import xavier_init, kaiming_init, equi_var_init

def fetch_dataset(loc, config, debug): 
    dataset = HeteroGraphGraphLabelDataset(
        file=loc,
        allowed_ring_size=config['allowed_ring_size'],
        allowed_charges=config['allowed_charges'],
        allowed_spins=config['allowed_spins'],
        element_set=config['element_set'],
        self_loop=True, 
        extra_keys=config['extra_keys'],
        target_list=config['target_list'],
        extra_dataset_info={}, 
        debug=debug, 
        log_scale_features=config['log_scale_features'],
        log_scale_targets=config['log_scale_targets'],
        standard_scale_features=config['standard_scale_features'],
        standard_scale_targets=config['standard_scale_targets'],
        verbose=False
    )
    return dataset


def get_charge_tmqm(batch_graph):
    global_feats = batch_graph.ndata["feat"]["global"]
    # 3th to 6th index inclusive
    ind_charges = (3, 6)
    charge_one_hot = global_feats[:, ind_charges[0] : ind_charges[1]]
    charge_one_hot = charge_one_hot.detach().numpy()
    charge_one_hot = list(np.argmax(charge_one_hot, axis=1) - 1)
    
    return charge_one_hot#, spin_one_hot



def test_and_predict_per_atom(dataset_test, model, batch_size=1024):
    statistics_dict = {}
    scaler_list = dataset_test.label_scalers   
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=batch_size, shuffle=False
    )
    preds_list_raw = []
    labels_list_raw = []
    n_atom_list = []
    charge_list_compiled = []

    model.eval()


    for batch_graph, batched_labels in data_loader:
        preds = model.forward(batch_graph, batch_graph.ndata["feat"])
        preds_raw = deepcopy(preds.detach())
        labels_raw = deepcopy(batched_labels)["global"]
        n_atoms = batch_graph.batch_num_nodes("atom")
        charge_list_test = get_charge_tmqm(batch_graph)
        charge_list_compiled += charge_list_test

        n_atom_list.append(n_atoms)
        preds_list_raw.append(preds_raw)
        labels_list_raw.append(labels_raw)

    charge_list_test = np.array(charge_list_compiled).reshape(-1)

    preds_raw = torch.cat(preds_list_raw, dim=0)
    labels_raw = torch.cat(labels_list_raw, dim=0)
    n_atom_list = torch.cat(n_atom_list, dim=0)

    for scaler in scaler_list[::-1]:
        labels_unscaled = scaler.inverse_feats({"global": labels_raw})["global"].view(-1, model.hparams.ntasks)
        preds_unscaled = scaler.inverse_feats({"global": preds_raw})["global"].view(-1, model.hparams.ntasks)

    
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
    
    #if r2_manual > 0.2:
    print(
        "Test Performance:\t r2: {:.4f}\t mae: {:.4f}\t  rmse: {:.4f}\t ewt: {:.4f}".format(
            float(r2_manual), float(mae_per_atom * 1000), float(mean_rmse_per_atom * 1000), float(ewt_prop * 100)
        )
    )
    print("--" * 50)
    statistics_dict["test"] = {
        "r2": r2_manual.numpy(), 
        "mae": mae_per_atom.numpy(),
        "mae_per_molecule": mae_per_molecule.numpy(), 
        "rmse": mean_rmse_per_atom.numpy(),
        "rmse_per_molecule": rmse_per_molecule.numpy(),
        "ewt": ewt_prop.numpy()
    }

    return preds_unscaled.numpy(), labels_unscaled.numpy(), statistics_dict, charge_list_test
    #else: 
    #    return False, False, False, False
    


def plot_joint_stratified(pred, label, charge_list, stats, save_loc):
    df_test = pd.DataFrame(
        {
            "preds": pred,
            "labels": label,
            "charge": charge_list
        }
    )

    

    r2 = stats["test"]['r2']
    mae = stats["test"]['mae']
    rmse = stats["test"]['rmse']
    ewt = stats["test"]['ewt']
    fig = plt.figure(figsize=(12, 10))
    # specify number of bins
    num_bins = 50
    # make jointplots
    sns.jointplot(
        x="preds",
        y="labels",
        data=df_test,
        # kind="reg",
        hue="charge",
        palette="colorblind",
        height=20,
        s=100, 
        edgecolor="black",
        space=0,
        ylim=(-0.5, 1.0),
        xlim=(-0.5, 1.0),
        # joint_kws={"line_kws": {"color": "red"}},
        # joint_kws={"gridsize": 80},
    ).set_axis_labels("Target(eV)", "Predicted(eV)")
    # set font sizes
    plt.xlabel("Target(eV)", fontsize=40)
    plt.ylabel("Predicted(eV)", fontsize=40)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)

    # rotate xticks
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    # add diagonal line
    plt.plot([-0.5, 1.0], [-0.5, 1.0], linewidth=5, color="red")
    plt.xlim(-0.5, 1.0)
    plt.ylim(-0.5, 1.0)
    # add statistics
    textstr = "$R^2$: {:.3f}\nMAE(meV/atom): {:.3f}\nRMSE(meV/atom): {:.3f}\nEwT: {:.1f}%".format(r2, mae*1000, rmse*1000, ewt*100)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.43, 0.27, textstr, transform=fig.transFigure, fontsize=32,
        verticalalignment='top', bbox=props)

    #["test_ewt_per"] * 100
    plt.legend(fontsize=30, loc="lower right", title="Charge", title_fontsize=30)
    # add title
    # save figure
    plt.savefig(save_loc, dpi=300)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="low", help="low, mid, high")
    parser.add_argument("--debug", action="store_true")
    results = parser.parse_args()
    level = str(results.level)
    debug = bool(results.debug)


    best_model_dict = {
        'qtaim': {
            'low': 'model_lightning_epoch=362-val_loss=0.0995.ckpt',
            'high': 'model_lightning_epoch=371-val_loss=0.0149.ckpt'
        },
        'no_qtaim': {
            'low': 'model_lightning_epoch=535-val_loss=0.0963.ckpt',
            'high': 'model_lightning_epoch=416-val_loss=0.0329.ckpt'
        }
    }

    elem_set = ['Ag', 'As', 'Au', 'B', 'Br', 'C', 'Cd', 'Cl', 'Co', 'Cr', 'Cu', 'F', 'Fe', 'H', 'Hf', 'Hg', 'I', 'Ir', 'La', 'Mn', 'Mo', 'N', 'Nb', 'Ni', 'O', 'Os', 'P', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'S', 'Sc', 'Se', 'Si', 'Ta', 'Tc', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr']

    dataset_dict = {    
        "qtaim_dataset": {
                "allowed_ring_size": [
                    4,
                    5,
                    6,
                    7
                ],
                "allowed_charges": [-1, 0, 1],
                "allowed_spins": [],
                "self_loop": True,
                "target_list": [
                    "corrected_E"
                ],
                "element_set": elem_set,
                "val_prop": 0.1,
                "test_prop": 0.1,
                "train_batch_size": 1024,
                "seed": 42,
                "debug": False, 
                "num_workers": 4,
                "per_atom": False,
                "extra_dataset_info": {},
                "log_scale_features": False,
                "log_scale_targets": False,
                "standard_scale_features": True,
                "standard_scale_targets": True,
                "extra_keys": {
                    "atom": [
                        "extra_feat_atom_Lagrangian_K",
                        "extra_feat_atom_Hamiltonian_K",
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
                        "extra_feat_atom_ellip_e_dens",
                        "extra_feat_atom_eta",
                        "extra_feat_atom_eig_hess"

                    ],
                    "bond": [
                        "extra_feat_bond_Lagrangian_K",
                        "extra_feat_bond_Hamiltonian_K",
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
                        "bond_length"
                    ],
                    "global": ["corrected_E", "charge"]
                }
            },
        "base_dataset": {
            "allowed_ring_size": [
                4,
                5,
                6,
                7
            ],
            "allowed_charges": [-1, 0, 1],
            "allowed_spins": [],
            "self_loop": True,
            "target_list": [
                "corrected_E"
            ],
            "element_set": elem_set,
            "val_prop": 0.1,
            "test_prop": 0.1,
            "train_batch_size": 1024,
            "seed": 42,
            "debug": True, 
            "num_workers": 4,
            "per_atom": True,
            "extra_dataset_info": {},
            "log_scale_features": False,
            "log_scale_targets": False,
            "standard_scale_features": True,
            "standard_scale_targets": True,
            "extra_keys": {
                "atom": [
                ],
                "bond": [
                    "bond_length"
                ],
                "global": ["corrected_E", "charge"]
            }
        },

    }

    test_dataset_low_loc = "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/low/new_parse_tmQM_wB97MV_TPSS_QTAIM_impute_aligned_test_charge_1_neg_1.pkl"
    test_dataset_high_loc = "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/high/new_parse_tmQMg_qtaim_impute_corrected_test_charge_1_neg_1.pkl"
    train_dataset_low_loc = "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/low/new_parse_tmQM_wB97MV_TPSS_QTAIM_impute_aligned_train_charge_0.pkl"
    train_dataset_high_loc = "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/new_parse/high/new_parse_tmQMg_qtaim_impute_corrected_train_charge_0.pkl"

    df_dict = {}
    


    
    if level == "low":
        dataset_low_no_qtaim = fetch_dataset(test_dataset_low_loc, config=dataset_dict["base_dataset"], debug=debug)
        dataset_low_qtaim = fetch_dataset(test_dataset_low_loc, config=dataset_dict["qtaim_dataset"], debug=debug)
        dataset_low_no_qtaim_train = fetch_dataset(train_dataset_low_loc, config=dataset_dict["base_dataset"], debug=debug)
        dataset_low_qtaim_train = fetch_dataset(train_dataset_low_loc, config=dataset_dict["qtaim_dataset"], debug=debug)
            
        df_dict = {
            "low_test": {
                "no_qtaim": dataset_low_no_qtaim,
                "qtaim": dataset_low_qtaim
            },
            "low_train": {
                "no_qtaim": dataset_low_no_qtaim_train,
                "qtaim": dataset_low_qtaim_train
            }
        }



    elif level == "high":
        dataset_high_no_qtaim = fetch_dataset(test_dataset_high_loc, config=dataset_dict["base_dataset"], debug=debug)
        dataset_high_qtaim = fetch_dataset(test_dataset_high_loc, config=dataset_dict["qtaim_dataset"], debug=debug)
        dataset_high_no_qtaim_train = fetch_dataset(train_dataset_high_loc, config=dataset_dict["base_dataset"], debug=debug)
        dataset_high_qtaim_train = fetch_dataset(train_dataset_high_loc, config=dataset_dict["qtaim_dataset"], debug=debug)

        df_dict = {
            "high_test": {
                "no_qtaim": dataset_high_no_qtaim,
                "qtaim": dataset_high_qtaim
            }, 
            "high_train": {
                "no_qtaim": dataset_high_no_qtaim_train,
                "qtaim": dataset_high_qtaim_train
            }
        }


    model_low_nq_root = "/home/santiagovargas/dev/qtaim_embed/data/saved_models/tmqm_reparse_0624/low/no_qtaim/"
    model_low_q_root = "/home/santiagovargas/dev/qtaim_embed/data/saved_models/tmqm_reparse_0624/low/qtaim/"
    model_high_nq_root = "/home/santiagovargas/dev/qtaim_embed/data/saved_models/tmqm_reparse_0624/high/no_qtaim/"
    model_high_q_root = "/home/santiagovargas/dev/qtaim_embed/data/saved_models/tmqm_reparse_0624/high/qtaim/"



    dict_model_loc = {
        "low":{
            "no_qtaim": {"root": model_low_nq_root, "models": os.listdir(model_low_nq_root)},
            "qtaim": {"root": model_low_q_root, "models": os.listdir(model_low_q_root)},
        },

        "high":{
            "no_qtaim": {"root": model_high_nq_root, "models": os.listdir(model_high_nq_root)},
            "qtaim": {"root": model_high_q_root, "models": os.listdir(model_high_q_root)},
        } 
    }




    #for level in ["low", "mid", "high"]:
    for descriptor in ["no_qtaim", "qtaim"]: 
        print("**"*20 + f"Processing {level} {descriptor}" + "**"*20)
        print("****"*20)
        print("****"*20)
        print("****"*20)
        model_loc = dict_model_loc[level][descriptor]["root"] + best_model_dict[descriptor][level]
        
        print(f"Processing {level} {descriptor} ")

        model_loc = os.path.join(model_loc)
        model_config = {
            "model": {
                "restore": True, 
                "restore_path": model_loc
                }
        }
        model = load_graph_level_model_from_config(model_config["model"])
        # reset model
        #kaiming_init(model)
        #xavier_init(model)

        dataset_train = df_dict[f"{level}_train"][descriptor]
        dataset_test = df_dict[f"{level}_test"][descriptor]


        dataloader_train = DataLoaderMoleculeGraphTask(
            dataset_train, batch_size=512, shuffle=True, num_workers=4
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_mae", min_delta=0.00, patience=100, verbose=False, mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        save_folder = "./test_ood_charge/{}/{}/".format(level, descriptor)

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
            check_val_every_n_epoch=1,
            strategy="auto",
            # default_root_dir=model_save_string,
            default_root_dir=save_folder,
            precision="bf16-mixed",
        )


        trainer.fit(model, dataloader_train)
        trainer.save_checkpoint("./test_ood_charge/{}_{}.ckpt".format(level, descriptor))

        preds, labels, stats, charge_list = test_and_predict_per_atom(dataset_test, model)
        print(preds.shape, labels.shape, charge_list.shape)
        # save predictions
        # convert numpy to list 
        
        preds = preds.reshape(-1).tolist()
        labels = labels.reshape(-1).tolist()
        charge_list = charge_list.reshape(-1).tolist()

        df_test = pd.DataFrame(
            {
            "preds": preds,
            "labels": labels,
            "charge": charge_list
            }
        )
        
        df_test.to_pickle(
            save_folder + "test.pkl".format(level, descriptor)
        )
        
        plot_joint_stratified(
            preds, 
            labels, 
            charge_list, 
            stats, 
            "./test_ood_charge/{}_{}.png".format(level, descriptor))


main()



