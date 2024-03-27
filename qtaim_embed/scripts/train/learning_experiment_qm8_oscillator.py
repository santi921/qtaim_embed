import json
import torch 
from sklearn.metrics import r2_score
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping
)

from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask

torch.set_float32_matmul_precision('high')



def get_datasets_qm8(loc_dict):

    qtaim_model_bl_dict ={
        'atom_feature_size': 33,
        'bond_feature_size': 29,
        'global_feature_size': 3,
        'conv_fn': 'ResidualBlock',
        'target_dict': {'global': ['f1-CC2', 'f2-CC2']},
        'dropout': 0.2,
        'batch_norm_tf': False,
        'activation': 'ReLU',
        'bias': True,
        'norm': 'both',
        'aggregate': 'sum',
        'n_conv_layers': 4,
        'lr': 0.044675426899321025,
        'weight_decay': 0.00001,
        'lr_plateau_patience': 10,
        'lr_scale_factor': 0.5,
        'scheduler_name': 'reduce_on_plateau',
        'loss_fn': 'mse',
        'resid_n_graph_convs': 2,
        'embedding_size': 80,
        'fc_layer_size': [512, 512],
        "shape_fc": "flat",
        'fc_dropout': 0.1,
        'fc_batch_norm': True,
        'fc_num_layers': 2,
        'n_fc_layers': 2,
        'global_pooling_fn': 'MeanPoolingThenCat',
        'ntypes_pool': ['atom', 'bond', 'global'],
        'ntypes_pool_direct_cat': ['global'],
        'lstm_iters': 15,
        'lstm_layers': 2,
        'num_heads': 3,
        'feat_drop': 0.2,
        'attn_drop': 0.1,
        'residual': False,
        'hidden_size': 64,
        'ntasks': 2,
        'num_heads_gat': 1,
        'dropout_feat_gat': 0.1,
        'dropout_attn_gat': 0.1,
        'hidden_size_gat': 8,
        'residual_gat': True,
        'classifier': False,
        'batch_norm': True,
        "restore": False,
        "fc_hidden_size_1": 256,
        'pooling_ntypes': ['atom', 'bond', 'global'],
        'pooling_ntypes_direct': ['global']
    }


    qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    qtaim_model_dict['atom_feature_size'] = 33
    qtaim_model_dict['bond_feature_size'] = 28
    
    
    non_qtaim_model_bl_dict = deepcopy(qtaim_model_bl_dict)
    non_qtaim_model_bl_dict['atom_feature_size'] = 12
    non_qtaim_model_bl_dict['bond_feature_size'] = 7
    
    non_qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    non_qtaim_model_dict['atom_feature_size'] = 12
    non_qtaim_model_dict['bond_feature_size'] = 6

    

    qtaim_keys = {
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
        "extra_feat_atom_energy_density",
        "extra_feat_atom_density_beta",
        "extra_feat_atom_density_alpha",
        "extra_feat_atom_spin_density",
        "extra_feat_atom_lol",
    ],
    "bond": [
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
        "extra_feat_bond_energy_density",
        "extra_feat_bond_density_beta",
        "extra_feat_bond_density_alpha",
        "extra_feat_bond_spin_density",
        "extra_feat_bond_lol",
    ],
        "global":  ["f1-CC2", "f2-CC2"]
    }

    qtaim_keys_bl = {
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
        "extra_feat_atom_energy_density",
        "extra_feat_atom_density_beta",
        "extra_feat_atom_density_alpha",
        "extra_feat_atom_spin_density",
        "extra_feat_atom_lol",
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
        "extra_feat_bond_energy_density",
        "extra_feat_bond_density_beta",
        "extra_feat_bond_density_alpha",
        "extra_feat_bond_spin_density",
        "extra_feat_bond_lol",
    ],
        "global":  ["f1-CC2", "f2-CC2"]
    }

    base_dict_bl = {
        "atom": [],
        "bond": ["bond_length"],
        "global": ["f1-CC2", "f2-CC2"],
    }

    base_dict = {
        "atom": [],
        "bond": [],
        "global": ["f1-CC2", "f2-CC2"],
    }


    dict_datasets={
        "qtaim":{}, 
        "qtaim_bl":{},
        "non_qtaim":{},
        "non_qtaim_bl":{}
    }

    for key, libe_loc in loc_dict.items():
        
        dataset_train_base = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[4, 5, 6, 7],
            element_set={'N', 'C', 'H', 'F', 'O'},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict,
            target_list=["f1-CC2", "f2-CC2"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_qtaim = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[4, 5, 6, 7],
            element_set={'N', 'C', 'H', 'F', 'O'},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys,
            target_list=["f1-CC2", "f2-CC2"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_qtaim_bl = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[4, 5, 6, 7],
            element_set={'N', 'C', 'H', 'F', 'O'},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys_bl,
            target_list=["f1-CC2", "f2-CC2"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_base_bl = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[4, 5, 6, 7],
            element_set={'N', 'C', 'H', 'F', 'O'},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict_bl,
            target_list=["f1-CC2", "f2-CC2"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )
        print(dataset_train_qtaim_bl.feature_size())
        print(dataset_train_qtaim.feature_size())
        print(dataset_train_base_bl.feature_size())
        print(dataset_train_base.feature_size())

        dict_datasets["qtaim"][key] = dataset_train_qtaim
        dict_datasets["non_qtaim"][key] = dataset_train_base
        dict_datasets["qtaim_bl"][key] = dataset_train_qtaim_bl
        dict_datasets["non_qtaim_bl"][key] = dataset_train_base_bl

    
    dict_keys = {}
    model_dict = {}

    dict_keys["non_qtaim"] = base_dict
    model_dict["non_qtaim"] = non_qtaim_model_dict
    
    dict_keys["non_qtaim_bl"] = base_dict_bl
    model_dict["non_qtaim_bl"] = non_qtaim_model_bl_dict

    dict_keys["qtaim"] = qtaim_keys
    model_dict["qtaim"] = qtaim_model_dict
    
    dict_keys["qtaim_bl"] = qtaim_keys_bl
    model_dict["qtaim_bl"] = qtaim_model_bl_dict

    return model_dict, dict_keys, dict_datasets




# main 
def main(): 
    results_dict = {}
    loc_dict = {
        "10": "../../../datasets/1205_splits/train_qm8_qtaim_1205_labelled_10.pkl",
        "100": "../../../datasets/1205_splits/train_qm8_qtaim_1205_labelled_100.pkl",
        "1000": "../../../datasets/1205_splits/train_qm8_qtaim_1205_labelled_1000.pkl",
        "10000": "../../../datasets/1205_splits/train_qm8_qtaim_1205_labelled_10000.pkl",
        "all": "../../../datasets/1205_splits/train_qm8_qtaim_1205_labelled.pkl",
        "test": "../../../datasets/1205_splits/test_qm8_qtaim_1205_labelled.pkl"
    }
    model_dict, dict_keys, dict_datasets = get_datasets_qm8(loc_dict)


    for keys in dict_datasets.keys():

        model_temp = load_graph_level_model_from_config(model_dict[keys])
        test_dataset = dict_datasets[keys]["test"]
        
        for name in dict_datasets[keys].keys():
            if name != "test":
                dataloader_train = DataLoaderMoleculeGraphTask(dict_datasets[keys][name], batch_size=256, shuffle=True, num_workers=0)
                dataloader_test = DataLoaderMoleculeGraphTask(test_dataset, batch_size=len(test_dataset.graphs), shuffle=False, num_workers=0)
                early_stopping_callback = EarlyStopping(
                    monitor="val_mae", min_delta=0.00, patience=100, verbose=False, mode="min"
                )
                lr_monitor = LearningRateMonitor(logging_interval="step")

                trainer = pl.Trainer(
                    max_epochs=1000,
                    accelerator="gpu",
                    gradient_clip_val=2.0,
                    devices=1,
                    accumulate_grad_batches=1,
                    enable_progress_bar=True,
                    callbacks=[
                        early_stopping_callback,
                        lr_monitor,
                    ],
                    enable_checkpointing=True,
                    strategy="auto",
                    #default_root_dir=model_save_string,
                    default_root_dir="./test_oscillator/",
                    precision="bf16-mixed",
                )
                
                trainer.fit(model_temp, dataloader_train)
                trainer.save_checkpoint(f"./libe_learning_test_oscillator/{keys}_{name}.ckpt")
                
                batch_graph, batched_labels = next(iter(dataloader_test))
                r2_metrics, mae_metrics, mse_metrics, _, _ = model_temp.evaluate_manually(
                    batch_graph,
                    batched_labels,
                    scaler_list=test_dataset.label_scalers,
                )
                # convert to numpy
                r2_metrics = r2_metrics.cpu().numpy()
                mae_metrics = mae_metrics.cpu().numpy()
                mse_metrics = mse_metrics.cpu().numpy()

                # convert to list 
                r2_metrics = r2_metrics.tolist()
                mae_metrics = mae_metrics.tolist()
                mse_metrics = mse_metrics.tolist()
                
                results_dict[f"{keys}_{name}"] = {
                    "r2_metrics": r2_metrics,
                    "mae_metrics": mae_metrics,
                    "mse_metrics": mse_metrics,
                }


    print(results_dict)
    # save results dict
    json.dump(results_dict, open("./qm8_oscillator_learning_results_dict.json", "w"), indent=4)

main()
