from copy import deepcopy
from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset


def get_datasets_qm9(loc_dict):

    qtaim_model_bl_dict = {
        "atom_feature_size": 31,
        "bond_feature_size": 27,
        "global_feature_size": 3,
        "conv_fn": "ResidualBlock",
        "target_dict": {"global": ["homo", "lumo", "gap"]},
        "dropout": 0.1,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "aggregate": "sum",
        "n_conv_layers": 10,
        "lr": 0.012034891741807852,
        "weight_decay": 5e-05,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.5,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 2,
        "embedding_size": 16,
        "fc_layer_size": [1024, 1024],
        "fc_dropout": 0,
        "fc_batch_norm": True,
        "n_fc_layers": 2,
        "global_pooling_fn": "GlobalAttentionPoolingThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 5,
        "lstm_layers": 3,
        "num_heads": 3,
        "feat_drop": 0.1,
        "attn_drop": 0,
        "residual": False,
        "hidden_size": 30,
        "ntasks": 1,
        "num_heads_gat": 3,
        "dropout_feat_gat": 0.2,
        "dropout_attn_gat": 0.1,
        "hidden_size": 30,
        "residual_gat": True,
        "batch_norm": True,
        "shape_fc": "flat",
        "fc_hidden_size_1": 1024,
        "fc_num_layers": 2,
        "classifier": False,
        "restore": False,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "bond_key": "bonds",
        "map_key": "extra_feat_bond_indices_qtaim",
    }

    qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    qtaim_model_dict["bond_feature_size"] = 26
    qtaim_model_dict["atom_feature_size"] = 31

    non_qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    non_qtaim_model_dict["atom_feature_size"] = 13
    non_qtaim_model_dict["bond_feature_size"] = 7

    non_qtaim_model_bl_dict = deepcopy(qtaim_model_bl_dict)
    non_qtaim_model_bl_dict["atom_feature_size"] = 13
    non_qtaim_model_bl_dict["bond_feature_size"] = 8

    qtaim_keys = {
        "atom": [
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
            "extra_feat_bond_lol",
        ],
        "global": ["homo", "lumo", "gap"],
    }

    qtaim_keys_bl = {
        "atom": [
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
            "extra_feat_atom_lol",
        ],
        "bond": [
            "extra_feat_bond_Lagrangian_K",
            "bond_length",
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
            "extra_feat_bond_lol",
        ],
        "global": ["homo", "lumo", "gap"],
    }

    base_dict_bl = {
        "atom": [],
        "bond": ["bond_length"],
        "global": ["homo", "lumo", "gap"],
    }

    base_dict = {
        "atom": [],
        "bond": [],
        "global": ["homo", "lumo", "gap"],
    }

    dict_datasets = {"qtaim": {}, "qtaim_bl": {}, "non_qtaim": {}, "non_qtaim_bl": {}}

    for key, libe_loc in loc_dict.items():

        dataset_train_base = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict,
            target_list=["homo", "lumo", "gap"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_qtaim = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys,
            target_list=["homo", "lumo", "gap"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_qtaim_bl = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys_bl,
            target_list=["homo", "lumo", "gap"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_base_bl = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict_bl,
            target_list=["homo", "lumo", "gap"],
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


def get_datasets_qm8(loc_dict):

    qtaim_model_bl_dict = {
        "atom_feature_size": 33,
        "bond_feature_size": 29,
        "global_feature_size": 3,
        "conv_fn": "ResidualBlock",
        "target_dict": {"global": ["E1-CC2", "E2-CC2"]},
        "dropout": 0.2,
        "batch_norm_tf": False,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "aggregate": "sum",
        "n_conv_layers": 4,
        "lr": 0.044675426899321025,
        "weight_decay": 0.00001,
        "lr_plateau_patience": 10,
        "lr_scale_factor": 0.5,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 2,
        "embedding_size": 80,
        "fc_layer_size": [512, 512],
        "shape_fc": "flat",
        "bond_key": "bonds",
        "map_key": "extra_feat_bond_indices_qtaim",
        "fc_dropout": 0.1,
        "fc_batch_norm": True,
        "fc_num_layers": 2,
        "n_fc_layers": 2,
        "global_pooling_fn": "MeanPoolingThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 15,
        "lstm_layers": 2,
        "num_heads": 3,
        "feat_drop": 0.2,
        "attn_drop": 0.1,
        "residual": False,
        "hidden_size": 64,
        "ntasks": 2,
        "num_heads_gat": 1,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "hidden_size": 8,
        "residual_gat": True,
        "classifier": False,
        "batch_norm": True,
        "restore": False,
        "fc_hidden_size_1": 256,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
    }

    qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    qtaim_model_dict["atom_feature_size"] = 33
    qtaim_model_dict["bond_feature_size"] = 28

    non_qtaim_model_bl_dict = deepcopy(qtaim_model_bl_dict)
    non_qtaim_model_bl_dict["atom_feature_size"] = 12
    non_qtaim_model_bl_dict["bond_feature_size"] = 7

    non_qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    non_qtaim_model_dict["atom_feature_size"] = 12
    non_qtaim_model_dict["bond_feature_size"] = 6

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
        "global": ["E1-CC2", "E2-CC2"],
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
        "global": ["E1-CC2", "E2-CC2"],
    }

    base_dict_bl = {
        "atom": [],
        "bond": ["bond_length"],
        "global": ["E1-CC2", "E2-CC2"],
    }

    base_dict = {
        "atom": [],
        "bond": [],
        "global": ["E1-CC2", "E2-CC2"],
    }

    dict_datasets = {"qtaim": {}, "qtaim_bl": {}, "non_qtaim": {}, "non_qtaim_bl": {}}

    for key, libe_loc in loc_dict.items():

        dataset_train_base = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[4, 5, 6, 7],
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict,
            target_list=["E1-CC2", "E2-CC2"],
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
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys,
            target_list=["E1-CC2", "E2-CC2"],
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
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys_bl,
            target_list=["E1-CC2", "E2-CC2"],
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
            element_set={"N", "C", "H", "F", "O"},
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict_bl,
            target_list=["E1-CC2", "E2-CC2"],
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


def get_datasets_libe(loc_dict):

    qtaim_model_bl_dict = {
        "atom_feature_size": 33,
        "bond_feature_size": 27,
        "global_feature_size": 9,
        "conv_fn": "ResidualBlock",
        "target_dict": {"global": ["corrected_E"]},
        "dropout": 0.2,
        "bond_key": "bonds",
        "map_key": "extra_feat_bond_indices_qtaim",
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "aggregate": "sum",
        "n_conv_layers": 6,
        "lr": 0.044675426899321025,
        "weight_decay": 5e-05,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.5,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 2,
        "embedding_size": 100,
        "fc_layer_size": [512, 512, 512],
        "fc_dropout": 0.1,
        "fc_batch_norm": True,
        "n_fc_layers": 3,
        "global_pooling_fn": "WeightAndSumThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 15,
        "lstm_layers": 2,
        "num_heads": 2,
        "feat_drop": 0.1,
        "attn_drop": 0.1,
        "residual": True,
        "hidden_size": 50,
        "ntasks": 1,
        "num_heads_gat": 2,
        "dropout_feat_gat": 0.2,
        "dropout_attn_gat": 0.1,
        "hidden_size": 50,
        "residual_gat": True,
        "shape_fc": "flat",
        "classifier": False,
        "fc_num_layers": 3,
        "batch_norm": True,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 512,
        "restore": False,
        "initializer": "kaiming",
    }

    non_qtaim_model_bl_dict = {
        "atom_feature_size": 16,
        "bond_feature_size": 8,
        "global_feature_size": 9,
        "conv_fn": "GraphConvDropoutBatch",
        "target_dict": {"global": ["shifted_rrho_ev_free_energy"]},
        "dropout": 0.2,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "aggregate": "sum",
        "n_conv_layers": 5,
        "lr": 0.01885852849843154,
        "weight_decay": 1e-05,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.5,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 2,
        "embedding_size": 50,
        "fc_layer_size": [1024, 512, 256],
        "fc_dropout": 0.1,
        "fc_batch_norm": True,
        "n_fc_layers": 3,
        "global_pooling_fn": "WeightAndSumThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 9,
        "lstm_layers": 2,
        "num_heads": 3,
        "feat_drop": 0.1,
        "attn_drop": 0.1,
        "residual": False,
        "hidden_size": 10,
        "ntasks": 1,
        "num_heads_gat": 3,
        "dropout_feat_gat": 0.1,
        "dropout_attn_gat": 0.1,
        "hidden_size": 10,
        "residual_gat": False,
        "batch_norm": True,
        "shape_fc": "cone",
        "classifier": False,
        "fc_num_layers": 3,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 512,
        "restore": False,
        "initializer": "kaiming",
    }

    qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    qtaim_model_dict["atom_feature_size"] = 33
    qtaim_model_dict["bond_feature_size"] = 26
    qtaim_model_dict["global_feature_size"] = 9

    non_qtaim_model_dict = deepcopy(non_qtaim_model_bl_dict)
    non_qtaim_model_dict["atom_feature_size"] = 16
    non_qtaim_model_dict["bond_feature_size"] = 6
    non_qtaim_model_dict["global_feature_size"] = 9

    qtaim_keys = {
        "atom": [
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
            "extra_feat_bond_lol",
        ],
        "global": ["corrected_E", "charge", "spin"],
    }

    qtaim_keys_bl = {
        "atom": [
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
            "extra_feat_bond_lol",
        ],
        "global": ["corrected_E", "charge", "spin"],
    }

    base_dict_bl = {
        "atom": [],
        "bond": ["bond_length"],
        "global": ["corrected_E", "charge", "spin"],
    }

    base_dict = {
        "atom": [],
        "bond": [],
        "global": ["corrected_E", "charge", "spin"],
    }

    dict_datasets = {"qtaim": {}, "qtaim_bl": {}, "non_qtaim": {}, "non_qtaim_bl": {}}

    for key, libe_loc in loc_dict.items():

        dataset_train_base = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"S", "Li", "O", "N", "H", "C", "F", "P"},
            allowed_charges=[-1, 0, 1],
            allowed_spins=[1, 2, 3],
            self_loop=True,
            extra_keys=base_dict,
            target_list=["corrected_E"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_qtaim = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"S", "Li", "O", "N", "H", "C", "F", "P"},
            allowed_charges=[-1, 0, 1],
            allowed_spins=[1, 2, 3],
            self_loop=True,
            extra_keys=qtaim_keys,
            target_list=["corrected_E"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_qtaim_bl = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"S", "Li", "O", "N", "H", "C", "F", "P"},
            allowed_charges=[-1, 0, 1],
            allowed_spins=[1, 2, 3],
            self_loop=True,
            extra_keys=qtaim_keys_bl,
            target_list=["corrected_E"],
            extra_dataset_info={},
            debug=False,
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )

        dataset_train_base_bl = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[3, 4, 5, 6, 7],
            element_set={"S", "Li", "O", "N", "H", "C", "F", "P"},
            allowed_charges=[-1, 0, 1],
            allowed_spins=[1, 2, 3],
            self_loop=True,
            extra_keys=base_dict_bl,
            target_list=["corrected_E"],
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


def get_datasets_tmqm_low(loc_dict):

    qtaim_model_bl_dict = {
        "atom_feature_size": 66,
        "bond_feature_size": 23,
        "global_feature_size": 6,
        "conv_fn": "ResidualBlock",
        "target_dict": {"global": ["corrected_E"]},
        "dropout": 0.1,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bond_key": "bonds",
        "map_key": "extra_feat_bond_indices_qtaim",
        "bias": True,
        "norm": "both",
        "edge_dropout": 0.1,
        "aggregate": "sum",
        "n_conv_layers": 10,
        "lr": 0.04031,
        "weight_decay": 2e-05,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.75,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mae",
        "resid_n_graph_convs": 4,
        "embedding_size": 50,
        "fc_layer_size": [512, 512, 512],
        "fc_dropout": 0.1,
        "fc_batch_norm": True,
        "n_fc_layers": 3,
        "global_pooling_fn": "WeightAndSumThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 9,
        "lstm_layers": 3,
        "num_heads": 2,
        "feat_drop": 0.2,
        "attn_drop": 0.2,
        "residual": False,
        "hidden_size": 50,
        "ntasks": 1,
        "num_heads_gat": 2,
        "dropout_feat_gat": 0.2,
        "dropout_attn_gat": 0.2,
        "hidden_size": 50,
        "residual_gat": False,
        "shape_fc": "flat",
        "classifier": False,
        "fc_num_layers": 3,
        "batch_norm": True,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 256,
        "restore": False,
        "initializer": "xavier",
    }

    non_qtaim_model_bl_dict = {
        "atom_feature_size": 51,
        "bond_feature_size": 7,
        "global_feature_size": 6,
        "conv_fn": "ResidualBlock",
        "target_dict": {"global": ["corrected_E"]},
        "dropout": 0.2,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "edge_dropout": 0.1,
        "aggregate": "sum",
        "n_conv_layers": 10,
        "lr": 0.04031,
        "weight_decay": 0.00001,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.75,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mae",
        "resid_n_graph_convs": 3,
        "embedding_size": 50,
        "fc_layer_size": [512, 512, 512],
        "fc_dropout": 0.0,
        "fc_batch_norm": True,
        "n_fc_layers": 2,
        "global_pooling_fn": "GlobalAttentionPoolingThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 9,
        "lstm_layers": 3,
        "num_heads": 2,
        "feat_drop": 0.0,
        "attn_drop": 0.1,
        "residual": True,
        "hidden_size": 50,
        "ntasks": 1,
        "num_heads_gat": 2,
        "dropout_feat_gat": 0.0,
        "dropout_attn_gat": 0.0,
        "hidden_size": 50,
        "residual_gat": True,
        "shape_fc": "flat",
        "classifier": False,
        "fc_num_layers": 2,
        "batch_norm": True,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 256,
        "restore": False,
        "initializer": "xavier",
    }

    qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    qtaim_model_dict["atom_feature_size"] = 66
    qtaim_model_dict["bond_feature_size"] = 22
    qtaim_model_dict["global_feature_size"] = 6

    non_qtaim_model_dict = deepcopy(non_qtaim_model_bl_dict)
    non_qtaim_model_dict["atom_feature_size"] = 51
    non_qtaim_model_dict["bond_feature_size"] = 6
    non_qtaim_model_dict["global_feature_size"] = 6

    qtaim_keys = {
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
            "extra_feat_atom_eig_hess",
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
        ],
        "global": ["corrected_E", "charge"],
    }

    qtaim_keys_bl = {
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
            "extra_feat_atom_eig_hess",
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
            "bond_length",
        ],
        "global": ["corrected_E", "charge"],
    }

    base_dict_bl = {
        "atom": [],
        "bond": ["bond_length"],
        "global": ["corrected_E", "charge"],
    }

    base_dict = {
        "atom": [],
        "bond": [],
        "global": ["corrected_E", "charge"],
    }

    dict_datasets = {"qtaim": {}, "qtaim_bl": {}, "non_qtaim": {}, "non_qtaim_bl": {}}
    elem_set = [
        "Ag",
        "As",
        "Au",
        "B",
        "Br",
        "C",
        "Cd",
        "Cl",
        "Co",
        "Cr",
        "Cu",
        "F",
        "Fe",
        "H",
        "Hf",
        "Hg",
        "I",
        "Ir",
        "La",
        "Mn",
        "Mo",
        "N",
        "Nb",
        "Ni",
        "O",
        "Os",
        "P",
        "Pd",
        "Pt",
        "Re",
        "Rh",
        "Ru",
        "S",
        "Sc",
        "Se",
        "Si",
        "Ta",
        "Tc",
        "Ti",
        "V",
        "W",
        "Y",
        "Zn",
        "Zr",
    ]

    for key, libe_loc in loc_dict.items():

        dataset_train_base = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[4, 5, 6, 7],
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict,
            target_list=["corrected_E"],
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
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys,
            target_list=["corrected_E"],
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
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys_bl,
            target_list=["corrected_E"],
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
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict_bl,
            target_list=["corrected_E"],
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


def get_datasets_tmqm_high(loc_dict):

    qtaim_model_bl_dict = {
        "atom_feature_size": 66,
        "bond_feature_size": 23,
        "global_feature_size": 6,
        "conv_fn": "ResidualBlock",
        "target_dict": {"global": ["corrected_E"]},
        "dropout": 0.1,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bond_key": "bonds",
        "map_key": "extra_feat_bond_indices_qtaim",
        "bias": True,
        "norm": "both",
        "aggregate": "sum",
        "n_conv_layers": 8,
        "lr": 0.016,
        "weight_decay": 2e-05,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.5,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 4,
        "embedding_size": 50,
        "fc_layer_size": [512, 512, 512],
        "fc_dropout": 0.1,
        "fc_batch_norm": True,
        "n_fc_layers": 3,
        "global_pooling_fn": "WeightAndSumThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 8,
        "lstm_layers": 2,
        "num_heads": 2,
        "feat_drop": 0.2,
        "attn_drop": 0.1,
        "residual": False,
        "hidden_size": 50,
        "ntasks": 1,
        "num_heads_gat": 2,
        "dropout_feat_gat": 0.2,
        "dropout_attn_gat": 0.1,
        "hidden_size": 50,
        "residual_gat": False,
        "shape_fc": "cone",
        "classifier": False,
        "fc_num_layers": 3,
        "batch_norm": False,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 512,
        "restore": False,
        "initializer": "xavier",
    }

    non_qtaim_model_bl_dict = {
        "atom_feature_size": 51,
        "bond_feature_size": 7,
        "global_feature_size": 6,
        "conv_fn": "ResidualBlock",
        "target_dict": {"global": ["shifted_rrho_ev_free_energy"]},
        "dropout": 0.2,
        "batch_norm_tf": True,
        "activation": "ReLU",
        "bias": True,
        "norm": "both",
        "aggregate": "sum",
        "n_conv_layers": 5,
        "lr": 0.054,
        "weight_decay": 1e-05,
        "lr_plateau_patience": 25,
        "lr_scale_factor": 0.5,
        "scheduler_name": "reduce_on_plateau",
        "loss_fn": "mse",
        "resid_n_graph_convs": 2,
        "embedding_size": 50,
        "fc_layer_size": [1024, 512, 256],
        "fc_dropout": 0.0,
        "fc_batch_norm": True,
        "n_fc_layers": 3,
        "global_pooling_fn": "GlobalAttentionPoolingThenCat",
        "ntypes_pool": ["atom", "bond", "global"],
        "ntypes_pool_direct_cat": ["global"],
        "lstm_iters": 15,
        "lstm_layers": 2,
        "num_heads": 2,
        "feat_drop": 0.2,
        "attn_drop": 0.1,
        "residual": False,
        "hidden_size": 30,
        "ntasks": 1,
        "num_heads_gat": 2,
        "dropout_feat_gat": 0.2,
        "dropout_attn_gat": 0.2,
        "hidden_size": 30,
        "residual_gat": True,
        "batch_norm": True,
        "shape_fc": "flat",
        "classifier": False,
        "fc_num_layers": 3,
        "pooling_ntypes": ["atom", "bond", "global"],
        "pooling_ntypes_direct": ["global"],
        "fc_hidden_size_1": 512,
        "restore": False,
        "initializer": "kaiming",
    }

    qtaim_model_dict = deepcopy(qtaim_model_bl_dict)
    qtaim_model_dict["atom_feature_size"] = 66
    qtaim_model_dict["bond_feature_size"] = 22
    qtaim_model_dict["global_feature_size"] = 6

    non_qtaim_model_dict = deepcopy(non_qtaim_model_bl_dict)
    non_qtaim_model_dict["atom_feature_size"] = 51
    non_qtaim_model_dict["bond_feature_size"] = 6
    non_qtaim_model_dict["global_feature_size"] = 6

    qtaim_keys = {
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
            "extra_feat_atom_eig_hess",
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
        ],
        "global": ["corrected_E", "charge"],
    }

    qtaim_keys_bl = {
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
            "extra_feat_atom_eig_hess",
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
            "bond_length",
        ],
        "global": ["corrected_E", "charge"],
    }

    base_dict_bl = {
        "atom": [],
        "bond": ["bond_length"],
        "global": ["corrected_E", "charge"],
    }

    base_dict = {
        "atom": [],
        "bond": [],
        "global": ["corrected_E", "charge"],
    }

    elem_set = [
        "Ag",
        "As",
        "Au",
        "B",
        "Br",
        "C",
        "Cd",
        "Cl",
        "Co",
        "Cr",
        "Cu",
        "F",
        "Fe",
        "H",
        "Hf",
        "Hg",
        "I",
        "Ir",
        "La",
        "Mn",
        "Mo",
        "N",
        "Nb",
        "Ni",
        "O",
        "Os",
        "P",
        "Pd",
        "Pt",
        "Re",
        "Rh",
        "Ru",
        "S",
        "Sc",
        "Se",
        "Si",
        "Ta",
        "Tc",
        "Ti",
        "V",
        "W",
        "Y",
        "Zn",
        "Zr",
    ]

    dict_datasets = {"qtaim": {}, "qtaim_bl": {}, "non_qtaim": {}, "non_qtaim_bl": {}}

    for key, libe_loc in loc_dict.items():

        dataset_train_base = HeteroGraphGraphLabelDataset(
            file=libe_loc,
            allowed_ring_size=[4, 5, 6, 7],
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict,
            target_list=["corrected_E"],
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
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys,
            target_list=["corrected_E"],
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
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=qtaim_keys_bl,
            target_list=["corrected_E"],
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
            element_set=elem_set,
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            extra_keys=base_dict_bl,
            target_list=["corrected_E"],
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
