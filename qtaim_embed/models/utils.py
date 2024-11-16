import torch
import numpy as np
import pytorch_lightning as pl
import pandas as pd
from qtaim_embed.models.graph_level.base_gcn import GCNGraphPred
from qtaim_embed.models.node_level.base_gcn import GCNNodePred
from qtaim_embed.models.graph_level.base_gcn_classifier import GCNGraphPredClassifier
from qtaim_embed.models.link_pred.link_model import GCNLinkPred
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask
from qtaim_embed.models.initializers import xavier_init, kaiming_init, equi_var_init


def load_graph_level_model_from_config(config):
    """
    returns model and optimizer from dict of parameters

    Args:
        dict_train(dict): dictionary
    Returns:
        model (pytorch model): model to train
        optimizer (pytorch optimizer obj): optimizer
    """
    if config["restore"]:
        print(":::RESTORING MODEL FROM EXISTING FILE:::")

        if config["restore_path"] != None:
            try:
                try:
                    model = GCNGraphPred.load_from_checkpoint(
                        checkpoint_path=config["restore_path"]
                    )
                    # model.to(device)
                    print(":::MODEL LOADED:::")
                    return model
                except:
                    model = GCNGraphPredClassifier.load_from_checkpoint(
                        checkpoint_path=config["restore_path"]
                    )
                    # model.to(device)
                    print(":::MODEL LOADED:::")
                    return model
            except:
                pass
            print(":::NO MODEL FOUND LOADING FRESH MODEL:::")
        else:
            if load_dir == None:
                load_dir = "./"

            try:
                model = GCNGraphPred.load_from_checkpoint(
                    checkpoint_path=load_dir + "/last.ckpt"
                )
                # model.to(device)
                print(":::MODEL LOADED:::")
                return model

            except:
                print(":::NO MODEL FOUND LOADING FRESH MODEL:::")

    shape_fc = config["shape_fc"]
    base_fc = config["fc_hidden_size_1"]

    if shape_fc == "flat":
        fc_layers = [base_fc for i in range(config["fc_num_layers"])]
    else:
        fc_layers = [int(base_fc / (2**i)) for i in range(config["fc_num_layers"])]
    if config["classifier"]:
        print(":::CLASSIFIER MODEL:::")
        model = GCNGraphPredClassifier(
            atom_input_size=config["atom_feature_size"],
            bond_input_size=config["bond_feature_size"],
            global_input_size=config["global_feature_size"],
            n_conv_layers=config["n_conv_layers"],
            resid_n_graph_convs=config["resid_n_graph_convs"],
            target_dict=config["target_dict"],
            conv_fn=config["conv_fn"],
            global_pooling=config["global_pooling_fn"],
            dropout=config["dropout"],
            batch_norm=config["batch_norm"],
            activation=config["activation"],
            bias=config["bias"],
            norm=config["norm"],
            aggregate=config["aggregate"],
            lr=config["lr"],
            scheduler_name="reduce_on_plateau",
            weight_decay=config["weight_decay"],
            lr_plateau_patience=config["lr_plateau_patience"],
            lr_scale_factor=config["lr_scale_factor"],
            loss_fn="cross_entropy",
            embedding_size=config["embedding_size"],
            fc_layer_size=fc_layers,
            fc_dropout=config["fc_dropout"],
            fc_batch_norm=config["fc_batch_norm"],
            lstm_iters=config["lstm_iters"],
            lstm_layers=config["lstm_layers"],
            output_dims=2,
            pooling_ntypes=["atom", "bond", "global"],
            pooling_ntypes_direct=["global"],
            num_heads_gat=config["num_heads_gat"],
            dropout_feat_gat=config["dropout_feat_gat"],
            dropout_attn_gat=config["dropout_attn_gat"],
            hidden_size=config["hidden_size"],
            residual_gat=config["residual_gat"],
        )
    else:
        print(":::REGRESSION MODEL:::")
        model = GCNGraphPred(
            atom_input_size=config["atom_feature_size"],
            bond_input_size=config["bond_feature_size"],
            global_input_size=config["global_feature_size"],
            n_conv_layers=config["n_conv_layers"],
            resid_n_graph_convs=config["resid_n_graph_convs"],
            target_dict=config["target_dict"],
            conv_fn=config["conv_fn"],
            global_pooling=config["global_pooling_fn"],
            dropout=config["dropout"],
            batch_norm=config["batch_norm"],
            activation=config["activation"],
            bias=config["bias"],
            norm=config["norm"],
            aggregate=config["aggregate"],
            lr=config["lr"],
            scheduler_name="reduce_on_plateau",
            weight_decay=config["weight_decay"],
            lr_plateau_patience=config["lr_plateau_patience"],
            lr_scale_factor=config["lr_scale_factor"],
            loss_fn=config["loss_fn"],
            embedding_size=config["embedding_size"],
            fc_layer_size=fc_layers,
            fc_dropout=config["fc_dropout"],
            fc_batch_norm=config["fc_batch_norm"],
            lstm_iters=config["lstm_iters"],
            lstm_layers=config["lstm_layers"],
            # output_dims=config["output_dims"],
            pooling_ntypes=["atom", "bond", "global"],
            pooling_ntypes_direct=["global"],
            num_heads_gat=config["num_heads_gat"],
            dropout_feat_gat=config["dropout_feat_gat"],
            dropout_attn_gat=config["dropout_attn_gat"],
            hidden_size=config["hidden_size"],
            residual_gat=config["residual_gat"],
        )
    # model.to(device)

    if config["initializer"] == "kaiming":
        print(":::USING KAIMING INITIALIZER:::")
        kaiming_init(model)

    elif config["initializer"] == "xavier":
        print(":::USING XAVIER INITIALIZER:::")
        xavier_init(model)

    elif config["initializer"] == "equi_var":
        print(":::USING EQUIVARIANCE INITIALIZER:::")
        equi_var_init(model)

    else:
        print(":::NO INITIALIZER USED:::")

    return model


def load_node_level_model_from_config(config):
    """
    returns model and optimizer from dict of parameters

    Args:
        dict_train(dict): dictionary
    Returns:
        model (pytorch model): model to train
        optimizer (pytorch optimizer obj): optimizer
    """
    if config["restore"]:
        print(":::RESTORING MODEL FROM EXISTING FILE:::")

        if config["restore_path"] != None:
            try:
                model = GCNNodePred.load_from_checkpoint(
                    checkpoint_path=config["restore_path"]
                )
                # model.to(device)
                print(":::MODEL LOADED:::")
                return model
            except:
                pass
            print(":::NO MODEL FOUND LOADING FRESH MODEL:::")
        else:
            if load_dir == None:
                load_dir = "./"

            try:
                model = GCNNodePred.load_from_checkpoint(
                    checkpoint_path=load_dir + "/last.ckpt"
                )
                # model.to(device)
                print(":::MODEL LOADED:::")
                return model

            except:
                print(":::NO MODEL FOUND LOADING FRESH MODEL:::")

    print(config)
    print(":::NODE-LEVEL REGRESSION MODEL:::")
    model = GCNNodePred(
        atom_input_size=config["atom_feature_size"],
        bond_input_size=config["bond_feature_size"],
        global_input_size=config["global_feature_size"],
        n_conv_layers=config["n_conv_layers"],
        resid_n_graph_convs=config["resid_n_graph_convs"],
        target_dict=config["target_dict"],
        conv_fn=config["conv_fn"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
        activation=config["activation"],
        bias=config["bias"],
        norm=config["norm"],
        aggregate=config["aggregate"],
        lr=config["lr"],
        scheduler_name="reduce_on_plateau",
        weight_decay=config["weight_decay"],
        lr_plateau_patience=config["lr_plateau_patience"],
        lr_scale_factor=config["lr_scale_factor"],
        loss_fn=config["loss_fn"],
        embedding_size=config["embedding_size"],
        num_heads_gat=config["num_heads_gat"],
        dropout_feat_gat=config["dropout_feat_gat"],
        dropout_attn_gat=config["dropout_attn_gat"],
        hidden_size=config["hidden_size"],
        residual_gat=config["residual_gat"],
    )
    # model.to(device)

    if config["initializer"] == "kaiming":
        print(":::USING KAIMING INITIALIZER:::")
        kaiming_init(model)

    elif config["initializer"] == "xavier":
        print(":::USING XAVIER INITIALIZER:::")
        xavier_init(model)

    elif config["initializer"] == "equi_var":
        print(":::USING EQUIVARIANCE INITIALIZER:::")
        equi_var_init(model)

    else:
        print(":::NO INITIALIZER USED:::")

    return model


def load_link_model_from_config(config):
    """
    returns model and optimizer from dict of parameters

    Args:
        dict_train(dict): dictionary
    Returns:
        model (pytorch model): model to train
        optimizer (pytorch optimizer obj): optimizer
    """
    if config["restore"]:
        print(":::RESTORING MODEL FROM EXISTING FILE:::")

        if config["restore_path"] != None:
            try:
                model = GCNLinkPred.load_from_checkpoint(
                    checkpoint_path=config["restore_path"]
                )
                # model.to(device)
                print(":::MODEL LOADED:::")
                return model
            except:
                pass
            print(":::NO MODEL FOUND LOADING FRESH MODEL:::")
        else:
            if load_dir == None:
                load_dir = "./"

            try:
                model = GCNLinkPred.load_from_checkpoint(
                    checkpoint_path=load_dir + "/last.ckpt"
                )
                # model.to(device)
                print(":::MODEL LOADED:::")
                return model

            except:
                print(":::NO MODEL FOUND LOADING FRESH MODEL:::")

    print(config)
    print(":::LINK-PRED MODEL:::")
    model = GCNLinkPred(
        input_size=config["input_size"],
        n_conv_layers=config["n_conv_layers"],
        target_dict=config["target_dict"],
        conv_fn=config["conv_fn"],
        resid_n_graph_convs=config["resid_n_graph_convs"],
        num_heads_gat=config["num_heads_gat"],
        dropout_feat_gat=config["dropout_feat_gat"],
        dropout_attn_gat=config["dropout_attn_gat"],
        residual_gat=config["residual_gat"],
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        batch_norm=config["batch_norm"],
        activation=config["activation"],
        bias=config["bias"],
        norm=config["norm"],
        lr=config["lr"],
        scheduler_name="reduce_on_plateau",
        weight_decay=config["weight_decay"],
        lr_plateau_patience=config["lr_plateau_patience"],
        lr_scale_factor=config["lr_scale_factor"],
        loss_fn=config["loss_fn"],
        embedding_size=config["embedding_size"],
        predictor=config["predictor"],
        predictor_param_dict=config["predictor_param_dict"],
        aggregator_type=config["aggregator_type"],
        
    )
    # model.to(device)

    if config["initializer"] == "kaiming":
        print(":::USING KAIMING INITIALIZER:::")
        kaiming_init(model)

    elif config["initializer"] == "xavier":
        print(":::USING XAVIER INITIALIZER:::")
        xavier_init(model)

    elif config["initializer"] == "equi_var":
        print(":::USING EQUIVARIANCE INITIALIZER:::")
        equi_var_init(model)

    else:
        print(":::NO INITIALIZER USED:::")

    return model


class LogParameters(pl.Callback):
    # weight and biases to tensorboard
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self.d_parameters = {}
        for n, p in pl_module.named_parameters():
            self.d_parameters[n] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:  # WARN: sanity_check is turned on by default
            lp = []
            tensorboard_logger_index = 0
            for n, p in pl_module.named_parameters():
                trainer.logger.experiment.add_histogram(
                    n, p.data, trainer.current_epoch
                )
                self.d_parameters[n].append(p.ravel().cpu().numpy())
                lp.append(p.ravel().cpu().numpy())

            p = np.concatenate(lp)
            trainer.logger.experiment.add_histogram(
                "Parameters", p, trainer.current_epoch
            )


def get_charge_spin_libe(batch_graph):
    global_feats = batch_graph.ndata["feat"]["global"]
    # 3th to 6th index inclusive
    ind_charges = (3, 6)
    ind_spins = (5, 8)
    charge_one_hot = global_feats[:, ind_charges[0] : ind_charges[1]]
    spin_one_hot = global_feats[:, ind_spins[0] : ind_spins[1]]
    charge_one_hot = charge_one_hot.detach().numpy()
    spin_one_hot = spin_one_hot.detach().numpy()
    charge_one_hot = list(np.argmax(charge_one_hot, axis=1) - 1)
    spin_one_hot = list(np.argmax(spin_one_hot, axis=1))

    return charge_one_hot, spin_one_hot


def get_charge_tmqm(batch_graph):
    global_feats = batch_graph.ndata["feat"]["global"]
    # 3th to 6th index inclusive
    ind_charges = (3, 6)
    charge_one_hot = global_feats[:, ind_charges[0] : ind_charges[1]]
    charge_one_hot = charge_one_hot.detach().numpy()
    charge_one_hot = list(np.argmax(charge_one_hot, axis=1) - 1)

    return charge_one_hot  # , spin_one_hot


def test_and_predict_libe(dataset_test, dataset_train, model):
    statistics_dict = {}

    ### Train set
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_train, batch_size=len(dataset_train.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader))
    charge_list_train, spin_list_train = get_charge_spin_libe(batch_graph)
    preds_train = model.forward(batch_graph, batch_graph.ndata["feat"])
    preds_train = preds_train.detach()

    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_train.label_scalers,
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]
    statistics_dict["train"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    print("--" * 50)
    print(
        "Performance training set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )

    ### Test set
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=len(dataset_test.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader))
    charge_list_test, spin_list_test = get_charge_spin_libe(batch_graph)
    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_test.label_scalers,
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]

    print(
        "Performance test set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )
    print("--" * 50)
    statistics_dict["test"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    preds_test = model.forward(batch_graph, batch_graph.ndata["feat"])
    label_list = torch.tensor(
        [i.ndata["labels"]["global"].tolist()[0][0] for i in dataset_test.graphs]
    )
    label_list_train = torch.tensor(
        [i.ndata["labels"]["global"].tolist()[0][0] for i in dataset_train.graphs]
    )

    for scaler in dataset_test.label_scalers:
        label_list_train = scaler.inverse_feats({"global": label_list_train})[
            "global"
        ].view(-1, 1)
        preds_test = scaler.inverse_feats({"global": preds_test})["global"].view(-1, 1)
        label_list = scaler.inverse_feats({"global": label_list})["global"].view(-1, 1)
        preds_train = scaler.inverse_feats({"global": preds_train})["global"].view(
            -1, 1
        )

    # return preds_test, preds_train, label_list, label_list_train, statistics_dict, charge_list_test, spin_list_test, charge_list_train, spin_list_train
    return {
        "preds_test": preds_test.detach().numpy(),
        "preds_train": preds_train.detach().numpy(),
        "label_list": label_list.detach().numpy(),
        "label_list_train": label_list_train.detach().numpy(),
        "statistics_dict": statistics_dict,
        "charge_list_test": charge_list_test,
        "spin_list_test": spin_list_test,
        "charge_list_train": charge_list_train,
        "spin_list_train": spin_list_train,
    }


def test_and_predict_tmqm(dataset_test, dataset_train, model):
    statistics_dict = {}

    ### Train set
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_train, batch_size=len(dataset_train.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader))
    charge_list_train = get_charge_tmqm(batch_graph)
    preds_train = model.forward(batch_graph, batch_graph.ndata["feat"])
    preds_train = preds_train.detach()

    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_train.label_scalers,
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]
    statistics_dict["train"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    print("--" * 50)
    print(
        "Performance training set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )

    ### Test set
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=len(dataset_test.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader))
    charge_list_test = get_charge_tmqm(batch_graph)
    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_test.label_scalers,
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]

    print(
        "Performance test set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )
    print("--" * 50)
    statistics_dict["test"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    preds_test = model.forward(batch_graph, batch_graph.ndata["feat"])
    label_list = torch.tensor(
        [i.ndata["labels"]["global"].tolist()[0][0] for i in dataset_test.graphs]
    )
    label_list_train = torch.tensor(
        [i.ndata["labels"]["global"].tolist()[0][0] for i in dataset_train.graphs]
    )

    for scaler in dataset_test.label_scalers:
        label_list_train = scaler.inverse_feats({"global": label_list_train})[
            "global"
        ].view(-1, 1)
        preds_test = scaler.inverse_feats({"global": preds_test})["global"].view(-1, 1)
        label_list = scaler.inverse_feats({"global": label_list})["global"].view(-1, 1)
        preds_train = scaler.inverse_feats({"global": preds_train})["global"].view(
            -1, 1
        )

    # return preds_test, preds_train, label_list, label_list_train, statistics_dict, charge_list_test, spin_list_test, charge_list_train, spin_list_train
    return {
        "preds_test": preds_test.detach().numpy(),
        "preds_train": preds_train.detach().numpy(),
        "label_list": label_list.detach().numpy(),
        "label_list_train": label_list_train.detach().numpy(),
        "statistics_dict": statistics_dict,
        "charge_list_test": charge_list_test,
        "charge_list_train": charge_list_train,
    }


def test_and_predict(dataset_test, dataset_train, model):
    statistics_dict = {}

    ### Train set
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_train, batch_size=len(dataset_train.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader))
    # charge_list_train, spin_list_train = get_charge_spin_libe(batch_graph)
    # preds_train = model.forward(batch_graph, batch_graph.ndata["feat"])
    # preds_train = preds_train.detach()

    (
        r2_pre,
        mae,
        mse,
        preds_unscaled_train,
        labels_unscaled_train,
    ) = model.evaluate_manually(
        batch_graph, batched_labels, scaler_list=dataset_train.label_scalers
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]
    statistics_dict["train"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    print("--" * 50)
    print(
        "Performance training set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )

    ### Test set
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=len(dataset_test.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader))
    # charge_list_test, spin_list_test = get_charge_spin_libe(batch_graph)
    (
        r2_pre,
        mae,
        mse,
        preds_unscaled_test,
        labels_unscaled_test,
    ) = model.evaluate_manually(
        batch_graph, batched_labels, scaler_list=dataset_test.label_scalers
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]

    print(
        "Performance test set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )
    print("--" * 50)
    statistics_dict["test"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    # return preds_test, preds_train, label_list, label_list_train, statistics_dict, charge_list_test, spin_list_test, charge_list_train, spin_list_train
    return {
        "preds_test": preds_unscaled_test.detach().numpy(),
        "preds_train": preds_unscaled_train.detach().numpy(),
        "label_list": labels_unscaled_test.detach().numpy(),
        "label_list_train": labels_unscaled_train.detach().numpy(),
        "statistics_dict": statistics_dict,
        # "charge_list_test": charge_list_test,
        # "spin_list_test": spin_list_test,
        # "charge_list_train": charge_list_train,
        # "spin_list_train": spin_list_train,
    }


def get_test_train_preds_as_df(results_dict, key="qtaim_full"):
    dict_test = {
        "preds": results_dict[key]["test_preds"].flatten(),
        "labels": results_dict[key]["test_labels"].flatten(),
    }
    dict_train = {
        "preds": results_dict[key]["train_preds"].flatten(),
        "labels": results_dict[key]["train_labels"].flatten(),
    }

    if "charge_list_test" in results_dict[key].keys():
        dict_test["charge"] = results_dict[key]["charge_list_test"]
        dict_test["spin"] = results_dict[key]["spin_list_test"]

    if "charge_list_train" in results_dict[key].keys():
        dict_train["charge"] = results_dict[key]["charge_list_train"]
        dict_train["spin"] = results_dict[key]["spin_list_train"]

    df_test = pd.DataFrame(dict_test)
    df_train = pd.DataFrame(dict_train)

    return df_test, df_train


def get_charge_tmqm(batch_graph):
    global_feats = batch_graph.ndata["feat"]["global"]
    # 3th to 6th index inclusive
    ind_charges = (3, 6)
    charge_one_hot = global_feats[:, ind_charges[0] : ind_charges[1]]
    charge_one_hot = charge_one_hot.detach().numpy()
    charge_one_hot = list(np.argmax(charge_one_hot, axis=1) - 1)

    return charge_one_hot  # , spin_one_hot


def test_and_predict_tmqm(dataset_test, model, batch_size=100):
    statistics_dict = {}

    ### Test set
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=batch_size, shuffle=False
    )
    pred_list = []
    label_list = []
    charge_list = []

    for i, (batch_graph, batched_labels) in enumerate(data_loader):
        # batch_graph, batched_labels = next(iter(data_loader))
        charge_list_test = get_charge_tmqm(batch_graph)

        r2_pre, mae, mse, pred, labels = model.evaluate_manually(
            batch_graph,
            batched_labels,
            scaler_list=dataset_test.label_scalers,
        )
        pred_list.append(pred)
        label_list.append(labels)
        charge_list.append(charge_list_test)


    preds_test = torch.cat(pred_list)
    label_list = torch.cat(label_list)
    # charge list isn't a tensor , concat w numpy
    charge_list_test = np.concatenate(charge_list)

    # manually compute r2, mae, mse
    y = label_list
    y_pred = preds_test
    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    ss_res = torch.sum((y - y_pred) ** 2)
    r2_pre = 1 - ss_res / ss_tot
    mae = torch.mean(torch.abs(y - y_pred))
    mse = torch.mean((y - y_pred) ** 2)
    print(
        "Performance test set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )
    print("--" * 50)
    statistics_dict["test"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    # return preds_test, preds_train, label_list, label_list_train, statistics_dict, charge_list_test, spin_list_test, charge_list_train, spin_list_train
    return {
        "preds_test": preds_test.detach().numpy(),
        "label_list": label_list.detach().numpy(),
        "charge_list_test": charge_list_test,
        "statistics_dict": statistics_dict,
    }
