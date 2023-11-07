import torch
import numpy as np
import pytorch_lightning as pl
from qtaim_embed.models.graph_level.base_gcn import GCNGraphPred
from qtaim_embed.models.graph_level.base_gcn_classifier import GCNGraphPredClassifier


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
            model = GCNGraphPred.load_from_checkpoint(
                checkpoint_path=config["restore_path"]
            )
            # model.to(device)
            print(":::MODEL LOADED:::")
            return model
        
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
            output_dims=config["output_dims"],
            pooling_ntypes=["atom", "bond", "global"],
            pooling_ntypes_direct=["global"],
        )
    # model.to(device)

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
