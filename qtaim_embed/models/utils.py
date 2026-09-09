import logging
import math
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

logger = logging.getLogger(__name__)


def get_grapher_config_from_model(model):
    """
    Extract grapher configuration from a model's hparams.

    Returns the grapher_config dict if present in hparams, else None.
    """
    if hasattr(model, 'hparams') and hasattr(model.hparams, 'grapher_config'):
        return model.hparams.grapher_config
    return None


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
        logger.info("RESTORING MODEL FROM EXISTING FILE")

        if config["restore_path"] is not None:
            try:
                try:
                    model = GCNGraphPred.load_from_checkpoint(
                        checkpoint_path=config["restore_path"]
                    )
                    logger.info("MODEL LOADED")
                    return model
                except Exception as e:
                    logger.warning(f"GCNGraphPred load failed: {e}, trying GCNGraphPredClassifier")
                    model = GCNGraphPredClassifier.load_from_checkpoint(
                        checkpoint_path=config["restore_path"]
                    )
                    logger.info("MODEL LOADED")
                    return model
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
            logger.warning("NO MODEL FOUND LOADING FRESH MODEL")
        else:
            load_dir = config.get("restore_dir", "./")
            try:
                model = GCNGraphPred.load_from_checkpoint(
                    checkpoint_path=load_dir + "/last.ckpt"
                )
                logger.info("MODEL LOADED")
                return model
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
                logger.warning("NO MODEL FOUND LOADING FRESH MODEL")

    shape_fc = config["shape_fc"]
    base_fc = config["fc_hidden_size_1"]

    if shape_fc == "flat":
        fc_layers = [base_fc for i in range(config["fc_num_layers"])]
    else:
        fc_layers = [int(base_fc / (2**i)) for i in range(config["fc_num_layers"])]
    if config["classifier"]:
        logger.info("CLASSIFIER MODEL")
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
            class_weights=config.get("class_weights", None),
            encoder_fn=config.get("encoder_fn", "none"),
            encoder_hidden=config.get("encoder_hidden", 64),
            encoder_cutoff=config.get("encoder_cutoff", 5.0),
            encoder_n_interactions=config.get("encoder_n_interactions", 3),
            encoder_num_gaussians=config.get("encoder_num_gaussians", 50),
            encoder_num_radial=config.get("encoder_num_radial", 6),
            encoder_lmax=config.get("encoder_lmax", 1),
            encoder_max_neighbors=config.get("encoder_max_neighbors", 16),
            encoder_tp=config.get("encoder_tp", "channelwise"),
            dense_grid=config.get("dense_grid", 16),
            bn_before_activation=config.get("bn_before_activation", True),
            global_aggr=config.get("global_aggr", "sum"),
        )
    else:
        logger.info("REGRESSION MODEL")
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
            compiled=config["compiled"],
            encoder_fn=config.get("encoder_fn", "none"),
            encoder_hidden=config.get("encoder_hidden", 64),
            encoder_cutoff=config.get("encoder_cutoff", 5.0),
            encoder_n_interactions=config.get("encoder_n_interactions", 3),
            encoder_num_gaussians=config.get("encoder_num_gaussians", 50),
            encoder_num_radial=config.get("encoder_num_radial", 6),
            encoder_lmax=config.get("encoder_lmax", 1),
            encoder_max_neighbors=config.get("encoder_max_neighbors", 16),
            encoder_tp=config.get("encoder_tp", "channelwise"),
            dense_grid=config.get("dense_grid", 16),
            bn_before_activation=config.get("bn_before_activation", True),
            global_aggr=config.get("global_aggr", "sum"),
        )
    # model.to(device)

    if config["initializer"] == "kaiming":
        logger.debug("Using kaiming initializer")
        kaiming_init(model)

    elif config["initializer"] == "xavier":
        logger.debug("Using xavier initializer")
        xavier_init(model)

    elif config["initializer"] == "equi_var":
        logger.debug("Using equivariance initializer")
        equi_var_init(model)

    else:
        logger.debug("No initializer used")

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
        logger.info("RESTORING MODEL FROM EXISTING FILE")

        if config["restore_path"] is not None:
            try:
                model = GCNNodePred.load_from_checkpoint(
                    checkpoint_path=config["restore_path"]
                )
                logger.info("MODEL LOADED")
                return model
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
            logger.warning("NO MODEL FOUND LOADING FRESH MODEL")
        else:
            load_dir = config.get("restore_dir", "./")
            try:
                model = GCNNodePred.load_from_checkpoint(
                    checkpoint_path=load_dir + "/last.ckpt"
                )
                logger.info("MODEL LOADED")
                return model
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
                logger.warning("NO MODEL FOUND LOADING FRESH MODEL")

    logger.debug(f"{config}")
    logger.info("NODE-LEVEL REGRESSION MODEL")
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
        compiled=config["compiled"],
        encoder_fn=config.get("encoder_fn", "none"),
        encoder_hidden=config.get("encoder_hidden", 64),
        encoder_cutoff=config.get("encoder_cutoff", 5.0),
        encoder_n_interactions=config.get("encoder_n_interactions", 3),
        encoder_num_gaussians=config.get("encoder_num_gaussians", 50),
        encoder_num_radial=config.get("encoder_num_radial", 6),
        encoder_lmax=config.get("encoder_lmax", 1),
        encoder_max_neighbors=config.get("encoder_max_neighbors", 16),
        encoder_tp=config.get("encoder_tp", "channelwise"),
        dense_grid=config.get("dense_grid", 16),
        bn_before_activation=config.get("bn_before_activation", True),
        global_aggr=config.get("global_aggr", "sum"),
    )
    # model.to(device)

    if config["initializer"] == "kaiming":
        logger.debug("Using kaiming initializer")
        kaiming_init(model)

    elif config["initializer"] == "xavier":
        logger.debug("Using xavier initializer")
        xavier_init(model)

    elif config["initializer"] == "equi_var":
        logger.debug("Using equivariance initializer")
        equi_var_init(model)

    else:
        logger.debug("No initializer used")

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
        logger.info("RESTORING MODEL FROM EXISTING FILE")

        if config["restore_path"] is not None:
            try:
                model = GCNLinkPred.load_from_checkpoint(
                    checkpoint_path=config["restore_path"]
                )
                logger.info("MODEL LOADED")
                return model
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
            logger.warning("NO MODEL FOUND LOADING FRESH MODEL")
        else:
            load_dir = config.get("restore_dir", "./")
            try:
                model = GCNLinkPred.load_from_checkpoint(
                    checkpoint_path=load_dir + "/last.ckpt"
                )
                logger.info("MODEL LOADED")
                return model
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")
                logger.warning("NO MODEL FOUND LOADING FRESH MODEL")

    logger.debug(f"{config}")
    logger.info("LINK-PRED MODEL")
    model = GCNLinkPred(
        input_size=config["input_size"],
        n_conv_layers=config["n_conv_layers"],
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
        compiled=config["compiled"],
    )
    # model.to(device)

    if config["initializer"] == "kaiming":
        logger.debug("Using kaiming initializer")
        kaiming_init(model)

    elif config["initializer"] == "xavier":
        logger.debug("Using xavier initializer")
        xavier_init(model)

    elif config["initializer"] == "equi_var":
        logger.debug("Using equivariance initializer")
        equi_var_init(model)

    else:
        logger.debug("No initializer used")

    return model


def load_bond_model_from_config(config):
    """Build (or restore) a GCNBondPred from the model section of a config dict."""
    from qtaim_embed.models.link_pred.bond_model import GCNBondPred

    if config.get("restore", False):
        path = config.get("restore_path") or (config.get("restore_dir", "./") + "/last.ckpt")
        try:
            model = GCNBondPred.load_from_checkpoint(checkpoint_path=path)
            logger.info("BOND MODEL LOADED FROM %s", path)
            return model
        except Exception as e:
            logger.warning(f"Checkpoint load failed: {e}; building a fresh model")

    keys = [
        "encoder_fn", "encoder_hidden", "encoder_cutoff", "encoder_n_interactions",
        "encoder_num_gaussians", "encoder_num_radial", "encoder_lmax", "encoder_max_neighbors",
        "encoder_tp",
        "use_atom_feat", "atom_input_size", "embedding_size", "pool_multiplier",
        "pair_rbf", "pair_rbf_n", "pair_rbf_cutoff", "pair_hidden", "pair_dropout",
        "activation", "lr", "weight_decay", "scheduler_name", "lr_plateau_patience",
        "lr_scale_factor", "threshold", "n_threshold_bins",
    ]
    kwargs = {k: config[k] for k in keys if k in config}
    logger.info("BOND MODEL (encoder_fn=%s)", kwargs.get("encoder_fn", "schnet"))
    model = GCNBondPred(**kwargs)

    init = config.get("initializer", None)
    if init == "kaiming":
        kaiming_init(model)
    elif init == "xavier":
        xavier_init(model)
    elif init == "equi_var":
        equi_var_init(model)
    return model


def convert_model_to_dense(model):
    """Copy of a trained conv_fn="ResidualBlock" model as "ResidualBlockDense".

    Rebuilds the model from its saved hyperparameters with the dense conv
    stack, loads every non-conv module's state (embedding, encoder, readout,
    heads, metrics) and maps each ResidualBlock onto its DenseResidualBlock
    with copy_residual_block_weights. Outputs match to fp32 precision.
    """
    import inspect
    from qtaim_embed.models.layers_dense import copy_residual_block_weights

    assert model.hparams.conv_fn == "ResidualBlock", "only ResidualBlock models convert"
    sig = inspect.signature(type(model).__init__).parameters
    # hparams carries ctor args plus derived names for a few of them
    aliases = {
        "batch_norm": "batch_norm_tf", "num_heads_gat": "num_heads",
        "dropout_feat_gat": "feat_drop", "dropout_attn_gat": "attn_drop",
        "residual_gat": "residual", "pooling_ntypes": "ntypes_pool",
        "pooling_ntypes_direct": "ntypes_pool_direct_cat",
    }
    kwargs = {}
    for k in sig:
        if k == "self":
            continue
        if k in model.hparams:
            kwargs[k] = model.hparams[k]
        elif aliases.get(k) in model.hparams:
            kwargs[k] = model.hparams[aliases[k]]
    act = kwargs.get("activation")
    if act is not None and not isinstance(act, str):
        kwargs["activation"] = type(act).__name__
    kwargs["conv_fn"] = "ResidualBlockDense"
    kwargs["compiled"] = False
    dense = type(model)(**kwargs)
    keep = {k: v for k, v in model.state_dict().items() if not k.startswith("conv_layers.")}
    missing, unexpected = dense.load_state_dict(keep, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("conv_layers.") for k in missing), missing
    for src, dst in zip(model.conv_layers, dense.conv_layers):
        copy_residual_block_weights(src, dst)
    dense.train(model.training)
    return dense


class LinearWarmup(pl.Callback):
    """Linear learning-rate warmup over the first `warmup_epochs` epochs.

    ReduceLROnPlateau (the project scheduler) cannot be chained with
    SequentialLR, so the warmup is a callback: it scales every param group
    from base_lr / n_steps up to base_lr on each training step of the warmup
    window and then stops touching the optimizer, leaving the plateau
    scheduler in charge. Fractional epochs are allowed. Used with the linear
    batch-size LR scaling rule (performance plan A1).
    """

    def __init__(self, warmup_epochs: float):
        super().__init__()
        self.warmup_epochs = float(warmup_epochs)
        self._base_lrs = None
        self._total_steps = 0

    def on_train_start(self, trainer, pl_module):
        if self._base_lrs is None:
            # first start; on a checkpoint resume the base LRs come from
            # load_state_dict, since the live param-group lr is mid-ramp
            self._base_lrs = [
                [g.get("initial_lr", g["lr"]) for g in opt.param_groups]
                for opt in trainer.optimizers
            ]
        n_batches = trainer.num_training_batches
        if not math.isfinite(n_batches):
            raise ValueError("LinearWarmup needs a sized train dataloader (num_training_batches is inf)")
        self._total_steps = int(round(self.warmup_epochs * n_batches))

    def state_dict(self):
        return {"base_lrs": self._base_lrs, "total_steps": self._total_steps}

    def load_state_dict(self, state_dict):
        self._base_lrs = state_dict.get("base_lrs")
        self._total_steps = int(state_dict.get("total_steps", 0))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        if self._total_steps <= 0 or step >= self._total_steps:
            return
        factor = (step + 1) / self._total_steps
        for opt, base in zip(trainer.optimizers, self._base_lrs):
            for group, lr in zip(opt.param_groups, base):
                group["lr"] = lr * factor


class LogParameters(pl.Callback):
    # weight and biases to tensorboard
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        self.d_parameters = {}
        for n, p in pl_module.named_parameters():
            self.d_parameters[n] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
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
    global_feats = batch_graph["global"].feat
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


def test_and_predict_libe(dataset_test, dataset_train, model):
    statistics_dict = {}

    ### Train set
    data_loader_train = DataLoaderMoleculeGraphTask(
        dataset_train, batch_size=len(dataset_train.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader_train))
    charge_list_train, spin_list_train = get_charge_spin_libe(batch_graph)
    feat_dict = {nt: batch_graph[nt].feat for nt in batch_graph.node_types if hasattr(batch_graph[nt], "feat")}
    preds_train = model.forward(batch_graph, feat_dict)
    preds_train = preds_train.detach()

    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        data_loader_train,
        scaler_list=dataset_train.label_scalers,
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]
    statistics_dict["train"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    logger.info("--" * 50)
    logger.info(
        "Performance training set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )

    ### Test set
    data_loader_test = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=len(dataset_test.graphs), shuffle=False
    )
    batch_graph, batched_labels = next(iter(data_loader_test))
    charge_list_test, spin_list_test = get_charge_spin_libe(batch_graph)
    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        data_loader_test,
        scaler_list=dataset_test.label_scalers,
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]

    logger.info(
        "Performance test set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )
    logger.info("--" * 50)
    statistics_dict["test"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    feat_dict = {nt: batch_graph[nt].feat for nt in batch_graph.node_types if hasattr(batch_graph[nt], "feat")}
    preds_test = model.forward(batch_graph, feat_dict)
    label_list = torch.tensor(
        [i["global"].labels.tolist()[0][0] for i in dataset_test.graphs]
    )
    label_list_train = torch.tensor(
        [i["global"].labels.tolist()[0][0] for i in dataset_train.graphs]
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


def test_and_predict(dataset_test, dataset_train, model):
    statistics_dict = {}

    ### Train set
    data_loader_train = DataLoaderMoleculeGraphTask(
        dataset_train, batch_size=len(dataset_train.graphs), shuffle=False
    )

    (
        r2_pre,
        mae,
        mse,
        preds_unscaled_train,
        labels_unscaled_train,
    ) = model.evaluate_manually(
        data_loader_train, scaler_list=dataset_train.label_scalers
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]
    statistics_dict["train"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    logger.info("--" * 50)
    logger.info(
        "Performance training set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )

    ### Test set
    data_loader_test = DataLoaderMoleculeGraphTask(
        dataset_test, batch_size=len(dataset_test.graphs), shuffle=False
    )

    (
        r2_pre,
        mae,
        mse,
        preds_unscaled_test,
        labels_unscaled_test,
    ) = model.evaluate_manually(
        data_loader_test, scaler_list=dataset_test.label_scalers
    )
    r2_pre = r2_pre.numpy()[0]
    mae = mae.numpy()[0]
    mse = mse.numpy()[0]

    logger.info(
        "Performance test set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )
    logger.info("--" * 50)
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
    global_feats = batch_graph["global"].feat
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

    # Use evaluate_manually with the full dataloader to get unscaled preds/labels
    _, _, _, preds_test, label_list = model.evaluate_manually(
        data_loader,
        scaler_list=dataset_test.label_scalers,
    )

    # Collect charge info per batch (requires iterating separately)
    charge_list = []
    for batch_graph, batched_labels in data_loader:
        charge_list_test = get_charge_tmqm(batch_graph)
        charge_list.append(charge_list_test)
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
    logger.info(
        "Performance test set:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )
    logger.info("--" * 50)
    statistics_dict["test"] = {"r2": r2_pre, "mae": mae, "mse": mse}

    # return preds_test, preds_train, label_list, label_list_train, statistics_dict, charge_list_test, spin_list_test, charge_list_train, spin_list_train
    return {
        "preds_test": preds_test.detach().numpy(),
        "label_list": label_list.detach().numpy(),
        "charge_list_test": charge_list_test,
        "statistics_dict": statistics_dict,
    }
