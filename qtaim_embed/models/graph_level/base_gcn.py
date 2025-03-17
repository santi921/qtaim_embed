# baseline GNN model for node-level regression
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn
from torchmetrics.wrappers import MultioutputWrapper
import torchmetrics
from dgl.nn.pytorch import GATConv

from qtaim_embed.utils.models import (
    get_layer_args,
    link_fmt_to_node_fmt,
    _split_batched_output,
)

from qtaim_embed.models.layers import (
    GraphConvDropoutBatch,
    ResidualBlock,
    UnifySize,
    Set2SetThenCat,
    SumPoolingThenCat,
    WeightAndSumThenCat,
    GlobalAttentionPoolingThenCat,
    MeanPoolingThenCat,
    WeightAndMeanThenCat,
)


class GCNGraphPred(pl.LightningModule):
    """
    Basic GNN model for graph-level regression
    Takes
        atom_input_size: int, dimension of atom features
        bond_input_size: int, dimension of bond features
        global_input_size: int, dimension of global features
        target_dict: dict, dictionary of targets
        n_conv_layers: int, number of convolution layers
        conv_fn: str "GraphConvDropoutBatch"
        dropout: float, dropout rate
        batch_norm: bool, whether to use batch norm
        activation: str, activation function
        bias: bool, whether to use bias
        norm: str, normalization type
        aggregate: str, aggregation type
        lr: float, learning rate
        scheduler_name: str, scheduler type
        weight_decay: float, weight decay
        lr_plateau_patience: int, patience for lr scheduler
        lr_scale_factor: float, scale factor for lr scheduler
        loss_fn: str, loss function
        resid_n_graph_convs: int, number of graph convolutions per residual block
        scalers: list, list of scalers applied to each node type
        embedding_size: int, size of embedding layer
        global_pooling: str, type of global pooling

    """

    def __init__(
        self,
        atom_input_size=12,
        bond_input_size=8,
        global_input_size=3,
        n_conv_layers=3,
        target_dict={"atom": "E"},
        conv_fn="GraphConvDropoutBatch",
        global_pooling="WeightAndSumThenCat",
        resid_n_graph_convs=None,
        num_heads_gat=2,
        dropout_feat_gat=0.2,
        dropout_attn_gat=0.2,
        hidden_size=128,
        residual_gat=True,
        dropout=0.2,
        batch_norm=True,
        activation="ReLU",
        bias=True,
        norm="both",
        aggregate="sum",
        lr=1e-3,
        scheduler_name="reduce_on_plateau",
        weight_decay=0.0,
        lr_plateau_patience=5,
        lr_scale_factor=0.5,
        loss_fn="mse",
        embedding_size=128,
        fc_layer_size=[128, 64],
        fc_dropout=0.0,
        fc_batch_norm=True,
        lstm_iters=3,
        lstm_layers=1,
        pooling_ntypes=["atom", "bond"],
        pooling_ntypes_direct=["global"],
        compiled=None
    ):
        super().__init__()
        self.learning_rate = lr

        # output_dims = 0
        # for k, v in target_dict.items():
        #    output_dims += len(v)

        assert (
            conv_fn == "GraphConvDropoutBatch"
            or conv_fn == "ResidualBlock"
            or conv_fn == "GATConv"
        ), (
            "conv_fn must be either GraphConvDropoutBatch, GATConv or ResidualBlock"
            + f"but got {conv_fn}"
        )

        if conv_fn == "ResidualBlock":
            assert resid_n_graph_convs is not None, (
                "resid_n_graph_convs must be specified for ResidualBlock"
                + f"but got {resid_n_graph_convs}"
            )

        assert global_pooling in [
            "WeightAndSumThenCat",
            "SumPoolingThenCat",
            "GlobalAttentionPoolingThenCat",
            "Set2SetThenCat",
            "MeanPoolingThenCat",
            "WeightandMeanThenCat",
        ], (
            "global_pooling must be either WeightAndSumThenCat, SumPoolingThenCat, MeanPoolingThenCat, WeightandMeanThenCat, or GlobalAttentionPoolingThenCat"
            + f"but got {global_pooling}"
        )

        params = {
            "atom_input_size": atom_input_size,
            "bond_input_size": bond_input_size,
            "global_input_size": global_input_size,
            "conv_fn": conv_fn,
            "target_dict": target_dict,
            "dropout": dropout,
            "batch_norm_tf": batch_norm,
            "activation": activation,
            "bias": bias,
            "norm": norm,
            "aggregate": aggregate,
            "n_conv_layers": n_conv_layers,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_plateau_patience": lr_plateau_patience,
            "lr_scale_factor": lr_scale_factor,
            "scheduler_name": scheduler_name,
            "loss_fn": loss_fn,
            "resid_n_graph_convs": resid_n_graph_convs,
            "embedding_size": embedding_size,
            "fc_layer_size": fc_layer_size,
            "fc_dropout": fc_dropout,
            "fc_batch_norm": fc_batch_norm,
            "n_fc_layers": len(fc_layer_size),
            "global_pooling": global_pooling,
            "ntypes_pool": pooling_ntypes,
            "ntypes_pool_direct_cat": pooling_ntypes_direct,
            "lstm_iters": lstm_iters,
            "lstm_layers": lstm_layers,
            "num_heads": num_heads_gat,
            "feat_drop": dropout_feat_gat,
            "attn_drop": dropout_attn_gat,
            "residual": residual_gat,
            "hidden_size": hidden_size,
            "ntasks": len(target_dict["global"]),
            "compiled": compiled,
        }

        self.hparams.update(params)
        self.save_hyperparameters()

        # convert string activation to function
        if self.hparams.activation is not None:
            self.activation = getattr(torch.nn, self.hparams.activation)()
        else:
            self.activation = None

        input_size = {
            "atom": self.hparams.atom_input_size,
            "bond": self.hparams.bond_input_size,
            "global": self.hparams.global_input_size,
        }
        # print("input size", input_size)
        self.embedding = UnifySize(
            input_dim=input_size,
            output_dim=self.hparams.embedding_size,
        )
        # self.embedding_output_size = self.hparams.embedding_size

        self.conv_layers = nn.ModuleList()

        if self.hparams.conv_fn == "GraphConvDropoutBatch":
            for i in range(self.hparams.n_conv_layers):
                # embedding_in = False
                # if i == 0:
                # embedding_in = True

                layer_args = get_layer_args(
                    self.hparams, i, activation=self.activation, embedding_in=True
                )
                # print("resid layer args", layer_args)

                self.conv_layers.append(
                    dglnn.HeteroGraphConv(
                        {
                            "a2b": GraphConvDropoutBatch(**layer_args["a2b"]),
                            "b2a": GraphConvDropoutBatch(**layer_args["b2a"]),
                            "a2g": GraphConvDropoutBatch(**layer_args["a2g"]),
                            "g2a": GraphConvDropoutBatch(**layer_args["g2a"]),
                            "b2g": GraphConvDropoutBatch(**layer_args["b2g"]),
                            "g2b": GraphConvDropoutBatch(**layer_args["g2b"]),
                            "a2a": GraphConvDropoutBatch(**layer_args["a2a"]),
                            "b2b": GraphConvDropoutBatch(**layer_args["b2b"]),
                            "g2g": GraphConvDropoutBatch(**layer_args["g2g"]),
                        },
                        aggregate=self.hparams.aggregate,
                    )
                )

        elif self.hparams.conv_fn == "ResidualBlock":
            layer_tracker = 0
            # embedding_in = True

            while layer_tracker < self.hparams.n_conv_layers:
                if (
                    layer_tracker + self.hparams.resid_n_graph_convs
                    > self.hparams.n_conv_layers - 1
                ):
                    # print("triggered output_layer args")
                    layer_ind = -1
                else:
                    layer_ind = layer_tracker

                layer_args = get_layer_args(
                    self.hparams,
                    layer_ind,
                    embedding_in=True,
                    activation=self.activation,
                )

                output_block = False

                if layer_ind == -1:
                    output_block = True

                self.conv_layers.append(
                    ResidualBlock(
                        layer_args,
                        resid_n_graph_convs=self.hparams.resid_n_graph_convs,
                        aggregate=self.hparams.aggregate,
                        output_block=output_block,
                    )
                )

                layer_tracker += self.hparams.resid_n_graph_convs

        elif self.hparams.conv_fn == "GATConv":
            for i in range(self.hparams.n_conv_layers):

                embedding_in = True

                layer_args = get_layer_args(
                    self.hparams, i, activation=self.activation, embedding_in=True
                )
                # print("resid layer args", layer_args)

                self.conv_layers.append(
                    dglnn.HeteroGraphConv(
                        {
                            "a2b": GATConv(**layer_args["a2b"]),
                            "b2a": GATConv(**layer_args["b2a"]),
                            "a2g": GATConv(**layer_args["a2g"]),
                            "g2a": GATConv(**layer_args["g2a"]),
                            "b2g": GATConv(**layer_args["b2g"]),
                            "g2b": GATConv(**layer_args["g2b"]),
                            "a2a": GATConv(**layer_args["a2a"]),
                            "b2b": GATConv(**layer_args["b2b"]),
                            "g2g": GATConv(**layer_args["g2g"]),
                        },
                        aggregate=self.hparams.aggregate,
                    )
                )

        self.conv_layers = nn.ModuleList(self.conv_layers)
        # print("conv layer out modes", self.conv_layers[-1].mods)

        # print("conv layer out feats", self.conv_layers[-1].out_feats)
        # conv_out_size = self.conv_layers[-1].out_feats

        if self.hparams.conv_fn == "GraphConvDropoutBatch":
            conv_out_size = {}
            for k, v in self.conv_layers[-1].mods.items():
                conv_out_size[k] = v.out_feats

        elif self.hparams.conv_fn == "ResidualBlock":
            conv_out_size = self.conv_layers[-1].out_feats

        elif self.hparams.conv_fn == "GATConv":
            conv_out_size = {}
            for k, v in self.conv_layers[-1].mods.items():
                conv_out_size[k] = v._out_feats

        self.conv_out_size = link_fmt_to_node_fmt(conv_out_size)

        ####################### readout starts here ######################
        if self.hparams.global_pooling == "WeightAndSumThenCat":
            readout_fn = WeightAndSumThenCat
        elif self.hparams.global_pooling == "SumPoolingThenCat":
            readout_fn = SumPoolingThenCat
        elif self.hparams.global_pooling == "GlobalAttentionPoolingThenCat":
            readout_fn = GlobalAttentionPoolingThenCat
        elif self.hparams.global_pooling == "Set2SetThenCat":
            readout_fn = Set2SetThenCat
        elif self.hparams.global_pooling == "MeanPoolingThenCat":
            readout_fn = MeanPoolingThenCat
        elif self.hparams.global_pooling == "WeightandMeanThenCat":
            readout_fn = WeightAndMeanThenCat

        list_in_feats = []
        for type_feat in self.hparams.pooling_ntypes:
            list_in_feats.append(self.conv_out_size[type_feat])

        self.readout_out_size = 0

        if self.hparams.global_pooling == "Set2SetThenCat":

            self.readout = readout_fn(
                n_iters=self.hparams.lstm_iters,
                n_layers=self.hparams.lstm_layers,
                in_feats=list_in_feats,
                ntypes=self.hparams.pooling_ntypes,
                ntypes_direct_cat=self.hparams.ntypes_pool_direct_cat,
            )
            for i in self.hparams.pooling_ntypes:
                if i not in self.hparams.ntypes_pool_direct_cat:
                    self.readout_out_size += self.conv_out_size[i] * 2
                else:
                    self.readout_out_size += self.conv_out_size[i]

        else:
            # print("other readout used")
            self.readout = readout_fn(
                ntypes=self.hparams.pooling_ntypes,
                in_feats=list_in_feats,
                ntypes_direct_cat=self.hparams.ntypes_pool_direct_cat,
            )

            for i in self.hparams.pooling_ntypes:
                if i in self.hparams.ntypes_pool_direct_cat:
                    self.readout_out_size += self.conv_out_size[i]
                else:
                    self.readout_out_size += self.conv_out_size[i]
        # if self.hparams.conv_fn == "GATConv":
        # self.readout_out_size = self.hparams.hidden_size * self.hparams.num_heads
        # print("readout out size", self.readout_out_size)
        # self.readout_out_size = readout_out_size
        self.loss = self.loss_function()
        ####################### fc starts here ######################
        self.fc_layers = nn.ModuleList()

        input_size = self.readout_out_size
        # print("readout in size", input_size)
        for i in range(self.hparams.n_fc_layers):
            out_size = self.hparams.fc_layer_size[i]
            self.fc_layers.append(nn.Linear(input_size, out_size))

            if self.hparams.fc_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(out_size))

            if self.activation is not None:
                self.fc_layers.append(self.activation)

            if self.hparams.fc_dropout > 0:
                self.fc_layers.append(nn.Dropout(self.hparams.fc_dropout))
            input_size = out_size

        self.fc_layers.append(nn.Linear(input_size, self.hparams.ntasks))

        # print("number of output dims", output_dims)
        print("... > number of tasks:", self.hparams.ntasks)

        # create multioutput wrapper for metrics
        self.train_r2 = MultioutputWrapper(
            torchmetrics.R2Score(), num_outputs=self.hparams.ntasks
        )
        self.train_torch_l1 = MultioutputWrapper(
            torchmetrics.MeanAbsoluteError(), num_outputs=self.hparams.ntasks
        )
        self.train_torch_mse = MultioutputWrapper(
            torchmetrics.MeanSquaredError(squared=False),
            num_outputs=self.hparams.ntasks,
        )
        self.val_r2 = MultioutputWrapper(
            torchmetrics.R2Score(), num_outputs=self.hparams.ntasks
        )
        self.val_torch_l1 = MultioutputWrapper(
            torchmetrics.MeanAbsoluteError(), num_outputs=self.hparams.ntasks
        )
        self.val_torch_mse = MultioutputWrapper(
            torchmetrics.MeanSquaredError(squared=False),
            num_outputs=self.hparams.ntasks,
        )
        self.test_r2 = MultioutputWrapper(
            torchmetrics.R2Score(), num_outputs=self.hparams.ntasks
        )
        self.test_torch_l1 = MultioutputWrapper(
            torchmetrics.MeanAbsoluteError(), num_outputs=self.hparams.ntasks
        )
        self.test_torch_mse = MultioutputWrapper(
            torchmetrics.MeanSquaredError(squared=False),
            num_outputs=self.hparams.ntasks,
        )

        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if compiled is not None
            else self.compiled_forward
        )

    def compiled_forward(self, graph, feat, eweight=None):
        feats = self.embedding(feat)
        for ind, conv in enumerate(self.conv_layers):
            feats = conv(graph, feats)

            if self.hparams.conv_fn == "GATConv":
                if ind < self.hparams.n_conv_layers - 1:
                    for k, v in feats.items():
                        feats[k] = v.reshape(
                            -1, self.hparams.num_heads * self.hparams.hidden_size
                        )
                else:
                    for k, v in feats.items():
                        feats[k] = v.reshape(-1, self.hparams.input_size[k])

        readout_feats = self.readout(graph, feats)
        for ind, layer in enumerate(self.fc_layers):
            readout_feats = layer(readout_feats)

        # print("preds shape:", readout_feats.shape)
        return readout_feats


    def forward(self, graph, feat, eweight=None):
        """
        Forward pass
        """
        # just use the compiled forward function
        return self.forward_fn(graph, feat, eweight)


    def loss_function(self):
        """
        Initialize loss function
        """
        if self.hparams.ntasks > 1:
            loss_fn = nn.ModuleList()
            for i in range(self.hparams.ntasks):
                if self.hparams.loss_fn == "mse":
                    loss_fn.append(torchmetrics.MeanSquaredError())
                elif self.hparams.loss_fn == "smape":
                    loss_fn.append(torchmetrics.SymmetricMeanAbsolutePercentageError())
                elif self.hparams.loss_fn == "mae":
                    loss_fn.append(torchmetrics.MeanAbsoluteError())
                else:
                    loss_fn.append(torchmetrics.MeanSquaredError())

        else:
            if self.hparams.loss_fn == "mse":
                loss_fn = torchmetrics.MeanSquaredError()

            elif self.hparams.loss_fn == "smape":
                loss_fn = torchmetrics.SymmetricMeanAbsolutePercentageError()
            elif self.hparams.loss_fn == "mae":
                loss_fn = torchmetrics.MeanAbsoluteError()
            else:
                loss_fn = torchmetrics.MeanSquaredError()

        return loss_fn

    def compute_loss(self, target, pred):
        """
        Compute loss
        """
        if self.hparams.ntasks > 1:
            loss = 0
            # print("target shape", target.shape)
            for i in range(self.hparams.ntasks):
                loss += self.loss[i](target[:, i], pred[:, i])
            return loss
        return self.loss(target, pred)

    def feature_at_each_layer(model, graph, feats):
        """
        Get the features at each layer before the final fully-connected layer.

        This is used for feature visualization to see how the model learns.

        Returns:
            dict: (layer_idx, feats), each feats is a list of
        """

        layer_idx = 0
        atom_feats, bond_feats, global_feats = {}, {}, {}

        feats = model.embedding(feats)
        bond_feats[layer_idx] = _split_batched_output(graph, feats["bond"], "bond")
        atom_feats[layer_idx] = _split_batched_output(graph, feats["atom"], "atom")
        global_feats[layer_idx] = _split_batched_output(
            graph, feats["global"], "global"
        )

        layer_idx += 1

        # gated layer
        for layer in model.conv_layers[:-1]:
            feats = layer(graph, feats)
            # store bond feature of each molecule
            bond_feats[layer_idx] = _split_batched_output(graph, feats["bond"], "bond")

            atom_feats[layer_idx] = _split_batched_output(graph, feats["atom"], "atom")

            global_feats[layer_idx] = _split_batched_output(
                graph, feats["global"], "global"
            )
            layer_idx += 1

    def shared_step(self, batch, mode, scalers=None):
        batch_graph, batch_label = batch
        logits = self.forward(
            batch_graph, batch_graph.ndata["feat"]
        )  # returns a dict of node types
        labels = batch_label["global"]
        all_loss = self.compute_loss(logits, labels)
        logits = logits.view(-1, self.hparams.ntasks)
        labels = labels.view(-1, self.hparams.ntasks)
        # log loss
        self.log(
            f"{mode}_loss",
            all_loss.sum(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
            sync_dist=True,
        )
        self.update_metrics(logits, labels, mode)

        return all_loss

    def training_step(self, batch, batch_idx):
        """
        Train step
        """
        loss = self.shared_step(batch, mode="train")
        return {"train_loss": loss, "loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Val step
        """
        return {"val_loss": self.shared_step(batch, mode="val")}

    def test_step(self, batch, batch_idx, scalers=None):
        # Todo
        return {"test_loss": self.shared_step(batch, mode="test", scalers=scalers)}

    def on_train_epoch_end(self):
        """
        Training epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="train")
        # get epoch number

        if self.trainer.current_epoch < 2:
            self.log("val_mae", 10**10, prog_bar=False)
        self.log("train_r2", r2.median(), prog_bar=False, sync_dist=True)
        self.log("train_mae", mae.mean(), prog_bar=False, sync_dist=True)
        self.log("train_mse", mse.mean(), prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """
        Validation epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="val")
        r2_median = r2.median().type(torch.float32)
        self.log("val_r2", r2_median, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae.mean(), prog_bar=False, sync_dist=True)
        self.log("val_mse", mse.mean(), prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        """
        Test epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="test")
        self.log("test_r2", r2.median(), prog_bar=False, sync_dist=True)
        self.log("test_mae", mae.mean(), prog_bar=False, sync_dist=True)
        self.log("test_mse", mse.mean(), prog_bar=False, sync_dist=True)

    def update_metrics(self, pred, target, mode):
        """
        Update metrics using torchmetrics interfaces
        """

        if mode == "train":
            self.train_r2.update(pred, target)
            self.train_torch_l1.update(pred, target)
            self.train_torch_mse.update(pred, target)
        elif mode == "val":
            self.val_r2.update(pred, target)
            self.val_torch_l1.update(pred, target)
            self.val_torch_mse.update(pred, target)

        elif mode == "test":
            self.test_r2.update(pred, target)
            self.test_torch_l1.update(pred, target)
            self.test_torch_mse.update(pred, target)

    def compute_metrics(self, mode):
        """
        Compute metrics using torchmetrics interfaces
        """

        if mode == "train":
            r2 = self.train_r2.compute()
            torch_l1 = self.train_torch_l1.compute()
            torch_mse = self.train_torch_mse.compute()
            self.train_r2.reset()
            self.train_torch_l1.reset()
            self.train_torch_mse.reset()

        elif mode == "val":
            r2 = self.val_r2.compute()
            torch_l1 = self.val_torch_l1.compute()
            torch_mse = self.val_torch_mse.compute()
            self.val_r2.reset()
            self.val_torch_l1.reset()
            self.val_torch_mse.reset()

        elif mode == "test":
            r2 = self.test_r2.compute()
            torch_l1 = self.test_torch_l1.compute()
            torch_mse = self.test_torch_mse.compute()
            self.test_r2.reset()
            self.test_torch_l1.reset()
            self.test_torch_mse.reset()

        return r2, torch_l1, torch_mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = self._config_lr_scheduler(optimizer)

        lr_scheduler = {"scheduler": scheduler, "monitor": "val_mae"}

        return [optimizer], [lr_scheduler]

    def _config_lr_scheduler(self, optimizer):
        scheduler_name = self.hparams["scheduler_name"].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.lr_scale_factor,
                patience=self.hparams.lr_plateau_patience,
                verbose=True,
            )

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {scheduler_name}")

        return scheduler

    def evaluate_manually(self, dataloader, scaler_list, per_atom=False):
        """
        Evaluate a set of data manually
        Takes
            feats: dict, dictionary of batched features
            scaler_list: list, list of scalers
        """
        # batch_graph, batch_label = batch

        # batch_label = batch_label["global"]
        preds_list_raw = []
        labels_list_raw = []
        n_atom_list = []

        self.eval()

        for batch_graph, batched_labels in dataloader:

            preds = self.forward(batch_graph, batch_graph.ndata["feat"])
            preds_raw = deepcopy(preds.detach())
            labels_raw = deepcopy(batched_labels)["global"]

            preds_list_raw.append(preds_raw)
            labels_list_raw.append(labels_raw)

            if per_atom:
                n_atoms = batch_graph.batch_num_nodes("atom")
                n_atom_list.append(n_atoms)

        preds_raw = torch.cat(preds_list_raw, dim=0)
        labels_raw = torch.cat(labels_list_raw, dim=0)

        if per_atom:
            n_atom_list = torch.cat(n_atom_list, dim=0)

        for scaler in scaler_list[::-1]:
            labels_unscaled = scaler.inverse_feats({"global": labels_raw})[
                "global"
            ].view(-1, self.hparams.ntasks)
            preds_unscaled = scaler.inverse_feats({"global": preds_raw})["global"].view(
                -1, self.hparams.ntasks
            )

        if per_atom:
            abs_diff = torch.abs(preds_unscaled - labels_unscaled)

            # n_atoms = batch_graph.batch_num_nodes("atom")
            # n_mols = batch_graph.batch_size
            abs_diff = torch.abs(preds_unscaled - labels_unscaled)
            n_atoms = n_atom_list
            y = labels_unscaled
            y_pred = preds_unscaled
            r2_manual = torchmetrics.functional.r2_score(y_pred, y)
            print("r2 manual", r2_manual)
            mae_per_atom = torch.mean(abs_diff / n_atom_list)
            mae_per_molecule = torch.mean(abs_diff)
            ewt_prop = torch.sum(abs_diff < 0.043) / len(abs_diff)
            rmse_per_molecule = torch.mean(torch.sqrt(torch.mean(abs_diff**2)))
            mse_per_atom = abs_diff**2 / n_atom_list
            mean_rmse_per_atom = torch.sqrt(torch.mean(mse_per_atom))

            return (
                mae_per_atom,
                mean_rmse_per_atom,
                ewt_prop,
                preds_unscaled,
                labels_unscaled,
            )

        else:

            r2_eval = MultioutputWrapper(
                torchmetrics.R2Score(), num_outputs=self.hparams.ntasks
            )
            mae_eval = MultioutputWrapper(
                torchmetrics.MeanAbsoluteError(), num_outputs=self.hparams.ntasks
            )
            mse_eval = MultioutputWrapper(
                torchmetrics.MeanSquaredError(squared=False),
                num_outputs=self.hparams.ntasks,
            )

            r2_eval.update(preds_unscaled, labels_unscaled)
            mae_eval.update(preds_unscaled, labels_unscaled)
            mse_eval.update(preds_unscaled, labels_unscaled)

            r2_val = r2_eval.compute()
            mae_val = mae_eval.compute()
            mse_val = mse_eval.compute()

            return r2_val, mae_val, mse_val, preds_unscaled, labels_unscaled
