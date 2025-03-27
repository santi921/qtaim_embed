# baseline GNN model for node-level regression
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn
from torchmetrics.wrappers import MultioutputWrapper
import torchmetrics

from qtaim_embed.utils.models import _split_batched_output, get_layer_args
from qtaim_embed.models.layers import GraphConvDropoutBatch, ResidualBlock, UnifySize
from dgl.nn.pytorch import GATConv


class GCNNodePred(pl.LightningModule):
    """
    Basic GNN model for node-level regression
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

    """

    def __init__(
        self,
        atom_input_size=12,
        bond_input_size=8,
        global_input_size=3,
        n_conv_layers=3,
        target_dict={"atom": "extra_feat_bond_esp_total"},
        conv_fn="GraphConvDropoutBatch",
        resid_n_graph_convs=None,
        dropout=0.2,
        batch_norm=True,
        activation=None,
        bias=True,
        norm="both",
        aggregate="sum",
        lr=1e-3,
        embedding_size=16,
        hidden_size=128,
        num_heads_gat=4,
        dropout_feat_gat=0.2,
        dropout_attn_gat=0.2,
        residual_gat=True,
        scheduler_name="reduce_on_plateau",
        weight_decay=0.0,
        lr_plateau_patience=5,
        lr_scale_factor=0.5,
        loss_fn="mse",
        compiled=None
    ):
        super().__init__()
        self.learning_rate = lr

        if "atom" not in target_dict or target_dict["atom"] == []:
            target_dict["atom"] = [None]
        if "bond" not in target_dict or target_dict["bond"] == []:
            target_dict["bond"] = [None]
        if "global" not in target_dict or target_dict["global"] == []:
            target_dict["global"] = [None]

        output_dims = 0
        # print("target dict", target_dict)
        for k, v in target_dict.items():
            if v != [None]:
                output_dims += len(v)

        assert conv_fn in ["GraphConvDropoutBatch", "ResidualBlock", "GATConv"], (
            "conv_fn must be either GraphConvDropoutBatch, GATConv or ResidualBlock"
            + f"but got {conv_fn}"
        )

        if conv_fn == "ResidualBlock":
            assert resid_n_graph_convs is not None, (
                "resid_n_graph_convs must be specified for ResidualBlock"
                + f"but got {resid_n_graph_convs}"
            )

        params = {
            "atom_input_size": atom_input_size,
            "bond_input_size": bond_input_size,
            "global_input_size": global_input_size,
            "conv_fn": conv_fn,
            "target_dict": target_dict,
            "output_dims": output_dims,
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
            "num_heads": num_heads_gat,
            "feat_drop": dropout_feat_gat,
            "attn_drop": dropout_attn_gat,
            "residual": residual_gat,
            "resid_n_graph_convs": resid_n_graph_convs,
            "hidden_size": hidden_size,
            "embedding_size": embedding_size,
            "compiled": compiled,
        }

        self.hparams.update(params)
        self.save_hyperparameters()

        # convert string activation to function
        if self.hparams.activation is not None:
            self.hparams.activation = getattr(torch.nn, self.hparams.activation)()

        input_size = {
            "atom": self.hparams.atom_input_size,
            "bond": self.hparams.bond_input_size,
            "global": self.hparams.global_input_size,
        }

        self.embedding = UnifySize(
            input_dim=input_size,
            output_dim=self.hparams.embedding_size,
        )

        self.conv_layers = nn.ModuleList()

        if self.hparams.conv_fn == "GraphConvDropoutBatch":
            for i in range(self.hparams.n_conv_layers):
                embedding_in = True
                layer_args = get_layer_args(
                    self.hparams,
                    i,
                    activation=self.hparams.activation,
                    embedding_in=embedding_in,
                )

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
                    activation=self.hparams.activation,
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

                layer_args = get_layer_args(
                    self.hparams,
                    i,
                    activation=self.hparams.activation,
                    embedding_in=True,
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
        # print(self.conv_layers)
        self.target_dict = target_dict

        self.loss = self.loss_function()

        print("number of output dims", output_dims)

        # create multioutput wrapper for metrics
        self.train_r2 = MultioutputWrapper(
            torchmetrics.R2Score(), num_outputs=output_dims
        )
        self.train_torch_l1 = MultioutputWrapper(
            torchmetrics.MeanAbsoluteError(), num_outputs=output_dims
        )
        self.train_torch_mse = MultioutputWrapper(
            torchmetrics.MeanSquaredError(squared=False), num_outputs=output_dims
        )
        self.val_r2 = MultioutputWrapper(
            torchmetrics.R2Score(), num_outputs=output_dims
        )
        self.val_torch_l1 = MultioutputWrapper(
            torchmetrics.MeanAbsoluteError(), num_outputs=output_dims
        )
        self.val_torch_mse = MultioutputWrapper(
            torchmetrics.MeanSquaredError(squared=False), num_outputs=output_dims
        )
        self.test_r2 = MultioutputWrapper(
            torchmetrics.R2Score(), num_outputs=output_dims
        )
        self.test_torch_l1 = MultioutputWrapper(
            torchmetrics.MeanAbsoluteError(), num_outputs=output_dims
        )
        self.test_torch_mse = MultioutputWrapper(
            torchmetrics.MeanSquaredError(squared=False), num_outputs=output_dims
        )


        self.forward_fn = (
            torch.compile(self.compiled_forward)
            if compiled
            else self.compiled_forward
        )

    def compiled_forward(self, graph, inputs):
        """
        Forward pass
        """
        feats = self.embedding(inputs)

        for ind, conv in enumerate(self.conv_layers):

            # if ind == 0:
            #    feats = conv(graph, inputs)
            # else:
            feats = conv(graph, feats)

            if self.hparams.conv_fn == "GATConv":
                if ind < self.hparams.n_conv_layers - 1:
                    for k, v in feats.items():
                        # print(k)
                        # print("feats shape", v.shape)
                        # print("conv ind", ind)

                        feats[k] = v.reshape(
                            -1, self.hparams.num_heads * self.hparams.hidden_size
                        )
                else:
                    for k, v in feats.items():
                        # print(k)
                        # print("feats shape", v.shape)
                        # print("conv ind", ind)
                        feats[k] = v.reshape(-1, len(self.target_dict[k]))

        # filter features if output is None for one of the node types
        for k, v in self.hparams.target_dict.items():
            if v == [None]:
                del feats[k]

        return feats

    def forward(self, graph, inputs):
        """
        Forward pass
        """
        return self.forward_fn(graph, inputs)


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

        return bond_feats, atom_feats, global_feats

    def shared_step(self, batch, mode):
        batch_graph, batch_label = batch
        logits_list = []
        labels_list = []
        logits = self.forward(
            batch_graph, batch_graph.ndata["feat"]
        )  # returns a dict of node types
        #print("lmdb batch", batch_graph, batch_label.keys())
        max_nodes = -1

        for target_type, target_list in self.hparams.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                labels = batch_label[target_type]
                logits_temp = logits[target_type]
                if max_nodes < logits_temp.shape[0]:
                    max_nodes = logits_temp.shape[0]
                logits_list.append(logits_temp)
                labels_list.append(labels)
        logits_list = [
            F.pad(i, (0, 0, 0, max_nodes - i.shape[0])) for i in logits_list
        ]  # unify node size
        labels_list = [
            F.pad(i, (0, 0, 0, max_nodes - i.shape[0])) for i in labels_list
        ]  # unify node size

        logits = torch.cat(logits_list, dim=1)
        labels = torch.cat(labels_list, dim=1)

         
        all_loss = self.compute_loss(logits, labels)
        
        # compat with older torchmetrics/dgl 
        if type(all_loss) == list: 
            all_loss = torch.stack(all_loss)


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

        return all_loss.sum()

    def loss_function(self):
        """
        Initialize loss function
        """
        if self.hparams.loss_fn == "mse":
            # make multioutput wrapper for mse
            loss_multi = MultioutputWrapper(
                torchmetrics.MeanSquaredError(), num_outputs=self.hparams.output_dims
            )
        elif self.hparams.loss_fn == "smape":
            loss_multi = MultioutputWrapper(
                torchmetrics.SymmetricMeanAbsolutePercentageError(),
                num_outputs=self.hparams.output_dims,
            )
        elif self.hparams.loss_fn == "mae":
            loss_multi = MultioutputWrapper(
                torchmetrics.MeanAbsoluteError(), num_outputs=self.hparams.output_dims
            )
        else:
            loss_multi = MultioutputWrapper(
                torchmetrics.MeanSquaredError(), num_outputs=self.hparams.output_dims
            )

        loss_fn = loss_multi
        return loss_fn

    def compute_loss(self, target, pred):
        """
        Compute loss
        """
        return self.loss(target, pred)

    def training_step(self, batch, batch_idx):
        """
        Train step
        """
        return self.shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        """
        Val step
        """
        return self.shared_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        # Todo
        return self.shared_step(batch, mode="test")

    def on_train_epoch_end(self):
        """
        Training epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="train")
        
        if type(r2) == list: 
            r2 = torch.stack(r2)
        if type(mae) == list:
            mae = torch.stack(mae)
        if type(mse) == list:
            mse = torch.stack(mse)

        # get epoch number
        if self.trainer.current_epoch == 0:
            self.log("val_mae", 10000000.0, prog_bar=False, sync_dist=True)

        self.log("train_r2", r2.median(), prog_bar=False, sync_dist=True)
        self.log("train_mae", mae.mean(), prog_bar=False, sync_dist=True)
        self.log("train_mse", mse.mean(), prog_bar=True, sync_dist=True)

        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                for i, target in enumerate(target_list):
                    self.log(
                        f"train_r2_{target_type}_{target}",
                        r2[i],
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"train_mae_{target_type}_{target}",
                        mae[i],
                        prog_bar=False,
                        sync_dist=True,
                    )

    def on_validation_epoch_end(self):
        """
        Validation epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="val")
        
        if type(r2) == list: 
            r2 = torch.stack(r2)
        if type(mae) == list:
            mae = torch.stack(mae)
        if type(mse) == list:
            mse = torch.stack(mse)

        r2_median = r2.median().type(torch.float32)
        self.log("val_r2", r2_median, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae.mean(), prog_bar=False, sync_dist=True)
        self.log("val_mse", mse.mean(), prog_bar=True, sync_dist=True)

        # log each target r2 and mae
        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                for i, target in enumerate(target_list):
                    self.log(
                        f"val_r2_{target_type}_{target}",
                        r2[i],
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"val_mae_{target_type}_{target}",
                        mae[i],
                        prog_bar=False,
                        sync_dist=True,
                    )

    def on_test_epoch_end(self):
        """
        Test epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="test")
        
        # compat with older torchmetrics/dgl 
        if type(r2) == list: 
            r2 = torch.stack(r2)
        if type(mae) == list:
            mae = torch.stack(mae)
        if type(mse) == list:
            mse = torch.stack(mse)

        self.log("test_r2", r2.median(), prog_bar=False, sync_dist=True)
        self.log("test_mae", mae.mean(), prog_bar=False, sync_dist=True)
        self.log("test_mse", mse.mean(), prog_bar=False, sync_dist=True)

        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                for i, target in enumerate(target_list):
                    self.log(
                        f"test_r2_{target_type}_{target}",
                        r2[i],
                        prog_bar=False,
                        sync_dist=True,
                    )
                    self.log(
                        f"test_mae_{target_type}_{target}",
                        mae[i],
                        prog_bar=False,
                        sync_dist=True,
                    )

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
                mode="max",
                factor=self.hparams.lr_scale_factor,
                patience=self.hparams.lr_plateau_patience,
                verbose=True,
            )

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {scheduler_name}")

        return scheduler

    def evaluate_manually(self, test_dataloader, scaler_list):
        """
        Evaluate a set of data manually
        Takes
            feats: dict, dictionary of batched features
            scaler_list: list, list of scalers
        """
        r2_dict = {}
        mae_dict = {}
        r2_eval = {}
        mae_eval = {}
        pred_dict = {}
        label_dict = {}
        # print("target dict: ", self.target_dict)

        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:

                r2_eval[target_type] = MultioutputWrapper(
                    torchmetrics.R2Score(),
                    num_outputs=len(self.target_dict[target_type]),
                )
                mae_eval[target_type] = MultioutputWrapper(
                    torchmetrics.MeanAbsoluteError(),
                    num_outputs=len(self.target_dict[target_type]),
                )
                pred_dict[target_type] = []
                label_dict[target_type] = []

        # batch_graph, batch_label = batch
        for batch_graph, batched_label in test_dataloader:
            preds = self.forward(batch_graph, batch_graph.ndata["feat"])
            # detach every tensor in dictionary
            preds_unscaled = {k: deepcopy(v.detach()) for k, v in preds.items()}
            # print("preds shape", preds_unscaled["atom"].shape)
            labels_unscaled = deepcopy(batched_label)

            for scaler in scaler_list:
                labels_unscaled = scaler.inverse_feats(labels_unscaled)
                preds_unscaled = scaler.inverse_feats(preds_unscaled)

            # max_nodes = -1
            for target_type, target_list in self.target_dict.items():
                if target_list != [None] and len(target_list) > 0:
                    r2_eval[target_type].update(
                        preds_unscaled[target_type], labels_unscaled[target_type]
                    )
                    mae_eval[target_type].update(
                        preds_unscaled[target_type], labels_unscaled[target_type]
                    )
                    pred_dict[target_type].append(preds_unscaled[target_type].numpy())
                    label_dict[target_type].append(labels_unscaled[target_type].numpy())

        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                r2_dict[target_type] = r2_eval[target_type].compute()
                mae_dict[target_type] = mae_eval[target_type].compute()
                pred_dict[target_type] = np.concatenate(pred_dict[target_type])
                label_dict[target_type] = np.concatenate(label_dict[target_type])

        return r2_dict, mae_dict, pred_dict, label_dict
