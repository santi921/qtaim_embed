# baseline GNN model for node-level regression
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import pytorch_lightning as pl
import dgl.nn.pytorch as dglnn
from torchmetrics.wrappers import MultioutputWrapper
import torchmetrics

from qtaim_embed.models.utils import _split_batched_output
from qtaim_embed.models.layers import GraphConvDropoutBatch

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
    """
    
    def __init__(
        self, 
        atom_input_size=12, 
        bond_input_size=8, 
        global_input_size=3,
        n_conv_layers=3,
        target_dict= {
            "atom": "extra_feat_bond_esp_total"
        },
        conv_fn = "GraphConvDropoutBatch", 
        dropout=0.2, 
        batch_norm=True,
        activation=None,
        bias=True,
        norm="both",
        aggregate="sum",
        lr=1e-3,
        scheduler_name="reduce_on_plateau",
        weight_decay=0.0,
        lr_plateau_patience=5,
        lr_scale_factor=0.5,
        loss_fn="mse",  
    ):


        super().__init__()
        self.learning_rate = lr


        output_dims = 0
        for k, v in target_dict.items():
            output_dims += len(v)

        params = {
            "atom_input_size": atom_input_size,
            "bond_input_size": bond_input_size,
            "global_input_size": global_input_size,
            "conv_fn": conv_fn,
            "target_dict": target_dict,
            "output_dims": output_dims,
            "dropout": dropout,
            "batch_norm": batch_norm,
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
        }

        self.hparams.update(params)
        self.save_hyperparameters()

        # convert string activation to function
        if self.hparams.activation is not None:
            self.hparams.activation = getattr(torch.nn, self.hparams.activation)()

        self.conv_layers = nn.ModuleList()
        for i in range(self.hparams.n_conv_layers):
            atom_out = self.hparams.atom_input_size
            bond_out = self.hparams.bond_input_size
            global_out = self.hparams.global_input_size

            # check if it's the last layer
            if i == self.hparams.n_conv_layers - 1:
                if "atom" in self.hparams.target_dict.keys():
                    atom_out = len(self.hparams.target_dict["atom"])
                if "bond" in self.hparams.target_dict.keys():
                    bond_out = len(self.hparams.target_dict["bond"])
                if "global" in self.hparams.target_dict.keys():
                    global_out = len(self.hparams.target_dict["global"])
            
            
            a2b_args = {
                "in_feats": self.hparams.atom_input_size,
                "out_feats": bond_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
                "allow_zero_in_degree": True,
            }

            b2a_args = {
                "in_feats": self.hparams.bond_input_size,
                "out_feats": atom_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
                "allow_zero_in_degree": True,

            }

            a2g_args = {
                "in_feats": self.hparams.atom_input_size,
                "out_feats": global_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
                "allow_zero_in_degree": True,

            }

            g2a_args = {
                "in_feats": self.hparams.global_input_size,
                "out_feats": atom_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
                "allow_zero_in_degree": True,

            }

            b2g_args = {
                "in_feats": self.hparams.bond_input_size,
                "out_feats": global_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
                "allow_zero_in_degree": True,

            }

            g2b_args = {
                "in_feats": self.hparams.global_input_size,
                "out_feats": bond_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
                "allow_zero_in_degree": True,

            }

            a2a_args = {
                "in_feats": self.hparams.atom_input_size,
                "out_feats": atom_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
            }

            b2b_args = {
                "in_feats": self.hparams.bond_input_size,
                "out_feats": bond_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
            }

            g2g_args = {
                "in_feats": self.hparams.global_input_size,
                "out_feats": global_out,
                "dropout": self.hparams.dropout,
                "batch_norm_tf": self.hparams.batch_norm,
                "activation": self.hparams.activation,
                "bias": self.hparams.bias,
                "norm": self.hparams.norm,
            }


            self.conv_layers.append(dglnn.HeteroGraphConv(
                {
                    "a2b": GraphConvDropoutBatch(**a2b_args),
                    "b2a": GraphConvDropoutBatch(**b2a_args),
                    "a2g": GraphConvDropoutBatch(**a2g_args),
                    "g2a": GraphConvDropoutBatch(**g2a_args),
                    "b2g": GraphConvDropoutBatch(**b2g_args),
                    "g2b": GraphConvDropoutBatch(**g2b_args),
                    "a2a": GraphConvDropoutBatch(**a2a_args),
                    "b2b": GraphConvDropoutBatch(**b2b_args),
                    "g2g": GraphConvDropoutBatch(**g2g_args),
                },
                aggregate=self.hparams.aggregate,
            )   
            )
         
        self.loss = self.loss_function()

        print("number of output dims", output_dims)

        # create multioutput wrapper for metrics
        self.train_r2 = MultioutputWrapper(torchmetrics.R2Score(), num_outputs=output_dims)
        self.train_torch_l1 = MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=output_dims)
        self.train_torch_mse = MultioutputWrapper(torchmetrics.MeanSquaredError(), num_outputs=output_dims)
        self.val_r2 = MultioutputWrapper(torchmetrics.R2Score(), num_outputs=output_dims)
        self.val_torch_l1 = MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=output_dims)
        self.val_torch_mse = MultioutputWrapper(torchmetrics.MeanSquaredError(), num_outputs=output_dims)
        self.test_r2 = MultioutputWrapper(torchmetrics.R2Score(), num_outputs=output_dims)
        self.test_torch_l1 = MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=output_dims)
        self.test_torch_mse = MultioutputWrapper(torchmetrics.MeanSquaredError(), num_outputs=output_dims)

    def forward(self, graph, inputs):
        """
        Forward pass
        """
        for ind, conv in enumerate(self.conv_layers):
            if ind == 0: 
                feats = conv(graph, inputs)
            else: 
                feats = conv(graph, feats)


        return feats  


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
        global_feats[layer_idx] = _split_batched_output(graph, feats["global"], "global")

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
        logits = self.forward(batch_graph, batch_graph.ndata["feat"])
        max_nodes = -1 
        for target_type, target_list in self.hparams.target_dict.items():
            if  target_list is not None and len(target_list) > 0:
                labels = batch_label[target_type]
                logits_temp = logits[target_type]
                if max_nodes < logits_temp.shape[0]:
                    max_nodes = logits_temp.shape[0]
                logits_list.append(logits_temp)
                labels_list.append(labels)
        logits_list = [F.pad(i, (0, 0, 0, max_nodes - i.shape[0])) for i in logits_list]
        labels_list = [F.pad(i, (0, 0, 0, max_nodes - i.shape[0])) for i in labels_list]
        logits = torch.cat(logits_list, dim=1)
        labels = torch.cat(labels_list, dim=1)
        
        all_loss = self.compute_loss(logits, labels)
        
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
            loss_multi = MultioutputWrapper(torchmetrics.MeanSquaredError(), num_outputs=self.hparams.output_dims)
        elif self.hparams.loss_fn == "smape":
            loss_multi = MultioutputWrapper(torchmetrics.SymmetricMeanAbsolutePercentageError(), num_outputs=self.hparams.output_dims)
        elif self.hparams.loss_fn == "mae":
            loss_multi = MultioutputWrapper(torchmetrics.MeanAbsoluteError(), num_outputs=self.hparams.output_dims)
        else:
            loss_multi = MultioutputWrapper(torchmetrics.MeanSquaredError(), num_outputs=self.hparams.output_dims)
        
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
        return self.shared_step(batch, mode="val")


    def on_train_epoch_end(self):
        """
        Training epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="train")
        # get epoch number 
        if self.trainer.current_epoch == 0:
            self.log("val_mae", 10**10, prog_bar=False)
        self.log("train_r2", r2.median(), prog_bar=False, sync_dist=True)
        self.log("train_mae", mae.mean(), prog_bar=False, sync_dist=True)
        self.log("train_mse", mse.mean(), prog_bar=False, sync_dist=True)


    def on_validation_epoch_end(self):
        """
        Validation epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="val")
        self.log("val_r2", r2.median(), prog_bar=True, sync_dist=True)
        self.log("val_mae", mae.mean(), prog_bar=True, sync_dist=True)
        self.log("val_mse", mse.mean(), prog_bar=False, sync_dist=True)


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

        #if self.stdev is not None:
        #    # print("stdev", self.stdev)
        #    torch_l1 = torch_l1 * self.stdev
        #    torch_mse = torch_mse * self.stdev * self.stdev
        #else:
        #    print("scaling is 1!" + "*" * 20)


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
                optimizer, mode="max", 
                factor=self.hparams.lr_scale_factor, 
                patience=self.hparams.lr_plateau_patience, 
                verbose=True
            )

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {scheduler_name}")

        return scheduler