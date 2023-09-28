# baseline GNN model for node-level regression
import torch
import pytorch_lightning as pl
import torchmetrics
import dgl.nn.pytorch as dglnn
from torchmetrics.wrappers import MultioutputWrapper
from qtaim_embed.models.layers import GraphConvDropoutBatch

class GCNNodePred(pl.LightningModule):
    def __init__(
        self, 
        atom_input_size=64, 
        bond_input_size=64, 
        global_input_size=64,
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
        ):
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


        super().__init__()

        self.atom_input_size = atom_input_size
        self.bond_input_size = bond_input_size
        self.global_input_size = global_input_size
        self.layer_conv = conv_fn
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.bias = bias
        self.norm = norm
        self.aggregate = aggregate
        self.n_conv_layers = n_conv_layers
        self.target_dict = target_dict
        self.output_dims = 0 

        for k, v in target_dict.items():
            self.output_dims += len(v)

        params = {
            "atom_input_size": atom_input_size,
            "bond_input_size": bond_input_size,
            "global_input_size": global_input_size,
            "conv_fn": conv_fn,
            "target_dict": target_dict,
            "output_dims": self.output_dims,
            "dropout": dropout,
            "batch_norm": batch_norm,
            "activation": activation,
            "bias": bias,
            "norm": norm,
            "aggregate": "sum",
            "n_conv_layers": n_conv_layers,
        }

        self.hparams.update(params)
        self.save_hyperparameters()

        # convert string activation to function
        if self.activation is not None:
            self.activation = getattr(torch.nn, self.activation)()

        self.conv_layers = []
        for i in range(self.n_conv_layers):
            
            atom_out = self.atom_input_size
            bond_out = self.bond_input_size
            global_out = self.global_input_size

            # check if it's the last layer
            if i == self.n_conv_layers - 1:
                if "atom" in self.target_dict.keys():
                    atom_out = len(self.target_dict["atom"])
                if "bond" in self.target_dict.keys():
                    bond_out = len(self.target_dict["bond"])
                if "global" in self.target_dict.keys():
                    global_out = len(self.target_dict["global"])

            a2b_args = {
                "in_feats": self.atom_input_size,
                "out_feats": bond_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,
            }

            b2a_args = {
                "in_feats": self.bond_input_size,
                "out_feats": atom_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,

            }

            a2g_args = {
                "in_feats": self.atom_input_size,
                "out_feats": global_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,

            }

            g2a_args = {
                "in_feats": self.global_input_size,
                "out_feats": atom_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,

            }

            b2g_args = {
                "in_feats": self.bond_input_size,
                "out_feats": global_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,   
                "bias": self.bias,
                "norm": self.norm,
             
            }

            g2b_args = {
                "in_feats": self.global_input_size,
                "out_feats": bond_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,

            }

            a2a_args = {
                "in_feats": self.atom_input_size,
                "out_feats": atom_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,

            }

            b2b_args = {
                "in_feats": self.bond_input_size,
                "out_feats": bond_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,

            }

            g2g_args = {
                "in_feats": self.global_input_size,
                "out_feats": global_out,
                "dropout": self.dropout,
                "batch_norm_tf": self.batch_norm,
                "activation": self.activation,
                "bias": self.bias,
                "norm": self.norm,

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
                aggregate=self.aggregate,
            )   
            )
        
  

    def forward(self, graph, inputs):
        """
        Forward pass
        """
        for conv in self.conv_layers:
            inputs = conv(graph, inputs)
        return inputs


    def feature_at_each_layer():
        """
        Get feature at each layer
        """

        
    def shared_step():
        # Todo
        pass


    def loss_function(self):
        """
        Initialize loss function
        """
        if self.hparams.loss_fn == "mse":
            # loss_fn = WeightedMSELoss(reduction="mean")
            loss_fn = torchmetrics.MeanSquaredError()
        elif self.hparams.loss_fn == "smape":
            loss_fn = torchmetrics.SymmetricMeanAbsolutePercentageError()
        elif self.hparams.loss_fn == "mae":
            # loss_fn = WeightedL1Loss(reduction="mean")
            loss_fn = torchmetrics.MeanAbsoluteError()
        else:
            loss_fn = torchmetrics.MeanSquaredError()

        return loss_fn
    

    def compute_loss(self, target, pred):
        """
        Compute loss
        """
        return self.loss(target, pred)


    def configure_optimizers():        
        pass
    def _config_lr_schedule():
        pass

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
        pass

    def training_epoch_end(self, outputs):
        """
        Training epoch end
        """
        r2, mae, mse = self.compute_metrics(outputs, mode="train")
        self.log("train_r2", r2, prog_bar=True, sync_dist=True)
        self.log("train_mae", mae, prog_bar=True, sync_dist=True)
        self.log("train_mse", mse, prog_bar=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        """
        Validation epoch end
        """
        r2, mae, mse = self.compute_metrics(outputs, mode="val")
        self.log("val_r2", r2, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae, prog_bar=True, sync_dist=True)
        self.log("val_mse", mse, prog_bar=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        """
        Test epoch end
        """
        r2, mae, mse = self.compute_metrics(outputs, mode="test")
        self.log("test_r2", r2, prog_bar=True, sync_dist=True)
        self.log("test_mae", mae, prog_bar=True, sync_dist=True)
        self.log("test_mse", mse, prog_bar=True, sync_dist=True)

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
            # l1 = self.val_l1.compute()
            r2 = self.val_r2.compute()
            torch_l1 = self.val_torch_l1.compute()
            torch_mse = self.val_torch_mse.compute()
            self.val_r2.reset()
            # self.val_l1.reset()
            self.val_torch_l1.reset()
            self.val_torch_mse.reset()

        elif mode == "test":
            # l1 = self.test_l1.compute()
            r2 = self.test_r2.compute()
            torch_l1 = self.test_torch_l1.compute()
            torch_mse = self.test_torch_mse.compute()
            self.test_r2.reset()
            # self.test_l1.reset()
            self.test_torch_l1.reset()
            self.test_torch_mse.reset()

        if self.stdev is not None:
            # print("stdev", self.stdev)
            torch_l1 = torch_l1 * self.stdev
            torch_mse = torch_mse * self.stdev * self.stdev
        else:
            print("scaling is 1!" + "*" * 20)


        return r2, torch_l1, torch_mse

