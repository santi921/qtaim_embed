import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from dgl.nn.pytorch import GATConv
from dgl.nn import SAGEConv

from qtaim_embed.utils.models import get_layer_args_homo, _split_batched_output

from qtaim_embed.models.layers_homo import (
    ResidualBlockHomo,
    UnifySize,
    MLPPredictor,
    DotPredictor,
    AttentionPredictor,
)

from qtaim_embed.models.layers import GraphConvDropoutBatch

from qtaim_embed.models.link_pred.losses import (
    HingeMetric,
    CrossEntropyMetric,
    AUCMetric,
    MarginMetric,
    F1Metric,
    AccuracyMetric,
)


class GCNLinkPred(pl.LightningModule):
    """
    Basic GNN model for link prediction
    Takes
        input_size: int, dimension of input features
        n_conv_layers: int, number of convolution layers
        conv_fn: str "GraphConvDropoutBatch"
        dropout: float, dropout rate
        batch_norm: bool, whether to use batch norm
        activation: str, activation function
        bias: bool, whether to use bias
        norm: str, normalization type
        lr: float, learning rate
        scheduler_name: str, scheduler type
        weight_decay: float, weight decay
        lr_plateau_patience: int, patience for lr scheduler
        lr_scale_factor: float, scale factor for lr scheduler
        loss_fn: str, loss function
        resid_n_graph_convs: int, number of graph convolutions per residual block
        scalers: list, list of scalers applied to each node type
        embedding_size: int, size of embedding layer
        predictor: str, predictor type
        predictor_param_dict: dict, dictionary of predictor parameters

    """

    def __init__(
        self,
        input_size=12,
        n_conv_layers=3,
        conv_fn="GraphConvDropoutBatch",
        resid_n_graph_convs=None,
        num_heads_gat=2,
        dropout_feat_gat=0.2,
        dropout_attn_gat=0.2,
        residual_gat=True,
        hidden_size=128,
        dropout=0.2,
        batch_norm=True,
        activation="ReLU",
        bias=True,
        norm="both",
        lr=1e-3,
        scheduler_name="reduce_on_plateau",
        weight_decay=0.0,
        lr_plateau_patience=5,
        lr_scale_factor=0.5,
        loss_fn="cross_entropy",
        embedding_size=128,
        predictor="Dot",
        predictor_param_dict={},
        aggregator_type="mean",
    ):
        super().__init__()
        self.learning_rate = lr

        assert conv_fn in [
            "GraphSAGE",
            "GATConv",
            "ResidualBlock",
            "GraphConvDropoutBatch",
        ], (
            "conv_fn must be either GraphConvDropoutBatch, GATConv, ResidualBlock, or GraphSAGE"
            + f" but got {conv_fn}",
        )

        if conv_fn == "ResidualBlock":
            assert resid_n_graph_convs is not None, (
                "resid_n_graph_convs must be specified for ResidualBlock"
                + f"but got {resid_n_graph_convs}"
            )

        # provide defaults
        if predictor_param_dict == {} and predictor == "MLP":
            predictor_param_dict = {
                "fc_layer_size": [512, 512],
                "fc_dropout": 0.2,
                "batch_norm": False,
                "activation": "ReLU",
            }
            print("...> Using default predictor parameters for MLP predictor!")

        params = {
            "input_size": input_size,
            "conv_fn": conv_fn,
            "dropout": dropout,
            "batch_norm_tf": batch_norm,
            "activation": activation,
            "bias": bias,
            "norm": norm,
            "n_conv_layers": n_conv_layers,
            "lr": lr,
            "weight_decay": weight_decay,
            "lr_plateau_patience": lr_plateau_patience,
            "lr_scale_factor": lr_scale_factor,
            "scheduler_name": scheduler_name,
            "loss_fn": loss_fn,
            "resid_n_graph_convs": resid_n_graph_convs,
            "embedding_size": embedding_size,
            "num_heads": num_heads_gat,
            "feat_drop": dropout_feat_gat,
            "attn_drop": dropout_attn_gat,
            "residual": residual_gat,
            "hidden_size": hidden_size,
            "predictor": predictor,
            "predictor_param_dict": predictor_param_dict,
            "aggregator_type": aggregator_type,
        }

        self.hparams.update(params)
        self.save_hyperparameters()

        # convert string activation to function
        if self.hparams.activation is not None:
            self.activation = getattr(torch.nn, self.hparams.activation)()

        else:
            self.activation = None

        # print("input size", input_size)
        self.embedding = UnifySize(
            input_dim=self.hparams.input_size,
            output_dim=self.hparams.embedding_size,
        )

        self.conv_layers = nn.ModuleList()

        if self.hparams.conv_fn == "GraphConvDropoutBatch":
            for i in range(self.hparams.n_conv_layers):

                layer_args = get_layer_args_homo(
                    self.hparams, i, activation=self.activation, embedding_in=True
                )
                #print("layer args: ", layer_args)
                self.conv_layers.append(GraphConvDropoutBatch(**layer_args["conv"]))

        elif self.hparams.conv_fn == "ResidualBlock":
            layer_tracker = 0

            while layer_tracker < self.hparams.n_conv_layers:
                if (
                    layer_tracker + self.hparams.resid_n_graph_convs
                    > self.hparams.n_conv_layers - 1
                ):
                    layer_ind = self.hparams.n_conv_layers - layer_tracker - 1
                else:
                    layer_ind = layer_tracker

                block_args = get_layer_args_homo(
                    self.hparams,
                    layer_ind,
                    embedding_in=True,
                    activation=self.activation,
                )
                # print("layer args: ", block_args)
                # print(layer_tracker)

                input_block = False
                if layer_tracker == 0:
                    # print("input block!")
                    input_block = True

                block_args["input_block"] = input_block
                block_args["resid_n_graph_convs"] = self.hparams.resid_n_graph_convs

                self.conv_layers.append(ResidualBlockHomo(**block_args))

                layer_tracker += self.hparams.resid_n_graph_convs

        elif self.hparams.conv_fn == "GATConv":
            for i in range(self.hparams.n_conv_layers):
                embedding_in = True
                layer_args = get_layer_args_homo(
                    self.hparams, i, activation=self.activation, embedding_in=True
                )

                self.conv_layers.append(GATConv(**layer_args["conv"]))

        elif self.hparams.conv_fn == "GraphSAGE":
            for i in range(self.hparams.n_conv_layers):

                layer_args = get_layer_args_homo(
                    self.hparams, i, activation=self.activation, embedding_in=True
                )
                #print("layer args:", layer_args)

                self.conv_layers.append(SAGEConv(**layer_args["conv"]))

        self.conv_layers = nn.ModuleList(self.conv_layers)

        if self.hparams.conv_fn in ["GraphSAGE", "GATConv"]:
            self.conv_out_size = self.conv_layers[-1]._out_feats
        else:
            self.conv_out_size = self.conv_layers[-1].out_feats
        # print(self.conv_layers)

        # self.conv_out_size = self.conv_layers[-1].out_feats

        ####################### predictor ######################
        #print("conv out: ", self.conv_out_size)

        if self.hparams.predictor == "MLP":
            self.predictor = MLPPredictor(
                h_feats=self.conv_out_size,  # out layer
                h_dims=self.hparams.predictor_param_dict["fc_layer_size"],
                dropout=self.hparams.predictor_param_dict["fc_dropout"],
                batch_norm=self.hparams.predictor_param_dict["batch_norm"],
                activation=self.hparams.predictor_param_dict["activation"],
            )

        elif self.hparams.predictor == "Dot":
            self.predictor = DotPredictor()

        elif self.hparams.predictor == "Attention":
            self.predictor = AttentionPredictor(in_feats=self.conv_out_size)

        # print("creating statistics")
        ###################### statistics ######################
        self.train_hinge = HingeMetric()  # .to(self.device)
        self.val_hinge = HingeMetric()  # .to(self.device)
        self.test_hinge = HingeMetric()  # .to(self.device)

        self.train_auc = AUCMetric()  # .to(self.device)
        self.val_auc = AUCMetric()  # .to(self.device)
        self.test_auc = AUCMetric()  # .to(self.device)

        self.train_accuracy = AccuracyMetric()  # .to(self.device)
        self.val_accuracy = AccuracyMetric()  # .to(self.device)
        self.test_accuracy = AccuracyMetric()  # .to(self.device)

        self.train_f1 = F1Metric()  # .to(self.device)
        self.val_f1 = F1Metric()  # .to(self.device)
        self.test_f1 = F1Metric()  # .to(self.device)

        self.loss = self.loss_function()

    def forward(self, pos_graph, neg_graph, inputs):
        """
        Forward pass
        """
        feats_embed = self.embedding(inputs)
        for ind, conv in enumerate(self.conv_layers):
            # print("conv ind: ", ind)
            if ind == 0:
                feats_pos = conv(pos_graph, feats_embed)
                feats_neg = conv(neg_graph, feats_embed)
            else:
                feats_pos = conv(pos_graph, feats_pos)
                feats_neg = conv(neg_graph, feats_neg)

            if self.hparams.conv_fn == "GATConv":
                if ind < self.hparams.n_conv_layers - 1:
                    # for k, v in feats.items():
                    feats_pos = feats_pos.reshape(
                        -1, self.hparams.num_heads * self.hparams.hidden_size
                    )
                    feats_neg = feats_neg.reshape(
                        -1, self.hparams.num_heads * self.hparams.hidden_size
                    )
                else:
                    # for k, v in feats.items():
                    feats_pos = feats_pos.reshape(-1, self.hparams.hidden_size)
                    feats_neg = feats_neg.reshape(-1, self.hparams.hidden_size)
        # print("predictor!")
        pos_pred = self.predictor(pos_graph, feats_pos)
        neg_pred = self.predictor(neg_graph, feats_neg)

        return pos_pred, neg_pred

    def loss_function(self):
        """
        Initialize loss function
        """

        if self.hparams.loss_fn == "auroc":
            loss_fn = AUCMetric()
        elif self.hparams.loss_fn == "margin":
            loss_fn = MarginMetric()
        elif self.hparams.loss_fn == "hinge":
            loss_fn = HingeMetric()
        elif self.hparams.loss_fn == "cross_entropy":
            loss_fn = CrossEntropyMetric()
        elif self.hparams.loss_fn == "accuracy":
            loss_fn = AccuracyMetric()
        elif self.hparams.loss_fn == "f1":
            loss_fn = F1Metric()
        else:
            loss_fn = CrossEntropyMetric()

        loss_fn = loss_fn  # .to(self.device)

        # print(self.device)
        # print(loss_fn.device)

        return loss_fn

    def compute_loss(self, target, pred):
        """
        Compute loss
        """
        # print("target device", target.device)
        # print("pred device", pred.device)
        # print("loss device", self.loss.device)
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

    def shared_step(self, batch, mode):

        # if mode == "train":
        positive_graph, negative_graph, feat = batch
        # else:
        #    positive_graph, negative_graph, negative_graph_explicit, feat = batch

        pred_pos, pred_neg = self.forward(
            positive_graph, negative_graph, feat
        )  # returns a dict of node types

        all_loss = self.compute_loss(pred_pos, pred_neg)
        # log loss
        self.log(
            f"{mode}_loss",
            all_loss.sum(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # if mode == "train":
        self.update_metrics(pred_pos, pred_neg, mode)
        # else:
        #    self.update_metrics(
        #        pred_pos, pred_neg, mode, target_neg=negative_graph_explicit
        #    )

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
        """
        Test step
        """
        return {"test_loss": self.shared_step(batch, mode="test")}

    def on_train_epoch_end(self):
        """
        Training epoch end
        """
        acc, f1, auc = self.compute_metrics(mode="train")

        if self.trainer.current_epoch < 2:
            self.log("val_f1", 0.0, prog_bar=False)
        self.log("train_f1", f1, prog_bar=True, sync_dist=True)
        self.log("train_auc", auc, prog_bar=False, sync_dist=True)
        self.log("train_accuracy", acc, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        """
        Validation epoch end
        """
        acc, f1, auc = self.compute_metrics(mode="val")
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_auc", auc, prog_bar=False, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=False, sync_dist=True)

    def on_test_epoch_end(self):
        """
        Test epoch end
        """
        acc, f1, auc = self.compute_metrics(mode="test")
        self.log("test_f1", f1, prog_bar=True, sync_dist=True)
        self.log("test_auc", auc, prog_bar=False, sync_dist=True)
        self.log("test_accuracy", acc, prog_bar=False, sync_dist=True)

    def update_metrics(self, pred, target, mode):
        """
        Update metrics using torchmetrics interfaces
        """

        if mode == "train":
            self.train_accuracy.update(pred, target)
            self.train_f1.update(pred, target)
            self.train_auc.update(pred, target)

        elif mode == "val":
            self.val_accuracy.update(pred, target)
            self.val_f1.update(pred, target)
            self.val_auc.update(pred, target)

        elif mode == "test":
            self.test_accuracy(pred, target)
            self.test_f1(pred, target)
            self.test_auc(pred, target)

    def compute_metrics(self, mode):
        """
        Compute metrics using torchmetrics interfaces
        """

        if mode == "train":
            acc = self.train_accuracy.compute()
            f1 = self.train_f1.compute()
            auc = self.train_auc.compute()
            self.train_accuracy.reset()
            self.train_f1.reset()
            self.train_auc.reset()

        elif mode == "val":
            acc = self.val_accuracy.compute()
            f1 = self.val_f1.compute()
            auc = self.val_auc.compute()
            self.val_accuracy.reset()
            self.val_f1.reset()
            self.val_auc.reset()

        elif mode == "test":
            acc = self.test_accuracy.compute()
            f1 = self.test_f1.compute()
            auc = self.test_auc.compute()
            self.test_accuracy.reset()
            self.test_f1.reset()
            self.test_auc.reset()

        return acc, f1, auc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = self._config_lr_scheduler(optimizer)

        lr_scheduler = {"scheduler": scheduler, "monitor": "val_f1"}

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

    def evaluate_manually(self, dataloader):
        """
        Evaluate a set of data manually
        Takes
            feats: dict, dictionary of batched features
            scaler_list: list, list of scalers
        """
        self.eval()

        metric_hinge = HingeMetric()
        metric_cross = CrossEntropyMetric()
        metric_margin = MarginMetric()
        metric_f1 = F1Metric()
        metric_acc = AccuracyMetric()
        metric_auc = AUCMetric()

        for positive_graph, negative_graph, feat in dataloader:
            positive_graph = positive_graph
            negative_graph = negative_graph
            pred_pos, pred_neg = self.forward(positive_graph, negative_graph, feat)

            metric_f1.update(pred_pos, pred_neg)
            metric_acc.update(pred_pos, pred_neg)
            metric_hinge.update(pred_pos, pred_neg)
            metric_auc.update(pred_pos, pred_neg)
            metric_margin.update(pred_pos, pred_neg)
            metric_cross.update(pred_pos, pred_neg)

        metric_cross_val = metric_cross.compute()
        metric_f1_val = metric_f1.compute()
        metric_hinge_val = metric_hinge.compute()
        metric_acc_val = metric_acc.compute()
        metric_auc_val = metric_auc.compute()
        metric_margin_val = metric_margin.compute()

        stats_dict = {
            "Accuracy": float(metric_acc_val.numpy()),
            "AUROC": float(metric_auc_val.numpy()),
            "Cross_entropy": float(metric_cross_val.numpy()),
            "F1": float(metric_f1_val.numpy()),
            "Hinge": float(metric_hinge_val.numpy()),
            "Margin": float(metric_margin_val.numpy()),
        }

        return stats_dict
