# baseline GNN model for node-level regression
import logging
from copy import deepcopy
import numpy as np

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

import pytorch_lightning as pl
from torch_geometric.nn import HeteroConv, GATConv
from torch_geometric.data import HeteroData
from torchmetrics.wrappers import MultioutputWrapper
import torchmetrics

from qtaim_embed.utils.models import _split_batched_output, get_layer_args
from qtaim_embed.models.layers import (
    GraphConvDropoutBatch,
    ResidualBlock,
    UnifySize,
    EDGE_TYPE_MAP,
)
from qtaim_embed.models.encoders import attach_encoder, check_encoder_hparams, encode_atom_inputs
from qtaim_embed.models.layers_dense import DenseHeteroBatch, DenseResidualBlock, to_dense_hetero
from qtaim_embed.models.optim import build_adam
import torch.autograd.profiler as profiler

from typing import List, Tuple, Dict, Optional


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
        encoder_fn: str, optional 3D encoder ("none", "schnet", "dimenetpp",
            "equivariant"); requires atom.pos/atom.z on the graphs and
            compiled=False. Output is concatenated onto atom features ahead
            of UnifySize.
        encoder_hidden: int, encoder output width added to the atom input dim
        encoder_cutoff: float, radius-graph cutoff in Angstrom
        encoder_n_interactions: int, number of encoder interaction blocks
        encoder_num_gaussians: int, schnet RBF size
        encoder_num_radial: int, dimenetpp/equivariant radial basis size
        encoder_lmax: int, max spherical harmonic l (equivariant only)
        encoder_max_neighbors: int, nearest-neighbor cap (dimenetpp only)
        encoder_tp: str, equivariant tensor product, "channelwise" (default) or "fully_connected"
        encoder_max_z: int, atomic-number embedding rows in the encoder (119 covers the table)
        dense_grid: int, ResidualBlockDense pads molecules to multiples of this many atoms/bonds
        bn_before_activation: bool, conv -> BN -> activation -> dropout instead of BN last (see docs/research/2026-09-tm-react-eval-divergence.md)
        global_aggr: str, "sum" (GraphConv add) or "mean" for the a2g / b2g relations into the global node
        compile_mode: str, torch.compile mode for the dense conv stack: "reduce-overhead" (CUDA graphs, default) or "default" (plain inductor; required with accumulate_grad_batches > 1, whose accumulated .grad tensors alias CUDA-graph outputs)

    """

    def __init__(
        self,
        atom_input_size: int = 12,
        bond_input_size: int = 8,
        global_input_size: int = 3,
        n_conv_layers: int = 3,
        target_dict: Dict[str, List[str]] = {"atom": "extra_feat_bond_esp_total"},
        conv_fn: str = "GraphConvDropoutBatch",
        resid_n_graph_convs: Optional[int] = None,
        dropout: float = 0.2,
        batch_norm: bool = True,
        activation: Optional[str] = None,
        bias: bool = True,
        norm: str = "both",
        aggregate: str = "sum",
        lr: float = 1e-3,
        embedding_size: int = 16,
        hidden_size: int = 128,
        num_heads_gat: int = 4,
        dropout_feat_gat: float = 0.2,
        dropout_attn_gat: float = 0.2,
        residual_gat: bool = True,
        scheduler_name: str = "reduce_on_plateau",
        weight_decay: float = 0.0,
        lr_plateau_patience: int = 5,
        lr_scale_factor: float = 0.5,
        loss_fn: str = "mse",
        compiled: bool = False,
        encoder_fn: str = "none",
        encoder_hidden: int = 64,
        encoder_cutoff: float = 5.0,
        encoder_n_interactions: int = 3,
        encoder_num_gaussians: int = 50,
        encoder_num_radial: int = 6,
        encoder_lmax: int = 1,
        encoder_max_neighbors: int = 16,
        encoder_tp: str = "channelwise",
        encoder_max_z: int = 119,
        dense_grid: int = 16,
        bn_before_activation: bool = False,
        global_aggr: str = "sum",
        compile_mode: str = "reduce-overhead",
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

        assert conv_fn in ["GraphConvDropoutBatch", "ResidualBlock", "ResidualBlockDense", "GATConv"], (
            "conv_fn must be either GraphConvDropoutBatch, GATConv or ResidualBlock"
            + f"but got {conv_fn}"
        )

        if conv_fn in ("ResidualBlock", "ResidualBlockDense"):
            assert resid_n_graph_convs is not None, (
                "resid_n_graph_convs must be specified for ResidualBlock"
                + f"but got {resid_n_graph_convs}"
            )

        check_encoder_hparams(encoder_fn, compiled, conv_fn)

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
            "encoder_fn": encoder_fn,
            "encoder_hidden": encoder_hidden,
            "encoder_cutoff": encoder_cutoff,
            "encoder_n_interactions": encoder_n_interactions,
            "encoder_num_gaussians": encoder_num_gaussians,
            "encoder_num_radial": encoder_num_radial,
            "encoder_lmax": encoder_lmax,
            "encoder_max_neighbors": encoder_max_neighbors,
            "encoder_tp": encoder_tp,
            "encoder_max_z": encoder_max_z,
            "dense_grid": dense_grid,
            "bn_before_activation": bn_before_activation,
            "global_aggr": global_aggr,
            "compile_mode": compile_mode,
        }

        self.hparams.update(params)
        self.save_hyperparameters()

        # convert string activation to function
        # checkpoints store the module (not the name) because hparams is
        # mutated after save_hyperparameters; accept both on reload
        if isinstance(self.hparams.activation, str):
            self.hparams.activation = getattr(torch.nn, self.hparams.activation)()

        self.encoder, encoder_width = attach_encoder(self.hparams)

        input_size = {
            "atom": self.hparams.atom_input_size + encoder_width,
            "bond": self.hparams.bond_input_size,
            "global": self.hparams.global_input_size,
        }

        self.embedding = UnifySize(
            input_dim=input_size,
            output_dim=self.hparams.embedding_size,
        )

        self.conv_layers = nn.ModuleList()

        # All short edge type names used for building HeteroConv dicts
        edge_types = ["a2b", "b2a", "a2g", "g2a", "b2g", "g2b", "a2a", "b2b", "g2g"]

        if self.hparams.conv_fn == "GraphConvDropoutBatch":
            for i in range(self.hparams.n_conv_layers):
                embedding_in = True
                layer_args = get_layer_args(
                    self.hparams,
                    i,
                    activation=self.hparams.activation,
                    embedding_in=embedding_in,
                )

                conv_dict = {
                    EDGE_TYPE_MAP[et]: GraphConvDropoutBatch(**layer_args[et])
                    for et in edge_types
                }
                self.conv_layers.append(
                    HeteroConv(conv_dict, aggr=self.hparams.aggregate)
                )

        elif self.hparams.conv_fn in ("ResidualBlock", "ResidualBlockDense"):
            block_cls = (
                DenseResidualBlock
                if self.hparams.conv_fn == "ResidualBlockDense"
                else ResidualBlock
            )
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
                    block_cls(
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

                conv_dict = {
                    EDGE_TYPE_MAP[et]: GATConv(**layer_args[et])
                    for et in edge_types
                }
                self.conv_layers.append(
                    HeteroConv(conv_dict, aggr=self.hparams.aggregate)
                )

        self.conv_layers = nn.ModuleList(self.conv_layers)
        self._dense_blocks_fn = self._run_dense_blocks
        # print(self.conv_layers)
        self.target_dict = target_dict

        self.loss = self.loss_function()

        logger.debug("Number of output dims: %d", output_dims)

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

        if compiled and hasattr(torch.version, "hip") and torch.version.hip is not None:
            import warnings
            warnings.warn("torch.compile disabled on ROCm (no benefit, potential instability)")
            compiled = False

        if self.hparams.conv_fn == "ResidualBlockDense":
            # compile only the padded block stack: shapes are static per bucket,
            # the encoder and the padding step stay eager. Bucketed loaders
            # produce ~25 distinct shapes, above dynamo's default cache of 8.
            if compiled:
                torch._dynamo.config.cache_size_limit = max(
                    torch._dynamo.config.cache_size_limit, 64
                )
                self._dense_blocks_fn = torch.compile(
                    self._run_dense_blocks, mode=compile_mode, dynamic=False
                )
            self.forward_fn = self.compiled_forward
        else:
            self.forward_fn = (
                torch.compile(self.compiled_forward, dynamic=True)
                if compiled
                else self.compiled_forward
            )

    def compiled_forward(self, graph: HeteroData, inputs: dict) -> dict:
        """
        Forward pass with JIT compatibility
        """

        inputs = encode_atom_inputs(self.encoder, graph, inputs)
        feats = self.embedding(inputs)

        # Extract edge_index_dict from PyG HeteroData graph
        edge_index_dict = graph.edge_index_dict

        if self.hparams.conv_fn == "ResidualBlockDense":
            feats = self._dense_conv_stack(graph, feats)
            conv_iter = ()
        else:
            conv_iter = enumerate(self.conv_layers)
        for ind, conv in conv_iter:
            feats = conv(feats, edge_index_dict)

            if self.hparams.conv_fn == "GATConv":
                for k in list(feats.keys()):
                    v = feats[k]
                    reshape_dim = (
                        self.hparams.num_heads * self.hparams.hidden_size
                        if ind < self.hparams.n_conv_layers - 1
                        else len(self.target_dict[k])
                    )
                    feats[k] = v.reshape(-1, reshape_dim)

        filtered_feats = {}
        for k, v in feats.items():
            if self.hparams.target_dict[k] != [None]:
                filtered_feats[k] = v
        feats = filtered_feats

        return feats

    def _dense_conv_stack(self, graph, feats):
        """conv_fn="ResidualBlockDense": pad to (N_b, B_b) blocks, run the blocks
        (as one compiled CUDA graph when compiled=True), return flat features."""
        # eval always runs the eager blocks (cuDNN batch norm on the valid rows,
        # no recompiles for ragged eval batches); training uses the compiled
        # CUDA graphs when compiled=True, keyed by the bucket's stamped shape
        fn = self._dense_blocks_fn if self.training else self._run_dense_blocks
        eager = fn == self._run_dense_blocks
        if not eager and self.hparams.get("compile_mode", "reduce-overhead") == "reduce-overhead":
            # new iteration for cudagraph trees: outputs of the previous replay
            # may now be overwritten (to_flat copies everything we keep)
            torch.compiler.cudagraph_mark_step_begin()
        dense = to_dense_hetero(graph, feats, grid=self.hparams.dense_grid,
                                shape=getattr(graph, "dense_shape", None), with_valid=eager)
        xa, xb, xg = fn(
            dense.x["atom"], dense.x["bond"], dense.x["global"],
            dense.inc_a2b, dense.inc_b2a, dense.mask["atom"], dense.mask["bond"],
            dense.valid["atom"] if eager else None, dense.valid["bond"] if eager else None,
        )
        return dense.to_flat({"atom": xa, "bond": xb, "global": xg})

    def _run_dense_blocks(self, xa, xb, xg, inc_a2b, inc_b2a, ma, mb, va=None, vb=None):
        # tensor-only signature so torch.compile sees static shapes per bucket
        dense = DenseHeteroBatch(x={}, mask={"atom": ma, "bond": mb}, inc_a2b=inc_a2b,
                                 inc_b2a=inc_b2a, num_graphs=xa.shape[0],
                                 valid=None if va is None else {"atom": va, "bond": vb})
        x = {"atom": xa, "bond": xb, "global": xg}
        for conv in self.conv_layers:
            x = conv(dense, x)
        return x["atom"], x["bond"], x["global"]

    def forward(self, graph: HeteroData, inputs: dict) -> dict:
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

        feats = encode_atom_inputs(model.encoder, graph, feats)
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

    def shared_step(self, batch: tuple, mode: str):
        batch_graph, batch_label = batch
        logits_list = []
        labels_list = []
        # Extract node features from PyG HeteroData
        feat_dict = {ntype: batch_graph[ntype].feat for ntype in batch_graph.node_types}
        logits = self.forward(
            batch_graph, feat_dict
        )  # returns a dict of node types
        with profiler.record_function("Post Forward"):
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

            # compute loss
            all_loss = self.compute_loss(logits, labels)

            # compat with older torchmetrics
            if isinstance(all_loss, list):
                all_loss = torch.stack(all_loss)

        # log loss
        self.log(
            f"{mode}_loss",
            all_loss.sum(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
            # rank-local on purpose: with sync_dist=True the rank-0 progress bar
            # all-reduces this value before on_*_epoch_end while the other ranks
            # are already inside the torchmetrics all-gather, and DDP deadlocks
            # (2026-09-09 smoke). The synced metrics come from torchmetrics.
            sync_dist=False,
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

    def compute_loss(self, target: torch.Tensor, pred: torch.Tensor):
        """
        Compute loss
        """
        return self.loss(target, pred)

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Train step
        """
        with torch.profiler.record_function("Forward Train Step"):
            return self.shared_step(batch, mode="train")

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
    ):
        """
        Optimizer step
        """
        with torch.profiler.record_function("Optimizer Step"):
            super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx)

    def validation_step(self, batch: tuple, batch_idx: int):
        """
        Val step
        """
        with torch.profiler.record_function("Forward Val Step"):
            return self.shared_step(batch, mode="val")

    def test_step(self, batch: tuple, batch_idx: int):
        with torch.profiler.record_function("Forward Test Step"):
            return self.shared_step(batch, mode="test")

    def backward(self, loss):
        with torch.profiler.record_function("Backward Pass"):
            super().backward(loss)

    def on_train_epoch_end(self):
        """
        Training epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="train")

        if isinstance(r2, list):
            r2 = torch.stack(r2)
        if isinstance(mae, list):
            mae = torch.stack(mae)
        if isinstance(mse, list):
            mse = torch.stack(mse)

        # get epoch number
        if self.trainer.current_epoch == 0:
            self.log("val_mae", 10000000.0, prog_bar=False, sync_dist=False)

        # TorchMetrics .compute() already syncs across ranks; sync_dist=False avoids double-sync
        self.log("train_r2", r2.median(), prog_bar=False, sync_dist=False)
        self.log("train_mae", mae.mean(), prog_bar=False, sync_dist=False)
        self.log("train_mse", mse.mean(), prog_bar=True, sync_dist=False)

        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                for i, target in enumerate(target_list):
                    self.log(
                        f"train_r2_{target_type}_{target}",
                        r2[i],
                        prog_bar=False,
                        sync_dist=False,
                    )
                    self.log(
                        f"train_mae_{target_type}_{target}",
                        mae[i],
                        prog_bar=False,
                        sync_dist=False,
                    )

    def on_validation_epoch_end(self):
        """
        Validation epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="val")

        if isinstance(r2, list):
            r2 = torch.stack(r2)
        if isinstance(mae, list):
            mae = torch.stack(mae)
        if isinstance(mse, list):
            mse = torch.stack(mse)

        r2_median = r2.median().type(torch.float32)
        self.log("val_r2", r2_median, prog_bar=True, sync_dist=False)
        self.log("val_mae", mae.mean(), prog_bar=False, sync_dist=False)
        self.log("val_mse", mse.mean(), prog_bar=True, sync_dist=False)

        # log each target r2 and mae
        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                for i, target in enumerate(target_list):
                    self.log(
                        f"val_r2_{target_type}_{target}",
                        r2[i],
                        prog_bar=False,
                        sync_dist=False,
                    )
                    self.log(
                        f"val_mae_{target_type}_{target}",
                        mae[i],
                        prog_bar=False,
                        sync_dist=False,
                    )

    def on_test_epoch_end(self):
        """
        Test epoch end
        """
        r2, mae, mse = self.compute_metrics(mode="test")

        # compat with older torchmetrics
        if isinstance(r2, list):
            r2 = torch.stack(r2)
        if isinstance(mae, list):
            mae = torch.stack(mae)
        if isinstance(mse, list):
            mse = torch.stack(mse)

        self.log("test_r2", r2.median(), prog_bar=False, sync_dist=False)
        self.log("test_mae", mae.mean(), prog_bar=False, sync_dist=False)
        self.log("test_mse", mse.mean(), prog_bar=False, sync_dist=False)

        for target_type, target_list in self.target_dict.items():
            if target_list != [None] and len(target_list) > 0:
                for i, target in enumerate(target_list):
                    self.log(
                        f"test_r2_{target_type}_{target}",
                        r2[i],
                        prog_bar=False,
                        sync_dist=False,
                    )
                    self.log(
                        f"test_mae_{target_type}_{target}",
                        mae[i],
                        prog_bar=False,
                        sync_dist=False,
                    )

    def update_metrics(self, pred: torch.Tensor, target: torch.Tensor, mode: str):
        """
        Update metrics using torchmetrics interfaces
        """
        with torch.profiler.record_function("update metrics"):
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

        _nan = torch.full((self.hparams.output_dims,), float("nan"))

        if mode == "train":
            try:
                r2 = self.train_r2.compute()
            except ValueError:
                r2 = _nan
            try:
                torch_l1 = self.train_torch_l1.compute()
            except ValueError:
                torch_l1 = _nan
            try:
                torch_mse = self.train_torch_mse.compute()
            except ValueError:
                torch_mse = _nan
            self.train_r2.reset()
            self.train_torch_l1.reset()
            self.train_torch_mse.reset()

        elif mode == "val":
            try:
                r2 = self.val_r2.compute()
            except ValueError:
                r2 = _nan
            try:
                torch_l1 = self.val_torch_l1.compute()
            except ValueError:
                torch_l1 = _nan
            try:
                torch_mse = self.val_torch_mse.compute()
            except ValueError:
                torch_mse = _nan
            self.val_r2.reset()
            self.val_torch_l1.reset()
            self.val_torch_mse.reset()

        elif mode == "test":
            try:
                r2 = self.test_r2.compute()
            except ValueError:
                r2 = _nan
            try:
                torch_l1 = self.test_torch_l1.compute()
            except ValueError:
                torch_l1 = _nan
            try:
                torch_mse = self.test_torch_mse.compute()
            except ValueError:
                torch_mse = _nan
            self.test_r2.reset()
            self.test_torch_l1.reset()
            self.test_torch_mse.reset()

        return r2, torch_l1, torch_mse

    def on_fit_start(self):
        # restored checkpoints bypass the config check in build_trainer
        if (
            self.hparams.get("compiled")
            and self.hparams.conv_fn == "ResidualBlockDense"
            and self.hparams.get("compile_mode", "reduce-overhead") == "reduce-overhead"
            and self.trainer.accumulate_grad_batches > 1
        ):
            raise ValueError(
                'compiled ResidualBlockDense with accumulate_grad_batches > 1 needs compile_mode="default" '
                "(CUDA-graph outputs alias the accumulated .grad tensors)"
            )

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = build_adam(
            self, params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
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
            )

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(f"Not supported lr scheduler: {scheduler_name}")

        return scheduler

    @torch.no_grad()
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
            # Extract node features from PyG HeteroData
            feat_dict = {ntype: batch_graph[ntype].feat for ntype in batch_graph.node_types}
            preds = self.forward(batch_graph, feat_dict)
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
