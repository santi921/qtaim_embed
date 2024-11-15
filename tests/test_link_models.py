# test statistics 
# test output shapes
import torch
import torch.nn.functional as F
import json

import pytorch_lightning as pl

from qtaim_embed.core.dataset import HeteroGraphNodeLabelDataset
from qtaim_embed.data.dataloader import DataLoaderLinkTaskHeterograph
from qtaim_embed.models.link_pred.link_model import GCNLinkPred

torch.set_float32_matmul_precision("medium")
torch.multiprocessing.set_sharing_strategy("file_system")


class TestLinkPred:

    dataset = HeteroGraphNodeLabelDataset(
        file="./data/low_train_50.pkl",
        allowed_ring_size=[4, 5, 6, 7],
        # allowed_charges=[-1, 0, 1],
        # allowed_spins=[1, 2, 3],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        extra_keys={
            "atom": [
                "extra_feat_atom_esp_total",
            ],
            "bond": ["bond_length", "extra_feat_bond_esp_total"],
            # "global": ["charge", "spin"]
            "global": ["charge"],
        },
        target_dict={
            "atom": [
                "extra_feat_atom_esp_total",
            ],
            "bond": ["extra_feat_bond_esp_total"],
            "global": [],
        },
        extra_dataset_info={},
        debug=False,
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
        filter_self_bonds=True,
    )

    dataloader = DataLoaderLinkTaskHeterograph(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )


    _, _, ft = next(iter(dataloader))
    node_len = ft.shape[1]

    def main_lightning(self, model='GCN_Dot'):

        if model == "GCN_Dot":
            n_conv_layers=11
            resid_n_graph_convs=3
            conv_fn="GraphConvDropoutBatch"
            predictor="Dot"
            num_heads_gat=2

        if model == "Resid_Dot":
            n_conv_layers=9
            resid_n_graph_convs=3
            conv_fn="ResidualBlock"
            predictor="Dot"
            num_heads_gat=2

        if model == "Resid_Off_Dot":
            n_conv_layers=11
            resid_n_graph_convs=3
            conv_fn="ResidualBlock"
            predictor="Dot"
            num_heads_gat=2

        if model == "GAT_Dot":
            n_conv_layers=2
            num_heads_gat=1
            conv_fn="GATConv"
            predictor="Dot"
            resid_n_graph_convs=3

        if model == "GAT_2_Dot":
            n_conv_layers=2
            num_heads_gat=2
            conv_fn="GATConv"
            predictor="Dot"
            resid_n_graph_convs=3
            

        if model == "SAGE_Dot":
            conv_fn="GraphSAGE"
            predictor="Dot"
            n_conv_layers=2
            num_heads_gat=2
            resid_n_graph_convs=3

        if model == "SAGE_Attention":
            conv_fn="GraphSAGE"
            predictor="Attention"
            n_conv_layers=2
            num_heads_gat=2
            resid_n_graph_convs=3
            

        if model == "GAT_2_Attention":
            n_conv_layers=2
            num_heads_gat=2
            resid_n_graph_convs=3
            conv_fn="GATConv"
            predictor="Attention"

        if model == "GCN_Attention":
            n_conv_layers=11
            resid_n_graph_convs=3
            conv_fn="GraphConvDropoutBatch"
            predictor="Attention"
            num_heads_gat=2

        if model == "Resid_Attention":
            n_conv_layers=9
            resid_n_graph_convs=3
            conv_fn="ResidualBlock"
            predictor="Attention"
            num_heads_gat=2

        if model == "GAT_2_MLP":
            n_conv_layers=2
            num_heads_gat=2

            resid_n_graph_convs=3
            conv_fn="GATConv"
            predictor="MLP"

        if model == "GCN_MLP":
            n_conv_layers=11
            resid_n_graph_convs=3
            conv_fn="GraphConvDropoutBatch"
            predictor="MLP"
            num_heads_gat=2

        if model == "SAGE_MLP":
            conv_fn="GraphSAGE"
            predictor="MLP"
            n_conv_layers=2
            num_heads_gat=2
            resid_n_graph_convs=3

        if model == "Resid_MLP":
            n_conv_layers=9
            resid_n_graph_convs=3
            conv_fn="ResidualBlock"
            predictor="MLP"
            num_heads_gat=2

        model = GCNLinkPred(
            input_size=self.ft.shape[1],
            dropout=0.1,
            activation="ReLU",
            n_conv_layers=n_conv_layers,
            num_heads_gat=num_heads_gat,
            resid_n_graph_convs=resid_n_graph_convs,
            conv_fn=conv_fn,
            predictor=predictor,
            norm="both",
            lr=0.01,
            weight_decay=0.0,
            loss_fn="cross_entropy",
            embedding_size=50,
            hidden_size=256,
                
        )


        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="gpu",
            devices=[0], 
            gradient_clip_val=10.0,
            accumulate_grad_batches=1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            strategy="auto",
        )

        trainer.fit(model, self.dataloader, self.dataloader)
        stat_dict = model.evaluate_manually(self.dataloader)
        print(
            json.dumps(stat_dict, indent=4)
        )
        return stat_dict

    def test_gcn_dot(self): 
        self.main_lightning("GCN_Dot")

    def test_gcn_attention(self):
        self.main_lightning("GCN_Attention")

    def test_gat_dot(self):
        self.main_lightning("GAT_2_Dot")

    def test_gat_attention(self):
        self.main_lightning("GAT_2_Attention")

    def test_sage_dot(self):
        self.main_lightning("SAGE_Dot")

    def test_sage_attention(self):
        self.main_lightning("SAGE_Attention")

    def test_resid_dot(self):
        self.main_lightning("Resid_Dot")

    def test_resid_attention(self):
        self.main_lightning("Resid_Attention")

    def test_resid_off_dot(self):
        self.main_lightning("Resid_Off_Dot")

    def test_gcn_mlp(self):
        self.main_lightning("GCN_MLP")

    def test_gat_mlp(self):
        self.main_lightning("GAT_2_MLP")

    def test_sage_mlp(self):
        self.main_lightning("SAGE_MLP")

    def test_resid_mlp(self):
        self.main_lightning("Resid_MLP")
