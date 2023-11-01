# Todo
# Test 1: construction and output shape
# Test 2: manually evaluation of statistics
import torch, dgl
from torch.nn import functional as F
from qtaim_embed.utils.tests import get_dataset_graph_level
from qtaim_embed.utils.data import get_default_graph_level_config
from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask

# def construct_default_model():


def test_save_load():
    # TODO
    pass


def test_manual_eval_graph_level_classifier():
    # TODO
    pass


def test_manual_eval_graph_level():
    dataset_graph_level = get_dataset_graph_level(
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_graph_level, batch_size=len(dataset_graph_level.graphs), shuffle=False
    )
    print(dataset_graph_level.feature_size())
    print(dataset_graph_level.target_dict)

    model_config = get_default_graph_level_config()
    model_config["model"]["atom_feature_size"] = dataset_graph_level.feature_size()[
        "atom"
    ]
    model_config["model"]["bond_feature_size"] = dataset_graph_level.feature_size()[
        "bond"
    ]
    model_config["model"]["global_feature_size"] = dataset_graph_level.feature_size()[
        "global"
    ]
    model_config["model"]["target_dict"]["global"] = dataset_graph_level.target_dict[
        "global"
    ]

    model = load_graph_level_model_from_config(model_config["model"])

    # get unscaled targets from dataset
    graphs_unscale = dataset_graph_level.unscale_targets(dataset_graph_level.graphs)
    # labels_raw = [i.ndata["labels"]["global"] for i in dataset_graph_level.graphs]
    labels_unscaled = [i.ndata["labels"]["global"] for i in graphs_unscale]
    labels_unscaled = torch.cat(labels_unscaled, dim=0)

    batch_graph, batched_labels = next(iter(data_loader))
    r2_pre, mae, mse = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_graph_level.label_scalers,
    )
    print("-" * 50)
    print(
        "Prior to training:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre, mae, mse
        )
    )

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(50):
        model.train()
        # training_loss = 0
        for step, (batch_graph, batch_label) in enumerate(data_loader):
            # forward propagation by using all nodes and extracting the user embeddings
            batch_graph, batch_label = next(iter(data_loader))
            labels = batch_label["global"]
            logits = model(batch_graph, batch_graph.ndata["feat"])
            loss = F.mse_loss(logits, labels)
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

    r2_post, mae, mse = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_graph_level.label_scalers,
    )
    print(
        "After 10 Epochs \t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_post, mae, mse
        )
    )

    assert r2_post > r2_pre, "R2 score did not improve after training"
