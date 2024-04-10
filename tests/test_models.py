import torch, dgl
import pytorch_lightning as pl
from torch.nn import functional as F
from qtaim_embed.utils.tests import (
    get_dataset_graph_level, 
    get_dataset_graph_level_multitask,
    get_datasets_graph_level_classifier
)
from qtaim_embed.utils.data import get_default_graph_level_config
from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask

# def construct_default_model():


def test_save_load():
    dataset_graph_level = get_dataset_graph_level(
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_graph_level, batch_size=len(dataset_graph_level.graphs), shuffle=False
    )

    model_config = get_default_graph_level_config()
    model_config["model"]["max_epochs"] = 50
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
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        enable_progress_bar=True,
        devices=1,
        strategy="auto",
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=16,
    )

    trainer.fit(model, data_loader)

    reload_config = model_config
    reload_config["model"]["restore"] = True
    reload_config["model"]["restore_path"] = "./test_save_load/lightning_logs/version_0/checkpoints/epoch=99-step=100.ckpt"
    model_reload = load_graph_level_model_from_config(reload_config["model"])

#test_save_load()
def test_manual_eval_graph_level_classifier():
    dataset_single, dataset_multi = get_datasets_graph_level_classifier(
        log_scale_features=True, 
        standard_scale_features=True
    )
    
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_single, batch_size=len(dataset_single.graphs), shuffle=False
    )

    model_config = get_default_graph_level_config()
    model_config["model"]["atom_feature_size"] = dataset_single.feature_size()[
        "atom"
    ]
    model_config["model"]["bond_feature_size"] = dataset_single.feature_size()[
        "bond"
    ]
    model_config["model"]["global_feature_size"] = dataset_single.feature_size()[
        "global"
    ]
    model_config["model"]["target_dict"]["global"] = dataset_single.target_dict[
        "global"
    ]

    model_config["model"]["classifier"] = True
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    batch_graph, batched_labels = next(iter(data_loader))

    acc_pre, auroc_pre, f1_pre = model.evaluate_manually(
        (batch_graph,batched_labels)
    )
    print("-" * 50)
    print(
        "Prior to training:\t acc: {:.4f}\t auroc: {:.4f}\t f1: {:.4f}".format(
            acc_pre, auroc_pre, f1_pre
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
            labels_one_hot = torch.argmax(labels, axis=2)
            labels_one_hot = labels_one_hot.reshape(-1)
            logits = model(batch_graph, batch_graph.ndata["feat"])
            logits_one_hot = torch.argmax(logits, axis=-1)
            loss = F.cross_entropy(logits, labels_one_hot)
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

    acc, auroc, f1 = model.evaluate_manually(
        (batch_graph,
        batched_labels)
    )
    print(
        "After 10 Epochs \t acc: {:.4f}\t auroc: {:.4f}\t f1: {:.4f}".format(
            acc, auroc, f1
        )
    )

    assert acc > acc_pre, "R2 score did not improve after training"


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
    #print(dataset_graph_level.feature_size())
    #print(dataset_graph_level.target_dict)

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
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    # get unscaled targets from dataset
    graphs_unscale = dataset_graph_level.unscale_targets(dataset_graph_level.graphs)
    # labels_raw = [i.ndata["labels"]["global"] for i in dataset_graph_level.graphs]
    labels_unscaled = [i.ndata["labels"]["global"] for i in graphs_unscale]
    labels_unscaled = torch.cat(labels_unscaled, dim=0)

    batch_graph, batched_labels = next(iter(data_loader))
    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_graph_level.label_scalers,
    )
    print("-" * 50)
    print(
        "Prior to training:\t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_pre.numpy()[0], mae.numpy()[0], mse.numpy()[0]
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

    r2_post, mae, mse, _, _ = model.evaluate_manually(
        batch_graph,
        batched_labels,
        scaler_list=dataset_graph_level.label_scalers,
    )
    print(
        "After 10 Epochs \t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_post.numpy()[0], mae.numpy()[0], mse.numpy()[0]
        )
    )

    assert r2_post.numpy() > r2_pre.numpy(), "R2 score did not improve after training"


def test_multi_task():
    dataset_graph_level = get_dataset_graph_level_multitask(
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )
    data_loader = DataLoaderMoleculeGraphTask(
        dataset_graph_level, batch_size=len(dataset_graph_level.graphs), shuffle=False
    )

    model_config = get_default_graph_level_config()
    model_config["model"]["max_epochs"] = 50
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
    model_config["model"]["initializer"] = None
    #model_config["model"]["output_dims"] = 1

    model = load_graph_level_model_from_config(model_config["model"])

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        enable_progress_bar=True,
        devices=1,
        strategy="auto",
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=16,
    )

    trainer.fit(model, data_loader)

