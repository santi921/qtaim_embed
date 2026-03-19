import os
import glob
import shutil
import tracemalloc

import pytest
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch_geometric.data import Batch

from qtaim_embed.utils.tests import (
    get_dataset_graph_level,
    get_dataset_graph_level_multitask,
    get_datasets_graph_level_classifier,
    make_hetero,
    make_hetero_graph,
    make_test_model,
)
from qtaim_embed.utils.data import get_default_graph_level_config
from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.models.graph_level.base_gcn import GCNGraphPred
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
    model_config["model"]["atom_feature_size"] = dataset_graph_level.feature_size[
        "atom"
    ]
    model_config["model"]["bond_feature_size"] = dataset_graph_level.feature_size[
        "bond"
    ]
    model_config["model"]["global_feature_size"] = dataset_graph_level.feature_size[
        "global"
    ]
    model_config["model"]["target_dict"]["global"] = dataset_graph_level.target_dict[
        "global"
    ]
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=16,
        log_every_n_steps=1,
    )

    trainer.fit(model, data_loader)

    reload_config = model_config
    reload_config["model"]["restore"] = True
    reload_config["model"][
        "restore_path"
    ] = "./test_save_load/lightning_logs/version_0/checkpoints/epoch=99-step=100.ckpt"
    model_reload = load_graph_level_model_from_config(reload_config["model"])


def test_manual_eval_graph_level_classifier():
    dataset_single, dataset_multi = get_datasets_graph_level_classifier(
        log_scale_features=True, standard_scale_features=True
    )

    data_loader = DataLoaderMoleculeGraphTask(
        dataset_single, batch_size=len(dataset_single.graphs), shuffle=False
    )

    model_config = get_default_graph_level_config()
    model_config["model"]["atom_feature_size"] = dataset_single.feature_size["atom"]
    model_config["model"]["bond_feature_size"] = dataset_single.feature_size["bond"]
    model_config["model"]["global_feature_size"] = dataset_single.feature_size["global"]
    model_config["model"]["target_dict"]["global"] = dataset_single.target_dict[
        "global"
    ]

    model_config["model"]["classifier"] = True
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    batch_graph, batched_labels = next(iter(data_loader))

    acc_pre, auroc_pre, f1_pre = model.evaluate_manually((batch_graph, batched_labels))
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
            feat_dict = {nt: batch_graph[nt].feat for nt in batch_graph.node_types}
            logits = model(batch_graph, feat_dict)
            logits_one_hot = torch.argmax(logits, axis=-1)
            loss = F.cross_entropy(logits, labels_one_hot)
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

    acc, auroc, f1 = model.evaluate_manually((batch_graph, batched_labels))
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

    model_config = get_default_graph_level_config()
    model_config["model"]["atom_feature_size"] = dataset_graph_level.feature_size[
        "atom"
    ]
    model_config["model"]["bond_feature_size"] = dataset_graph_level.feature_size[
        "bond"
    ]
    model_config["model"]["global_feature_size"] = dataset_graph_level.feature_size[
        "global"
    ]
    model_config["model"]["target_dict"]["global"] = dataset_graph_level.target_dict[
        "global"
    ]
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    # get unscaled targets from dataset
    graphs_unscale = dataset_graph_level.unscale_targets(dataset_graph_level.graphs)
    labels_unscaled = [i["global"].labels for i in graphs_unscale]
    labels_unscaled = torch.cat(labels_unscaled, dim=0)

    batch_graph, batched_labels = next(iter(data_loader))
    r2_pre, mae, mse, _, _ = model.evaluate_manually(
        data_loader,
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
            feat_dict = {nt: batch_graph[nt].feat for nt in batch_graph.node_types}
            logits = model(batch_graph, feat_dict)
            loss = F.mse_loss(logits, labels)
            # backward propagation
            opt.zero_grad()
            loss.backward()
            opt.step()

    r2_post, mae, mse, _, _ = model.evaluate_manually(
        data_loader,
        scaler_list=dataset_graph_level.label_scalers,
    )
    print(
        "After 10 Epochs \t r2: {:.4f}\t mae: {:.4f}\t mse: {:.4f}".format(
            r2_post.numpy()[0], mae.numpy()[0], mse.numpy()[0]
        )
    )

    assert r2_post.numpy() > r2_pre.numpy(), "R2 score did not improve after training"


def test_checkpoint_load_resume():
    """Test that a model can be saved, loaded from checkpoint, and resume training."""
    ckpt_dir = "./test_ckpt_resume/"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

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
    model_config["model"]["atom_feature_size"] = dataset_graph_level.feature_size["atom"]
    model_config["model"]["bond_feature_size"] = dataset_graph_level.feature_size["bond"]
    model_config["model"]["global_feature_size"] = dataset_graph_level.feature_size[
        "global"
    ]
    model_config["model"]["target_dict"]["global"] = dataset_graph_level.target_dict[
        "global"
    ]
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    # Phase 1: train for 3 epochs, save checkpoint
    # Use same loader for val so ReduceLROnPlateau has val_mae
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        enable_checkpointing=True,
        default_root_dir=ckpt_dir,
        precision="16-mixed",
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer.fit(model, data_loader, val_dataloaders=data_loader)

    # Find the saved checkpoint
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "**/*.ckpt"), recursive=True)
    assert len(ckpt_files) > 0, "No checkpoint files found after training"
    ckpt_path = ckpt_files[0]

    # Phase 2: load from checkpoint and verify weights match
    model_loaded = GCNGraphPred.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.cpu()
    model_loaded.cpu()
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model_loaded.named_parameters()
    ):
        assert n1 == n2, f"Parameter name mismatch: {n1} vs {n2}"
        assert torch.allclose(p1, p2), f"Parameter {n1} differs after reload"

    # Phase 3: resume training from checkpoint for more epochs
    trainer2 = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        default_root_dir=ckpt_dir,
        precision="16-mixed",
        log_every_n_steps=1,
        enable_progress_bar=False,
    )
    trainer2.fit(
        model_loaded, data_loader, val_dataloaders=data_loader, ckpt_path=ckpt_path
    )

    # Verify resumed training ran (global_step advanced beyond checkpoint)
    assert trainer2.global_step > 0, "No training steps after resume"

    # Phase 4: also test the config-based restore path
    model_config["model"]["restore"] = True
    model_config["model"]["restore_path"] = ckpt_path
    model_restored = load_graph_level_model_from_config(model_config["model"])
    assert model_restored is not None, "Config-based restore returned None"

    # Cleanup
    shutil.rmtree(ckpt_dir, ignore_errors=True)


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
    model_config["model"]["atom_feature_size"] = dataset_graph_level.feature_size[
        "atom"
    ]
    model_config["model"]["bond_feature_size"] = dataset_graph_level.feature_size[
        "bond"
    ]
    model_config["model"]["global_feature_size"] = dataset_graph_level.feature_size[
        "global"
    ]
    model_config["model"]["target_dict"]["global"] = dataset_graph_level.target_dict[
        "global"
    ]
    model_config["model"]["initializer"] = None
    # model_config["model"]["output_dims"] = 1

    model = load_graph_level_model_from_config(model_config["model"])

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_checkpointing=True,
        default_root_dir="./test_save_load/",
        precision=16,
        log_every_n_steps=1,
    )

    trainer.fit(model, data_loader)


# ---------------------------------------------------------------------------
# Edge case model pipeline tests
# ---------------------------------------------------------------------------


def _forward_edge_case(model, graph):
    """Extract features from graph and run model forward."""
    feats = {
        nt: graph[nt].feat
        for nt in graph.node_types
        if hasattr(graph[nt], "feat")
    }
    with torch.no_grad():
        return model(graph, feats)


def _make_large_graph(num_atoms: int, num_bonds: int):
    """Build a synthetic graph with many atoms and bonds."""
    a2b = [(i % num_atoms, i) for i in range(num_bonds)]
    b2a = [(i, (i + 1) % num_atoms) for i in range(num_bonds)]
    return make_hetero(num_atoms, num_bonds, a2b, b2a, self_loop=True)


# Feature sizes from make_hetero: atom=2, bond=3, global=4
_ATOM_FEAT = 2
_BOND_FEAT = 3
_GLOBAL_FEAT = 4


class TestEdgeCases:
    """Test extreme graph configurations through the full model pipeline."""

    def _model(self, pooling_fn="SumPoolingThenCat"):
        return make_test_model(
            _ATOM_FEAT, _BOND_FEAT, _GLOBAL_FEAT, pooling_fn=pooling_fn
        )

    def test_zero_bond_full_pipeline(self):
        """5 atoms, 0 bonds -> model produces (1,1), no NaN/Inf."""
        graph, _ = make_hetero(num_atoms=5, num_bonds=0, a2b=[], b2a=[])
        batched = Batch.from_data_list([graph])
        model = self._model()
        out = _forward_edge_case(model, batched)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all(), f"Non-finite output: {out}"

    def test_single_node_per_type(self):
        """1 atom, 1 bond, 1 global -> model works."""
        graph, _ = make_hetero(
            num_atoms=1, num_bonds=1, a2b=[(0, 0)], b2a=[(0, 0)]
        )
        batched = Batch.from_data_list([graph])
        model = self._model()
        out = _forward_edge_case(model, batched)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all(), f"Non-finite output: {out}"

    def test_large_graph_200_atoms(self):
        """200 atoms, 300 bonds -> no numerical blowup."""
        graph, _ = _make_large_graph(200, 300)
        batched = Batch.from_data_list([graph])
        model = self._model()
        out = _forward_edge_case(model, batched)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all(), f"Non-finite output: {out}"

    def test_batch_size_1(self):
        """Single graph batch through model."""
        graph, _ = make_hetero_graph()
        batched = Batch.from_data_list([graph])
        model = self._model()
        out = _forward_edge_case(model, batched)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all()

    def test_batch_size_16(self):
        """16 identical graphs batched."""
        graph, _ = make_hetero_graph()
        graphs = [graph.clone() for _ in range(16)]
        batched = Batch.from_data_list(graphs)
        model = self._model()
        out = _forward_edge_case(model, batched)
        assert out.shape == (16, 1)
        assert torch.isfinite(out).all()

    def test_mixed_batch_sizes(self):
        """1-atom graph + 50-atom graph in same batch, both finite."""
        tiny, _ = make_hetero(
            num_atoms=1, num_bonds=1, a2b=[(0, 0)], b2a=[(0, 0)]
        )
        large, _ = _make_large_graph(50, 75)
        batched = Batch.from_data_list([tiny, large])
        model = self._model()
        out = _forward_edge_case(model, batched)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all(), f"Non-finite: {out}"

    @pytest.mark.parametrize(
        "pooling_fn",
        [
            "SumPoolingThenCat",
            "MeanPoolingThenCat",
            "WeightAndSumThenCat",
            "WeightAndMeanThenCat",
            "GlobalAttentionPoolingThenCat",
            "Set2SetThenCat",
        ],
    )
    def test_zero_bond_all_pooling_fns(self, pooling_fn):
        """Zero-bond graph works with every pooling function."""
        graph, _ = make_hetero(num_atoms=3, num_bonds=0, a2b=[], b2a=[])
        batched = Batch.from_data_list([graph])
        model = self._model(pooling_fn=pooling_fn)
        out = _forward_edge_case(model, batched)
        assert out.shape == (1, 1)
        assert torch.isfinite(out).all(), (
            f"{pooling_fn} produced non-finite output on zero-bond graph: {out}"
        )


# ---------------------------------------------------------------------------
# Logger integration tests
# ---------------------------------------------------------------------------


class TestLoggerIntegration:
    """Verify logging backends work with the training pipeline."""

    def test_tensorboard_creates_event_files(self, tmp_path):
        """Train 2 epochs with TensorBoardLogger, verify event files exist."""
        from pytorch_lightning.loggers import TensorBoardLogger

        dataset = get_dataset_graph_level(
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )
        dl = DataLoaderMoleculeGraphTask(
            dataset, batch_size=len(dataset.graphs), shuffle=False
        )

        model = make_test_model(
            atom_feat_size=dataset.feature_size["atom"],
            bond_feat_size=dataset.feature_size["bond"],
            global_feat_size=dataset.feature_size["global"],
            target_dict={"global": dataset.target_dict["global"]},
        )

        tb_logger = TensorBoardLogger(save_dir=str(tmp_path), name="test_tb")
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1,
            logger=tb_logger,
        )
        trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl)

        log_dir = os.path.join(str(tmp_path), "test_tb")
        assert os.path.isdir(log_dir), f"TensorBoard log dir not created: {log_dir}"
        event_files = glob.glob(
            os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True
        )
        assert len(event_files) > 0, "No TensorBoard event files found"

    def test_wandb_logger_offline(self, tmp_path):
        """Train 2 epochs with WandbLogger in offline mode, verify run dir created."""
        import wandb
        from pytorch_lightning.loggers import WandbLogger

        dataset = get_dataset_graph_level(
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )
        dl = DataLoaderMoleculeGraphTask(
            dataset, batch_size=len(dataset.graphs), shuffle=False
        )

        model = make_test_model(
            atom_feat_size=dataset.feature_size["atom"],
            bond_feat_size=dataset.feature_size["bond"],
            global_feat_size=dataset.feature_size["global"],
            target_dict={"global": dataset.target_dict["global"]},
        )

        wandb_logger = WandbLogger(
            project="test_project",
            save_dir=str(tmp_path),
            offline=True,
        )
        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            log_every_n_steps=1,
            logger=wandb_logger,
        )
        trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl)

        # Verify wandb created an offline run directory
        wandb_dir = tmp_path / "wandb"
        assert wandb_dir.is_dir(), f"wandb dir not created at {wandb_dir}"
        offline_runs = list(wandb_dir.glob("offline-run-*"))
        assert len(offline_runs) > 0, "No offline run directories found"
        wandb.finish()


class TestMemoryProfiling:
    """Verify no memory regressions in training pipeline."""

    def test_training_host_memory(self):
        """Measure host (CPU) memory during a short training run.

        Ensures no excessive memory bloat from graph construction,
        batching, or the torch.split shared-storage issue.
        """
        dataset = get_dataset_graph_level(
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )
        dl = DataLoaderMoleculeGraphTask(
            dataset, batch_size=len(dataset.graphs), shuffle=False
        )

        model = make_test_model(
            atom_feat_size=dataset.feature_size["atom"],
            bond_feat_size=dataset.feature_size["bond"],
            global_feat_size=dataset.feature_size["global"],
            target_dict={"global": dataset.target_dict["global"]},
        )

        tracemalloc.start()

        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="cpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl)

        current_mb, peak_mb = [x / 1e6 for x in tracemalloc.get_traced_memory()]
        tracemalloc.stop()

        # Sanity threshold: small test dataset should stay well under 512 MB
        assert peak_mb < 512, (
            f"Peak host memory {peak_mb:.1f} MB exceeds 512 MB threshold"
        )
        # Current allocation should be much lower than peak (no leaks)
        assert current_mb < peak_mb, (
            f"Current memory ({current_mb:.1f} MB) not below peak ({peak_mb:.1f} MB)"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_training_gpu_memory(self):
        """Measure GPU peak memory during a short training run."""
        dataset = get_dataset_graph_level(
            log_scale_features=False,
            log_scale_targets=False,
            standard_scale_features=True,
            standard_scale_targets=True,
        )
        dl = DataLoaderMoleculeGraphTask(
            dataset, batch_size=len(dataset.graphs), shuffle=False
        )

        model = make_test_model(
            atom_feat_size=dataset.feature_size["atom"],
            bond_feat_size=dataset.feature_size["bond"],
            global_feat_size=dataset.feature_size["global"],
            target_dict={"global": dataset.target_dict["global"]},
        )

        torch.cuda.reset_peak_memory_stats()

        trainer = pl.Trainer(
            max_epochs=5,
            accelerator="gpu",
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl)

        peak_mb = torch.cuda.max_memory_allocated() / 1e6

        # Small test model should stay well under 1 GB
        assert peak_mb < 1024, (
            f"GPU peak memory {peak_mb:.1f} MB exceeds 1024 MB threshold"
        )


def test_train_with_rbf_features():
    """End-to-end: dataset with RBF bond features trains without error."""
    from pathlib import Path
    from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset

    data_file = str(Path(__file__).parent / "data" / "labelled_data.pkl")

    dataset = HeteroGraphGraphLabelDataset(
        file=data_file,
        allowed_ring_size=[3, 4, 5, 6, 7],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        element_set=[],
        extra_keys={
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["rbf_bessel_10"],
            "global": ["extra_feat_global_E1_CAM"],
        },
        target_list=["extra_feat_global_E1_CAM"],
        extra_dataset_info={},
        debug=True,
        log_scale_features=False,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
        bond_key="bonds",
        map_key="extra_feat_bond_indices_qtaim",
        rbf_cutoff=5.0,
    )

    data_loader = DataLoaderMoleculeGraphTask(
        dataset, batch_size=len(dataset.graphs), shuffle=False
    )

    model_config = get_default_graph_level_config()
    model_config["model"]["atom_feature_size"] = dataset.feature_size["atom"]
    model_config["model"]["bond_feature_size"] = dataset.feature_size["bond"]
    model_config["model"]["global_feature_size"] = dataset.feature_size["global"]
    model_config["model"]["target_dict"]["global"] = dataset.target_dict["global"]
    model_config["model"]["initializer"] = None

    model = load_graph_level_model_from_config(model_config["model"])

    trainer = pl.Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, data_loader)

    # verify bond feature width: 7 ring features + 10 RBF = 17
    assert dataset.feature_size["bond"] == 17
