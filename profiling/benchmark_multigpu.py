#!/usr/bin/env python3
"""
Minimal 2-GPU DDP test for PyG graph-level model.
Run with: torchrun --nproc_per_node=2 profiling/test_multigpu.py
"""
import json
import os
import sys
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

torch.set_float32_matmul_precision("high")


class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self._start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.perf_counter() - self._start
        self.epoch_times.append(elapsed)
        rank = trainer.global_rank
        epoch = trainer.current_epoch
        print(f"[rank {rank}] Epoch {epoch} train done in {elapsed:.2f}s", flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        rank = trainer.global_rank
        epoch = trainer.current_epoch
        print(f"[rank {rank}] Epoch {epoch} validation done", flush=True)


class BatchCountCallback(Callback):
    """Track batch counts per rank to detect DDP divergence."""
    def __init__(self):
        super().__init__()
        self.train_batches = 0
        self.val_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_batches = batch_idx + 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_batches = batch_idx + 1

    def on_train_epoch_end(self, trainer, pl_module):
        rank = trainer.global_rank
        print(f"[rank {rank}] train_batches={self.train_batches}, val_batches={self.val_batches}", flush=True)
        self.train_batches = 0
        self.val_batches = 0


def load_config(num_workers=0, persistent_workers=False):
    config_path = "profiling/graph_configs/graph_baseline_w0.json"
    with open(config_path) as f:
        config = json.load(f)
    config["model"]["max_epochs"] = 3
    config["optim"]["num_workers"] = num_workers
    config["optim"]["persistent_workers"] = persistent_workers
    config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]
    return config


def run_benchmark(config, num_gpus, strategy="auto", label=""):
    from qtaim_embed.core.datamodule import LMDBDataModule
    from qtaim_embed.models.utils import load_graph_level_model_from_config

    dm = LMDBDataModule(config=config)
    dm.setup(stage="fit")
    feature_size = dm.train_dataset.feature_size

    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]

    model = load_graph_level_model_from_config(config["model"])

    timer = TimingCallback()
    batch_counter = BatchCountCallback()

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision="bf16-mixed",
        gradient_clip_val=5.0,
        enable_checkpointing=False,
        enable_progress_bar=False,
        callbacks=[timer, batch_counter],
    )

    rank = trainer.global_rank
    t0 = time.perf_counter()
    trainer.fit(model, dm)
    total = time.perf_counter() - t0

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
        for i, t in enumerate(timer.epoch_times):
            print(f"  Epoch {i}: {t:.2f}s")
        avg = sum(timer.epoch_times[1:]) / max(len(timer.epoch_times) - 1, 1)
        print(f"  Avg epoch (excl warmup): {avg:.2f}s")
        print(f"  Total: {total:.1f}s")
        print(f"{'=' * 60}", flush=True)

    return timer.epoch_times


def main():
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))

    if num_gpus > 1:
        # DDP mode: run with workers=4
        config = load_config(num_workers=4, persistent_workers=True)
        run_benchmark(config, num_gpus, strategy="ddp",
                      label=f"{num_gpus}x GPU, DDP, workers=4")
    else:
        # Single GPU: run both worker configs
        print("Running 1-GPU benchmarks...\n", flush=True)

        config = load_config(num_workers=0)
        run_benchmark(config, 1, label="1x GPU, workers=0")

        config = load_config(num_workers=4, persistent_workers=True)
        run_benchmark(config, 1, label="1x GPU, workers=4")


if __name__ == "__main__":
    main()
