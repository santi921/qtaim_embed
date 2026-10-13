"""Shared pl.Trainer construction for the training entry points.

Ten scripts build a Trainer by hand and drift (num_sanity_val_steps reached
three of them, LinearWarmup another three). New scripts call build_trainer;
existing ones migrate as they are touched (docs/todos/037).
"""

from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy


def build_trainer(
    config: dict,
    loggers: list,
    callbacks: List[Callback],
    accelerator: str = "auto",
    default_root_dir: Optional[str] = None,
) -> pl.Trainer:
    """Trainer from config["model"]["max_epochs"] and config["optim"].

    Honours optim.num_devices, num_nodes, gradient_clip_val,
    accumulate_grad_batches, strategy ("ddp" -> DDPStrategy with unused-parameter
    detection), precision, num_sanity_val_steps (default 2), warmup_epochs
    (adds LinearWarmup when > 0) and dataset.bucketing (disables Lightning's
    distributed sampler, BucketBatchSampler shards by rank itself).
    """
    optim = config["optim"]
    model_cfg = config.get("model", {})
    if (
        model_cfg.get("compiled")
        and model_cfg.get("conv_fn") == "ResidualBlockDense"
        and optim.get("accumulate_grad_batches", 1) > 1
        and model_cfg.get("compile_mode", "reduce-overhead") == "reduce-overhead"
    ):
        raise ValueError(
            "accumulate_grad_batches > 1 with compiled ResidualBlockDense needs "
            'model.compile_mode = "default": accumulated .grad tensors alias CUDA-graph '
            "outputs and are overwritten by the next replay (2026-09-09)."
        )
    callbacks = list(callbacks)
    if optim.get("warmup_epochs", 0) > 0:
        from qtaim_embed.models.utils import LinearWarmup

        callbacks.append(LinearWarmup(optim["warmup_epochs"]))
    return pl.Trainer(
        max_epochs=config["model"]["max_epochs"],
        accelerator=accelerator,
        devices=optim.get("num_devices", 1),
        num_nodes=optim.get("num_nodes", 1),
        gradient_clip_val=optim.get("gradient_clip_val", 0.0),
        accumulate_grad_batches=optim.get("accumulate_grad_batches", 1),
        enable_progress_bar=True,
        callbacks=callbacks,
        enable_checkpointing=True,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if optim.get("strategy", "auto") == "ddp"
            else optim.get("strategy", "auto")
        ),
        default_root_dir=default_root_dir or config["dataset"].get("log_save_dir"),
        # BucketBatchSampler shards itself by rank; Lightning must not wrap it
        use_distributed_sampler=not config["dataset"].get("bucketing", False),
        logger=loggers,
        precision=optim.get("precision", 32),
        num_sanity_val_steps=optim.get("num_sanity_val_steps", 2),
    )


def best_ckpt_path(trainer: pl.Trainer) -> Optional[str]:
    """Path of the best checkpoint saved during this run, or None.

    `trainer.test(model, dm)` with no ckpt_path scores whatever weights the
    model currently holds, i.e. the last epoch, not the checkpoint that was
    selected on the monitored metric. Pass this as ckpt_path to score the
    selected checkpoint instead. None (nothing was saved) keeps the current
    weights, which is the only sensible fallback.
    """
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
            return cb.best_model_path
    return None
