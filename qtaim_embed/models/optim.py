"""Optimizer construction shared by every LightningModule in the package.

Single place for the fused-Adam policy so the five models cannot drift:
fused Adam unscales gradients internally, and Lightning refuses to combine
that with gradient clipping under mixed precision (see commit 0f82e55), so
fused is only requested when the attached trainer does not clip. Fused also
requires CUDA parameters; the RuntimeError fallback covers CPU runs.
"""

from typing import Iterable, Optional

import pytorch_lightning as pl
import torch


def trainer_clips_gradients(module: pl.LightningModule) -> bool:
    """True when the module's trainer (if attached) has gradient_clip_val > 0."""
    trainer = getattr(module, "_trainer", None)
    clip: Optional[float] = getattr(trainer, "gradient_clip_val", None) if trainer is not None else None
    return bool(clip)


def build_adam(
    module: pl.LightningModule,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    """Adam with fused kernels when safe (CUDA available, no gradient clipping)."""
    params = list(params)
    use_fused = torch.cuda.is_available() and not trainer_clips_gradients(module)
    try:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, fused=use_fused)
    except RuntimeError:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
