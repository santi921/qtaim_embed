import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from qtaim_embed.models.utils import LinearWarmup


class _Tiny(pl.LightningModule):
    def __init__(self, lr=0.1):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self.lr = lr
        self.seen_lrs = []

    def training_step(self, batch, batch_idx):
        self.seen_lrs.append(self.trainer.optimizers[0].param_groups[0]["lr"])
        x, y = batch
        return torch.nn.functional.mse_loss(self.lin(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def _run(warmup_epochs, steps_per_epoch=4, epochs=3):
    ds = TensorDataset(torch.randn(steps_per_epoch, 2), torch.randn(steps_per_epoch, 1))
    model = _Tiny()
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[LinearWarmup(warmup_epochs)],
    )
    trainer.fit(model, DataLoader(ds, batch_size=1))
    return model.seen_lrs


class TestLinearWarmup:
    def test_ramps_linearly_then_holds(self):
        lrs = _run(warmup_epochs=1, steps_per_epoch=4, epochs=2)
        assert lrs[:4] == pytest.approx([0.025, 0.05, 0.075, 0.1])
        assert lrs[4:] == pytest.approx([0.1] * 4)

    def test_fractional_epochs(self):
        lrs = _run(warmup_epochs=0.5, steps_per_epoch=4, epochs=1)
        assert lrs == pytest.approx([0.05, 0.1, 0.1, 0.1])

    def test_zero_is_noop(self):
        lrs = _run(warmup_epochs=0, steps_per_epoch=3, epochs=1)
        assert lrs == pytest.approx([0.1, 0.1, 0.1])


    def test_resume_mid_warmup_keeps_base_lr(self, tmp_path):
        ds = TensorDataset(torch.randn(4, 2), torch.randn(4, 1))
        common = dict(accelerator="cpu", logger=False, enable_progress_bar=False,
                      enable_model_summary=False, default_root_dir=str(tmp_path))
        ckpt = ModelCheckpoint(dirpath=str(tmp_path), save_last=True, save_top_k=0)
        first = _Tiny()
        pl.Trainer(max_epochs=1, callbacks=[LinearWarmup(2), ckpt], **common).fit(
            first, DataLoader(ds, batch_size=1))
        assert first.seen_lrs == pytest.approx([0.0125, 0.025, 0.0375, 0.05])
        second = _Tiny()
        pl.Trainer(max_epochs=2, callbacks=[LinearWarmup(2), ModelCheckpoint(dirpath=str(tmp_path), save_top_k=0)],
                   **common).fit(second, DataLoader(ds, batch_size=1), ckpt_path=str(tmp_path / "last.ckpt"))
        assert second.seen_lrs == pytest.approx([0.0625, 0.075, 0.0875, 0.1])
