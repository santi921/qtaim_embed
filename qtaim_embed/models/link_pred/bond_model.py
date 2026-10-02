"""Bond classifier (T3): QTAIM bond-path existence for geometric candidate pairs.

Design (docs/plans/2026-09-08-feat-t3-bond-classifier-plan.md):
- atom embeddings from a learned element embedding plus, optionally, a 3D
  encoder over atom.pos / atom.z (encoder_fn in ENCODER_FNS);
- candidates from candidate_pairs at pool_multiplier x (rcov_i + rcov_j);
- labels read off the graph's a2b connectivity inside the step;
- symmetric pair head -> one logit per pair, BCE-with-logits;
- decision threshold calibrated on validation to maximise macro-F1.

No message passing over the graph's bond nodes: those are the labels.
"""

import logging
from typing import Dict, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from qtaim_embed.data.bonds import bond_pairs_from_heterograph, candidate_labels
from qtaim_embed.models.encoders import attach_encoder, check_encoder_hparams
from qtaim_embed.models.encoders.neighbors import RCOV, candidate_pairs
from qtaim_embed.models.link_pred.baselines import scores_from_counts
from qtaim_embed.models.optim import build_adam
from qtaim_embed.models.link_pred.pair_head import SymmetricPairHead

logger = logging.getLogger(__name__)


class BinnedBinaryStats(Metric):
    """Confusion counts on a fixed probability-threshold grid.

    O(n_bins) state, DDP-reducible, and enough to read precision, recall,
    F1 and macro-F1 at any threshold and to pick the macro-F1-optimal one.
    """

    full_state_update: bool = False

    def __init__(self, n_bins: int = 101):
        super().__init__()
        self.register_buffer("thresholds", torch.linspace(0.0, 1.0, n_bins))
        zeros = torch.zeros(n_bins, dtype=torch.long)
        for name in ("tp", "fp", "tn", "fn"):
            self.add_state(name, zeros.clone(), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        y = target.bool().unsqueeze(1)
        pred = probs.unsqueeze(1) >= self.thresholds.to(probs.device).unsqueeze(0)
        self.tp += (pred & y).sum(0)
        self.fp += (pred & ~y).sum(0)
        self.fn += (~pred & y).sum(0)
        self.tn += (~pred & ~y).sum(0)

    def compute(self) -> Dict[str, torch.Tensor]:
        out = scores_from_counts(self.tp, self.fp, self.tn, self.fn)
        out["thresholds"] = self.thresholds
        return out

    def total(self) -> int:
        """Number of scored pairs (threshold 0 classifies every pair positive)."""
        return int(self.tp[0] + self.fp[0] + self.tn[0] + self.fn[0])

    def at(self, threshold: float) -> Dict[str, float]:
        """Metrics at the grid threshold nearest to `threshold`."""
        out = self.compute()
        idx = int(torch.argmin((out["thresholds"] - float(threshold)).abs()))
        return {k: float(v[idx]) for k, v in out.items() if k != "thresholds"}

    def best_threshold(self) -> float:
        """Grid threshold maximising macro-F1."""
        out = self.compute()
        return float(out["thresholds"][int(torch.argmax(out["macro_f1"]))])


class GCNBondPred(pl.LightningModule):
    """Geometry-aware bond-path classifier over candidate atom pairs.

    Args:
        encoder_fn: "none" (element embedding only; a learned distance rule),
            "schnet", "dimenetpp" or "equivariant".
        encoder_*: forwarded to build_encoder.
        use_atom_feat: concatenate atom.feat into the atom embedding. Only
            safe when the atom features do not derive from the bond list
            (see the leakage rules in the plan). Requires atom_input_size.
        embedding_size: width of the projected atom embedding fed to the head.
        pool_multiplier: candidate radius as a multiple of rcov_i + rcov_j.
        pair_*: SymmetricPairHead settings. pair_rbf_cutoff=None resolves to
            pool_multiplier * 2 * max(rcov), the largest possible candidate
            distance, so the Bessel envelope never zeroes a real candidate.
        threshold: fixed decision threshold; None calibrates on validation.
    """

    def __init__(
        self,
        encoder_fn: str = "schnet",
        encoder_hidden: int = 64,
        encoder_cutoff: float = 5.0,
        encoder_n_interactions: int = 3,
        encoder_num_gaussians: int = 50,
        encoder_num_radial: int = 6,
        encoder_lmax: int = 1,
        encoder_max_neighbors: int = 32,
        encoder_tp: str = "channelwise",
        encoder_max_z: int = 119,
        use_atom_feat: bool = False,
        atom_input_size: int = 0,
        embedding_size: int = 64,
        max_z: int = 119,
        pool_multiplier: float = 2.0,
        pair_rbf: str = "bessel",
        pair_rbf_n: int = 50,
        pair_rbf_cutoff: Optional[float] = None,
        pair_hidden: Sequence[int] = (256, 128),
        pair_dropout: float = 0.1,
        activation: str = "SiLU",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_name: str = "reduce_on_plateau",
        lr_plateau_patience: int = 10,
        lr_scale_factor: float = 0.5,
        threshold: Optional[float] = None,
        n_threshold_bins: int = 101,
    ):
        super().__init__()
        check_encoder_hparams(encoder_fn)
        if use_atom_feat:
            assert atom_input_size > 0, "atom_input_size is required when use_atom_feat=True"
        self.save_hyperparameters()
        self.learning_rate = lr
        if pair_rbf_cutoff is None:
            pair_rbf_cutoff = float(pool_multiplier * 2.0 * RCOV.max())
            self.hparams.pair_rbf_cutoff = pair_rbf_cutoff

        self.encoder, encoder_width = attach_encoder(self.hparams)
        self.z_embedding = nn.Embedding(max_z, embedding_size)
        in_dim = embedding_size + encoder_width
        if use_atom_feat:
            in_dim += atom_input_size
        act = getattr(nn, activation)
        self.atom_proj = nn.Sequential(nn.Linear(in_dim, embedding_size), act())
        self.pair_head = SymmetricPairHead(
            in_dim=embedding_size,
            hidden=list(pair_hidden),
            rbf=pair_rbf,
            rbf_n=pair_rbf_n,
            rbf_cutoff=pair_rbf_cutoff,
            dropout=pair_dropout,
            activation=activation,
            use_ratio=True,
        )

        self.register_buffer("rcov", RCOV.clone())
        self.register_buffer(
            "threshold", torch.tensor(0.5 if threshold is None else float(threshold))
        )
        self.calibrate_threshold = threshold is None

        self.metrics = nn.ModuleDict(
            {
                f"m_{mode}": nn.ModuleDict(
                    {
                        "auroc": BinaryAUROC(thresholds=200),
                        "ap": BinaryAveragePrecision(thresholds=200),
                        "binned": BinnedBinaryStats(n_threshold_bins),
                    }
                )
                for mode in ("train", "val", "test")
            }
        )

    # ------------------------------------------------------------------ model
    @staticmethod
    def _batch_vector(graph: HeteroData) -> Optional[torch.Tensor]:
        atom = graph["atom"]
        return atom.batch if "batch" in atom else None

    def embed_atoms(self, graph: HeteroData) -> torch.Tensor:
        atom = graph["atom"]
        parts = [self.z_embedding(atom.z)]
        if self.encoder is not None:
            h_enc = self.encoder(atom.pos, atom.z, self._batch_vector(graph))
            parts.append(h_enc.to(parts[0].dtype))
        if self.hparams.use_atom_feat:
            parts.append(atom.feat.to(parts[0].dtype))
        return self.atom_proj(torch.cat(parts, dim=-1))

    def candidates(
        self, graph: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Candidate pairs (i, j, d_ij, rcov_i + rcov_j) at pool_multiplier."""
        atom = graph["atom"]
        i, j, d = candidate_pairs(
            atom.pos, atom.z, self._batch_vector(graph), self.rcov, self.hparams.pool_multiplier
        )
        r_ref = self.rcov[atom.z[i]] + self.rcov[atom.z[j]]
        return i, j, d, r_ref

    def forward(
        self, graph: HeteroData
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (logits, i, j, d_ij, r_ref) over the candidate pairs."""
        h = self.embed_atoms(graph)
        i, j, d, r_ref = self.candidates(graph)
        if i.numel() == 0:
            return h.new_zeros(0), i, j, d, r_ref
        logits = self.pair_head(h, i, j, d.to(h.dtype), r_ref.to(h.dtype))
        return logits, i, j, d, r_ref

    @torch.no_grad()
    def predict_bonds(self, graph: HeteroData, threshold: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Candidate pairs with probabilities and the thresholded decision."""
        was_training = self.training
        self.eval()
        logits, i, j, d, _ = self(graph)
        if was_training:
            self.train()
        prob = torch.sigmoid(logits.float())
        thr = float(self.threshold) if threshold is None else float(threshold)
        return {"i": i, "j": j, "distance": d, "prob": prob, "pred": prob >= thr, "threshold": thr}

    # --------------------------------------------------------------- training
    def shared_step(self, graph: HeteroData, mode: str) -> Optional[torch.Tensor]:
        """Loss for one batch, or None when the batch has no candidate pairs
        (Lightning skips the optimizer step on None)."""
        logits, i, j, d, r_ref = self(graph)
        if logits.numel() == 0:
            return None
        num_nodes = int(graph["atom"].num_nodes)
        y = candidate_labels(i, j, bond_pairs_from_heterograph(graph), num_nodes)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        n_graphs = int(graph.num_graphs) if hasattr(graph, "num_graphs") else 1
        self.log(
            f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True,
            batch_size=n_graphs, sync_dist=True,
        )
        probs = torch.sigmoid(logits.detach().float())
        m = self.metrics[f"m_{mode}"]
        m["auroc"].update(probs, y.long())
        m["ap"].update(probs, y.long())
        m["binned"].update(probs, y)
        return loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> Optional[torch.Tensor]:
        return self.shared_step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> Optional[torch.Tensor]:
        return self.shared_step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> Optional[torch.Tensor]:
        return self.shared_step(batch, "test")

    def _epoch_end(self, mode: str) -> None:
        m = self.metrics[f"m_{mode}"]
        binned = m["binned"]
        if binned.total() == 0:
            for x in m.values():
                x.reset()
            return
        if mode == "val" and self.calibrate_threshold and not self.trainer.sanity_checking:
            self.threshold.fill_(binned.best_threshold())
        stats = binned.at(float(self.threshold))
        self.log(f"{mode}_auroc", m["auroc"].compute(), sync_dist=False)
        self.log(f"{mode}_ap", m["ap"].compute(), sync_dist=False)
        for key in ("precision", "recall", "f1_pos", "macro_f1"):
            self.log(f"{mode}_{key}", stats[key], prog_bar=(key == "macro_f1"), sync_dist=False)
        self.log(f"{mode}_threshold", float(self.threshold), sync_dist=False)
        for x in m.values():
            x.reset()

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        optimizer = build_adam(
            self,
            (p for p in self.parameters() if p.requires_grad),
            lr=self.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        name = (self.hparams.scheduler_name or "none").lower()
        if name == "none":
            return optimizer
        if name == "cosine":
            t_max = int(getattr(self.trainer, "max_epochs", None) or 100)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=self.learning_rate * 0.01
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        if name != "reduce_on_plateau":
            raise ValueError(f"scheduler_name must be none, cosine or reduce_on_plateau, got {name}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_scale_factor,
            patience=self.hparams.lr_plateau_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"},
        }
