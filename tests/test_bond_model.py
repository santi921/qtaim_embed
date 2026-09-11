"""GCNBondPred: metrics, forward per encoder, training against the distance rule."""

from functools import partial
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from torch_geometric.data import Batch

from qtaim_embed.core.datamodule import LMDBBondDataModule
from qtaim_embed.core.dataset import LMDBMoleculeDataset
from qtaim_embed.data.bonds import bond_pairs_from_heterograph, candidate_labels
from qtaim_embed.data.lmdb import TransformMol
from qtaim_embed.models.encoders import ENCODER_FNS
from qtaim_embed.models.encoders.neighbors import RCOV, candidate_pairs
from qtaim_embed.models.link_pred.baselines import binary_scores, fit_distance_rule
from qtaim_embed.models.link_pred.bond_model import BinnedBinaryStats, GCNBondPred
from qtaim_embed.models.utils import load_bond_model_from_config
from qtaim_embed.utils.data import get_default_bond_level_config

DATA_DIR = Path(__file__).parent / "data" / "lmdb_link"


def _dataset(split):
    return LMDBMoleculeDataset(
        {"src": str(DATA_DIR / split / "molecule.lmdb")},
        transform=partial(TransformMol, dtype="float32"),
    )


def _batch(split="train", n=6):
    ds = _dataset(split)
    return Batch.from_data_list([ds[i] for i in range(min(n, len(ds)))])


def _small_config(encoder_fn="none"):
    cfg = get_default_bond_level_config()
    cfg["dataset"]["train_lmdb"] = str(DATA_DIR / "train" / "molecule.lmdb")
    cfg["dataset"]["val_lmdb"] = str(DATA_DIR / "val" / "molecule.lmdb")
    cfg["dataset"]["test_lmdb"] = str(DATA_DIR / "test" / "molecule.lmdb")
    cfg["model"].update(
        encoder_fn=encoder_fn, encoder_hidden=16, encoder_n_interactions=1,
        encoder_num_gaussians=10, encoder_num_radial=4, embedding_size=16,
        pair_hidden=[32], pair_rbf_n=16, pair_dropout=0.0, lr=5e-3,
    )
    cfg["optim"].update(train_batch_size=16, num_workers=0, pin_memory=False, precision=32)
    return cfg


def test_binned_stats_matches_exact_counts_and_picks_best_threshold():
    torch.manual_seed(0)
    y = (torch.rand(500) > 0.5)
    probs = torch.where(y, torch.rand(500) * 0.5 + 0.5, torch.rand(500) * 0.5)
    probs[:50] = 1 - probs[:50]  # inject noise
    m = BinnedBinaryStats(n_bins=101)
    m.update(probs[:250], y[:250])
    m.update(probs[250:], y[250:])
    at = m.at(0.5)
    exact = binary_scores(probs >= 0.5, y)
    assert at["precision"] == pytest.approx(exact.precision, abs=1e-6)
    assert at["recall"] == pytest.approx(exact.recall, abs=1e-6)
    assert at["macro_f1"] == pytest.approx(exact.macro_f1, abs=1e-6)
    best = m.best_threshold()
    assert 0.0 <= best <= 1.0
    assert m.at(best)["macro_f1"] >= at["macro_f1"] - 1e-9


@pytest.mark.parametrize("encoder_fn", ENCODER_FNS)
def test_forward_and_step_each_encoder(encoder_fn):
    torch.manual_seed(0)
    model = GCNBondPred(
        encoder_fn=encoder_fn, encoder_hidden=16, encoder_n_interactions=1,
        encoder_num_gaussians=10, encoder_num_radial=4, embedding_size=16,
        pair_hidden=[32], pair_rbf_n=16,
    )
    batch = _batch("train", 4)
    logits, i, j, d, r_ref = model(batch)
    assert logits.shape == i.shape == j.shape == d.shape == r_ref.shape
    assert i.numel() > 0
    assert bool((i < j).all())
    loss = model.shared_step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    out = model.predict_bonds(batch)
    assert out["prob"].shape == i.shape and out["pred"].dtype == torch.bool
    assert 0.0 <= out["threshold"] <= 1.0


def test_default_rbf_cutoff_covers_heaviest_candidate():
    model = GCNBondPred(encoder_fn="none", embedding_size=8, pair_hidden=[8], pair_rbf_n=8, pool_multiplier=2.0)
    assert model.hparams.pair_rbf_cutoff == pytest.approx(float(2.0 * 2.0 * RCOV.max()))
    d_max = torch.tensor([float(2.0 * 2.0 * RCOV.max()) * 0.99])
    assert model.pair_head.expand_distance(d_max).abs().sum() > 0
    fixed = GCNBondPred(encoder_fn="none", embedding_size=8, pair_hidden=[8], pair_rbf_n=8, pair_rbf_cutoff=6.0)
    assert fixed.hparams.pair_rbf_cutoff == 6.0


def test_use_atom_feat_requires_size():
    with pytest.raises(AssertionError):
        GCNBondPred(encoder_fn="none", use_atom_feat=True, atom_input_size=0)
    batch = _batch("train", 2)
    model = GCNBondPred(encoder_fn="none", use_atom_feat=True, atom_input_size=int(batch["atom"].feat.shape[1]), embedding_size=8, pair_hidden=[8], pair_rbf_n=4)
    logits, i, *_ = model(batch)
    assert logits.shape == i.shape


def test_training_matches_or_beats_distance_rule_on_fixture(tmp_path):
    pl.seed_everything(0)
    cfg = _small_config("none")
    cfg["model"]["max_epochs"] = 40
    cfg["dataset"]["log_save_dir"] = str(tmp_path)
    dm = LMDBBondDataModule(cfg)
    dm.setup("fit")
    model = load_bond_model_from_config(cfg["model"])
    trainer = pl.Trainer(
        max_epochs=cfg["model"]["max_epochs"], accelerator="cpu", devices=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        num_sanity_val_steps=0, default_root_dir=str(tmp_path),
    )
    trainer.fit(model, dm)
    val_f1 = float(trainer.callback_metrics["val_macro_f1"])
    assert 0.0 < float(model.threshold) < 1.0

    # distance rule fitted on the same validation pairs
    vb = Batch.from_data_list([g for g in dm.val_dataset])
    atom = vb["atom"]
    i, j, d = candidate_pairs(atom.pos, atom.z, atom.batch, RCOV, pool_multiplier=cfg["model"]["pool_multiplier"])
    y = candidate_labels(i, j, bond_pairs_from_heterograph(vb), int(atom.num_nodes))
    r_ref = RCOV[atom.z[i]] + RCOV[atom.z[j]]
    best_k, table = fit_distance_rule(d, r_ref, y)
    rule_f1 = table[best_k].macro_f1
    print(f"val macro-F1 model={val_f1:.4f} rule(k={best_k})={rule_f1:.4f}")
    assert val_f1 > 0.9
    assert val_f1 >= rule_f1 - 0.02

    # checkpoint round trip keeps the calibrated threshold and predictions
    ckpt = tmp_path / "bond.ckpt"
    trainer.save_checkpoint(str(ckpt))
    loaded = GCNBondPred.load_from_checkpoint(str(ckpt))
    assert float(loaded.threshold) == pytest.approx(float(model.threshold))
    a = model.predict_bonds(vb)
    b = loaded.predict_bonds(vb)
    assert torch.allclose(a["prob"], b["prob"], atol=1e-6)
    assert bool((a["pred"] == b["pred"]).all())


def test_datamodule_batches_and_eval_script(tmp_path):
    cfg = _small_config("none")
    dm = LMDBBondDataModule(cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    # DataLoaderLMDB(with_labels=False): the batched HeteroData alone, direct collate
    assert not isinstance(batch, tuple)
    assert "pos" in batch["atom"] and "z" in batch["atom"]
    assert int(batch.num_graphs) == cfg["optim"]["train_batch_size"]
    assert batch["atom"].batch.max().item() == batch.num_graphs - 1
    from qtaim_embed.scripts.eval.eval_bond_baselines import main as eval_main

    eval_main(["--lmdb", str(DATA_DIR / "train"), "--out_dir", str(tmp_path / "bl"), "--pool_multipliers", "2.0", "3.0", "--k_step", "0.05"])
    assert (tmp_path / "bl" / "README.md").exists()
    assert (tmp_path / "bl" / "pool_summary.csv").exists()
