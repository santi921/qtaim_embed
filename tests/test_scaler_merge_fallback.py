"""Regressions from the 2026-09-08 review: merge_scalers on stored-moment scalers,
NaN-free finalize on constant columns, and the shared fused-Adam policy."""

import pytest
import torch

from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative, merge_scalers
from qtaim_embed.models.optim import build_adam, trainer_clips_gradients
from qtaim_embed.utils.scalers import _transform, guard_std
import numpy as np


def _from_moments(mean, std, n):
    return HeteroGraphStandardScalerIterative(
        features_tf=True,
        mean={"atom": mean.clone()},
        std={"atom": std.clone()},
        dict_node_sizes={"atom": n},
        finalized=True,
    )


def test_merge_scalers_accepts_scalers_built_from_mean_std():
    torch.manual_seed(0)
    a = torch.randn(1000, 3, dtype=torch.float64) * 2 + 1
    b = torch.randn(500, 3, dtype=torch.float64) * 0.5 - 3
    s1 = _from_moments(a.mean(0), a.std(0, unbiased=False), 1000)
    s2 = _from_moments(b.mean(0), b.std(0, unbiased=False), 500)
    merged = merge_scalers([s1, s2], finalize_merged=True)
    both = torch.cat([a, b])
    assert torch.allclose(merged._mean["atom"], both.mean(0), atol=1e-9)
    assert torch.allclose(merged._std["atom"], both.std(0, unbiased=False), atol=1e-9)


def test_merge_scalers_rejects_unfinalized_without_moments():
    s = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={"atom": torch.zeros(2)}, std={"atom": torch.ones(2)},
        dict_node_sizes={"atom": 5}, finalized=False,
    )
    with pytest.raises(AssertionError):
        merge_scalers([s, s])


def test_finalize_constant_column_has_no_nan():
    c = 0.2820947917738781
    s = HeteroGraphStandardScalerIterative(features_tf=True)
    n = 1000
    s._mean = {"atom": torch.tensor([c, 1.0], dtype=torch.float64)}
    # sum_x2 accumulated one sample at a time reproduces cancellation error
    acc = torch.zeros(2, dtype=torch.float64)
    for _ in range(n):
        acc += torch.tensor([c, 1.0], dtype=torch.float64) ** 2
    s._sum_x2 = {"atom": acc}
    s.dict_node_sizes = {"atom": n}
    s.finalize()
    std = s._std["atom"]
    assert torch.isfinite(std).all()
    assert torch.allclose(std, torch.ones(2, dtype=torch.float64))


class _Trainer:
    def __init__(self, clip):
        self.gradient_clip_val = clip


class _Module(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.lin = torch.nn.Linear(2, 1)
        self._trainer = _Trainer(clip)


def test_build_adam_disables_fused_when_clipping():
    m = _Module(clip=5.0)
    assert trainer_clips_gradients(m)
    opt = build_adam(m, m.parameters(), lr=1e-3)
    assert opt.param_groups[0].get("fused") in (False, None)
    m2 = _Module(clip=0.0)
    assert not trainer_clips_gradients(m2)
    opt2 = build_adam(m2, m2.parameters(), lr=1e-3)
    # CPU parameters: fused either disabled up front or fell back to plain Adam
    assert isinstance(opt2, torch.optim.Adam)


def test_guard_std_relative_tolerance_torch_and_numpy():
    mean = torch.tensor([1.0, 0.1, 5.0, 100.0], dtype=torch.float64)
    std = torch.tensor([1.4e-17, 1e-3, 0.0, 1e-13], dtype=torch.float64)
    out = guard_std(std.clone(), mean)
    assert out.tolist() == [1.0, 1e-3, 1.0, 1.0]
    out_np = guard_std(std.numpy().copy(), mean.numpy())
    assert out_np.tolist() == [1.0, 1e-3, 1.0, 1.0]


def test_transform_std_matches_sklearn_on_float_noise_constant():
    x = torch.ones(200, 2, dtype=torch.float64)
    x[0, 0] += 4e-16
    x[:, 1] = torch.linspace(0, 1, 200, dtype=torch.float64)
    rst, mean, std = _transform(x, copy=True)
    assert std[0] == 1.0
    assert abs(rst[:, 0]).max() < 1e-12
    assert 0.2 < std[1] < 0.4


def _graph(n_atoms, with_feat=True, width=3):
    from torch_geometric.data import HeteroData

    g = HeteroData()
    g["atom"].num_nodes = n_atoms
    if with_feat:
        g["atom"].feat = torch.randn(n_atoms, width)
    return g


def test_update_mixed_batch_fits_every_graph_with_track():
    torch.manual_seed(0)
    bare, g1, g2 = _graph(4, with_feat=False), _graph(5), _graph(7)
    s = HeteroGraphStandardScalerIterative(features_tf=True)
    s.update([bare, g1, g2])  # first graph lacks feat: old code skipped all three
    assert s.dict_node_sizes == {"atom": 12}
    s.finalize()
    both = torch.cat([g1["atom"].feat, g2["atom"].feat]).double()
    assert torch.allclose(s.mean["atom"], both.mean(0), atol=1e-9)
    out = s([bare, g1, g2])
    assert not hasattr(out[0]["atom"], "feat")
    assert torch.allclose(out[1]["atom"].feat.double().mean(0) * 5 + out[2]["atom"].feat.double().mean(0) * 7, torch.zeros(3, dtype=torch.float64), atol=1e-6)


def test_zero_observation_feature_scaler_warns_and_raises_on_apply(caplog):
    import logging

    s = HeteroGraphStandardScalerIterative(features_tf=True)
    s.update([_graph(3, with_feat=False)])
    with caplog.at_level(logging.WARNING, logger="qtaim_embed.data.processing"):
        s.finalize()
    assert "zero observations" in caplog.text
    with pytest.raises(ValueError, match="no statistics"):
        s([_graph(3)])
    # graphs without the track still pass through untouched
    assert s([_graph(2, with_feat=False)])[0]["atom"].num_nodes == 2
