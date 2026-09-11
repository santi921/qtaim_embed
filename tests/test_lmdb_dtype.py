"""Tests for graph feature-dtype handling on LMDB load.

Scaled LMDBs carry float64 `feat` tensors (the scaler stores mean/std as
float64), which collide with float32/bf16 model weights at the first linear.
`TransformMol` downcasts `feat` to a configurable dtype (default float32) on
load; the LMDB datamodules plumb `config["dataset"]["dtype"]` through to it.
These tests pin that behavior: `labels` follow the same dtype, edge_index stays long.
"""
import pytest
import torch

from qtaim_embed.data.lmdb import (
    TransformMol,
    serialize_graph,
    _cast_graph_floats,
    _resolve_dtype,
)
from qtaim_embed.core.datamodule import LMDBDataModule, LMDBLinkDataModule
from qtaim_embed.utils.tests import make_hetero_graph


def _graph_with_dtypes(feat_dtype=torch.float64, label_dtype=torch.float64):
    """make_hetero_graph() with feat forced to feat_dtype and a labels tensor."""
    graph, _ = make_hetero_graph()
    for nt in graph.node_types:
        graph[nt].feat = graph[nt].feat.to(feat_dtype)
    # attach a label tensor on the atom store to check how it is cast
    graph["atom"].labels = torch.ones(
        graph["atom"].num_nodes, 2, dtype=label_dtype
    )
    return graph


def _serialized(graph):
    return {"molecule_graph": serialize_graph(graph, ret=True)}


def test_default_downcasts_feat_to_float32():
    out = TransformMol(_serialized(_graph_with_dtypes(feat_dtype=torch.float64)))
    for nt in out.node_types:
        assert out[nt].feat.dtype == torch.float32


def test_labels_follow_feature_dtype():
    # scaled LMDBs carry float64 labels; they are cast with feat so the loss
    # sees matching dtypes (performance plan A1)
    out = TransformMol(_serialized(_graph_with_dtypes(label_dtype=torch.float64)))
    assert out["atom"].labels.dtype == torch.float32


def test_dtype_none_leaves_labels_alone():
    out = TransformMol(
        _serialized(_graph_with_dtypes(label_dtype=torch.float64)), dtype=None
    )
    assert out["atom"].labels.dtype == torch.float64


def test_edge_index_stays_long():
    out = TransformMol(_serialized(_graph_with_dtypes()))
    for store in out.edge_stores:
        assert store.edge_index.dtype == torch.long


def test_dtype_override_upcasts_to_float64():
    # source feat is float32; requesting float64 must upcast
    out = TransformMol(
        _serialized(_graph_with_dtypes(feat_dtype=torch.float32)),
        dtype=torch.float64,
    )
    for nt in out.node_types:
        assert out[nt].feat.dtype == torch.float64


def test_dtype_string_equivalent_to_torch_dtype():
    src = _serialized(_graph_with_dtypes(feat_dtype=torch.float64))
    src2 = _serialized(_graph_with_dtypes(feat_dtype=torch.float64))
    out_str = TransformMol(src, dtype="float32")
    out_torch = TransformMol(src2, dtype=torch.float32)
    for nt in out_str.node_types:
        assert out_str[nt].feat.dtype == out_torch[nt].feat.dtype == torch.float32


def test_dtype_none_disables_cast():
    out = TransformMol(
        _serialized(_graph_with_dtypes(feat_dtype=torch.float64)),
        dtype=None,
    )
    for nt in out.node_types:
        assert out[nt].feat.dtype == torch.float64


def test_invalid_dtype_raises():
    with pytest.raises(ValueError):
        TransformMol(_serialized(_graph_with_dtypes()), dtype="float8")


def test_already_heterodata_branch_casts_in_place_and_returns_dict():
    # when molecule_graph is already a HeteroData (not bytes), the dict is
    # returned and the graph is cast in place
    graph = _graph_with_dtypes(feat_dtype=torch.float64)
    data_object = {"molecule_graph": graph}
    out = TransformMol(data_object)
    assert out is data_object
    assert out["molecule_graph"]["atom"].feat.dtype == torch.float32


def test_resolve_dtype_accepts_dtype_string_and_none():
    assert _resolve_dtype(torch.float16) == torch.float16
    assert _resolve_dtype("bfloat16") == torch.bfloat16
    assert _resolve_dtype(None) is None
    with pytest.raises(ValueError):
        _resolve_dtype("not_a_dtype")


def test_cast_graph_floats_none_is_noop():
    graph = _graph_with_dtypes(feat_dtype=torch.float64)
    same = _cast_graph_floats(graph, None)
    assert same["atom"].feat.dtype == torch.float64


@pytest.mark.parametrize("dm_cls", [LMDBDataModule, LMDBLinkDataModule])
def test_datamodule_feature_dtype_default_and_override(dm_cls):
    base = {"dataset": {"train_lmdb": "/nonexistent/train.lmdb"}}
    # __init__ does not touch disk, so a fake path is fine
    assert dm_cls(config=base)._feature_dtype == "float32"

    override = {
        "dataset": {"train_lmdb": "/nonexistent/train.lmdb", "dtype": "float64"}
    }
    assert dm_cls(config=override)._feature_dtype == "float64"


def test_labels_never_below_float32():
    # bf16 features keep full-precision regression targets
    out = TransformMol(_serialized(_graph_with_dtypes()), dtype="bfloat16")
    assert out["atom"].feat.dtype == torch.bfloat16
    assert out["atom"].labels.dtype == torch.float32
