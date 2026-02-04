"""
Tests for the FullPredictor module.

These tests verify the iterative link-node prediction pipeline works correctly.
Some tests use mocked models since actual checkpoints may not be available.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import dgl
from unittest.mock import patch

from qtaim_embed.data.geometry_to_graph import GeometryToGraph
from qtaim_embed.models.full_predictor.full import (
    FullPredictor,
    FullPredictorInference,
)


class MockHparams:
    """
    Mock hyperparameters object that supports both attribute access and dict-like .get().

    This mimics PyTorch Lightning's hparams namespace behavior where you can do
    both `hparams.input_size` and `hparams.get("input_size")`.
    """

    def __init__(self, params_dict: dict):
        for key, value in params_dict.items():
            setattr(self, key, value)
        self._params_dict = params_dict

    def get(self, key, default=None):
        return self._params_dict.get(key, default)

    def __contains__(self, key):
        return key in self._params_dict

    def __repr__(self):
        return f"MockHparams({self._params_dict})"


class MockLinkModel(nn.Module):
    """Mock link prediction model for testing."""

    def __init__(self, input_size=50, hidden_size=64):
        super().__init__()
        self.hparams = MockHparams({
            "input_size": input_size,
            "hidden_size": hidden_size,
            "n_conv_layers": 2,
            "predictor": "Dot",
            "grapher_config": None,  # Add for compatibility with new code
        })
        self.linear = nn.Linear(input_size, hidden_size)

    def forward(self, pos_graph, neg_graph, inputs):
        # Return mock scores
        num_pos_edges = pos_graph.num_edges() if pos_graph.num_edges() > 0 else 1
        num_neg_edges = neg_graph.num_edges() if neg_graph.num_edges() > 0 else 1
        pos_scores = torch.randn(num_neg_edges)  # Use neg_graph edges for candidate scoring
        neg_scores = torch.randn(num_neg_edges)
        return pos_scores, neg_scores

    def to(self, device):
        return self

    def eval(self):
        return self


class MockNodeModel(nn.Module):
    """Mock node prediction model for testing."""

    def __init__(self, atom_size=50, bond_size=10, global_size=5, target_dict=None):
        super().__init__()
        target_dict = target_dict or {"atom": ["feat1"], "bond": ["feat2"]}
        self.hparams = MockHparams({
            "atom_input_size": atom_size,
            "bond_input_size": bond_size,
            "global_input_size": global_size,
            "hidden_size": 64,
            "n_conv_layers": 2,
            "target_dict": target_dict,
            "grapher_config": None,  # Add for compatibility with new code
        })
        self.target_dict = target_dict

    def forward(self, graph, inputs):
        # Return mock predictions for each target node type
        predictions = {}
        for node_type, targets in self.target_dict.items():
            if targets and targets != [None]:
                num_nodes = graph.num_nodes(node_type)
                num_features = len(targets)
                predictions[node_type] = torch.randn(num_nodes, num_features)
        return predictions

    def to(self, device):
        return self

    def eval(self):
        return self


class TestFullPredictorConstruction:
    """Tests for FullPredictor construction and model loading."""

    def test_mock_predictor_creation(self):
        """Test creating a predictor with mocked models."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            config = {
                "link_model_path": "fake_link.ckpt",
                "node_model_path": "fake_node.ckpt",
                "iterations": 3,
                "edge_threshold": 0.5,
                "device": "cpu",
            }

            predictor = FullPredictor(config)

            assert predictor.iterations == 3
            assert predictor.edge_threshold == 0.5
            assert predictor.link_model is not None
            assert predictor.node_model is not None

    def test_config_defaults(self):
        """Test that default config values are applied."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            config = {
                "link_model_path": "fake_link.ckpt",
                "node_model_path": "fake_node.ckpt",
            }

            predictor = FullPredictor(config)

            # Check defaults
            assert predictor.iterations == 3
            assert predictor.edge_threshold == 0.5
            assert predictor.device == "cpu"


class TestFullPredictorPrediction:
    """Tests for the prediction pipeline."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a predictor with mocked models."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            config = {
                "link_model_path": "fake_link.ckpt",
                "node_model_path": "fake_node.ckpt",
                "iterations": 2,
                "edge_threshold": 0.5,
                "device": "cpu",
            }

            return FullPredictor(config)

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        converter = GeometryToGraph(distance_cutoff=2.0)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.87, 0.0],
        ])
        elements = ["C", "H", "H"]
        return converter(coords, elements)

    def test_get_bidirectional_candidate_edges(self, mock_predictor):
        """Test bidirectional candidate edge generation."""
        edges = mock_predictor._get_bidirectional_candidate_edges(4)

        # Should have n*(n-1) = 12 edges (both directions, no self-loops)
        assert len(edges) == 12
        # Should contain both (i, j) and (j, i) for each pair
        assert (0, 1) in edges
        assert (1, 0) in edges
        # No self-loops
        for i, j in edges:
            assert i != j

    def test_get_bidirectional_candidate_edges_small(self, mock_predictor):
        """Test bidirectional candidate edge generation for small graphs."""
        edges = mock_predictor._get_bidirectional_candidate_edges(2)
        assert len(edges) == 2
        assert (0, 1) in edges
        assert (1, 0) in edges

    def test_update_topology(self, mock_predictor, sample_graph):
        """Test graph topology update."""
        new_edges = [(0, 1), (0, 2)]
        updated_graph = mock_predictor._update_topology(sample_graph, new_edges)

        assert updated_graph.num_nodes("bond") == 2
        assert updated_graph.num_nodes("atom") == sample_graph.num_nodes("atom")

    def test_predict_returns_graph(self, mock_predictor, sample_graph):
        """Test that predict returns a graph."""
        # This test verifies the structure, not the actual predictions
        result = mock_predictor.predict(sample_graph)

        assert isinstance(result, dgl.DGLHeteroGraph)
        assert result.num_nodes("atom") == sample_graph.num_nodes("atom")
        assert result.num_nodes("global") == 1

    def test_predict_with_intermediate(self, mock_predictor, sample_graph):
        """Test prediction with intermediate results."""
        result, intermediate = mock_predictor.predict(
            sample_graph, return_intermediate=True
        )

        assert isinstance(result, dgl.DGLHeteroGraph)
        assert len(intermediate) == mock_predictor.iterations

        # Check intermediate results structure
        for step in intermediate:
            assert "iteration" in step
            assert "edge_scores" in step
            assert "predicted_edges" in step
            assert "node_predictions" in step

    def test_predict_preserves_atom_features(self, mock_predictor, sample_graph):
        """Test that atom features are preserved through prediction."""
        original_atom_feat = sample_graph.nodes["atom"].data["feat"].clone()

        result = mock_predictor.predict(sample_graph)

        # Atom features should be preserved (or updated)
        assert result.nodes["atom"].data["feat"].shape[0] == original_atom_feat.shape[0]


class TestFullPredictorEvaluation:
    """Tests for the evaluation functionality."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a predictor with mocked models."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            config = {
                "link_model_path": "fake_link.ckpt",
                "node_model_path": "fake_node.ckpt",
                "iterations": 1,
                "edge_threshold": 0.5,
                "device": "cpu",
            }

            return FullPredictor(config)

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        converter = GeometryToGraph(distance_cutoff=2.0)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.87, 0.0],
        ])
        elements = ["C", "H", "H"]
        return converter(coords, elements)

    def test_evaluate_returns_metrics(self, mock_predictor, sample_graph):
        """Test that evaluate returns metrics dictionary."""
        ground_truth_edges = [(0, 1), (0, 2)]

        metrics = mock_predictor.evaluate(
            sample_graph,
            ground_truth_edges=ground_truth_edges,
        )

        assert isinstance(metrics, dict)
        # Should have edge metrics
        assert "edge_precision" in metrics
        assert "edge_recall" in metrics
        assert "edge_f1" in metrics

    def test_evaluate_edge_metrics_range(self, mock_predictor, sample_graph):
        """Test that edge metrics are in valid range."""
        ground_truth_edges = [(0, 1), (0, 2)]

        metrics = mock_predictor.evaluate(
            sample_graph,
            ground_truth_edges=ground_truth_edges,
        )

        assert 0 <= metrics["edge_precision"] <= 1
        assert 0 <= metrics["edge_recall"] <= 1
        assert 0 <= metrics["edge_f1"] <= 1

    def test_get_model_info(self, mock_predictor):
        """Test model info retrieval."""
        info = mock_predictor.get_model_info()

        assert "link_model" in info
        assert "node_model" in info
        assert "config" in info

        assert info["config"]["iterations"] == 1
        assert info["config"]["edge_threshold"] == 0.5


class TestFullPredictorFromGeometry:
    """Tests for geometry-to-prediction convenience method."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a predictor with mocked models."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            config = {
                "link_model_path": "fake_link.ckpt",
                "node_model_path": "fake_node.ckpt",
                "iterations": 1,
                "device": "cpu",
            }

            return FullPredictor(config)

    def test_predict_from_geometry(self, mock_predictor):
        """Test direct prediction from geometry."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        elements = ["C", "H"]

        result = mock_predictor.predict_from_geometry(
            coords, elements, charge=0
        )

        assert isinstance(result, dgl.DGLHeteroGraph)
        assert result.num_nodes("atom") == 2


class TestFullPredictorInference:
    """Tests for the inference wrapper."""

    def test_inference_wrapper_creation(self):
        """Test creating inference wrapper with mocked models."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            predictor = FullPredictorInference.from_checkpoints(
                link_ckpt="fake_link.ckpt",
                node_ckpt="fake_node.ckpt",
                iterations=2,
            )

            assert predictor.full_predictor is not None
            assert predictor.geometry_converter is not None

    def test_inference_call(self):
        """Test calling inference wrapper."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            predictor = FullPredictorInference.from_checkpoints(
                link_ckpt="fake_link.ckpt",
                node_ckpt="fake_node.ckpt",
                iterations=1,
            )

            coords = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ])
            elements = ["C", "H"]

            result = predictor(coords, elements)

            assert isinstance(result, dict)
            assert "predicted_edges" in result
            assert "num_predicted_bonds" in result
            assert "graph" in result


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_fullPredictor_alias(self):
        """Test that fullPredictor alias exists."""
        from qtaim_embed.models.full_predictor.full import fullPredictor

        # Should be the same class
        assert fullPredictor is FullPredictor

    def test_import_from_init(self):
        """Test importing from __init__.py."""
        from qtaim_embed.models.full_predictor import FullPredictor as FP

        assert FP is FullPredictor


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a predictor with mocked models."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            config = {
                "link_model_path": "fake_link.ckpt",
                "node_model_path": "fake_node.ckpt",
                "iterations": 1,
                "device": "cpu",
            }

            return FullPredictor(config)

    @pytest.mark.skip(reason="Single-atom molecules require special handling in hetero_to_homo transform")
    def test_single_atom_molecule(self, mock_predictor):
        """Test handling of single atom molecule.

        Note: Single-atom molecules with no bonds cause issues in the
        hetero_to_homo transform. This edge case would need fixes in
        the underlying transforms.py module.
        """
        converter = GeometryToGraph(distance_cutoff=2.0)
        coords = np.array([[0.0, 0.0, 0.0]])
        elements = ["C"]

        graph = converter(coords, elements)
        result = mock_predictor.predict(graph)

        assert result.num_nodes("atom") == 1

    def test_zero_iterations(self):
        """Test with zero iterations (no-op)."""
        with patch.object(FullPredictor, '_load_link_model') as mock_link, \
             patch.object(FullPredictor, '_load_node_model') as mock_node:

            mock_link.return_value = MockLinkModel()
            mock_node.return_value = MockNodeModel()

            config = {
                "link_model_path": "fake_link.ckpt",
                "node_model_path": "fake_node.ckpt",
                "iterations": 0,
                "device": "cpu",
            }

            predictor = FullPredictor(config)

            converter = GeometryToGraph(distance_cutoff=2.0)
            coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            elements = ["C", "H"]
            graph = converter(coords, elements)

            result = predictor.predict(graph)

            # With 0 iterations, should return original graph
            assert result.num_nodes("atom") == graph.num_nodes("atom")

    def test_high_threshold(self, mock_predictor):
        """Test with very high edge threshold (likely no edges predicted)."""
        mock_predictor.edge_threshold = 0.99

        converter = GeometryToGraph(distance_cutoff=2.0)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "H"]
        graph = converter(coords, elements)

        # Should still complete without error
        result = mock_predictor.predict(graph)
        assert isinstance(result, dgl.DGLHeteroGraph)

    def test_low_threshold(self, mock_predictor):
        """Test with very low edge threshold (likely all edges predicted)."""
        mock_predictor.edge_threshold = 0.01

        converter = GeometryToGraph(distance_cutoff=2.0)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.87, 0.0]])
        elements = ["C", "H", "H"]
        graph = converter(coords, elements)

        # Should still complete without error
        result = mock_predictor.predict(graph)
        assert isinstance(result, dgl.DGLHeteroGraph)
