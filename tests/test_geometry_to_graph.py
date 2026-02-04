"""
Tests for GeometryToGraph module.

Tests the conversion of molecular geometry (xyz coordinates + elements) to
DGL heterographs for use in the iterative link-node prediction pipeline.
"""

import numpy as np
import torch

from qtaim_embed.data.geometry_to_graph import (
    GeometryToGraph,
    update_graph_topology,
    edges_from_predictions,
    DEFAULT_ELEMENT_SET,
)


class TestGeometryToGraph:
    """Tests for the GeometryToGraph class."""

    def test_simple_molecule_creation(self):
        """Test creating a graph from a simple H2 molecule."""
        converter = GeometryToGraph(distance_cutoff=2.0)

        # H2 molecule: two hydrogens 0.74 Angstroms apart
        coords = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        elements = ["H", "H"]

        graph = converter(coords, elements)

        # Check node counts
        assert graph.num_nodes("atom") == 2
        assert graph.num_nodes("bond") == 1  # One H-H bond
        assert graph.num_nodes("global") == 1

        # Check that atom features exist and have correct shape
        atom_feats = graph.nodes["atom"].data["feat"]
        assert atom_feats.shape[0] == 2
        # Features: element one-hot + coordinates
        expected_feat_dim = len(DEFAULT_ELEMENT_SET) + 3
        assert atom_feats.shape[1] == expected_feat_dim

    def test_water_molecule(self):
        """Test creating a graph from a water molecule."""
        converter = GeometryToGraph(distance_cutoff=1.2)

        # Water molecule (simplified geometry)
        coords = np.array([
            [0.0, 0.0, 0.0],      # O
            [0.96, 0.0, 0.0],     # H
            [-0.24, 0.93, 0.0],   # H
        ])
        elements = ["O", "H", "H"]

        graph = converter(coords, elements)

        # Check node counts
        assert graph.num_nodes("atom") == 3
        # Should have 2 O-H bonds (H-H distance > 1.2)
        assert graph.num_nodes("bond") == 2
        assert graph.num_nodes("global") == 1

    def test_no_bonds_molecule(self):
        """Test handling of isolated atoms (no bonds)."""
        converter = GeometryToGraph(distance_cutoff=1.0)

        # Two atoms far apart
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        elements = ["C", "C"]

        graph = converter(coords, elements)

        # Should create a dummy bond node
        assert graph.num_nodes("atom") == 2
        assert graph.num_nodes("bond") == 1  # Dummy bond
        assert graph.num_nodes("global") == 1

    def test_element_one_hot_encoding(self):
        """Test that element one-hot encoding is correct."""
        converter = GeometryToGraph(distance_cutoff=2.0, include_coordinates=False)

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "H"]

        graph = converter(coords, elements)
        atom_feats = graph.nodes["atom"].data["feat"]

        # Get indices of C and H in element set
        c_idx = DEFAULT_ELEMENT_SET.index("C")
        h_idx = DEFAULT_ELEMENT_SET.index("H")

        # Check one-hot encoding
        assert atom_feats[0, c_idx] == 1.0
        assert atom_feats[1, h_idx] == 1.0
        # Other positions should be 0
        assert atom_feats[0, h_idx] == 0.0
        assert atom_feats[1, c_idx] == 0.0

    def test_coordinates_in_features(self):
        """Test that coordinates are included in features when requested."""
        converter = GeometryToGraph(distance_cutoff=2.0, include_coordinates=True)

        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        elements = ["C", "H"]

        graph = converter(coords, elements)
        atom_feats = graph.nodes["atom"].data["feat"]

        # Coordinates should be at the end of the feature vector
        assert np.allclose(atom_feats[0, -3:].numpy(), [1.0, 2.0, 3.0])
        assert np.allclose(atom_feats[1, -3:].numpy(), [4.0, 5.0, 6.0])

    def test_bond_features(self):
        """Test that bond features are correctly computed."""
        converter = GeometryToGraph(
            distance_cutoff=2.0,
            include_distances=True,
        )

        # Two atoms 1.5 Angstroms apart
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        elements = ["C", "H"]

        graph = converter(coords, elements)
        bond_feats = graph.nodes["bond"].data["feat"]

        # Check distance is in bond features
        assert bond_feats.shape[0] == 1
        # First feature should be the distance
        assert np.allclose(bond_feats[0, 0].item(), 1.5, atol=1e-5)

    def test_global_features(self):
        """Test that global features are correctly set."""
        converter = GeometryToGraph(distance_cutoff=2.0)

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elements = ["C", "H", "H"]

        graph = converter(coords, elements, charge=1, spin_multiplicity=2)
        global_feats = graph.nodes["global"].data["feat"]

        # Should have [charge, spin, num_atoms]
        assert global_feats.shape == (1, 3)
        assert global_feats[0, 0].item() == 1.0  # charge
        assert global_feats[0, 1].item() == 2.0  # spin
        assert global_feats[0, 2].item() == 3.0  # num_atoms

    def test_edge_types_present(self):
        """Test that all expected edge types are present."""
        converter = GeometryToGraph(distance_cutoff=2.0, self_loop=True)

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "H"]

        graph = converter(coords, elements)

        # Check all edge types exist
        expected_etypes = [
            ("atom", "a2b", "bond"),
            ("bond", "b2a", "atom"),
            ("atom", "a2g", "global"),
            ("global", "g2a", "atom"),
            ("bond", "b2g", "global"),
            ("global", "g2b", "bond"),
            ("atom", "a2a", "atom"),
            ("bond", "b2b", "bond"),
            ("global", "g2g", "global"),
        ]

        for etype in expected_etypes:
            assert etype in graph.canonical_etypes

    def test_no_self_loops(self):
        """Test graph creation without self-loops."""
        converter = GeometryToGraph(distance_cutoff=2.0, self_loop=False)

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "H"]

        graph = converter(coords, elements)

        # Self-loop edge types should not exist
        assert ("atom", "a2a", "atom") not in graph.canonical_etypes
        assert ("bond", "b2b", "bond") not in graph.canonical_etypes
        assert ("global", "g2g", "global") not in graph.canonical_etypes

    def test_tensor_input(self):
        """Test that torch tensor input is handled correctly."""
        converter = GeometryToGraph(distance_cutoff=2.0)

        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "H"]

        graph = converter(coords, elements)

        assert graph.num_nodes("atom") == 2
        assert graph.num_nodes("bond") == 1

    def test_custom_element_set(self):
        """Test using a custom element set."""
        custom_elements = ["X", "Y", "Z"]
        converter = GeometryToGraph(
            distance_cutoff=2.0,
            element_set=custom_elements,
            include_coordinates=False,
        )

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["X", "Y"]

        graph = converter(coords, elements)
        atom_feats = graph.nodes["atom"].data["feat"]

        # Should have 3 features (custom element set size)
        assert atom_feats.shape[1] == 3
        # X should be one-hot encoded at index 0
        assert atom_feats[0, 0] == 1.0
        # Y should be one-hot encoded at index 1
        assert atom_feats[1, 1] == 1.0

    def test_feature_info(self):
        """Test get_feature_info method."""
        converter = GeometryToGraph(
            distance_cutoff=2.0,
            include_coordinates=True,
            include_distances=True,
        )

        info = converter.get_feature_info()

        assert "atom" in info
        assert "bond" in info
        assert "global" in info

        # Atom: element one-hot + coordinates
        assert info["atom"]["feat_size"] == len(DEFAULT_ELEMENT_SET) + 3
        # Bond: distance + direction vector
        assert info["bond"]["feat_size"] == 4
        # Global: charge, spin, num_atoms
        assert info["global"]["feat_size"] == 3

    def test_mol_id_stored(self):
        """Test that molecule ID is stored on the graph."""
        converter = GeometryToGraph(distance_cutoff=2.0)

        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "H"]

        graph = converter(coords, elements, mol_id="test_mol_123")

        assert hasattr(graph, "mol_name")
        assert graph.mol_name == "test_mol_123"


class TestUpdateGraphTopology:
    """Tests for the update_graph_topology function."""

    def test_update_adds_edges(self):
        """Test that updating topology adds new edges."""
        # Create initial graph with 3 atoms and 1 edge
        converter = GeometryToGraph(distance_cutoff=1.2)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        elements = ["C", "C", "C"]

        old_graph = converter(coords, elements)

        # Update with new edges
        new_edges = [(0, 1), (1, 2)]
        new_graph = update_graph_topology(old_graph, new_edges)

        # Should now have 2 bonds
        assert new_graph.num_nodes("bond") == 2
        # Should preserve atom count
        assert new_graph.num_nodes("atom") == 3
        # Should preserve atom features
        assert torch.allclose(
            new_graph.nodes["atom"].data["feat"],
            old_graph.nodes["atom"].data["feat"],
        )

    def test_update_removes_edges(self):
        """Test that updating topology can remove edges."""
        converter = GeometryToGraph(distance_cutoff=2.5)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        elements = ["C", "C", "C"]

        old_graph = converter(coords, elements)
        old_bond_count = old_graph.num_nodes("bond")

        # Update with fewer edges
        new_edges = [(0, 1)]
        new_graph = update_graph_topology(old_graph, new_edges)

        # Should have only 1 bond
        assert new_graph.num_nodes("bond") == 1
        assert new_graph.num_nodes("bond") < old_bond_count

    def test_update_empty_edges(self):
        """Test handling of empty edge list."""
        converter = GeometryToGraph(distance_cutoff=2.0)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = ["C", "H"]

        old_graph = converter(coords, elements)

        # Update with no edges
        new_edges = []
        new_graph = update_graph_topology(old_graph, new_edges)

        # Should create dummy bond
        assert new_graph.num_nodes("bond") == 1


class TestEdgesFromPredictions:
    """Tests for the edges_from_predictions function."""

    def test_basic_thresholding(self):
        """Test that thresholding works correctly."""
        # Scores for 3 atoms: edges (0,1), (0,2), (1,2)
        scores = torch.tensor([0.8, 0.3, 0.6])
        num_atoms = 3

        edges = edges_from_predictions(scores, num_atoms, threshold=0.5)

        # Should return edges with score > 0.5
        assert (0, 1) in edges
        assert (0, 2) not in edges
        assert (1, 2) in edges
        assert len(edges) == 2

    def test_all_above_threshold(self):
        """Test when all scores are above threshold."""
        scores = torch.tensor([0.9, 0.8, 0.7])
        num_atoms = 3

        edges = edges_from_predictions(scores, num_atoms, threshold=0.5)

        assert len(edges) == 3

    def test_all_below_threshold(self):
        """Test when all scores are below threshold."""
        scores = torch.tensor([0.1, 0.2, 0.3])
        num_atoms = 3

        edges = edges_from_predictions(scores, num_atoms, threshold=0.5)

        assert len(edges) == 0

    def test_larger_molecule(self):
        """Test with a larger molecule."""
        num_atoms = 5
        # Number of upper triangular edges: n*(n-1)/2 = 5*4/2 = 10
        scores = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.05])

        edges = edges_from_predictions(scores, num_atoms, threshold=0.5)

        # Scores > 0.5: 0.9, 0.8, 0.7, 0.6 (0.5 is not > 0.5)
        assert len(edges) == 4


class TestIntegration:
    """Integration tests for the geometry_to_graph module."""

    def test_roundtrip_conversion(self):
        """Test that graph can be created and topology updated."""
        converter = GeometryToGraph(distance_cutoff=1.5)

        # Create a simple methane-like structure
        coords = np.array([
            [0.0, 0.0, 0.0],      # C
            [1.09, 0.0, 0.0],     # H
            [-0.36, 1.03, 0.0],   # H
            [-0.36, -0.51, 0.89], # H
            [-0.36, -0.51, -0.89], # H
        ])
        elements = ["C", "H", "H", "H", "H"]

        graph = converter(coords, elements)

        # Verify initial structure
        assert graph.num_nodes("atom") == 5

        # Simulate a prediction that modifies topology
        new_edges = [(0, 1), (0, 2), (0, 3), (0, 4)]  # All H bound to C
        updated_graph = update_graph_topology(graph, new_edges)

        assert updated_graph.num_nodes("atom") == 5
        assert updated_graph.num_nodes("bond") == 4

        # Verify features are preserved
        assert torch.allclose(
            updated_graph.nodes["atom"].data["feat"],
            graph.nodes["atom"].data["feat"],
        )

    def test_complete_pipeline(self):
        """Test the complete geometry -> graph -> update pipeline."""
        # Create converter
        converter = GeometryToGraph(
            distance_cutoff=2.0,
            include_coordinates=True,
            include_distances=True,
        )

        # CO2 molecule
        coords = np.array([
            [0.0, 0.0, 0.0],    # C
            [1.16, 0.0, 0.0],   # O
            [-1.16, 0.0, 0.0],  # O
        ])
        elements = ["C", "O", "O"]

        # Create initial graph
        graph = converter(coords, elements, charge=0, mol_id="CO2")

        # Check structure
        assert graph.num_nodes("atom") == 3
        assert graph.num_nodes("global") == 1
        assert hasattr(graph, "mol_name")

        # Simulate link prediction scores
        # 3 atoms -> 3 potential edges: (0,1), (0,2), (1,2)
        scores = torch.tensor([0.95, 0.95, 0.1])  # C-O bonds likely, O-O unlikely
        predicted_edges = edges_from_predictions(scores, 3, threshold=0.5)

        # Should predict C-O bonds
        assert (0, 1) in predicted_edges
        assert (0, 2) in predicted_edges
        assert (1, 2) not in predicted_edges

        # Update topology
        final_graph = update_graph_topology(graph, predicted_edges)
        assert final_graph.num_nodes("bond") == 2


