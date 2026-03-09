"""
GeometryToGraph module for building initial heterographs from molecular geometry.

This module creates initial graph representations from xyz coordinates and element types,
using distance cutoffs to determine connectivity. It is designed for use in the
iterative link-node prediction pipeline.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial.distance import cdist
from torch_geometric.data import HeteroData

from qtaim_embed.data.grapher import build_hetero_graph_skeleton

# Default element set for one-hot encoding (common organic + transition metals)
DEFAULT_ELEMENT_SET = [
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "B", "Si", "Se", "Te", "As",
    "Li", "Na", "K", "Mg", "Ca",
    "Fe", "Co", "Ni", "Cu", "Zn", "Mn", "Cr", "V", "Ti",
    "Pd", "Pt", "Au", "Ag", "Ru", "Rh", "Ir", "Os",
    "Mo", "W", "Re", "Ta", "Nb", "Zr", "Hf",
]


class GeometryToGraph:
    """
    Converts molecular geometry (xyz coordinates + element types) to a PyG HeteroData graph.

    This class builds an initial graph representation using distance cutoffs for edge
    determination, suitable for use in iterative link prediction -> node prediction loops.

    Args:
        distance_cutoff: Maximum distance (in Angstroms) for initial edge creation.
            Default is 1.8 A which captures most covalent bonds.
        element_set: List of elements for one-hot encoding. If None, uses DEFAULT_ELEMENT_SET.
        self_loop: Whether to add self-loops for atom, bond, and global nodes.
        include_coordinates: Whether to include xyz coordinates in atom features.
        include_distances: Whether to include bond distances in bond features.
        min_distance: Minimum distance threshold to avoid self-connections (default 0.1 A).

    Example:
        >>> converter = GeometryToGraph(distance_cutoff=2.0)
        >>> coords = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0]])
        >>> elements = ["C", "H", "H"]
        >>> graph = converter(coords, elements)
    """

    def __init__(
        self,
        distance_cutoff: float = 1.8,
        element_set: Optional[List[str]] = None,
        self_loop: bool = True,
        include_coordinates: bool = True,
        include_distances: bool = True,
        min_distance: float = 0.1,
    ):
        self.distance_cutoff = distance_cutoff
        self.element_set = element_set if element_set is not None else DEFAULT_ELEMENT_SET
        self.self_loop = self_loop
        self.include_coordinates = include_coordinates
        self.include_distances = include_distances
        self.min_distance = min_distance

        # Build element to index mapping
        self.element_to_idx = {elem: idx for idx, elem in enumerate(self.element_set)}

    def __call__(
        self,
        coords: Union[np.ndarray, torch.Tensor],
        elements: List[str],
        charge: int = 0,
        spin_multiplicity: int = 1,
        mol_id: Optional[str] = None,
    ) -> HeteroData:
        """
        Build a HeteroData graph from coordinates and element types.

        Args:
            coords: Atomic coordinates of shape (N, 3) in Angstroms.
            elements: List of element symbols of length N.
            charge: Molecular charge (default 0).
            spin_multiplicity: Spin multiplicity (default 1 for singlet).
            mol_id: Optional molecule identifier.

        Returns:
            PyG HeteroData with atom, bond, and global nodes.
        """
        coords = self._to_numpy(coords)

        # Compute pairwise distances
        distances = cdist(coords, coords)

        # Find edges based on distance cutoff
        edges = self._get_edges_from_distances(distances)

        # Build heterograph structure
        graph = self._build_heterodata(len(elements), edges)

        # Add node features
        self._add_atom_features(graph, elements, coords)
        self._add_bond_features(graph, edges, coords, distances)
        self._add_global_features(graph, charge, spin_multiplicity, len(elements))

        # Add metadata
        if mol_id is not None:
            graph.mol_name = mol_id

        return graph

    def _to_numpy(self, arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _get_edges_from_distances(
        self,
        distances: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Determine edges based on distance cutoff.

        Returns list of (atom_i, atom_j) tuples where i < j.
        """
        rows, cols = np.triu_indices(distances.shape[0], k=1)
        d = distances[rows, cols]
        mask = (d > self.min_distance) & (d <= self.distance_cutoff)
        return list(zip(rows[mask].tolist(), cols[mask].tolist()))

    def _build_heterodata(
        self,
        num_atoms: int,
        edges: List[Tuple[int, int]],
    ) -> HeteroData:
        """
        Build the PyG HeteroData graph structure.

        Creates atom, bond, and global nodes with appropriate edge types:
        - a2b, b2a: atom-bond connections
        - a2g, g2a: atom-global connections
        - b2g, g2b: bond-global connections
        - a2a, b2b, g2g: self-loops (if enabled)
        """
        return build_hetero_graph_skeleton(num_atoms, edges, self_loop=self.self_loop)

    def _add_atom_features(
        self,
        graph: HeteroData,
        elements: List[str],
        coords: np.ndarray,
    ) -> None:
        """Add features to atom nodes."""
        num_atoms = len(elements)

        # One-hot encoding of elements
        element_one_hot = np.zeros((num_atoms, len(self.element_set)), dtype=np.float32)
        for i, elem in enumerate(elements):
            if elem in self.element_to_idx:
                element_one_hot[i, self.element_to_idx[elem]] = 1.0

        features = [element_one_hot]

        if self.include_coordinates:
            features.append(coords.astype(np.float32))

        atom_feats = np.concatenate(features, axis=1)
        graph["atom"].feat = torch.tensor(atom_feats, dtype=torch.float32)

    def _add_bond_features(
        self,
        graph: HeteroData,
        edges: List[Tuple[int, int]],
        coords: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        """Add features to bond nodes."""
        num_bonds = max(len(edges), 1)  # At least 1 for dummy bond

        features = []

        if self.include_distances:
            bond_distances = np.zeros((num_bonds, 1), dtype=np.float32)
            for bond_idx, (i, j) in enumerate(edges):
                bond_distances[bond_idx, 0] = distances[i, j]
            features.append(bond_distances)

        # Add bond direction vector (normalized) - vectorized
        bond_vectors = np.zeros((num_bonds, 3), dtype=np.float32)
        if edges:
            src_idx = np.array([e[0] for e in edges])
            dst_idx = np.array([e[1] for e in edges])
            vecs = coords[dst_idx] - coords[src_idx]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 1e-6, norms, 1.0)
            bond_vectors[:len(edges)] = vecs / norms
        features.append(bond_vectors)

        if len(edges) == 0:
            bond_feats = np.zeros((1, sum(f.shape[1] for f in features)), dtype=np.float32)
        else:
            bond_feats = np.concatenate(features, axis=1)

        graph["bond"].feat = torch.tensor(bond_feats, dtype=torch.float32)

    def _add_global_features(
        self,
        graph: HeteroData,
        charge: int,
        spin_multiplicity: int,
        num_atoms: int,
    ) -> None:
        """Add features to global node."""
        global_feats = np.array([[
            float(charge),
            float(spin_multiplicity),
            float(num_atoms),
        ]], dtype=np.float32)

        graph["global"].feat = torch.tensor(global_feats, dtype=torch.float32)

    @classmethod
    def from_molecule_wrapper(
        cls,
        mol_wrapper,
        distance_cutoff: float = 1.8,
        use_mol_bonds: bool = False,
        **kwargs,
    ) -> HeteroData:
        """
        Create a graph from a MoleculeWrapper object.

        Args:
            mol_wrapper: MoleculeWrapper instance from qtaim_embed.
            distance_cutoff: Distance cutoff for edge creation.
            use_mol_bonds: If True, use bonds from MoleculeWrapper instead of
                distance cutoff. Default False (uses distance cutoff for initial guess).
            **kwargs: Additional arguments passed to GeometryToGraph constructor.

        Returns:
            PyG HeteroData graph.
        """
        converter = cls(distance_cutoff=distance_cutoff, **kwargs)

        coords = mol_wrapper.coords
        elements = mol_wrapper.species
        charge = mol_wrapper.charge
        mol_id = mol_wrapper.id

        if use_mol_bonds:
            edges = list(mol_wrapper.bonds.keys())
            graph = converter._build_heterodata(len(elements), edges)

            distances = cdist(coords, coords)

            converter._add_atom_features(graph, elements, coords)
            converter._add_bond_features(graph, edges, coords, distances)
            converter._add_global_features(graph, charge, 1, len(elements))

            if mol_id is not None:
                graph.mol_name = mol_id

            return graph
        else:
            return converter(coords, elements, charge=charge, mol_id=mol_id)

    def get_feature_info(self) -> Dict[str, Dict[str, int]]:
        """
        Return information about feature dimensions.

        Returns:
            Dictionary with feature sizes for each node type.
        """
        atom_dim = len(self.element_set)
        if self.include_coordinates:
            atom_dim += 3

        bond_dim = 3  # direction vector
        if self.include_distances:
            bond_dim += 1

        global_dim = 3  # charge, spin, num_atoms

        return {
            "atom": {"feat_size": atom_dim},
            "bond": {"feat_size": bond_dim},
            "global": {"feat_size": global_dim},
        }

    @classmethod
    def from_grapher_config(
        cls,
        grapher_config: Dict,
        distance_cutoff: float = 1.8,
        **kwargs,
    ) -> "GeometryToGraph":
        """
        Create GeometryToGraph using element_set from a grapher configuration.

        This factory method extracts the element_set from a grapher config (as stored
        in model hyperparameters) and creates a GeometryToGraph instance that will
        produce graphs with matching element one-hot encoding.

        Args:
            grapher_config: Dictionary containing grapher configuration, typically
                extracted from model.hparams.grapher_config. Must contain 'element_set'.
            distance_cutoff: Maximum distance for initial edge creation.
            **kwargs: Additional arguments passed to GeometryToGraph constructor.

        Returns:
            GeometryToGraph instance configured to match the training featurization.

        Example:
            >>> grapher_config = model.hparams.get("grapher_config")
            >>> converter = GeometryToGraph.from_grapher_config(
            ...     grapher_config, distance_cutoff=2.0
            ... )
            >>> graph = converter(coords, elements)
        """
        element_set = grapher_config.get("element_set")
        if element_set is None:
            raise ValueError("grapher_config must contain 'element_set'")

        return cls(
            distance_cutoff=distance_cutoff,
            element_set=element_set,
            **kwargs,
        )


def edges_from_predictions(
    scores: torch.Tensor,
    num_atoms: int,
    threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Convert link prediction scores to edge list.

    Args:
        scores: Tensor of edge prediction scores. Should be of shape
            (num_potential_edges,) where potential edges are ordered as
            (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ..., (n-2, n-1).
        num_atoms: Number of atoms in the molecule.
        threshold: Score threshold above which an edge is predicted.
        symmetric: If True, scores represent upper triangular only.

    Returns:
        List of (atom_i, atom_j) tuples for predicted edges.
    """
    i_idx, j_idx = torch.triu_indices(num_atoms, num_atoms, offset=1)
    mask = scores > threshold
    return list(zip(i_idx[mask].tolist(), j_idx[mask].tolist()))


def update_graph_topology(
    old_graph: HeteroData,
    new_edges: List[Tuple[int, int]],
    self_loop: bool = True,
) -> HeteroData:
    """
    Create a new HeteroData graph with updated edge topology while preserving node features.

    This function is used in the iterative prediction loop to update the graph
    structure based on link prediction results.

    Args:
        old_graph: The existing HeteroData graph.
        new_edges: New list of edges (atom pairs).
        self_loop: Whether to include self-loops.

    Returns:
        New HeteroData graph with updated topology and preserved atom features.
    """
    num_atoms = old_graph["atom"].num_nodes
    num_bonds = max(len(new_edges), 1)

    new_graph = build_hetero_graph_skeleton(num_atoms, new_edges, self_loop=self_loop)

    # Copy atom and global features from old graph
    new_graph["atom"].feat = old_graph["atom"].feat.clone()
    new_graph["global"].feat = old_graph["global"].feat.clone()

    # Initialize bond features to zero (topology changed; values recomputed by node model)
    old_bond_feat_dim = old_graph["bond"].feat.shape[1]
    new_graph["bond"].feat = torch.zeros(
        (num_bonds, old_bond_feat_dim), dtype=torch.float32
    )

    return new_graph
