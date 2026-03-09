from typing import List, Tuple

import torch
from torch_geometric.data import HeteroData


def build_hetero_graph_skeleton(
    num_atoms: int,
    edges: List[Tuple[int, int]],
    self_loop: bool = True,
) -> HeteroData:
    """Build a HeteroData graph with standard molecular edge types.

    Creates atom, bond, and global nodes with edge types:
    a2b, b2a, a2g, g2a, b2g, g2b, and optionally a2a, b2b, g2g self-loops.

    Args:
        num_atoms: Number of atom nodes.
        edges: List of (atom_i, atom_j) bond tuples.
        self_loop: Whether to include self-loop edge types.

    Returns:
        HeteroData with edge structure (no node features).
    """
    num_bonds = len(edges)

    if num_bonds == 0:
        num_bonds = 1
        a2b_src, a2b_dst = [0], [0]
        b2a_src, b2a_dst = [0], [0]
    else:
        a2b_src, a2b_dst = [], []
        b2a_src, b2a_dst = [], []
        for bond_idx, (u, v) in enumerate(edges):
            a2b_src.extend([u, v])
            a2b_dst.extend([bond_idx, bond_idx])
            b2a_src.extend([bond_idx, bond_idx])
            b2a_dst.extend([u, v])

    a2g_src = list(range(num_atoms))
    a2g_dst = [0] * num_atoms
    g2a_src = [0] * num_atoms
    g2a_dst = list(range(num_atoms))
    b2g_src = list(range(num_bonds))
    b2g_dst = [0] * num_bonds
    g2b_src = [0] * num_bonds
    g2b_dst = list(range(num_bonds))

    data = HeteroData()
    data["atom"].num_nodes = num_atoms
    data["bond"].num_nodes = num_bonds
    data["global"].num_nodes = 1

    data["atom", "a2b", "bond"].edge_index = torch.tensor(
        [a2b_src, a2b_dst], dtype=torch.long
    )
    data["bond", "b2a", "atom"].edge_index = torch.tensor(
        [b2a_src, b2a_dst], dtype=torch.long
    )
    data["atom", "a2g", "global"].edge_index = torch.tensor(
        [a2g_src, a2g_dst], dtype=torch.long
    )
    data["global", "g2a", "atom"].edge_index = torch.tensor(
        [g2a_src, g2a_dst], dtype=torch.long
    )
    data["bond", "b2g", "global"].edge_index = torch.tensor(
        [b2g_src, b2g_dst], dtype=torch.long
    )
    data["global", "g2b", "bond"].edge_index = torch.tensor(
        [g2b_src, g2b_dst], dtype=torch.long
    )

    if self_loop:
        a2a_nodes = list(range(num_atoms))
        b2b_nodes = list(range(num_bonds))
        data["atom", "a2a", "atom"].edge_index = torch.tensor(
            [a2a_nodes, a2a_nodes], dtype=torch.long
        )
        data["bond", "b2b", "bond"].edge_index = torch.tensor(
            [b2b_nodes, b2b_nodes], dtype=torch.long
        )
        data["global", "g2g", "global"].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )

    return data


class HeteroCompleteGraphFromMolWrapper:
    """ """

    def __init__(
        self,
        atom_featurizer=None,
        bond_featurizer=None,
        global_featurizer=None,
        self_loop=True,
    ):
        self.self_loop = self_loop
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer
        self.feat_names = None

    def build_graph(self, mol):
        bonds = list(mol.bonds.keys())
        num_atoms = len(mol.coords)

        data = build_hetero_graph_skeleton(num_atoms, bonds, self_loop=self.self_loop)
        data.mol_name = mol.id

        return data

    def featurize(self, data, mol, ret_feat_names=False, **kwargs):
        if self.atom_featurizer is not None:
            feat_dict, feat_atom = self.atom_featurizer(mol, **kwargs)
            for key, val in feat_dict.items():
                data["atom"][key] = val

        if self.bond_featurizer is not None:
            feat_dict, feat_bond = self.bond_featurizer(mol, **kwargs)
            for key, val in feat_dict.items():
                data["bond"][key] = val

        if self.global_featurizer is not None:
            feat_dict, globe_feat = self.global_featurizer(mol, **kwargs)
            for key, val in feat_dict.items():
                data["global"][key] = val

        if ret_feat_names or self.feat_names is None:
            feat_names = {}
            if self.atom_featurizer is not None:
                feat_names["atom"] = feat_atom
            if self.bond_featurizer is not None:
                feat_names["bond"] = feat_bond
            if self.global_featurizer is not None:
                feat_names["global"] = globe_feat
            self.feat_names = feat_names
            if ret_feat_names:
                return data, feat_names

        return data
