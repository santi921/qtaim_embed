import torch
from torch.distributions import Bernoulli
from torch_geometric.data import HeteroData, Data

from qtaim_embed.data.grapher import build_hetero_graph_skeleton
from qtaim_embed.utils.grapher import (
    get_bond_list_from_heterograph,
    get_fts_from_het_graph,
    construct_homograph_blank,
)


class DropBondHeterograph:
    r"""Randomly drop bond nodes from a heterogeneous graph.

    Inspired by DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
    <https://arxiv.org/abs/1907.10903> and Graph Contrastive Learning with Augmentations
    <https://arxiv.org/abs/2010.13902>.

    Parameters
    ----------
    p : float, optional
        Probability of a bond node being dropped.
    drop_node_type : list, optional
        List of node types to drop.
    drop_edge_types : list, optional
        List of edge type relation names that involve the dropped node types.
    """

    def __init__(
        self,
        p=0.5,
        drop_node_type=["bond"],
        drop_edge_types=["a2b", "b2a", "g2b", "b2g", "b2b"],
    ):
        self.p = p
        self.dist = Bernoulli(p)
        self.drop_node_type = drop_node_type
        self.drop_edge_types = drop_edge_types

    def __call__(self, g):
        g = g.clone()
        # Fast path
        if self.p == 0:
            return g

        for c_ntype in self.drop_node_type:
            num_nodes = g[c_ntype].num_nodes
            samples = self.dist.sample(torch.Size([num_nodes]))
            keep_mask = ~samples.bool()
            keep_ids = torch.where(keep_mask)[0]

            if keep_ids.numel() == 0:
                # Keep at least one node to avoid empty graph
                keep_ids = torch.tensor([0])
                keep_mask = torch.zeros(num_nodes, dtype=torch.bool)
                keep_mask[0] = True

            # Build node ID remapping
            new_ids = torch.full((num_nodes,), -1, dtype=torch.long)
            new_ids[keep_ids] = torch.arange(keep_ids.numel())

            # Update node features for the dropped type
            for key in g[c_ntype].keys():
                attr = getattr(g[c_ntype], key)
                if isinstance(attr, torch.Tensor) and attr.size(0) == num_nodes:
                    setattr(g[c_ntype], key, attr[keep_ids])
            g[c_ntype].num_nodes = keep_ids.numel()

            # Update all edge types that reference this node type
            for edge_type in g.edge_types:
                src_type, rel, dst_type = edge_type
                edge_index = g[edge_type].edge_index

                if src_type == c_ntype:
                    # Remap source nodes, filter edges with removed nodes
                    mask = keep_mask[edge_index[0]]
                    edge_index = edge_index[:, mask]
                    edge_index[0] = new_ids[edge_index[0]]
                    g[edge_type].edge_index = edge_index

                if dst_type == c_ntype:
                    # Remap destination nodes, filter edges with removed nodes
                    mask = keep_mask[edge_index[1]]
                    edge_index = edge_index[:, mask]
                    edge_index[1] = new_ids[edge_index[1]]
                    g[edge_type].edge_index = edge_index

        return g


class hetero_to_homo:
    """Convert a heterogeneous molecular graph to a homogeneous graph.

    Atom nodes become graph nodes, bond nodes become edge features.
    Optionally concatenates global features onto atom and bond features.
    """

    def __init__(self, concat_global=False):
        self.concat_global = concat_global
        self.global_feat_len = None

    def __call__(self, graph):
        graph = graph.clone()
        edge_list, id_list = get_bond_list_from_heterograph(graph)
        atom_ft, bond_ft, global_ft = get_fts_from_het_graph(graph)

        # if concat_global, add global_ft to atom_ft and bond_ft at each node
        if self.concat_global:
            if self.global_feat_len is None:
                self.global_feat_len = global_ft.shape[1]
            global_ft_atom = global_ft.repeat(atom_ft.shape[0], 1)
            atom_ft = torch.cat([atom_ft, global_ft_atom], dim=1)
            global_ft_bond = global_ft.repeat(bond_ft.shape[0], 1)
            bond_ft = torch.cat([bond_ft, global_ft_bond], dim=1)

        homo = construct_homograph_blank(graph["atom"].feat, edge_list)

        homo.ft = atom_ft
        # Store bond features as edge attributes
        if homo.edge_index.size(1) > 0:
            homo.edge_ft = bond_ft

        return homo


class homo_to_hetero:
    """Convert a homogeneous graph back to a heterogeneous molecular graph.

    Reconstructs atom, bond, and global node types from a homogeneous graph
    where global features were concatenated.
    """

    def __init__(self, global_feat_len, self_loop=True):
        self.global_feat_len = global_feat_len
        self.self_loop = self_loop

    def __call__(self, graph):
        graph = graph.clone()
        atom_ft = graph.ft
        bond_ft = graph.edge_ft if hasattr(graph, 'edge_ft') else None
        atom_ft_base = atom_ft[:, 0 : -self.global_feat_len]
        global_ft = atom_ft[0, -self.global_feat_len :].reshape(1, -1)

        num_atoms = graph.num_nodes
        edge_index = graph.edge_index
        num_bonds = edge_index.size(1)

        if bond_ft is not None:
            bond_ft_base = bond_ft[:, 0 : -self.global_feat_len]
        else:
            bond_ft_base = torch.zeros(num_bonds, 0)

        # Convert edge_index tensor to list of (u, v) tuples for skeleton builder
        edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        data = build_hetero_graph_skeleton(num_atoms, edges, self_loop=self.self_loop)

        # Set node features
        data["atom"].feat = atom_ft_base
        data["bond"].feat = bond_ft_base
        data["global"].feat = global_ft
        return data
