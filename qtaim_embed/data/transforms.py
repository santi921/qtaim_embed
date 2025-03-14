import torch
from torch.distributions import Bernoulli
from dgl.transforms import BaseTransform
import dgl

from qtaim_embed.utils.grapher import (
    get_bond_list_from_heterograph,
    get_fts_from_het_graph,
    construct_homograph_blank,
)

class DropBondHeterograph(BaseTransform):
    r"""Randomly drop edges, as described in
    `DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
    <https://arxiv.org/abs/1907.10903>`__ and `Graph Contrastive Learning with Augmentations
    <https://arxiv.org/abs/2010.13902>`__.

    Takes
    ----------
    p : float, optional
        Probability of an edge to be dropped.
    drop_node_type : list, optional
        list of node types to drop
    drop_edge_types : list, optional
        list of edge types to drop that correspond to the node types in drop_node_type

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

        for c_etype in self.drop_node_type:
            samples = self.dist.sample(torch.Size([g.num_nodes(c_etype)]))
            node_ids_to_remove = g.nodes(ntype=c_etype)[samples.bool()]
            g.remove_nodes(node_ids_to_remove, ntype=c_etype)

        return g


class hetero_to_homo(BaseTransform):
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

        homo = construct_homograph_blank(graph.nodes["atom"].data["feat"], edge_list)

        homo.ndata["ft"] = atom_ft
        homo.edata["ft"] = bond_ft

        return homo


class homo_to_hetero(BaseTransform):
    def __init__(self, global_feat_len, self_loop=True):
        self.global_feat_len = global_feat_len
        self.self_loop = self_loop

    def __call__(self, graph):

        graph = graph.clone()
        atom_ft = graph.ndata["ft"]
        bond_ft = graph.edata["ft"]
        atom_ft_base = atom_ft[:, 0 : -self.global_feat_len]
        bond_ft_base = bond_ft[:, 0 : -self.global_feat_len]
        global_ft = atom_ft[0, -self.global_feat_len :].reshape(1, -1)
        node_list = graph.nodes()
        edges_raw_u, edges_raw_v = graph.edges()
        bond_list = [[edges_raw_u[i], edges_raw_v[i]] for i in range(len(edges_raw_u))]

        num_atoms = len(node_list)
        num_bonds = len(bond_list)

        a2b = []
        b2a = []

        if num_bonds == 0:
            num_bonds = 1
            a2b = [(0, 0)]
            b2a = [(0, 0)]

        else:
            a2b = []
            b2a = []
            for b in range(num_bonds):
                u = bond_list[b][0]
                v = bond_list[b][1]
                b2a.extend([[b, u], [b, v]])
                a2b.extend([[u, b], [v, b]])

        a2g = [(a, 0) for a in range(num_atoms)]
        g2a = [(0, a) for a in range(num_atoms)]
        b2g = [(b, 0) for b in range(num_bonds)]
        g2b = [(0, b) for b in range(num_bonds)]

        edges_dict = {
            ("atom", "a2b", "bond"): a2b,
            ("bond", "b2a", "atom"): b2a,
            ("atom", "a2g", "global"): a2g,
            ("global", "g2a", "atom"): g2a,
            ("bond", "b2g", "global"): b2g,
            ("global", "g2b", "bond"): g2b,
        }
        if self.self_loop:
            a2a = [(i, i) for i in range(num_atoms)]
            b2b = [(i, i) for i in range(num_bonds)]
            g2g = [(0, 0)]
            edges_dict.update(
                {
                    ("atom", "a2a", "atom"): a2a,
                    ("bond", "b2b", "bond"): b2b,
                    ("global", "g2g", "global"): g2g,
                }
            )

        g = dgl.heterograph(edges_dict)

        # update node and edge features
        g.nodes["atom"].data["feat"] = atom_ft_base
        g.nodes["bond"].data["feat"] = bond_ft_base
        g.nodes["global"].data["feat"] = global_ft
        return g
