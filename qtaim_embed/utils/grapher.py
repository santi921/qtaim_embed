import torch 
import dgl 
import numpy as np 

from qtaim_embed.data.grapher import HeteroCompleteGraphFromMolWrapper
from qtaim_embed.data.featurizer import (
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,
)


def get_grapher(
    element_set,
    atom_keys=[],
    bond_keys=[],
    global_keys=[],
    allowed_ring_size=[],
    allowed_charges=None,
    allowed_spins=None,
    self_loop=True,
    atom_featurizer_tf=True,
    bond_featurizer_tf=True,
    global_featurizer_tf=True,
):
    if not atom_featurizer_tf:
        atom_featurizer = None
    else:
        atom_featurizer = AtomFeaturizerGraphGeneral(
            selected_keys=atom_keys,
            element_set=element_set,
            allowed_ring_size=allowed_ring_size,
        )

    if not bond_featurizer_tf:
        bond_featurizer = None
    else:
        bond_featurizer = BondAsNodeGraphFeaturizerGeneral(
            selected_keys=bond_keys,
            allowed_ring_size=allowed_ring_size,
        )

    if not global_featurizer_tf:
        global_featurizer = None
    else:
        global_featurizer = GlobalFeaturizerGraph(
            selected_keys=global_keys,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
        )

    grapher = HeteroCompleteGraphFromMolWrapper(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=self_loop,
    )
    return grapher



def compare_graphs(g1, g2): 
    """
        Compare two graphs, return true if they match in node types, number, and features
    Takes
        g2, g1(dgl.Heterograph)
    Returns 
        compare(bool): whether they're equal or not.
    """
    node_types = ["atom", "bond", "global"]
    edge_types = ["a2b", "b2a", "a2g", "g2a", "g2g", "a2a", "b2b"]
    
    for nt in node_types: 
        if g1.num_nodes(nt) != g2.num_nodes(nt):
            return False
        ft1 = g1.nodes[nt].data["feat"]
        ft2 = g2.nodes[nt].data["feat"]
        if torch.any(ft1 != ft2): return False

    for et in edge_types: 
        u1, v1 = g1.edges(etype=et)
        u2, v2 = g2.edges(etype=et)
        if torch.any(u1 != u2): return False
        if torch.any(v1 != v2): return False


    return True


def get_bond_list_from_heterograph(het_graph):
    """
    Get list of bonds from heterograph
    Takes: 
        het_graph(dgl heterograph): graph to convert
    Returns: 
        a list of lists of bonds
    """

    edge_list = []
    id_list = []
    nodes, bond_id = het_graph.edges(etype="a2b")
    for i in range(int(len(nodes)/2)):
        a = nodes[2*i]#.tolist()
        b = nodes[2*i+1]#.tolist()
        id = bond_id[2*i]#.tolist()
        edge_list.append([a,b])
        id_list.append(id)
    
    return np.array(edge_list), np.array(id_list)


def get_fts_from_het_graph(het_graph):
    """
    Just get features from hetereograph
    Takes:
        heterograph(dgl.Heterograph)
    Returns: 
        atom, bond, and global feature tensors
    """
    atom_ft = het_graph.nodes["atom"].data["feat"]
    bond_ft = het_graph.nodes["bond"].data["feat"]
    global_ft = het_graph.nodes["global"].data["feat"]
    return atom_ft, bond_ft, global_ft


def construct_homograph_blank(node_list, bond_list):
    # construct graph with node list and bond list
    g = dgl.graph(([],[]))
    for i in range(len(node_list)):
        g.add_nodes(1)
    g.add_edges(bond_list[:,0], bond_list[:,1])
    return g 

