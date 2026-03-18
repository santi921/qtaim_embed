import torch
import numpy as np

from qtaim_embed.data.grapher import HeteroCompleteGraphFromMolWrapper
from qtaim_embed.data.featurizer import (
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,
)


def get_grapher(
    element_set,
    atom_keys,
    bond_keys=[],
    global_keys=[],
    allowed_ring_size=[],
    allowed_charges=None,
    allowed_spins=None,
    self_loop=True,
    atom_featurizer_tf=True,
    bond_featurizer_tf=True,
    global_featurizer_tf=True,
    rbf_cutoff=5.0,
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
            rbf_cutoff=rbf_cutoff,
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
        Compare two graphs, return true if they match in node types, number, and features.
    Takes
        g1, g2 (torch_geometric.data.HeteroData)
    Returns
        compare(bool): whether they are equal or not.
    """
    node_types = ["atom", "bond", "global"]
    edge_types = [
        ("atom", "a2b", "bond"),
        ("bond", "b2a", "atom"),
        ("atom", "a2g", "global"),
        ("global", "g2a", "atom"),
        ("global", "g2g", "global"),
        ("atom", "a2a", "atom"),
        ("bond", "b2b", "bond"),
    ]

    for nt in node_types:
        if g1[nt].num_nodes != g2[nt].num_nodes:
            return False
        ft1 = g1[nt].feat
        ft2 = g2[nt].feat
        if torch.any(ft1 != ft2):
            return False

    for et in edge_types:
        if et not in g1.edge_types or et not in g2.edge_types:
            # If edge type missing from both, that is consistent
            if et not in g1.edge_types and et not in g2.edge_types:
                continue
            return False
        ei1 = g1[et].edge_index
        ei2 = g2[et].edge_index
        if ei1.shape != ei2.shape:
            return False
        if torch.any(ei1 != ei2):
            return False

    return True


def get_bond_list_from_heterograph(het_graph):
    """
    Get list of bonds from heterograph.
    Takes:
        het_graph (torch_geometric.data.HeteroData): graph to convert
    Returns:
        a list of lists of bonds
    """

    edge_list = []
    id_list = []
    edge_index = het_graph["atom", "a2b", "bond"].edge_index
    nodes = edge_index[0]  # source (atom) indices
    bond_id = edge_index[1]  # destination (bond) indices
    for i in range(int(len(nodes) / 2)):
        a = nodes[2 * i]
        b = nodes[2 * i + 1]
        id = bond_id[2 * i]
        edge_list.append([a, b])
        id_list.append(id)

    return np.array(edge_list), np.array(id_list)


def get_fts_from_het_graph(het_graph):
    """
    Just get features from heterograph.
    Takes:
        het_graph (torch_geometric.data.HeteroData)
    Returns:
        atom, bond, and global feature tensors
    """
    atom_ft = het_graph["atom"].feat
    bond_ft = het_graph["bond"].feat
    global_ft = het_graph["global"].feat
    return atom_ft, bond_ft, global_ft


def construct_homograph_blank(node_list, bond_list):
    """
    Construct a simple homogeneous graph from a node list and bond list.
    Uses torch_geometric.data.Data.
    """
    from torch_geometric.data import Data

    num_nodes = len(node_list)
    if len(bond_list) > 0:
        edge_index = torch.tensor(
            [bond_list[:, 0].tolist(), bond_list[:, 1].tolist()], dtype=torch.long
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    g = Data(num_nodes=num_nodes, edge_index=edge_index)
    return g


def get_element_list_heterograph(g, dataset):
    """
    Get element list from heterograph.
    Takes:
        g (torch_geometric.data.HeteroData): heterograph
        dataset: dataset object
    Returns:
        element_list: list of elements
    """

    elem_name_ind = [
        i
        for i in range(len(dataset.feature_names["atom"]))
        if "chemical_symbol" in dataset.feature_names["atom"][i]
    ]
    elem_names = [
        i.split("_")[-1]
        for ind, i in enumerate(dataset.feature_names["atom"])
        if "chemical_symbol" in dataset.feature_names["atom"][ind]
    ]
    element_info = g["atom"].feat
    element_info = element_info[:, elem_name_ind]
    element_list = []
    # iterate through vertical and add minimum of each column to element_info
    for i in range(element_info.shape[1]):
        element_info[:, i] = element_info[:, i] - torch.min(element_info[:, i])
        if torch.max(element_info[:, i]) != 0:
            element_info[:, i] = element_info[:, i] / torch.max(element_info[:, i])

    # convert from one hot to element names
    for i in range(element_info.shape[0]):
        element_list.append(elem_names[torch.argmax(element_info[i])])

    return element_list
