import pandas as pd
from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.molwrapper import mol_wrappers_from_df
from qtaim_embed.utils.tests import get_data


class TestGrapher:
    df_test = get_data()

    def test_graph_nodes(self):
        atom_keys = [
            "extra_feat_atom_esp_total",
        ]
        bond_keys = [
            "extra_feat_bond_esp_total",
        ]
        mol_wrappers, element_set = mol_wrappers_from_df(self.df_test, [], [])

        list_atom_num = [mol.num_atoms for mol in mol_wrappers]
        list_bond_num = [len(mol.bonds) for mol in mol_wrappers]

        grapher = get_grapher(
            element_set,
            atom_keys=atom_keys,
            bond_keys=bond_keys,
            global_keys=[],
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=None,
            self_loop=True,
        )

        graph_list = []
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)
            graph_list.append(graph)

        for ind in range(len(graph_list)):
            # shape_atom_feats = graph.ndata["feat"]["atom"].shape
            num_atoms_mol_wrapper = list_atom_num[ind]
            num_atoms = graph_list[ind].num_nodes("atom")
            assert num_atoms == num_atoms_mol_wrapper
            num_bonds_mol_wrapper = list_bond_num[ind]
            num_bonds = graph_list[ind].num_nodes("bond")
            assert num_bonds_mol_wrapper == num_bonds
