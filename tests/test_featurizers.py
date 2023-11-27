import pandas as pd
from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.molwrapper import mol_wrappers_from_df
from qtaim_embed.utils.tests import get_data, get_data_spin_charge


class TestGrapher:
    df_test = get_data()
    df_test_spin_charge = get_data_spin_charge()

    def test_node_sizes(self):
        atom_keys = [
            "extra_feat_atom_esp_total",
        ]
        bond_keys = [
            "extra_feat_bond_esp_total",
        ]
        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test,
            bond_key="bonds",
            atom_keys=atom_keys,
            bond_keys=bond_keys,
            global_keys=[],
        )

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
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)

        for graph in graph_list:
            shape_atom_feats = graph.ndata["feat"]["atom"].shape
            num_atoms = graph.num_nodes("atom")
            assert shape_atom_feats[0] == num_atoms
            shape_bond_feats = graph.ndata["feat"]["bond"].shape
            num_bonds = graph.num_nodes("bond")
            assert shape_bond_feats[0] == num_bonds
            shape_global_feats = graph.ndata["feat"]["global"].shape
            assert shape_global_feats[0] == 1

    def test_atom_featurizer(self):
        atom_keys = [
            "extra_feat_atom_esp_total",
            "extra_feat_atom_esp_nuc",
        ]
        col_test = pd.DataFrame(self.df_test["extra_feat_atom_esp_total"])

        test_vals = [col_test.iloc[ind][0][0] for ind in range(len(col_test))]
        # print(test_vals)
        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test,
            bond_key="bonds",
            atom_keys=atom_keys,
            bond_keys=[],
            global_keys=[],
        )

        grapher = get_grapher(
            element_set,
            atom_keys=atom_keys,
            bond_keys=[],
            global_keys=[],
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=None,
            self_loop=True,
            bond_featurizer_tf=False,
        )
        grapher_bare = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=[],
            allowed_ring_size=[],
            allowed_charges=None,
            self_loop=True,
            bond_featurizer_tf=False,
        )

        graph_list_bare, graph_list = [], []
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)

            graph, names_full = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)

            graph_bare = grapher_bare.build_graph(mol)
            graph_bare, names = grapher_bare.featurize(
                graph_bare, mol, ret_feat_names=True
            )
            graph_list_bare.append(graph_bare)

        for ind in range(len(graph_list)):
            assert graph_list[ind].ndata["feat"]["atom"][0, -2] == test_vals[ind]
            assert (
                graph_list[ind].ndata["feat"]["atom"].shape[1]
                == graph_list_bare[ind].ndata["feat"]["atom"].shape[1] + 8
            )

    def test_bond_featurizer(self):
        # test qtaim AND bond length as features!!!

        bond_keys = [
            "extra_feat_bond_esp_total",
            "extra_feat_bond_esp_nuc",
            "bond_length",
        ]

        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test, bond_key="bonds", atom_keys=[], bond_keys=bond_keys
        )

        list_test = []
        for wrapper in mol_wrappers:
            bond_list = list(wrapper.bond_features.keys())

            list_test.append(
                wrapper.bond_features[bond_list[3]]["extra_feat_bond_esp_total"]
            )

        grapher = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=bond_keys,
            global_keys=[],
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=None,
            self_loop=True,
            atom_featurizer_tf=False,
        )
        grapher_bare = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=[],
            allowed_ring_size=[3],
            allowed_charges=None,
            self_loop=True,
            atom_featurizer_tf=False,
        )

        graph_list_bare, graph_list = [], []
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)

            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)
            graph_bare = grapher_bare.build_graph(mol)
            graph_bare, names = grapher_bare.featurize(
                graph_bare, mol, ret_feat_names=True
            )
            graph_list_bare.append(graph_bare)

        for ind in range(len(graph_list)):
            assert graph_list[ind].ndata["feat"]["bond"][3, -2] == list_test[ind]

            assert (
                graph_list[ind].ndata["feat"]["bond"].shape[1]
                == graph_list_bare[ind].ndata["feat"]["bond"].shape[1] + 7
            )

    def test_global_featurizers(self):
        global_keys = ["ids"]

        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test,
            bond_key="bonds",
            atom_keys=[],
            bond_keys=[],
            global_keys=global_keys,
        )
        # for mol in mol_wrappers:
        #    print(mol.global_features)

        grapher = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=global_keys,
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=None,
            self_loop=True,
            atom_featurizer_tf=False,
            bond_featurizer_tf=False,
        )
        grapher_bare = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=[],
            allowed_ring_size=[],
            allowed_charges=None,
            self_loop=True,
            atom_featurizer_tf=False,
            bond_featurizer_tf=False,
        )

        graph_list_bare, graph_list = [], []
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)

            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_list.append(graph)
            graph_bare = grapher_bare.build_graph(mol)
            graph_bare, names = grapher_bare.featurize(
                graph_bare, mol, ret_feat_names=True
            )

            graph_list_bare.append(graph_bare)

        id_list = [2, 10, 300]
        for ind in range(len(graph_list)):
            assert (
                graph_list[ind].ndata["feat"]["global"].shape[1]
                == graph_list_bare[ind].ndata["feat"]["global"].shape[1] + 1
            )
            assert int(graph_list[ind].ndata["feat"]["global"][0][-1]) == id_list[ind]

    def test_spin_charge_encoding(self):
        global_keys = ["spin", "charge"]

        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test_spin_charge,
            bond_key="bonds",
            atom_keys=[],
            bond_keys=[],
            global_keys=global_keys,
        )

        grapher = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=global_keys,
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=[-1, 0, 1],
            allowed_spins=[1, 2, 3],
            self_loop=True,
            atom_featurizer_tf=False,
            bond_featurizer_tf=False,
        )
        grapher_spin = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=global_keys,
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=None,
            allowed_spins=[1, 2, 3],
            self_loop=True,
            atom_featurizer_tf=False,
            bond_featurizer_tf=False,
        )
        grapher_charge = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=global_keys,
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=[-1, 0, 1],
            allowed_spins=None,
            self_loop=True,
            atom_featurizer_tf=False,
            bond_featurizer_tf=False,
        )
        grapher_bare = get_grapher(
            element_set,
            atom_keys=[],
            bond_keys=[],
            global_keys=[],
            allowed_ring_size=[],
            allowed_charges=None,
            allowed_spins=None,
            self_loop=True,
            atom_featurizer_tf=False,
            bond_featurizer_tf=False,
        )

        graph_list_bare, graph_list = [], []
        graph_list_spin, graph_list_charge = [], []
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)
            graph_spin_only = grapher_spin.build_graph(mol)
            graph_charge_only = grapher_charge.build_graph(mol)
            graph_bare = grapher_bare.build_graph(mol)

            graph, names_full = grapher.featurize(graph, mol, ret_feat_names=True)
            graph_bare, names = grapher_bare.featurize(
                graph_bare, mol, ret_feat_names=True
            )
            graph_spin_only, names_spin = grapher_spin.featurize(
                graph_spin_only, mol, ret_feat_names=True
            )
            graph_charge_only, names_charge = grapher_charge.featurize(
                graph_charge_only, mol, ret_feat_names=True
            )

            graph_list.append(graph)
            graph_list_bare.append(graph_bare)
            graph_list_spin.append(graph_spin_only)
            graph_list_charge.append(graph_charge_only)

        for ind in range(len(graph_list)):
            assert (
                graph_list[ind].ndata["feat"]["global"].shape[1]
                == graph_list_bare[ind].ndata["feat"]["global"].shape[1] + 6
            )
            assert (
                graph_list_spin[ind].ndata["feat"]["global"].shape[1]
                == graph_list_bare[ind].ndata["feat"]["global"].shape[1] + 3
            )
            assert (
                graph_list_charge[ind].ndata["feat"]["global"].shape[1]
                == graph_list_bare[ind].ndata["feat"]["global"].shape[1] + 3
            )


# test = TestGrapher()
# test.test_node_sizes()
