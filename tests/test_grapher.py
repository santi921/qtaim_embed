import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.molwrapper import mol_wrappers_from_df
from qtaim_embed.data.lmdb import serialize_graph, load_graph_from_serialized
from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative
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
        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test, bond_key="bonds", map_key="extra_feat_bond_indices_qtaim"
        )

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
            num_atoms = graph_list[ind]["atom"].num_nodes
            assert num_atoms == num_atoms_mol_wrapper
            num_bonds_mol_wrapper = list_bond_num[ind]
            num_bonds = graph_list[ind]["bond"].num_nodes
            assert num_bonds_mol_wrapper == num_bonds

    def test_pos_z(self):
        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
            atom_keys=["extra_feat_atom_esp_total"],
            bond_keys=["extra_feat_bond_esp_total"],
        )
        grapher = get_grapher(
            element_set,
            atom_keys=["extra_feat_atom_esp_total"],
            bond_keys=["extra_feat_bond_esp_total"],
            global_keys=[],
            allowed_ring_size=[3, 4, 5, 6, 7],
            allowed_charges=None,
            self_loop=True,
        )

        graphs = []
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)
            pos = graph["atom"].pos
            z = graph["atom"].z
            assert pos.shape == (mol.num_atoms, 3)
            assert pos.dtype == torch.float32
            assert np.allclose(pos.numpy(), mol.coords, atol=1e-6)
            assert z.shape == (mol.num_atoms,)
            assert z.dtype == torch.long
            assert z.tolist() == list(mol.pymatgen_mol.atomic_numbers)
            graphs.append(grapher.featurize(graph, mol))

        # serialization roundtrip preserves pos/z
        loaded = load_graph_from_serialized(serialize_graph(graphs[0]))
        assert torch.equal(loaded["atom"].pos, graphs[0]["atom"].pos)
        assert torch.equal(loaded["atom"].z, graphs[0]["atom"].z)
        assert loaded.mol_name == graphs[0].mol_name

        # batching concatenates pos/z along the atom dimension
        batched = Batch.from_data_list(graphs[:2])
        n_total = graphs[0]["atom"].num_nodes + graphs[1]["atom"].num_nodes
        assert batched["atom"].pos.shape == (n_total, 3)
        assert batched["atom"].z.shape == (n_total,)

        # feature scaler must leave pos/z untouched
        pos_before = graphs[0]["atom"].pos.clone()
        z_before = graphs[0]["atom"].z.clone()
        scaler = HeteroGraphStandardScalerIterative(
            features_tf=True, mean={}, std={}
        )
        scaler.update(graphs)
        scaler.finalize()
        scaled = scaler(graphs)
        assert torch.equal(scaled[0]["atom"].pos, pos_before)
        assert torch.equal(scaled[0]["atom"].z, z_before)


# tester = TestGrapher()
# tester.test_graph_nodes()
