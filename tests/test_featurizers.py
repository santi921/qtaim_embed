import pandas as pd
import pytest
import torch

from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.molwrapper import mol_wrappers_from_df
from qtaim_embed.utils.tests import get_data, get_data_spin_charge


class TestGrapher:
    df_test = get_data()
    df_test_spin_charge = get_data_spin_charge()

    def test_node_sizes(self):  # check
        atom_keys = [
            "extra_feat_atom_esp_total",
        ]
        bond_keys = [
            "extra_feat_bond_esp_total",
        ]
        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
            shape_atom_feats = graph["atom"].feat.shape
            num_atoms = graph["atom"].num_nodes
            assert shape_atom_feats[0] == num_atoms
            shape_bond_feats = graph["bond"].feat.shape
            num_bonds = graph["bond"].num_nodes
            assert shape_bond_feats[0] == num_bonds
            shape_global_feats = graph["global"].feat.shape
            assert shape_global_feats[0] == 1

    def test_atom_featurizer(self):
        atom_keys = [
            "extra_feat_atom_esp_total",
            "extra_feat_atom_esp_nuc",
        ]
        col_test = pd.DataFrame(self.df_test["extra_feat_atom_esp_total"])

        test_vals = [col_test.iloc[ind, 0][0] for ind in range(len(col_test))]
        # print(test_vals)
        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
            assert graph_list[ind]["atom"].feat[0, -2] == test_vals[ind]
            assert (
                graph_list[ind]["atom"].feat.shape[1]
                == graph_list_bare[ind]["atom"].feat.shape[1] + 8
            )

    def test_bond_featurizer(self):  # check!
        # test qtaim AND bond length as features!!!

        bond_keys = [
            "extra_feat_bond_esp_total",
            "extra_feat_bond_esp_nuc",
            "bond_length",
            "boo_2",
        ]

        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
            atom_keys=[],
            bond_keys=bond_keys,
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
            assert graph_list[ind]["bond"].feat[3, -2] == list_test[ind]

            assert (
                graph_list[ind]["bond"].feat.shape[1]
                == graph_list_bare[ind]["bond"].feat.shape[1] + 16
            )

    def test_global_featurizers(self):
        global_keys = ["ids"]

        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
                graph_list[ind]["global"].feat.shape[1]
                == graph_list_bare[ind]["global"].feat.shape[1] + 1
            )
            assert int(graph_list[ind]["global"].feat[0][-1]) == id_list[ind]

    def test_spin_charge_encoding(self):
        global_keys = ["spin", "charge"]

        mol_wrappers, element_set = mol_wrappers_from_df(
            self.df_test_spin_charge,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
                graph_list[ind]["global"].feat.shape[1]
                == graph_list_bare[ind]["global"].feat.shape[1] + 6
            )
            assert (
                graph_list_spin[ind]["global"].feat.shape[1]
                == graph_list_bare[ind]["global"].feat.shape[1] + 3
            )
            assert (
                graph_list_charge[ind]["global"].feat.shape[1]
                == graph_list_bare[ind]["global"].feat.shape[1] + 3
            )


import torch
from qtaim_embed.utils.descriptors import sinusoidal_bessel_rbf, gaussian_rbf


class TestSinusoidalBesselRBF:
    def test_output_shape(self):
        d = torch.tensor([1.0, 2.0, 3.0])
        out = sinusoidal_bessel_rbf(d, n_basis=20, cutoff=5.0)
        assert out.shape == (3, 20)

    def test_zero_at_cutoff(self):
        d = torch.tensor([5.0])
        out = sinusoidal_bessel_rbf(d, n_basis=20, cutoff=5.0)
        assert torch.allclose(out, torch.zeros(1, 20), atol=1e-5)

    def test_zero_distance_no_nan(self):
        d = torch.tensor([0.0])
        out = sinusoidal_bessel_rbf(d, n_basis=20, cutoff=5.0)
        assert not torch.any(torch.isnan(out))

    def test_beyond_cutoff_is_zero(self):
        d = torch.tensor([6.0, 10.0])
        out = sinusoidal_bessel_rbf(d, n_basis=20, cutoff=5.0)
        assert torch.allclose(out, torch.zeros(2, 20), atol=1e-6)

    def test_different_cutoffs(self):
        d = torch.tensor([2.0])
        out_5 = sinusoidal_bessel_rbf(d, n_basis=20, cutoff=5.0)
        out_10 = sinusoidal_bessel_rbf(d, n_basis=20, cutoff=10.0)
        assert not torch.allclose(out_5, out_10)

    def test_nonzero_in_range(self):
        d = torch.tensor([1.5])
        out = sinusoidal_bessel_rbf(d, n_basis=20, cutoff=5.0)
        assert out.abs().sum() > 0


class TestGaussianRBF:
    def test_output_shape(self):
        d = torch.tensor([1.0, 2.0])
        out = gaussian_rbf(d, n_basis=50, cutoff=5.0)
        assert out.shape == (2, 50)

    def test_peak_at_center(self):
        n_basis = 10
        cutoff = 5.0
        centers = torch.linspace(0, cutoff, n_basis)
        for i, c in enumerate(centers):
            out = gaussian_rbf(c.unsqueeze(0), n_basis, cutoff)
            assert out[0, i] == out[0].max()

    def test_bounded_output(self):
        d = torch.linspace(0, 10, 100)
        out = gaussian_rbf(d, n_basis=50, cutoff=5.0)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_nonzero_in_range(self):
        d = torch.tensor([2.5])
        out = gaussian_rbf(d, n_basis=20, cutoff=5.0)
        assert out.abs().sum() > 0


class TestRBFFeaturizerIntegration:
    df_test = get_data()

    def test_rbf_bessel_featurizer(self):
        atom_keys = ["extra_feat_atom_esp_total"]
        bond_keys = ["rbf_bessel_20"]
        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
            self_loop=True,
            rbf_cutoff=5.0,
        )
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            bond_names = names["bond"]
            # Should have ring features (7) + 20 RBF features
            rbf_names = [n for n in bond_names if "rbf_bessel_20" in n]
            assert len(rbf_names) == 20
            assert "rbf_bessel_20_0" in bond_names
            assert "rbf_bessel_20_19" in bond_names

    def test_rbf_gaussian_featurizer(self):
        atom_keys = ["extra_feat_atom_esp_total"]
        bond_keys = ["rbf_gaussian_10"]
        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
            self_loop=True,
            rbf_cutoff=5.0,
        )
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            bond_names = names["bond"]
            rbf_names = [n for n in bond_names if "rbf_gaussian_10" in n]
            assert len(rbf_names) == 10

    def test_rbf_plus_boo_featurizer(self):
        atom_keys = ["extra_feat_atom_esp_total"]
        bond_keys = ["rbf_bessel_10", "boo_2"]
        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
            self_loop=True,
            rbf_cutoff=5.0,
        )
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            bond_names = names["bond"]
            rbf_names = [n for n in bond_names if "rbf_bessel_10" in n]
            boo_names = [n for n in bond_names if "boo_2" in n]
            assert len(rbf_names) == 10
            assert len(boo_names) == 9  # (2+1)^2 = 9

    def test_no_rbf_backward_compatible(self):
        atom_keys = ["extra_feat_atom_esp_total"]
        bond_keys = ["extra_feat_bond_esp_total", "bond_length"]
        mol_wrappers, element_set = mol_wrappers_from_df(
            df=self.df_test,
            bond_key="bonds",
            map_key="extra_feat_bond_indices_qtaim",
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
            self_loop=True,
        )
        for mol in mol_wrappers:
            graph = grapher.build_graph(mol)
            graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
            bond_names = names["bond"]
            # Should have ring features (7) + bond_length (1) + esp_total (1) = 9
            assert "bond_length" in bond_names
            assert "extra_feat_bond_esp_total" in bond_names
            # No RBF names
            rbf_names = [n for n in bond_names if "rbf_" in n]
            assert len(rbf_names) == 0

    def test_zero_bond_molecule_with_rbf(self):
        """Zero-bond fallback path produces correct feature width when RBF is enabled."""
        import numpy as np
        from qtaim_embed.data.featurizer import BondAsNodeGraphFeaturizerGeneral

        n_basis = 16
        bond_keys = ["rbf_bessel_16"]

        # Minimal mock that satisfies the featurizer interface
        class MockMol:
            bonds = {}
            bond_features = {}
            coords = np.zeros((1, 3))
            num_atoms = 1

        featurizer = BondAsNodeGraphFeaturizerGeneral(
            selected_keys=bond_keys,
            allowed_ring_size=[3, 4, 5, 6, 7],
            rbf_cutoff=5.0,
        )

        feats, names = featurizer(MockMol())
        bond_feat_tensor = feats["feat"]

        # num_feats = len(selected_keys)=1 + 7 (ring/metal) + (n_basis-1)=15 = 23
        expected_width = 1 + 7 + (n_basis - 1)
        assert bond_feat_tensor.shape == (1, expected_width), (
            f"Expected shape (1, {expected_width}), got {bond_feat_tensor.shape}"
        )
        assert not torch.any(torch.isnan(bond_feat_tensor))


class TestZeroBondFallbackWidth:
    """Zero-bond molecules must get the same bond-feat width as bonded ones.

    The fallback row width is computed independently of the per-bond path;
    a mismatch (e.g. the old hardcoded ring-block size) poisons a dataset
    with rows the scaler cannot broadcast against.
    """

    def _featurize(
        self, bonds: list[tuple[int, int]]
    ) -> tuple[torch.Tensor, list[str]]:
        from qtaim_embed.core.molwrapper import (
            create_wrapper_mol_from_atoms_and_bonds,
        )
        from qtaim_embed.data.featurizer import BondAsNodeGraphFeaturizerGeneral

        mol = create_wrapper_mol_from_atoms_and_bonds(
            species=["O", "H"],
            coords=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            bonds=bonds,
            bond_features={b: {"rho": 0.3, "odd_boo_name": 1.5} for b in bonds},
        )
        # all five width terms: ring block, bond_length, boo, rbf, scalars
        # (one scalar deliberately contains "boo_" without the prefix)
        featurizer = BondAsNodeGraphFeaturizerGeneral(
            selected_keys=["rbf_gaussian_50", "boo_1", "bond_length", "rho", "odd_boo_name"],
            allowed_ring_size=[3, 4, 5, 6, 7, 8],
        )
        feat_dict, names = featurizer(mol)
        return feat_dict["feat"], names

    def test_zero_bond_width_matches_bonded(self):
        feats_bonded, names = self._featurize([(0, 1)])
        feats_empty, names_empty = self._featurize([])
        assert feats_bonded.shape[1] == len(names)
        assert feats_empty.shape[1] == feats_bonded.shape[1]
        assert names_empty == names

    def test_zero_bond_fallback_boo_l0_matches_real_bonds(self):
        """boo_1_0 (|Y00|) is identically 1.0 for every real bond, so the
        fallback row must carry 1.0 there too - a 0 in a column the scaler
        sees as constant scaled to (0-1)/eps = -1e6 and poisoned predictions
        for any batch containing a zero-bond molecule."""
        feats_bonded, names = self._featurize([(0, 1)])
        feats_empty, _ = self._featurize([])
        col = names.index("boo_1_0")
        assert feats_bonded[0, col] == 1.0
        assert feats_empty[0, col] == 1.0

    def test_substring_keys_stay_scalar_and_names_do_not_accumulate(self):
        feats, names = self._featurize([(0, 1)])
        assert names[-2:] == ["rho", "odd_boo_name"]
        assert feats[0, names.index("bond_length")] == pytest.approx(5.0)
        assert feats[0, names.index("odd_boo_name")] == pytest.approx(1.5)
        from qtaim_embed.data.featurizer import BondAsNodeGraphFeaturizerGeneral

        f = BondAsNodeGraphFeaturizerGeneral(selected_keys=None, allowed_ring_size=[])
        assert f.selected_keys == []

    def test_duplicate_expanding_keys_raise(self):
        from qtaim_embed.core.molwrapper import create_wrapper_mol_from_atoms_and_bonds
        from qtaim_embed.data.featurizer import BondAsNodeGraphFeaturizerGeneral

        mol = create_wrapper_mol_from_atoms_and_bonds(
            species=["O", "H"], coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            bonds=[(0, 1)], bond_features={(0, 1): {}},
        )
        for keys in (["boo_1", "boo_2"], ["rbf_bessel_10", "rbf_gaussian_10"], ["rbf_cutoff"]):
            with pytest.raises(ValueError):
                BondAsNodeGraphFeaturizerGeneral(selected_keys=keys, allowed_ring_size=[])(mol)
