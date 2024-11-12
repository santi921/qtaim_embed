"""
Featurize a molecule heterograph of atom, bond, and global nodes with RDkit.
"""

import torch
import dgl
import os
import numpy as np
from rdkit.Chem.rdchem import GetPeriodicTable
from qtaim_embed.utils.descriptors import (
    one_hot_encoding,
    h_count_and_degree,
    ring_features_from_atom_full,
    ring_features_for_bonds_full,
    find_rings,
)

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class BaseFeaturizer:
    def __init__(self, dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self._feature_size = 0
        self._feature_name = []

    @property
    def feature_size(self):
        """
        Returns:
            an int of the feature size.
        """
        return self._feature_size

    @property
    def feature_name(self):
        """
        Returns:
            a list of the names of each feature. Should be of the same length as
            `feature_size`.
        """

        return self._feature_name

    def __call__(self, mol, **kwargs):
        """
        Returns:
            A dictionary of the features.
        """
        raise NotImplementedError


class BondAsNodeGraphFeaturizerGeneral(BaseFeaturizer):
    """BaseFeaturizer
    Featurize all bonds in a molecule.

    The bond indices will be preserved, i.e. feature i corresponds to atom i.
    The number of features will be equal to the number of bonds in the molecule,
    so this is suitable for the case where we represent bond as graph nodes.

    See Also:
        BondAsEdgeBidirectedFeaturizer
    """

    def __init__(self, dtype="float32", selected_keys=[], allowed_ring_size=[]):
        super(BaseFeaturizer, self).__init__()
        self._feature_size = 0
        self._feature_name = []
        self.selected_keys = selected_keys
        self.dtype = dtype
        self.allowed_ring_size = allowed_ring_size
        if allowed_ring_size == []:
            print(
                "NOTE: No ring size if no ring features are enabled, metal/nonmetal bonds are also off"
            )
        print("selected bond keys", selected_keys)

    def __call__(self, mol, **kwargs):
        """
        Parameters
        ----------
        mol : molwrapper object with xyz positions(coord) + electronic information

        Returns
        -------
            Dictionary for bond features
        """

        feats, no_metal_binary = [], []
        num_atoms = 0

        bond_list = list(mol.bonds)
        num_bonds = len(bond_list)
        num_atoms = int(mol.num_atoms)
        features = mol.bond_features
        xyz_coordinates = mol.coords

        # count number of keys in features
        num_feats = len(self.selected_keys)
        num_feats += 7

        if num_bonds == 0:
            ft = [0.0 for _ in range(num_feats)]
            feats = [ft]

        else:
            feats = []
            if self.allowed_ring_size != []:
                cycles = find_rings(
                    num_atoms, bond_list, self.allowed_ring_size, edges=True
                )
                no_metal_binary = [1 for i in range(num_bonds)]
                ring_dict = ring_features_for_bonds_full(
                    bond_list, no_metal_binary, cycles, self.allowed_ring_size
                )
                ring_dict_keys = list(ring_dict.keys())

            for ind, bond in enumerate(bond_list):
                ft = []
                if self.allowed_ring_size != []:
                    if tuple(bond) in ring_dict_keys:
                        ft.append(ring_dict[tuple(bond)][0])  # metal
                        ft.append(ring_dict[tuple(bond)][1])
                        ft += ring_dict[tuple(bond)][2]  # one hot ring
                    else:
                        ft += [0, 0]
                        ft += [0 for i in range(len(self.allowed_ring_size))]

                # check that features_flatten isn't empty lists

                if "bond_length" in self.selected_keys:
                    bond_len = np.sqrt(
                        np.sum(
                            np.square(
                                np.array(xyz_coordinates[bond[0]])
                                - np.array(xyz_coordinates[bond[1]])
                            )
                        )
                    )
                    ft.append(bond_len)

                if self.selected_keys != None:
                    for key in self.selected_keys:
                        if key != "bond_length":
                            ft.append(features[bond][key])

                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))

        if self.allowed_ring_size != []:
            self._feature_name = ["metal bond"]
            self._feature_name += ["ring inclusion"] + [
                "ring size_{}".format(i) for i in self.allowed_ring_size
            ]

        if "bond_length" in self.selected_keys:
            self._feature_name += ["bond_length"]

        if self.selected_keys != []:
            for key in self.selected_keys:
                if key != "bond_length":
                    self._feature_name.append(key)

            # self._feature_name += self.selected_keys
        self._feature_size = len(self._feature_name)
        return {"feat": feats}, self._feature_name


class AtomFeaturizerGraphGeneral(BaseFeaturizer):

    """
    Featurize atoms in a molecule.

    Mimimum set of info without hybridization info.
    """

    def __init__(
        self,
        element_set,
        selected_keys=None,
        allowed_ring_size=[],
        dtype="float32",
    ):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        print("element set in featurizer", element_set)
        print("selected atomic keys", selected_keys)

        self.dtype = dtype
        self._feature_size = 0
        self._feature_name = []
        self.selected_keys = selected_keys
        self.allowed_ring_size = allowed_ring_size
        self.element_set = element_set

    def __call__(self, mol, **kwargs):
        """
        Args:
            mol: molecular wraper object w/electronic info

            Also `extra_feats_info` should be provided as `kwargs` as additional info.

        Returns:
            Dictionary of atom features
        """

        features = mol.atom_features
        feats, bond_list = [], []
        num_atoms = len(mol.coords)
        species_sites = mol.species
        bond_list_tuple = list(mol.bonds.keys())
        atom_num = len(species_sites)
        [bond_list.append(list(bond)) for bond in bond_list_tuple]

        if self.allowed_ring_size != []:
            cycles = find_rings(atom_num, bond_list, edges=False)
            ring_info = ring_features_from_atom_full(
                num_atoms, cycles, self.allowed_ring_size
            )
        # print("features", features)
        for atom_ind in range(num_atoms):
            ft = []
            atom_element = species_sites[atom_ind]
            h_count, degree = h_count_and_degree(atom_ind, bond_list, species_sites)

            ft.append(degree)
            ft.append(h_count)

            if self.allowed_ring_size != []:
                ring_inclusion, ring_size_list = ring_info[atom_ind]
                ft.append(ring_inclusion)
                ft += ring_size_list

            ft += one_hot_encoding((atom_element), list(self.element_set))
            if self.selected_keys != None:
                for key in self.selected_keys:
                    ft.append(features[atom_ind][key])
            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        # self._feature_size = feats.shape[1]
        # print("feature size", self._feature_size)
        if self.allowed_ring_size != []:
            self._feature_name = (
                ["total_degree", "total_H", "is_in_ring"]
                + ["ring_size_{}".format(i) for i in self.allowed_ring_size]
                + ["chemical_symbol_{}".format(i) for i in list(self.element_set)]
            )
        else:
            self._feature_name = ["total_degree", "total_H"] + [
                "chemical_symbol_{}".format(i) for i in list(self.element_set)
            ]

        if self.selected_keys != None:
            self._feature_name += self.selected_keys
        self._feature_size = len(self._feature_name)
        return {"feat": feats}, self._feature_name


class GlobalFeaturizerGraph(BaseFeaturizer):
    """
    Featurize the global state of a molecules using number of atoms, number of bonds,
    molecular weight, and optionally charge and solvent environment.


    Args:
        allowed_charges (list, optional): charges allowed the the molecules to take.
        solvent_environment (list, optional): solvent environment in which the
        calculations for the molecule take place
    """

    def __init__(
        self,
        allowed_charges=[],
        allowed_spins=[],
        selected_keys=[],
        dtype="float32",
    ):
        super(BaseFeaturizer, self).__init__()
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self.allowed_charges = allowed_charges
        self.selected_keys = selected_keys
        self.allowed_spins = allowed_spins
        self._feature_size = 0
        self._feature_name = []
        print("selected global keys", selected_keys)

    def __call__(self, mol, **kwargs):
        """
        mol can either be an molwrapper object
        """
        pt = GetPeriodicTable()
        num_atoms, mw = 0, 0
        atom_types = list(mol.composition_dict.keys())
        for atom in atom_types:
            num_atom_type = int(mol.composition_dict[atom])
            num_atoms += num_atom_type
            mw += num_atom_type * pt.GetAtomicWeight(atom)

        g = [
            num_atoms,
            len(mol.bonds),
            mw,
        ]

        if self.allowed_charges is not None and self.allowed_charges != []:
            if self.allowed_charges is not None and self.allowed_charges != []:
                g += one_hot_encoding(
                    mol.global_features["charge"], self.allowed_charges
                )

        if self.allowed_spins is not None and self.allowed_spins != []:
            if self.allowed_spins is not None and self.allowed_spins != []:
                g += one_hot_encoding(mol.global_features["spin"], self.allowed_spins)

        self._feature_name = ["num atoms", "num bonds", "molecule weight"]
        if self.allowed_charges is not None:
            self._feature_name += ["charge one hot"] * len(self.allowed_charges)
        if self.allowed_spins is not None:
            self._feature_name += ["spin one hot"] * len(self.allowed_spins)

        # add extra features
        if self.selected_keys != []:
            for key in self.selected_keys:
                if key != "spin" and key != "charge":
                    self._feature_name.append(key)
                    g += [mol.global_features[key]]
        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))
        self._feature_size = len(self._feature_name)

        return {"feat": feats}, self._feature_name
