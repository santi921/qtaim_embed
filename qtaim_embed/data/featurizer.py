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
        self._feature_size = None
        self._feature_name = None

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

    def __init__(
        self, dtype="float32", selected_keys=[], allowed_ring_size=[3, 4, 5, 6, 7]
    ):
        super(BaseFeaturizer, self).__init__()
        self._feature_size = None
        self._feature_name = None
        self.selected_keys = selected_keys
        self.dtype = dtype
        self.allowed_ring_size = allowed_ring_size

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

        allowed_ring_size = [3, 4, 5, 6, 7]

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
            if self.length_featurizer:
                ft += [0.0 for _ in range(len(self.length_featurizer.feature_name))]
            feats = [ft]

        else:
            features_flatten = []
            for i in bond_list:
                feats_flatten_temp = []
                for key in self.selected_keys:
                    if key != "bond_length":
                        feats_flatten_temp.append(features[i][key])
                features_flatten.append(feats_flatten_temp)

            feats = []

            cycles = find_rings(num_atoms, bond_list, allowed_ring_size, edges=True)
            no_metal_binary = [1 for i in range(num_bonds)]
            ring_dict = ring_features_for_bonds_full(
                bond_list, no_metal_binary, cycles, allowed_ring_size
            )
            ring_dict_keys = list(ring_dict.keys())

            for ind, bond in enumerate(bond_list):
                ft = []

                if tuple(bond) in ring_dict_keys:
                    ft.append(ring_dict[tuple(bond)][0])  # metal
                    ft.append(ring_dict[tuple(bond)][1])  #
                    ft += ring_dict[tuple(bond)][2]  # one hot ring
                else:
                    ft += [0, 0]
                    ft += [0 for i in range(len(allowed_ring_size))]

                # check that features_flatten isn't empty lists
                if features_flatten[ind] != []:
                    ft += features_flatten[ind]

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

                # ft += features[bond[0]] # check that index is correct
                feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["metal bond"] + ["ring inclusion"] + ["ring size"] * 5 + self.selected_keys
        )

        return {"feat": feats}, self._feature_name


class AtomFeaturizerGraphGeneral(BaseFeaturizer):

    """
    Featurize atoms in a molecule.

    Mimimum set of info without hybridization info.
    """

    def __init__(self, element_set, selected_keys=[], dtype="float32"):
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self._feature_size = None
        self._feature_name = None
        self.selected_keys = selected_keys
        self.element_set = element_set

    def __call__(self, mol, **kwargs):
        """
        Args:
            mol: molecular wraper object w/electronic info

            Also `extra_feats_info` should be provided as `kwargs` as additional info.

        Returns:
            Dictionary of atom features
        """

        allowed_ring_size = [3, 4, 5, 6, 7]

        features = mol.atom_features
        features_flatten, feats, bond_list = [], [], []
        num_atoms = len(mol.coords)
        species_sites = mol.species
        bond_list_tuple = list(mol.bonds.keys())
        # print("atom feats,", features)
        # print(features)
        for i in range(num_atoms):
            feats_flatten_temp = []
            for key in self.selected_keys:
                # print(key, features[key])
                feats_flatten_temp.append(features[i][key])
            features_flatten.append(feats_flatten_temp)

        atom_num = len(species_sites)
        [bond_list.append(list(bond)) for bond in bond_list_tuple]
        cycles = find_rings(atom_num, bond_list, edges=False)
        ring_info = ring_features_from_atom_full(num_atoms, cycles, allowed_ring_size)

        for atom_ind in range(num_atoms):
            ft = []
            atom_element = species_sites[atom_ind]
            h_count, degree = h_count_and_degree(atom_ind, bond_list, species_sites)
            ring_inclusion, ring_size_list = ring_info[atom_ind]
            ft.append(degree)
            ft.append(ring_inclusion)
            ft.append(h_count)

            ft += features_flatten[atom_ind]
            ft += one_hot_encoding((atom_element), list(self.element_set))
            ft += ring_size_list
            feats.append(ft)

        feats = torch.tensor(feats, dtype=getattr(torch, self.dtype))
        self._feature_size = feats.shape[1]
        self._feature_name = (
            ["total degree", "is in ring", "total H"]
            + self.selected_keys
            + ["chemical symbol"] * len(list(self.element_set))
            + ["ring size"] * len(allowed_ring_size)
        )

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
        allowed_charges=None,
        dtype="float32",
    ):
        super(BaseFeaturizer, self).__init__()
        if dtype not in ["float32", "float64"]:
            raise ValueError(
                "`dtype` should be `float32` or `float64`, but got `{}`.".format(dtype)
            )
        self.dtype = dtype
        self.allowed_charges = allowed_charges

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

        if self.allowed_charges is not None:
            try:
                feats_info = kwargs["extra_feats_info"]
            except KeyError as e:
                raise KeyError(
                    "{} `extra_feats_info` needed for {}.".format(
                        e, self.__class__.__name__
                    )
                )

            if self.allowed_charges is not None:
                g += one_hot_encoding(feats_info["charge"], self.allowed_charges)

        feats = torch.tensor([g], dtype=getattr(torch, self.dtype))

        self._feature_size = feats.shape[1]
        self._feature_name = ["num atoms", "num bonds", "molecule weight"]
        if self.allowed_charges is not None:
            self._feature_name += ["charge one hot"] * len(self.allowed_charges)

        return {"feat": feats}, self._feature_name
