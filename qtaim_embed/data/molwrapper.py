from tqdm import tqdm
from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.descriptors import (
    get_atom_feats,
    get_bond_features,
    get_global_features,
    elements_from_pmg,
)


def mol_wrappers_from_df(df, atom_keys=[], bond_keys=[], global_keys=[]):
    """
    Creates a list of MoleculeWrapper objects from a dataframe
    Takes:
        df: dataframe with the following columns:
            - molecule_graph: dgl graph
            - molecule: pymatgen molecule
            - ids: molecule id
            - names: molecule name
            - bonds: list of bonds
        atom_keys: list of atom keys to be used as features
        bond_keys: list of bond keys to be used as features
    Returns:
        mol_wrappers: list of MoleculeWrapper objects
    """

    element_set = set()
    mol_wrappers = []
    print("... > creating MoleculeWrapper objects")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        charge = 0
        free_energy = 0
        bonds = row.bonds

        id_combined = str(row.ids) + "_" + row.names
        bonds = {tuple(sorted(b)): None for b in bonds}

        atom_feats, bond_feats, global_features = {}, {}, {}

        if global_keys != []:
            global_features = get_global_features(row, global_keys)
        if atom_keys != []:
            atom_feats = get_atom_feats(row, atom_keys)
        if bond_keys != []:
            bond_feats = get_bond_features(
                row,
                "extra_feat_bond_indices_qtaim",
                keys=bond_keys,
            )

        mol_graph = row.molecule_graph
        pmg_mol = row.molecule
        elements = elements_from_pmg(pmg_mol)
        element_set.update(elements)

        # filter for bond_feats = -1
        if bond_feats != -1 and atom_feats != 1:
            mol_wrapper = MoleculeWrapper(
                mol_graph,
                functional_group=None,
                free_energy=None,
                id=id_combined,
                non_metal_bonds=None,
                atom_features=atom_feats,
                bond_features=bond_feats,
                global_features=global_features,
                original_atom_ind=None,
                original_bond_mapping=None,
            )
            mol_wrappers.append(mol_wrapper)

    return mol_wrappers, element_set
