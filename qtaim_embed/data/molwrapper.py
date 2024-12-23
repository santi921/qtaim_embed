from tqdm import tqdm
from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.descriptors import (
    get_atom_feats,
    get_bond_features,
    get_global_features,
    elements_from_pmg,
)


def mol_wrappers_from_df(
    df,
    bond_key=None,
    map_key=None,
    atom_keys=[],
    bond_keys=[],
    global_keys=[],
    filter_self_bonds=True,
):
    """
    Creates a list of MoleculeWrapper objects from a dataframe
    Takes:
        df: dataframe with the following columns:
            - molecule_graph: dgl graph
            - molecule: pymatgen molecule
            - ids: molecule id
            - names: molecule name
            - bonds: list of bonds
        bond_key: bond key to be used as features
        map_key: key to map bond features to bond keys
        atom_keys: list of atom keys to be used as features
        bond_keys: list of bond keys to be used as features
        filter_self_bonds: whether to filter self bonds
    Returns:
        mol_wrappers: list of MoleculeWrapper objects
    """

    element_set = set()
    mol_wrappers = []
    print("... > creating MoleculeWrapper objects")
    bond_feats_error_count = 0
    atom_feats_error_count = 0
    #print("... > bond_key: ", bond_key)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        charge = 0
        free_energy = 0
        bonds = row[bond_key]
        # bonds = row.bonds
        if "names" not in row.index:
            id_combined = str(row.ids) + "_" + str(row.ids)
        else:
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
                map_key=map_key,
                bond_key=bond_key,
                keys=bond_keys,
            )

        mol_graph = row.molecule_graph
        # print(mol_graph)
        pmg_mol = row.molecule
        elements = elements_from_pmg(pmg_mol)
        element_set.update(elements)


        if len(row[bond_key]) == 1:
            bonds = row[bond_key][0]
        else:
            bonds = row[bond_key]

        if filter_self_bonds:
            bonds = {tuple(sorted(b)): None for b in bonds if b[0] != b[1]}
            bond_feats = {k: v for k, v in bond_feats.items() if k[0] != k[1]}

        if bond_feats != -1 and atom_feats != -1:
            mol_wrapper = MoleculeWrapper(
                mol_graph,
                functional_group=None,
                free_energy=None,
                id=id_combined,
                bonds=bonds,
                non_metal_bonds=bonds,  # TODO: fix this
                atom_features=atom_feats,
                bond_features=bond_feats,
                global_features=global_features,
                original_atom_ind=None,
                original_bond_mapping=None,
            )
            mol_wrappers.append(mol_wrapper)
        else:
            if bond_feats == -1:
                bond_feats_error_count += 1
            if atom_feats == -1:
                atom_feats_error_count += 1
                
    print("... > bond_feats_error_count: ", bond_feats_error_count)
    print("... > atom_feats_error_count: ", atom_feats_error_count)
    # sort element set
    element_set = set(sorted(list(element_set)))
    return mol_wrappers, element_set
