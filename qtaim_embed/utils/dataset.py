import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDetermineBonds

from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.analysis.graphs import MoleculeGraph

def gather_atom_level_stats(dataset_dev):
    atoms_in = [
        i.split("_")[-1]
        for i in dataset_dev.exclude_names["atom"]
        if "chemical_symbol_" in i
    ]
    atom_feats_qtaim = [
        i for i in dataset_dev.include_names["atom"] if "extra_feat_atom_" in i
    ]
    feat_dict = {}
    feat_dict_summary = {}
    feat_dict_complete = {}

    for probe_atom_type in atoms_in:
        print("capturing stats for atom type: ", probe_atom_type)
        for probe_descriptor in atom_feats_qtaim:
            # probe_atom_type = "C"
            # probe_descriptor = "extra_feat_atom_Lagrangian_K"

            for graph in dataset_dev.graphs:
                probe_ind = dataset_dev.exclude_names["atom"].index(
                    "chemical_symbol_" + probe_atom_type
                )
                probe_col = graph.ndata["feat"]["atom"][:, probe_ind]

                probe_desc_ind = dataset_dev.include_names["atom"].index(
                    probe_descriptor
                )
                atom_type_positive_ind = np.where(probe_col == 1)[0]
                feat_at_atom = graph.ndata["labels"]["atom"][
                    atom_type_positive_ind, probe_desc_ind
                ]

                if probe_atom_type not in feat_dict.keys():
                    feat_dict[probe_atom_type] = {}
                if probe_descriptor not in feat_dict[probe_atom_type].keys():
                    feat_dict[probe_atom_type][probe_descriptor] = []
                feat_dict[probe_atom_type][probe_descriptor].extend(feat_at_atom)

            feat_dict[probe_atom_type][probe_descriptor] = np.array(
                feat_dict[probe_atom_type][probe_descriptor]
            )

    for probe_descriptor in atom_feats_qtaim:
        for probe_atom_type in atoms_in:
            if probe_descriptor not in feat_dict_complete.keys():
                feat_dict_complete[probe_descriptor] = []
            feat_dict_complete[probe_descriptor].extend(
                feat_dict[probe_atom_type][probe_descriptor]
            )

    for k, v in feat_dict.items():
        if k not in feat_dict_summary.keys():
            feat_dict_summary[k] = {}
        for sub_k, sub_v in v.items():
            dict_summary_stats = {
                "mean": np.mean(sub_v),
                "std": np.std(sub_v),
                "min": np.min(sub_v),
                "max": np.max(sub_v),
                "mode": stats.mode(sub_v)[0],
            }
            feat_dict_summary[k][sub_k] = dict_summary_stats

    return feat_dict, feat_dict_complete, feat_dict_summary


def gather_bond_level_stats(dataset_dev):
    bond_feats_qtaim = [
        i for i in dataset_dev.include_names["bond"] if "extra_feat_bond_" in i
    ]
    feat_dict_summary = {}
    feat_dict_complete = {}

    for probe_descriptor in bond_feats_qtaim:
        for graph in dataset_dev.graphs:
            probe_desc_ind = dataset_dev.include_names["bond"].index(probe_descriptor)
            feat_at_atom = graph.ndata["labels"]["bond"][:, probe_desc_ind]

            if probe_descriptor not in feat_dict_complete.keys():
                feat_dict_complete[probe_descriptor] = []
            feat_dict_complete[probe_descriptor].extend(feat_at_atom)

        feat_dict_complete[probe_descriptor] = np.array(
            feat_dict_complete[probe_descriptor]
        )

    for k, v in feat_dict_complete.items():
        if k not in feat_dict_summary.keys():
            feat_dict_summary[k] = {}
        dict_summary_stats = {
            "mean": np.mean(v),
            "std": np.std(v),
            "min": np.min(v),
            "max": np.max(v),
            "mode": stats.mode(v)[0],
        }
        feat_dict_summary[k] = dict_summary_stats
    return feat_dict_complete, feat_dict_summary


def print_summary_complete(feat_complete):
    """
    Helper function to print summary of complete features
        feat_complete - dict of lists
    """
    for k, v in feat_complete.items():
        try:
            print("{}:\t mean: {:.3f} std:{:.3f} min: {:.3f} max: {:.3f}".format(k[16:], v["mean"], v["std"], v["min"], v["max"]))
        except:
            print("{}:\t mean: {:.3f} std:{:.3f} min: {:.3f} max: {:.3f}".format(k[16:], np.mean(v), np.std(v), np.min(v), np.max(v)))


def print_summary_atom_level(feat_dict_atoms):
    """
    Helper function to print summary of atom level features
        feat_dict_atoms - dict of dicts of lists
    """
    for k_super, v_super in feat_dict_atoms.items():
        print(k_super)
        print("====================================")
        for k, v in v_super.items():
            print("{}:\t mean: {:.3f} std:{:.3f} min: {:.3f} max: {:.3f}".format(k[16:], v["mean"], v["std"], v["min"], v["max"]))


def plot_violin_from_complete_dict(feat_dict_complete, plot_per_row=3, line_width=2, name="violin_plots.png"):
    num_feats_to_plot = len(feat_dict_complete.keys())
    num_rows = int(np.ceil(num_feats_to_plot / plot_per_row))
    fig, axs = plt.subplots(num_rows, plot_per_row, figsize=(2 * num_rows, 6 * plot_per_row))
    axs = axs.flatten()

    for i, (k, v) in enumerate(feat_dict_complete.items()):
        v_std = np.std(v)
        v_mean = np.mean(v) 
        #filter out outliers
        v = np.array(v)
        tf_has_outliers = np.abs(v - v_mean) > 3*v_std
        num_outliers = np.sum(tf_has_outliers)
        percent_outliers = num_outliers/len(v)

        if percent_outliers > 0.03:
            if np.min(v) > 0:
                sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='tomato')
                axs[i].set_yscale("log")
                axs[i].set_ylabel("log(feature value)", fontsize=15)
            else: 
                v = v[v < v_mean + 3*v_std]
                v = v[v > v_mean - 3*v_std]
                axs[i].set_ylabel("feature value", fontsize=15)
                sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='cornflowerblue')          
            
        elif percent_outliers > 0.00:
            v = v[v < v_mean + 3*v_std]
            v = v[v > v_mean - 3*v_std]
            axs[i].set_ylabel("feature value", fontsize=15)
            sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='cornflowerblue')
        else:
            sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='mediumseagreen')
            axs[i].set_ylabel("feature value", fontsize=15)      

        axs[i].set_title(k[16:], fontsize=18)
        axs[i].tick_params(axis="x", labelrotation=90, labelsize=15)
        axs[i].grid()
    # remove empty subplots
    for i in range(num_feats_to_plot, len(axs)):
        axs[i].remove()

    plt.tight_layout()
    #plt.show()
    # save the figure
    plt.savefig(name, dpi=300)


def plot_violin_from_atom_dict(feat_dict, atom_plot, plot_per_row=3, line_width=2):
    feat_atom = feat_dict[atom_plot]
    num_feats_to_plot = len(feat_atom.keys())
    num_rows = int(np.ceil(num_feats_to_plot / plot_per_row))

    fig, axs = plt.subplots(num_rows, plot_per_row, figsize=(2 * num_rows, 6 * plot_per_row))
    axs = axs.flatten()
    fig.suptitle("{} atom features".format(atom_plot), fontsize=22)

    for i, (k, v) in enumerate(feat_atom.items()):
        v_std = np.std(v)
        v_mean = np.mean(v) 
        #filter out outliers
        v = np.array(v)
        #v = v[v < v_mean + 3*v_std]
        #v = v[v > v_mean - 3*v_std]
        tf_has_outliers = np.abs(v - v_mean) > 3*v_std
        num_outliers = np.sum(tf_has_outliers)
        percent_outliers = num_outliers/len(v)

        if percent_outliers > 0.03:
            # check that there are no negative values
            if np.min(v) > 0:
                sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='tomato')
                axs[i].set_yscale("log")
                axs[i].set_ylabel("log(feature value)", fontsize=15)
            else: 
                v = v[v < v_mean + 3*v_std]
                v = v[v > v_mean - 3*v_std]
                axs[i].set_ylabel("feature value", fontsize=15)
                sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='cornflowerblue')                
            
        elif percent_outliers > 0.00:
            v = v[v < v_mean + 3*v_std]
            v = v[v > v_mean - 3*v_std]
            axs[i].set_ylabel("feature value", fontsize=15)
            sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='cornflowerblue')
        else:
            sns.violinplot(y=v, ax=axs[i], linewidth=line_width, color='mediumseagreen')
            axs[i].set_ylabel("feature value", fontsize=15)      

        axs[i].set_title(k[16:], fontsize=17)
        axs[i].tick_params(axis="x", labelrotation=90, labelsize=15)
        axs[i].grid()
    # remove empty subplots
    for i in range(num_feats_to_plot, len(axs)):
        axs[i].remove()

    plt.tight_layout()
    # change spacing between suptitle and subplots
    plt.subplots_adjust(top=0.94)

    #plt.show()
    # save figure
    plt.savefig("atom_features_{}.png".format(atom_plot), dpi=300)


def get_bond_guess(atomic_elements, atomic_positions, charge=None):
    """
    Takes elements and positions and returns a list of bond guesses
    Takes:
        atomic_elements(list) - list of atomic elements 
        atomic_positions(list) - list of atomic positions
    Returns:
        bonds_as_inds(list) - list of bond guesses
    """
    # Add atoms to the molecule
    molecule_string = "{}\ncomment\n".format(len(atomic_elements))
    for atomic_position, atomic_element in zip(atomic_positions, atomic_elements):
        molecule_string += atomic_element + "\t" + str(atomic_position[0]) + "\t" + str(atomic_position[1]) + "\t" + str(atomic_position[2]) + "\n"
    molecule_string += "\n"
    # create molecule object 
    molecule = Chem.MolFromXYZBlock(molecule_string)

    if charge:
        rdDetermineBonds.DetermineConnectivity(molecule, charge = charge)
    
    else:
        rdDetermineBonds.DetermineConnectivity(molecule)
        
    bonds = molecule.GetBonds()
    bonds_as_inds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in bonds]
    
    return bonds_as_inds




def get_bond_guess_dataset(dataset, mee=False, check_charge=False): 
    """
    Takes a dataset and returns a list of bond guesses
    Takes:
        dataset(HeteroGraphGraphLabelClassifierDataset) - dataset to pull values from 
        mee(bool) - whether to use metal edge extender to guess bonds
    Returns:
        bond_guesses(list) - list of bond guesses
    """
    bond_guesses = []
    if mee:
        pes = OpenBabelNN()
        for graph_no_bond in dataset.data:
            bonded_graph = pes.get_bonded_structure(graph_no_bond.pymatgen_mol)
            bonded_graph = metal_edge_extender(bonded_graph)
            bond_guesses.append([list(i) for i in pes.get_bonded_structure(bonded_graph.molecule).graph.edges(data=False)])

    else:
        elements = get_elements_from_ft(dataset)
        positions = get_positions(dataset)
        if check_charge:
            charge_list = [mol_wrapper.charge for mol_wrapper in dataset.data]
        
        bond_guesses = []
        for ind, element in enumerate(elements):
            if check_charge:
                bond_guesses.append(get_bond_guess(element, positions[ind], charge=charge_list[ind]))
            else:
                bond_guesses.append(get_bond_guess(element, positions[ind]))
    return bond_guesses


def get_elements_from_ft(dataset): 
    """
    From a dataset object, pull elements from molecular atom features 
    Takes:
        dataset(HeteroGraphGraphLabelClassifierDataset): dataset to pull values from 
    Returns:
        list_lens(list of lists) - list with elements
    """
    list_atom_names = dataset.exclude_names["atom"]
    list_pos = []
    list_elements = []
    ret_list = []
    for ind, name in enumerate(list_atom_names): 
        if "chemical_symbol" in name: 
            list_pos.append(ind)
            list_elements.append(name.split("_")[-1])
    
    for graph in dataset.graphs:
        ft_atom = graph.ndata["feat"]["atom"]
        ft_un_atom_position = ft_atom[:, list_pos]
        ft_un_atom_position = ft_un_atom_position.argmax(axis=1)
        ft_atom_elements = [list_elements[i] for i in ft_un_atom_position]
        ret_list.append(ft_atom_elements)
    return ret_list


def get_bond_lengths(dataset):
    """
    From a dataset object, pull bond lengths from molecular bond features 
    Takes:
        dataset(HeteroGraphGraphLabelClassifierDataset): dataset to pull values from 
    Returns:
        list_lens(list of lists) - list with bond distances
    """

    list_lens = []
    ind_bond_length = dataset.exclude_names["bond"].index("bond_length")
    graphs_unscale = dataset.unscale_features(dataset.graphs)
    for graph in graphs_unscale:
        ft_un = graph.ndata["feat"]["bond"][:, ind_bond_length]
        list_lens.append(ft_un.tolist())
    return list_lens


def get_positions(dataset):
    """
    From a dataset object, pull coordinates from molecular bond features
    Takes:
        dataset(HeteroGraphGraphLabelClassifierDataset): dataset to pull values from 
    Returns:
        list_lens(list of lists) - list with atom positions 
    """
    position_list = []
    for mol_wrapper in dataset.data:
        #print(mol_wrapper.coords)
        position_list.append(mol_wrapper.coords)
    return position_list


