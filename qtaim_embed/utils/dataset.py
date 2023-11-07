import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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