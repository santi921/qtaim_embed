#!/usr/bin/env python3

import argparse
from qtaim_embed.core.dataset import HeteroGraphNodeLabelDataset
from qtaim_embed.utils.dataset import (
    gather_atom_level_stats,
    gather_bond_level_stats,
    print_summary_complete,
    print_summary_atom_level,
    plot_violin_from_complete_dict,
    plot_violin_from_atom_dict,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_plot",
        type=str,
        default="",
        help="path to dataset to plot",
    )
    dataset_loc = parser.parse_args().dataset_plot

    dataset = HeteroGraphNodeLabelDataset(
        file=dataset_loc,
        standard_scale_features=False,
        standard_scale_targets=False,
        log_scale_features=False,
        log_scale_targets=False,
        allowed_ring_size=[5],
        allowed_charges=None,
        allowed_spins=None,
        self_loop=True,
        debug=False,
        extra_keys={
            "atom": [
                "extra_feat_atom_Lagrangian_K",
                "extra_feat_atom_Hamiltonian_K",
                "extra_feat_atom_e_density",
                "extra_feat_atom_lap_e_density",
                "extra_feat_atom_e_loc_func",
                "extra_feat_atom_ave_loc_ion_E",
                "extra_feat_atom_delta_g_promolecular",
                "extra_feat_atom_delta_g_hirsh",
                "extra_feat_atom_esp_nuc",
                "extra_feat_atom_esp_e",
                "extra_feat_atom_esp_total",
                "extra_feat_atom_grad_norm",
                "extra_feat_atom_lap_norm",
                "extra_feat_atom_eig_hess",
                "extra_feat_atom_det_hessian",
                "extra_feat_atom_ellip_e_dens",
                "extra_feat_atom_eta",
                "extra_feat_atom_energy_density",
                "extra_feat_atom_density_beta",
                "extra_feat_atom_density_alpha",
                "extra_feat_atom_spin_density",
                "extra_feat_atom_lol",
            ],
            "bond": [
                "extra_feat_bond_Lagrangian_K",
                "extra_feat_bond_Hamiltonian_K",
                "extra_feat_bond_e_density",
                "extra_feat_bond_lap_e_density",
                "extra_feat_bond_e_loc_func",
                "extra_feat_bond_ave_loc_ion_E",
                "extra_feat_bond_delta_g_promolecular",
                "extra_feat_bond_delta_g_hirsh",
                "extra_feat_bond_esp_nuc",
                "extra_feat_bond_esp_e",
                "extra_feat_bond_esp_total",
                "extra_feat_bond_grad_norm",
                "extra_feat_bond_lap_norm",
                "extra_feat_bond_eig_hess",
                "extra_feat_bond_det_hessian",
                "extra_feat_bond_ellip_e_dens",
                "extra_feat_bond_eta",
                "extra_feat_bond_energy_density",
                "extra_feat_bond_density_beta",
                "extra_feat_bond_density_alpha",
                "extra_feat_bond_spin_density",
                "extra_feat_bond_lol",
                "bond_length",
            ],
            "global": [],
        },
        target_dict={
            "atom": [
                "extra_feat_atom_Lagrangian_K",
                "extra_feat_atom_Hamiltonian_K",
                "extra_feat_atom_e_density",
                "extra_feat_atom_lap_e_density",
                "extra_feat_atom_e_loc_func",
                "extra_feat_atom_ave_loc_ion_E",
                "extra_feat_atom_delta_g_promolecular",
                "extra_feat_atom_delta_g_hirsh",
                "extra_feat_atom_esp_nuc",
                "extra_feat_atom_esp_e",
                "extra_feat_atom_esp_total",
                "extra_feat_atom_grad_norm",
                "extra_feat_atom_lap_norm",
                "extra_feat_atom_eig_hess",
                "extra_feat_atom_det_hessian",
                "extra_feat_atom_ellip_e_dens",
                "extra_feat_atom_eta",
                "extra_feat_atom_energy_density",
                "extra_feat_atom_density_beta",
                "extra_feat_atom_density_alpha",
                "extra_feat_atom_spin_density",
                "extra_feat_atom_lol",
            ],
            "bond": [
                "extra_feat_bond_Lagrangian_K",
                "extra_feat_bond_Hamiltonian_K",
                "extra_feat_bond_e_density",
                "extra_feat_bond_lap_e_density",
                "extra_feat_bond_e_loc_func",
                "extra_feat_bond_ave_loc_ion_E",
                "extra_feat_bond_delta_g_promolecular",
                "extra_feat_bond_delta_g_hirsh",
                "extra_feat_bond_esp_nuc",
                "extra_feat_bond_esp_e",
                "extra_feat_bond_esp_total",
                "extra_feat_bond_grad_norm",
                "extra_feat_bond_lap_norm",
                "extra_feat_bond_eig_hess",
                "extra_feat_bond_det_hessian",
                "extra_feat_bond_ellip_e_dens",
                "extra_feat_bond_eta",
                "extra_feat_bond_energy_density",
                "extra_feat_bond_density_beta",
                "extra_feat_bond_density_alpha",
                "extra_feat_bond_spin_density",
                "extra_feat_bond_lol",
                "bond_length",
            ],
            "global": [],
        },
        extra_dataset_info={},
    )

    (
        feat_dict_atoms,
        feat_dict_complete_atoms,
        feat_dict_summary_atoms,
    ) = gather_atom_level_stats(dataset)
    feat_dict_complete_bonds, feat_dict_summary_bonds = gather_bond_level_stats(dataset)

    print("... > atom_summary stats")
    print_summary_complete(feat_dict_complete_atoms)
    print("... > bond_summary stats")
    print_summary_complete(feat_dict_summary_bonds)
    print("... > atom level stats")
    print_summary_atom_level(feat_dict_summary_atoms)

    plot_violin_from_complete_dict(
        feat_dict_complete_atoms,
        plot_per_row=3,
        line_width=2,
        name="violin_plots_atoms.png",
    )
    plot_violin_from_complete_dict(
        feat_dict_complete_bonds,
        plot_per_row=3,
        line_width=2,
        name="violin_plots_bonds.png",
    )

    atoms_in_dataset = list(feat_dict_atoms.keys())

    for atom in atoms_in_dataset:
        plot_violin_from_atom_dict(feat_dict_atoms, atom, plot_per_row=3, line_width=2)


main()
