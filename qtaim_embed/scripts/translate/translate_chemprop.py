#!/usr/bin/env python3

"""
This script qm9 or qm8 datasets into the format for training with chemprop with atomic qtaim features
"""

import argparse
import pandas as pd
import openbabel as ob
from rdkit import RDLogger

ob_log_handler = ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
RDLogger.DisableLog("rdApp.*")

from qtaim_embed.utils.translation import translate_qm8, translate_qm9


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_loc", type=str, default="./qm9_test.json")
    parser.add_argument("-out_loc", type=str, default="./qm9_test_qtaim.json")
    parser.add_argument(
        "-out_loc_qtaim", type=str, default="./qm9_test_qtaim_feats.json"
    )
    parser.add_argument("-dataset_type", type=str, default="qm9")

    args = parser.parse_args()
    dataset_loc = str(args.dataset_loc)
    out_loc = str(args.out_loc)
    out_loc_qtaim = str(args.out_loc_qtaim)
    dataset_type = str(args.dataset_type)

    atom_qtaim_keys = [
        "extra_feat_atom_Hamiltonian_K",
        "extra_feat_atom_e_density",
        "extra_feat_atom_lap_e_density",
        "extra_feat_atom_ave_loc_ion_E",
        "extra_feat_atom_esp_e",
        "extra_feat_atom_esp_total",
        "extra_feat_atom_ellip_e_dens",
        "extra_feat_atom_energy_density",
    ]

    if dataset_loc.split(".")[-1] == "pkl":
        df = pd.read_pickle(dataset_loc)
    elif dataset_loc.split(".")[-1] == "json":
        df = pd.read_json(dataset_loc)

    if dataset_type == "qm8":
        translate_qm8(df, out_loc, out_loc_qtaim, atom_qtaim_keys)
    else:
        translate_qm9(df, out_loc, out_loc_qtaim, atom_qtaim_keys)


main()
