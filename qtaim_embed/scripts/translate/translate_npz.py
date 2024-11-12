"""
This script qm9 or qm8 datasets into the format for training with dimenet, schnet, or painn
"""

import argparse
import pandas as pd
from qtaim_embed.utils.translation import (
    get_molecule_translation_dimenet_qm8,
    get_molecule_translation_dimenet_qm9,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_loc", type=str, default="./qm9.json")
    parser.add_argument("-out_loc", type=str, default="./qm9_dimenet.json")
    parser.add_argument("-dataset_type", type=str, default="qm9")

    args = parser.parse_args()
    dataset_loc = str(args.dataset_loc)
    out_loc = str(args.out_loc)
    dataset_type = str(args.dataset_type)

    if dataset_loc.split(".")[-1] == "pkl":
        df = pd.read_pickle(dataset_loc)
    elif dataset_loc.split(".")[-1] == "json":
        df = pd.read_json(dataset_loc)

    if dataset_type == "qm8":
        get_molecule_translation_dimenet_qm8(df, out_loc)
    else:
        get_molecule_translation_dimenet_qm9(df, out_loc)


main()
