#!/usr/bin/env python3
"""
Convert SQLite workflow databases to pickle DataFrames for QTAIM-embed classifier training.

Reads structures from oact_utils-style SQLite databases, constructs pymatgen
Molecule and MoleculeGraph objects, and outputs pickle files with binary
success/failure labels suitable for HeteroGraphGraphLabelClassifierDataset.

Usage:
    python db2pkl_classifier.py \
        --db_paths /path/to/db1.db /path/to/db2.db \
        --output /path/to/output.pkl \
        [--neighbor_strategy jmol]
"""

import argparse
import logging
import sqlite3
from io import StringIO
from typing import List, Tuple

import numpy as np
import pandas as pd
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import JmolNN, CutOffDictNN
from pymatgen.core import Molecule
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_xyz_geometry(geometry_str: str, charge: int, spin: int) -> Molecule:
    """
    Parse XYZ geometry string from SQLite DB into a pymatgen Molecule.

    The geometry column stores XYZ format: first line is natoms,
    second line is blank/comment, then atom lines.
    """
    lines = geometry_str.strip().split("\n")
    # First line is atom count, second is comment/blank
    species = []
    coords = []
    for line in lines[2:]:  # skip natoms + comment line
        parts = line.split()
        if len(parts) >= 4:
            species.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    mol = Molecule(
        species,
        coords,
        charge=charge,
        spin_multiplicity=spin,
    )
    return mol


def build_molecule_graph(mol: Molecule, strategy: str = "jmol") -> MoleculeGraph:
    """Build a MoleculeGraph from a Molecule using the specified neighbor strategy."""
    if strategy == "jmol":
        nn = JmolNN()
    elif strategy == "cutoff":
        nn = CutOffDictNN()
    else:
        raise ValueError(f"Unknown neighbor strategy: {strategy}")

    mg = MoleculeGraph.with_local_env_strategy(mol, nn)
    return mg


def extract_bonds_from_graph(mg: MoleculeGraph) -> List[Tuple[int, int]]:
    """Extract bond list from MoleculeGraph edges."""
    bonds = []
    for u, v in mg.graph.edges():
        bonds.append((u, v))
    return bonds


def load_structures_from_db(db_path: str) -> pd.DataFrame:
    """Load completed and failed structures from a SQLite DB."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT id, elements, natoms, status, charge, spin, geometry, metal
        FROM structures
        WHERE status IN ('completed', 'failed')
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    logger.info("Loaded %d rows from %s", len(df), db_path)
    return df


def convert_db_to_classifier_df(
    db_paths: List[str],
    neighbor_strategy: str = "jmol",
) -> pd.DataFrame:
    """
    Convert one or more SQLite DBs to a DataFrame suitable for
    HeteroGraphGraphLabelClassifierDataset.

    Returns DataFrame with columns:
        molecule, molecule_graph, ids, names, bonds, success
    """
    all_dfs = []
    for path in db_paths:
        all_dfs.append(load_structures_from_db(path))

    raw_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        "Total structures: %d (completed=%d, failed=%d)",
        len(raw_df),
        (raw_df["status"] == "completed").sum(),
        (raw_df["status"] == "failed").sum(),
    )

    # Deduplicate by geometry hash (same structure may appear in multiple DBs)
    raw_df = raw_df.drop_duplicates(subset=["geometry"], keep="first")
    logger.info("After dedup: %d structures", len(raw_df))

    molecules = []
    molecule_graphs = []
    ids_list = []
    names_list = []
    bonds_list = []
    success_list = []
    skipped = 0

    for idx, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Building molecules"):
        try:
            mol = parse_xyz_geometry(row["geometry"], row["charge"], row["spin"])
            mg = build_molecule_graph(mol, strategy=neighbor_strategy)
            bonds = extract_bonds_from_graph(mg)

            # Skip molecules with no bonds detected
            if len(bonds) == 0:
                skipped += 1
                continue

            molecules.append(mol)
            molecule_graphs.append(mg)
            ids_list.append(row["id"])
            names_list.append(f"{row['metal']}_{row['id']}")
            bonds_list.append([bonds])  # wrapped in list to match expected format
            success_list.append(1.0 if row["status"] == "completed" else 0.0)

        except Exception as e:
            logger.warning("Skipping row %d: %s", row["id"], str(e))
            skipped += 1
            continue

    logger.info(
        "Built %d molecules, skipped %d",
        len(molecules),
        skipped,
    )

    result_df = pd.DataFrame(
        {
            "molecule": molecules,
            "molecule_graph": molecule_graphs,
            "ids": ids_list,
            "names": names_list,
            "bonds": bonds_list,
            "success": success_list,
        }
    )

    # Log class balance
    n_success = (result_df["success"] == 1.0).sum()
    n_fail = (result_df["success"] == 0.0).sum()
    logger.info("Class balance: success=%d (%.1f%%), fail=%d (%.1f%%)",
                n_success, 100 * n_success / len(result_df),
                n_fail, 100 * n_fail / len(result_df))

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Convert oact_utils SQLite DBs to classifier pickle for QTAIM-embed"
    )
    parser.add_argument(
        "--db_paths",
        nargs="+",
        required=True,
        help="Paths to SQLite .db files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file path",
    )
    parser.add_argument(
        "--neighbor_strategy",
        type=str,
        default="jmol",
        choices=["jmol", "cutoff"],
        help="Neighbor detection strategy for bond construction (default: jmol)",
    )

    args = parser.parse_args()

    df = convert_db_to_classifier_df(
        db_paths=args.db_paths,
        neighbor_strategy=args.neighbor_strategy,
    )

    df.to_pickle(args.output)
    logger.info("Saved %d structures to %s", len(df), args.output)


if __name__ == "__main__":
    main()
