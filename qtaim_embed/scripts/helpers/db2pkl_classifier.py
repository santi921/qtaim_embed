#!/usr/bin/env python3
"""
Convert SQLite workflow databases to pickle DataFrames for QTAIM-embed classifier training.

Supports two DB modes per file:
  --db_paths_extended  running/ready/timeout/to_run are also treated as failed
  --db_paths_strict    only 'failed' counts as failed; other non-completed statuses excluded

Bond detection uses numpy vectorized distance computation and multiprocessing.

Usage:
    python db2pkl_classifier.py \
        --db_paths_extended db1.db db2.db \
        --db_paths_strict db3.db \
        --output output.pkl \
        --num_workers 8
"""

import argparse
import logging
import sqlite3
from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Molecule
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EXTENDED_FAIL_STATUSES = ("failed", "running", "ready", "timeout", "to_run")
STRICT_FAIL_STATUSES = ("failed",)


def _get_jmol_params() -> Tuple[dict, float, float]:
    nn = JmolNN()
    return nn.el_radius, nn.tol, getattr(nn, "min_bond_distance", 0.4)


def _compute_bonds_numpy(
    coords: np.ndarray,
    species: List[str],
    el_radius: dict,
    tol: float,
    min_bond_distance: float,
) -> List[Tuple[int, int]]:
    n = len(coords)
    if n <= 1:
        return []
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    radii = np.array([el_radius.get(s, 1.5) for s in species])
    cutoffs = radii[:, None] + radii[None, :]
    mask = np.triu((dists < cutoffs + tol) & (dists > min_bond_distance), k=1)
    i_idx, j_idx = np.where(mask)
    return list(zip(i_idx.tolist(), j_idx.tolist()))


def _process_row(
    row_data: tuple,
    el_radius: dict,
    tol: float,
    min_bond_distance: float,
) -> Optional[tuple]:
    geometry, charge, spin, row_id, metal, status = row_data
    try:
        lines = geometry.strip().split("\n")
        species, coords_list = [], []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                species.append(parts[0])
                coords_list.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if len(species) == 0:
            return None

        coords = np.array(coords_list)
        bonds = _compute_bonds_numpy(coords, species, el_radius, tol, min_bond_distance)

        if len(bonds) == 0:
            return None

        mol = Molecule(species, coords_list, charge=charge, spin_multiplicity=spin)
        bond_dict = {(u, v): {} for u, v in bonds}
        mg = MoleculeGraph.with_edges(mol, bond_dict)

        name = f"{metal}_{row_id}"
        success = 1.0 if status == "completed" else 0.0
        return (mol, mg, row_id, name, [bonds], success)

    except Exception as e:
        logger.warning("Skipping row %s (%s): %s", row_id, type(e).__name__, e)
        return None


def load_structures_from_db(db_path: str, fail_statuses: tuple) -> pd.DataFrame:
    all_statuses = ("completed",) + fail_statuses
    placeholders = ",".join("?" for _ in all_statuses)
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT id, elements, natoms, status, charge, spin, geometry, metal
        FROM structures
        WHERE status IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=all_statuses)
    conn.close()
    logger.info(
        "Loaded %d rows from %s (statuses: completed + %s)",
        len(df), db_path, fail_statuses,
    )
    return df


def convert_db_to_classifier_df(
    db_paths_extended: List[str],
    db_paths_strict: List[str],
    num_workers: int = 1,
) -> pd.DataFrame:
    all_dfs = []
    for path in db_paths_extended:
        all_dfs.append(load_structures_from_db(path, EXTENDED_FAIL_STATUSES))
    for path in db_paths_strict:
        all_dfs.append(load_structures_from_db(path, STRICT_FAIL_STATUSES))

    if not all_dfs:
        raise ValueError("no databases given: pass --db_paths_extended and/or --db_paths_strict")
    raw_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(
        "Total structures: %d (completed=%d, other=%d)",
        len(raw_df),
        (raw_df["status"] == "completed").sum(),
        (raw_df["status"] != "completed").sum(),
    )

    # A geometry that completed in one DB must not be relabelled failed by a
    # non-terminal copy in another: keep the completed row on duplicates.
    raw_df = raw_df.sort_values("status", key=lambda s: (s != "completed").astype(int), kind="stable")
    raw_df = raw_df.drop_duplicates(subset=["geometry"], keep="first")
    logger.info("After dedup: %d structures", len(raw_df))

    el_radius, tol, min_bond_distance = _get_jmol_params()
    worker_fn = partial(_process_row, el_radius=el_radius, tol=tol, min_bond_distance=min_bond_distance)

    row_data = [
        (row["geometry"], row["charge"], row["spin"], row["id"], row["metal"], row["status"])
        for _, row in raw_df.iterrows()
    ]

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, row_data, chunksize=32),
                total=len(row_data),
                desc="Building molecules",
            ))
    else:
        results = [worker_fn(r) for r in tqdm(row_data, desc="Building molecules")]

    molecules, molecule_graphs, ids_list, names_list, bonds_list, success_list = [], [], [], [], [], []
    skipped = 0
    for r in results:
        if r is None:
            skipped += 1
            continue
        mol, mg, row_id, name, bonds, success = r
        molecules.append(mol)
        molecule_graphs.append(mg)
        ids_list.append(row_id)
        names_list.append(name)
        bonds_list.append(bonds)
        success_list.append(success)

    logger.info("Built %d molecules, skipped %d", len(molecules), skipped)

    result_df = pd.DataFrame({
        "molecule": molecules,
        "molecule_graph": molecule_graphs,
        "ids": ids_list,
        "names": names_list,
        "bonds": bonds_list,
        "success": success_list,
    })

    n_success = (result_df["success"] == 1.0).sum()
    n_fail = (result_df["success"] == 0.0).sum()
    logger.info(
        "Class balance: success=%d (%.1f%%), fail=%d (%.1f%%)",
        n_success, 100 * n_success / len(result_df),
        n_fail, 100 * n_fail / len(result_df),
    )

    return result_df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert oact_utils SQLite DBs to classifier pickle for QTAIM-embed"
    )
    parser.add_argument(
        "--db_paths_extended", nargs="*", default=[],
        help="DBs where running/ready/timeout/to_run are also treated as failed",
    )
    parser.add_argument(
        "--db_paths_strict", nargs="*", default=[],
        help="DBs where only 'failed' counts as failed (others excluded)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output pickle file path")
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of parallel workers (default: 1)",
    )
    # backward compat: the original script selected WHERE status IN ('completed', 'failed'),
    # which is the strict mode. Queued/running rows were never labelled.
    parser.add_argument(
        "--db_paths", nargs="*", default=[],
        help="legacy alias for --db_paths_strict (completed + failed only)",
    )

    args = parser.parse_args(argv)
    if not (args.db_paths_extended or args.db_paths_strict or args.db_paths):
        parser.error("pass at least one of --db_paths_extended, --db_paths_strict, --db_paths")

    df = convert_db_to_classifier_df(
        db_paths_extended=args.db_paths_extended,
        db_paths_strict=args.db_paths_strict + args.db_paths,
        num_workers=args.num_workers,
    )

    df.to_pickle(args.output)
    logger.info("Saved %d structures to %s", len(df), args.output)


if __name__ == "__main__":
    main()
