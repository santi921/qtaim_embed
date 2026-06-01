#!/usr/bin/env python3
"""
Split a single merged LMDB into train/val/test LMDBs.

Handles both qtaim_generator format (string mol-ID keys, bare graph bytes)
and qtaim_embed format (integer keys, {"molecule_graph": bytes} wrapper).
Output is always qtaim_embed-compatible.

Usage:
    python split_lmdb.py \
        -src /p/lustre5/vargas58/graphs/charge_only/tm_react/merged/merged.lmdb \
        -out_dir /p/lustre5/vargas58/graphs/charge_only/tm_react/merged/ \
        --val_prop 0.1 \
        --test_prop 0.1 \
        --seed 42
"""

import argparse
from qtaim_embed.data.lmdb import split_lmdb_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, required=True, help="path to source .lmdb file")
    parser.add_argument("-out_dir", type=str, required=True, help="parent dir for train/val/test subdirs")
    parser.add_argument("--val_prop", type=float, default=0.1)
    parser.add_argument("--test_prop", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lmdb_name", type=str, default="data.lmdb")
    parser.add_argument(
        "--method",
        type=str,
        default="random",
        choices=["random", "composition"],
        help="random: uniform shuffle by key. composition: group by molecular "
        "formula so same-formula molecules share a split (prevents leakage "
        "between near-identical structures such as trajectory steps).",
    )
    args = parser.parse_args()

    result = split_lmdb_file(
        src_path=args.src,
        out_dir=args.out_dir,
        val_prop=args.val_prop,
        test_prop=args.test_prop,
        seed=args.seed,
        lmdb_name=args.lmdb_name,
        method=args.method,
    )

    sizes = result["sizes"]
    print(f"Split complete -- train: {sizes['train']}, val: {sizes['val']}, test: {sizes['test']}")
    print(f"  train -> {result['train']}")
    print(f"  val   -> {result['val']}")
    print(f"  test  -> {result['test']}")


if __name__ == "__main__":
    main()
