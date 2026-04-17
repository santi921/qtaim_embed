#!/usr/bin/env python3
"""
Split a single merged LMDB into train/val/test LMDBs.

Usage:
    python split_lmdb.py \
        -src /p/lustre5/vargas58/graphs/charge_only/tm_react/merged/merged.lmdb \
        -out_dir /p/lustre5/vargas58/graphs/charge_only/tm_react/merged/ \
        --val_prop 0.1 \
        --test_prop 0.1 \
        --seed 42
"""

import argparse
import os
import pickle
import random
import shutil

import lmdb
from tqdm import tqdm


_DEFAULT_MAP_SIZE = 1099511627776


def _safe_map_size(path: str, desired: int = _DEFAULT_MAP_SIZE) -> int:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)
    free = shutil.disk_usage(parent).free
    safe = int(free * 0.9)
    return min(desired, safe) if safe > 0 else desired


def open_src(src_path: str):
    return lmdb.open(
        src_path,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=False,
    )


def write_split(src_env, indices: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    db = lmdb.open(
        out_path,
        map_size=_safe_map_size(out_path),
        subdir=False,
        meminit=False,
        map_async=True,
    )
    with src_env.begin() as src_txn:
        for new_idx, src_idx in enumerate(tqdm(indices, desc=f"writing {os.path.basename(out_path)}")):
            raw = src_txn.get(f"{src_idx}".encode("ascii"))
            txn = db.begin(write=True)
            txn.put(f"{new_idx}".encode("ascii"), raw)
            txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(indices), protocol=-1))
    txn.commit()
    db.sync()
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-src", type=str, required=True, help="path to merged.lmdb file")
    parser.add_argument("-out_dir", type=str, required=True, help="parent dir for train/val/test subdirs")
    parser.add_argument("--val_prop", type=float, default=0.1)
    parser.add_argument("--test_prop", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lmdb_name", type=str, default="data.lmdb", help="filename used inside each split subdir")
    args = parser.parse_args()

    src_env = open_src(args.src)
    with src_env.begin() as txn:
        n = pickle.loads(txn.get("length".encode("ascii")))
    print(f"Source LMDB has {n} entries")

    rng = random.Random(args.seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_test = int(n * args.test_prop)
    n_val = int(n * args.val_prop)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    print(f"Split sizes -- train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    write_split(src_env, train_idx, os.path.join(args.out_dir, "train", args.lmdb_name))
    write_split(src_env, val_idx,   os.path.join(args.out_dir, "val",   args.lmdb_name))
    write_split(src_env, test_idx,  os.path.join(args.out_dir, "test",  args.lmdb_name))

    src_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
