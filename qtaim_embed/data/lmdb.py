import logging
import os
import io
import shutil
import hashlib
import lmdb
import pickle
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

logger = logging.getLogger(__name__)

from qtaim_embed.core.dataset import Subset

scalar = 1 / 1024

# Default LMDB map size: 1 TB
_DEFAULT_MAP_SIZE = 1099511627776


def _safe_map_size(lmdb_path: str, desired: int = _DEFAULT_MAP_SIZE) -> int:
    """Return a map_size that fits on disk, capped at 90% of free space."""
    parent = os.path.dirname(os.path.abspath(lmdb_path)) or "."
    os.makedirs(parent, exist_ok=True)
    free = shutil.disk_usage(parent).free
    safe = int(free * 0.9)
    return min(desired, safe) if safe > 0 else desired



def TransformMol(data_object):
    serialized_graph = data_object["molecule_graph"]
    # check if serialized_graph is already a PyG HeteroData or if it is bytes
    if isinstance(serialized_graph, HeteroData):
        return data_object

    graph = load_graph_from_serialized(serialized_graph)
    return graph


def serialize_graph(graph, ret=True):
    """Serialize a PyG HeteroData graph to bytes using torch.save with BytesIO."""
    buf = io.BytesIO()
    torch.save(graph, buf)
    if ret:
        return buf.getvalue()


def load_graph_from_serialized(serialized_graph):
    """Load a PyG HeteroData graph from serialized bytes."""
    buf = io.BytesIO(serialized_graph)
    graph = torch.load(buf, weights_only=False, map_location='cpu')
    return graph


def write_molecule_lmdb(graphs, lmdb_dir, lmdb_name, global_values, chunk: int = -1):
    """
    Write the molecule graphs to lmdb
    Takes:
        graphs: list of molecule graphs
        lmdb_dir: directory to save the lmdb
        lmdb_name: name of the lmdb
        global_values: dictionary of global values
        chunk: chunk size(default of -1 means no chunking)
    Returns:
        None
    """

    os.makedirs(lmdb_dir, exist_ok=True)

    key_template = ["molecule_graph"]

    dataset = [{k: v for k, v in zip(key_template, values)} for values in zip(graphs)]

    if chunk > 0:
        dataset_chunk = []
        for i in range(0, len(dataset), chunk):
            dataset_chunk.append(dataset[i : i + chunk])

        for ind, chunk in enumerate(dataset_chunk):
            # create lmdb for each chunk
            lmdb_chunk_name = f"{lmdb_name}_{ind}.lmdb"
            lmdb_chunk_dir = os.path.join(lmdb_dir, lmdb_chunk_name)
            db = lmdb.open(
                lmdb_chunk_dir,
                map_size=_safe_map_size(lmdb_chunk_dir),
                subdir=False,
                meminit=False,
                map_async=True,
            )
            # write samples
            for ind, sample in enumerate(chunk):
                # sample_index = sample["molecule_index"]
                sample_index = ind
                txn = db.begin(write=True)
                txn.put(
                    # let index of molecule identical to index of sample
                    f"{sample_index}".encode("ascii"),
                    pickle.dumps(sample, protocol=-1),
                )
                txn.commit()
            # write properties.
            txn = db.begin(write=True)
            txn.put("length".encode("ascii"), pickle.dumps(len(chunk), protocol=-1))
            txn.commit()

            for key, value in global_values.items():
                # print(key, value)
                txn = db.begin(write=True)
                txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
                txn.commit()
            # write the chunk size
            txn = db.begin(write=True)
            txn.put(
                "length_chunk".encode("ascii"), pickle.dumps(len(chunk), protocol=-1)
            )
            txn.commit()
            db.sync()
            db.close()

    else:
        db = lmdb.open(
            lmdb_dir + lmdb_name,
            map_size=_safe_map_size(lmdb_dir + lmdb_name),
            subdir=False,
            meminit=False,
            map_async=True,
        )

        # write samples
        for ind, sample in enumerate(dataset):
            # sample_index = sample["molecule_index"]
            sample_index = ind
            txn = db.begin(write=True)
            txn.put(
                # let index of molecule identical to index of sample
                f"{sample_index}".encode("ascii"),
                pickle.dumps(sample, protocol=-1),
            )
            txn.commit()

        # write properties.
        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(len(dataset), protocol=-1))
        txn.commit()

        for key, value in global_values.items():
            # print(key, value)
            txn = db.begin(write=True)
            txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
            txn.commit()

        db.sync()
        db.close()


def _serialize_graphs_parallel(graphs, num_workers):
    """
    Serialize PyG graphs in parallel (LMDB conversion only).

    Args:
        graphs: List of PyG HeteroData graphs
        num_workers: Number of parallel workers

    Returns:
        List of serialized graphs (bytes)
    """
    from multiprocessing import Pool, cpu_count
    from qtaim_embed.data.parallel_utils import serialize_graph_worker

    actual_workers = min(num_workers, len(graphs), cpu_count())
    args_list = [(idx, graph) for idx, graph in enumerate(graphs)]
    serialized = [None] * len(graphs)

    with Pool(processes=actual_workers) as pool:
        results = pool.imap_unordered(
            serialize_graph_worker,
            args_list,
            chunksize=max(1, len(args_list) // (actual_workers * 4))
        )

        for idx, serialized_bytes in tqdm(
            results,
            total=len(graphs),
            desc="Serializing graphs (parallel)"
        ):
            if serialized_bytes is not None:
                serialized[idx] = serialized_bytes

    return serialized


def construct_lmdb_and_save_dataset(dataset: str, lmdb_dir: str, chunk: int = -1, save_scalers=False, num_workers: int = 1):
    """
    Converts dataset to lmdb and saves it to the specified directory.
    Streams graphs one at a time to avoid holding all serialized data in memory.
    Takes:
        dataset: dataset object
        lmdb_dir: directory to save the lmdb
    Returns:
        None
    """

    if isinstance(dataset, Subset):
        src = dataset.dataset
        graph_iter = (src.graphs[ind] for ind in dataset.indices)
        num_graphs = len(dataset.indices)
    else:
        src = dataset
        graph_iter = iter(dataset.graphs)
        num_graphs = len(dataset.graphs)

    feature_size = src.feature_size
    feature_names = src.feature_names
    element_set = src.element_set
    log_scale_features = src.log_scale_features
    allowed_charges = src.allowed_charges
    allowed_spins = src.allowed_spins
    allowed_ring_size = src.allowed_ring_size
    target_dict = src.target_dict
    extra_dataset_info = src.extra_dataset_info

    if save_scalers:
        feature_scalers = src.feature_scalers
        label_scalers = src.label_scalers

    global_dict = {
        "feature_size": feature_size,
        "feature_names": feature_names,
        "element_set": element_set,
        "allowed_ring_size": allowed_ring_size,
        "allowed_charges": allowed_charges,
        "allowed_spins": allowed_spins,
        "target_dict": target_dict,
        "extra_dataset_info": extra_dataset_info,
        "log_scale_features": log_scale_features,
    }

    os.makedirs(lmdb_dir, exist_ok=True)

    logger.info(f"Streaming {num_graphs} molecules to LMDB")

    WRITE_BATCH_SIZE = 1000

    def _write_metadata(db, length, global_dict):
        txn = db.begin(write=True)
        txn.put("length".encode("ascii"), pickle.dumps(length, protocol=-1))
        for key, value in global_dict.items():
            txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
        txn.commit()

    if chunk > 0:
        # Chunked streaming: one LMDB file per chunk
        chunk_idx = 0
        local_ind = 0
        lmdb_chunk_name = f"molecule.lmdb_{chunk_idx}.lmdb"
        lmdb_chunk_path = os.path.join(lmdb_dir, lmdb_chunk_name)
        db = lmdb.open(lmdb_chunk_path, map_size=_safe_map_size(lmdb_chunk_path),
                        subdir=False, meminit=False, map_async=True)

        txn = db.begin(write=True)
        batch_count = 0
        for ind, graph in enumerate(graph_iter):
            if local_ind >= chunk:
                # Commit any pending writes before closing chunk
                txn.commit()
                # Close current chunk, write metadata
                _write_metadata(db, local_ind, global_dict)
                txn_meta = db.begin(write=True)
                txn_meta.put("length_chunk".encode("ascii"), pickle.dumps(local_ind, protocol=-1))
                txn_meta.commit()
                db.sync()
                db.close()
                # Open next chunk
                chunk_idx += 1
                local_ind = 0
                batch_count = 0
                lmdb_chunk_name = f"molecule.lmdb_{chunk_idx}.lmdb"
                lmdb_chunk_path = os.path.join(lmdb_dir, lmdb_chunk_name)
                db = lmdb.open(lmdb_chunk_path, map_size=_safe_map_size(lmdb_chunk_path),
                                subdir=False, meminit=False, map_async=True)
                txn = db.begin(write=True)

            serialized = serialize_graph(graph)
            sample = {"molecule_graph": serialized}
            txn.put(f"{local_ind}".encode("ascii"), pickle.dumps(sample, protocol=-1))
            batch_count += 1
            local_ind += 1

            if batch_count >= WRITE_BATCH_SIZE:
                txn.commit()
                txn = db.begin(write=True)
                batch_count = 0

        txn.commit()

        # Close final chunk
        _write_metadata(db, local_ind, global_dict)
        txn = db.begin(write=True)
        txn.put("length_chunk".encode("ascii"), pickle.dumps(local_ind, protocol=-1))
        txn.commit()
        db.sync()
        db.close()

    else:
        # Single file streaming
        lmdb_path = os.path.join(lmdb_dir, "molecule.lmdb")
        db = lmdb.open(
            lmdb_path,
            map_size=_safe_map_size(lmdb_path),
            subdir=False,
            meminit=False,
            map_async=True,
        )

        txn = db.begin(write=True)
        batch_count = 0
        for ind, graph in enumerate(graph_iter):
            serialized = serialize_graph(graph)
            sample = {"molecule_graph": serialized}
            txn.put(
                f"{ind}".encode("ascii"),
                pickle.dumps(sample, protocol=-1),
            )
            batch_count += 1
            if batch_count >= WRITE_BATCH_SIZE:
                txn.commit()
                txn = db.begin(write=True)
                batch_count = 0
        txn.commit()

        _write_metadata(db, num_graphs, global_dict)
        db.sync()
        db.close()

    if save_scalers:
        if feature_scalers == []:
            logger.warning("No feature scalers found in dataset. Skipping scaler save.")
        else:
            for scaler in feature_scalers:
                scaler.save_scaler(os.path.join(lmdb_dir, "feature_scaler_{}.pt".format(scaler.name)))
        if label_scalers == []:
            logger.warning("No label scalers found in dataset. Skipping label scaler save.")
        else:
            for scaler in label_scalers:
                scaler.save_scaler(os.path.join(lmdb_dir, "label_scaler_{}.pt".format(scaler.name)))

def combined_mean_std(mean_list, std_list, count_list):
    """
    Calculate the combined mean and standard deviation of multiple datasets.

    :param mean_list: List of means of the datasets.
    :param std_list: List of standard deviations of the datasets.
    :param count_list: List of number of data points in each dataset.
    :return: Combined mean and standard deviation.
    """
    # Calculate total number of data points
    total_count = sum(count_list)

    # Calculate combined mean
    combined_mean = (
        sum(mean * count for mean, count in zip(mean_list, count_list)) / total_count
    )

    # Calculate combined variance
    combined_variance = sum(
        (
            (std**2) * (count - 1) + count * (mean - combined_mean) ** 2
            for mean, std, count in zip(mean_list, std_list, count_list)
        )
    ) / (total_count - len(mean_list))

    # Calculate combined standard deviation
    combined_std = combined_variance**0.5

    return combined_mean, combined_std


# ---------------------------------------------------------------------------
# LMDB splitting utilities
# ---------------------------------------------------------------------------

import random as _random


def open_lmdb_readonly(path: str, readahead: bool = False):
    """Open an LMDB file for reading.

    readahead defaults to False (best for the random-access dataloader). Pass
    readahead=True for sequential full-file scans (e.g. the split passes),
    where OS prefetch is a large win on networked filesystems like Lustre.
    """
    return lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=readahead,
        meminit=False,
        subdir=False,
    )


_LMDB_METADATA_KEYS = {b"length", b"feature_size", b"feature_names", b"element_set",
                        b"allowed_ring_size", b"allowed_charges", b"allowed_spins",
                        b"target_dict", b"extra_dataset_info", b"log_scale_features",
                        b"length_chunk", b"scaled", b"processed_source_keys",
                        b"scaler_finalized"}

# Metadata copied verbatim into each split LMDB so the dataloader works.
# Derived from _LMDB_METADATA_KEYS minus per-split bookkeeping ("length" is
# rewritten per split) and scaling state (a fresh split is unscaled).
_COPY_META = _LMDB_METADATA_KEYS - {b"length", b"length_chunk", b"scaled",
                                    b"processed_source_keys", b"scaler_finalized"}


def write_lmdb_split(src_env, keys: list, out_path: str) -> None:
    """
    Write one split (train/val/test) from a source LMDB.

    Handles two source formats:
      - qtaim_generator format: string mol-ID keys, value = pickle.dumps(graph_bytes)
      - qtaim_embed format: integer keys, value = pickle.dumps({"molecule_graph": graph_bytes})

    Output is always qtaim_embed-compatible: integer keys, {"molecule_graph": bytes} wrapper,
    and a "length" metadata key.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    db = lmdb.open(
        out_path,
        map_size=_safe_map_size(out_path),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    split_name = os.path.basename(os.path.dirname(out_path)) or os.path.basename(out_path)
    WRITE_BATCH_SIZE = 1000
    with src_env.begin() as src_txn:
        # detect format once on the first key
        raw0 = src_txn.get(keys[0]) if keys else None
        is_embed_fmt = raw0 is not None and isinstance(pickle.loads(raw0), dict)

        txn = db.begin(write=True)
        batch_count = 0
        for new_idx, src_key in enumerate(tqdm(keys, desc=f"writing {split_name}")):
            raw = src_txn.get(src_key)
            if is_embed_fmt:
                entry = raw
            else:
                entry = pickle.dumps({"molecule_graph": pickle.loads(raw)}, protocol=-1)
            txn.put(f"{new_idx}".encode("ascii"), entry)
            batch_count += 1
            if batch_count >= WRITE_BATCH_SIZE:
                txn.commit()
                txn = db.begin(write=True)
                batch_count = 0
        txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(keys), protocol=-1))
    # Copy dataset metadata so LMDBMoleculeDataset.feature_names/feature_size/target_dict work
    with src_env.begin() as src_txn:
        for meta_key in _COPY_META:
            val = src_txn.get(meta_key)
            if val is not None:
                txn.put(meta_key, val)
    txn.commit()
    db.sync()
    db.close()


def _assign_formula_to_split(formula: str, ratios, seed: int) -> int:
    """Deterministically map a formula string to a split index (0/1/2).

    Mirrors qtaim_gen.source.utils.splits.assign_formula_to_split exactly
    (SHA-256 of "{formula}_{seed}" -> [0, 1) -> cumulative-ratio bucket) so
    composition splits produced here match the converter --split path.
    Reimplemented rather than imported because qtaim_gen depends on
    qtaim_embed; importing it here would be a circular dependency.
    """
    hash_input = f"{formula}_{seed}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % 10000 / 10000.0
    cumulative = 0.0
    for i, ratio in enumerate(ratios):
        cumulative += ratio
        if hash_val < cumulative:
            return i
    return len(ratios) - 1


def _element_columns_from_feature_names(feature_names) -> list:
    """Return [(col_index, element_symbol), ...] for chemical_symbol_* atom feats."""
    atom_names = feature_names.get("atom", []) if isinstance(feature_names, dict) else []
    prefix = "chemical_symbol_"
    return [
        (i, name[len(prefix):])
        for i, name in enumerate(atom_names)
        if isinstance(name, str) and name.startswith(prefix)
    ]


def _formula_from_graph(graph, elem_cols, formula_cache: dict = None) -> str:
    """Reconstruct a pymatgen formula string from atom element one-hot columns.

    Sums each chemical_symbol_* one-hot column across atoms to recover element
    counts, then formats via pymatgen Composition so the string matches
    build_formula_map_from_structure_lmdb's convention.

    formula_cache (optional) memoizes the count-tuple -> formula mapping; a
    dataset has far fewer distinct compositions than graphs, so this avoids
    rebuilding pymatgen Composition for every record.
    """
    from pymatgen.core import Composition

    feat = graph["atom"].feat
    width = feat.shape[1]
    counts = {}
    for idx, sym in elem_cols:
        if idx >= width:
            raise ValueError(
                f"element column index {idx} ({sym}) exceeds atom.feat width {width}; "
                "feature_names does not align with stored graphs"
            )
        c = int(round(float(feat[:, idx].sum().item())))
        if c > 0:
            counts[sym] = c
    if not counts:
        return ""
    cache_key = tuple(sorted(counts.items()))
    if formula_cache is not None and cache_key in formula_cache:
        return formula_cache[cache_key]
    formula = Composition(counts).formula.replace(" ", "")
    if formula_cache is not None:
        formula_cache[cache_key] = formula
    return formula


def split_lmdb_file(
    src_path: str,
    out_dir: str,
    val_prop: float = 0.1,
    test_prop: float = 0.1,
    seed: int = 42,
    lmdb_name: str = "data.lmdb",
    method: str = "random",
) -> dict:
    """
    Split a single LMDB into train/val/test LMDBs.

    method:
      "random"      -- uniform shuffle by key (default; backward compatible).
      "composition" -- group by molecular formula (derived from each graph's
                       chemical_symbol_* one-hot columns) and assign whole
                       formula groups to a split via a deterministic hash.
                       All molecules sharing a formula land in the same split,
                       which prevents leakage between near-identical structures
                       (e.g. trajectory steps of one system). Requires
                       'feature_names' metadata in the source LMDB.

    Returns a dict with keys "train", "val", "test" mapping to output paths
    and "sizes" mapping to {"train": int, "val": int, "test": int}.
    """
    if val_prop + test_prop >= 1.0:
        raise ValueError(f"val_prop + test_prop must be < 1.0, got {val_prop + test_prop}")
    if method not in ("random", "composition"):
        raise ValueError(f"method must be 'random' or 'composition', got {method!r}")

    # One readahead=True env. Every pass reads sequentially: the derive scan
    # uses a cursor, and the write pass sorts each split's keys into storage
    # order (below) so its reads sweep the file forward too. Sequential reads
    # with OS prefetch are the difference between minutes and hours on Lustre.
    src_env = open_lmdb_readonly(src_path, readahead=True)
    approx_total = src_env.stat()["entries"]

    if method == "random":
        with src_env.begin() as txn:
            all_keys = [
                k
                for k, _ in tqdm(txn.cursor(), desc="scanning source", total=approx_total)
                if k not in _LMDB_METADATA_KEYS
            ]
        n = len(all_keys)
        rng = _random.Random(seed)
        rng.shuffle(all_keys)
        n_test = int(n * test_prop)
        n_val = int(n * val_prop)
        n_train = n - n_val - n_test
        splits = {
            "train": all_keys[:n_train],
            "val": all_keys[n_train : n_train + n_val],
            "test": all_keys[n_train + n_val :],
        }
    else:  # composition
        with src_env.begin() as txn:
            fn_raw = txn.get(b"feature_names")
            scaled_raw = txn.get(b"scaled")
        # composition needs raw 0/1 one-hot element columns; scaled features
        # turn them into standardized floats and silently corrupt formulas.
        if scaled_raw is not None and pickle.loads(scaled_raw):
            raise ValueError(
                "composition split requires unscaled features (chemical_symbol_* columns "
                "must be 0/1 one-hots), but the source LMDB is marked scaled. "
                "Run the composition split before scaling."
            )
        if fn_raw is None:
            raise ValueError(
                "composition split requires 'feature_names' metadata in the source LMDB"
            )
        elem_cols = _element_columns_from_feature_names(pickle.loads(fn_raw))
        if not elem_cols:
            raise ValueError(
                "no 'chemical_symbol_*' columns found in atom feature_names; "
                "cannot derive formulas for composition split"
            )
        ratios = (1.0 - val_prop - test_prop, val_prop, test_prop)
        split_names = ("train", "val", "test")
        by_formula = {}
        n_failed = 0
        n_empty = 0
        formula_cache = {}
        # single sequential pass collects keys and derives formulas together
        # (avoids a separate full key-scan over the source).
        with src_env.begin() as txn:
            for k, v in tqdm(txn.cursor(), desc="deriving formulas", total=approx_total):
                if k in _LMDB_METADATA_KEYS:
                    continue
                try:
                    obj = pickle.loads(v)
                    graph_bytes = obj["molecule_graph"] if isinstance(obj, dict) else obj
                    graph = (
                        load_graph_from_serialized(graph_bytes)
                        if isinstance(graph_bytes, (bytes, bytearray))
                        else graph_bytes
                    )
                    formula = _formula_from_graph(graph, elem_cols, formula_cache)
                except Exception as e:
                    # a single bad record must not abort the whole split; route to train
                    logger.warning(
                        f"composition split: formula derivation failed for key {k!r}: {e}; "
                        "routing to train"
                    )
                    formula = ""
                    n_failed += 1
                else:
                    if formula == "":
                        n_empty += 1
                by_formula.setdefault(formula, []).append(k)
        splits = {name: [] for name in split_names}
        for formula, group in by_formula.items():
            # empty/failed formulas go to train, matching the converter --split path
            # (qtaim_gen.partition_keys_by_composition routes missing formulas to train).
            if formula == "":
                splits["train"].extend(group)
            else:
                splits[split_names[_assign_formula_to_split(formula, ratios, seed)]].extend(group)
        n = sum(len(g) for g in splits.values())
        if n_failed or n_empty:
            logger.info(
                f"composition split: routed {n_failed} failed + {n_empty} empty-formula "
                "molecules to train"
            )
        logger.info(
            f"composition split: {n} molecules across {len(by_formula)} unique formulas"
        )

    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    logger.info(f"Total molecules: {n}  ->  train={n_train}, val={n_val}, test={n_test}")

    out_paths = {}
    for split_name, keys in splits.items():
        out_path = os.path.join(out_dir, split_name, lmdb_name)
        # sort into LMDB storage order so write-pass reads sweep the source
        # sequentially (random per-key reads crawl on Lustre); membership is
        # unchanged, only the 0..n re-indexing order.
        write_lmdb_split(src_env, sorted(keys), out_path)
        out_paths[split_name] = out_path

    src_env.close()

    return {
        "train": out_paths["train"],
        "val": out_paths["val"],
        "test": out_paths["test"],
        "sizes": {k: len(v) for k, v in splits.items()},
    }
