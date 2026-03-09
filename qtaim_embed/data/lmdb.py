import os
import io
import shutil
import lmdb
import pickle
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

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
    graph = torch.load(buf, weights_only=False)
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

    if type(dataset) == Subset:
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

    print(f"...> streaming {num_graphs} molecules to lmdb")

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
            print("No feature scalers found in dataset. Skipping scaler save.")
        else:
            for scaler in feature_scalers:
                scaler.save_scaler(os.path.join(lmdb_dir, "feature_scaler_{}.pt".format(scaler.name)))
        if label_scalers == []:
            print("No label scalers found in dataset. Skipping label scaler save.")
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
