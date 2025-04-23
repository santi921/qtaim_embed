import os
import dgl
import lmdb
import pickle
import tempfile


from qtaim_embed.core.dataset import Subset

scalar = 1 / 1024


def TransformMol(data_object):
    serialized_graph = data_object["molecule_graph"]
    # check if serialized_graph is DGL graph or if it is chunk
    if isinstance(serialized_graph, dgl.DGLGraph):
        return data_object
    elif isinstance(serialized_graph, dgl.DGLHeteroGraph):
        return data_object

    dgl_graph = load_dgl_graph_from_serialized(serialized_graph)
    # data_object["molecule_graph"] = dgl_graph
    return dgl_graph


def serialize_dgl_graph(dgl_graph, ret=True):
    # import pdb
    # pdb.set_trace()
    # Create a temporary file
    with tempfile.NamedTemporaryFile() as tmpfile:
        # Save the graph to the temporary file

        dgl.save_graphs(tmpfile.name, [dgl_graph])

        # Read the content of the file
        if ret:
            tmpfile.seek(0)
            serialized_data = tmpfile.read()
    if ret: 
        return serialized_data


def load_dgl_graph_from_serialized(serialized_graph):
    with tempfile.NamedTemporaryFile(mode="wb", delete=True) as tmpfile:
        tmpfile.write(serialized_graph)
        tmpfile.flush()  # Ensure all data is written

        # Rewind the file to the beginning before reading
        tmpfile.seek(0)

        # Load the graph using the file handle
        graphs, _ = dgl.load_graphs(tmpfile.name)

    return graphs[0]  # Assuming there's only one graph


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
                map_size=int(1099511627776 * 2),
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
            map_size=int(1099511627776 * 2),
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


def construct_lmdb_and_save_dataset(dataset: str, lmdb_dir: str, chunk: int = -1):
    """
    Converts dataset to lmdb and saves it to the specified directory
    Takes:
        dataset: dataset object
        lmdb_dir: directory to save the lmdb
    Returns:
        None
    """

    if type(dataset) == Subset:
        feature_size = dataset.dataset.feature_size
        feature_names = dataset.dataset.feature_names
        element_set = dataset.dataset.element_set
        log_scale_features = dataset.dataset.log_scale_features
        allowed_charges = dataset.dataset.allowed_charges
        allowed_spins = dataset.dataset.allowed_spins
        allowed_ring_size = dataset.dataset.allowed_ring_size
        target_dict = dataset.dataset.target_dict
        extra_dataset_info = dataset.dataset.extra_dataset_info
        # List of Molecules
        dgl_graphs_serialized = [
            serialize_dgl_graph(dataset.dataset.graphs[ind]) for ind in dataset.indices
        ]

    else:
        feature_size = dataset.feature_size
        feature_names = dataset.feature_names
        element_set = dataset.element_set
        log_scale_features = dataset.log_scale_features
        allowed_charges = dataset.allowed_charges
        allowed_spins = dataset.allowed_spins
        allowed_ring_size = dataset.allowed_ring_size
        target_dict = dataset.target_dict
        extra_dataset_info = dataset.extra_dataset_info
        # List of Molecules
        dgl_graphs_serialized = [serialize_dgl_graph(g) for g in dataset.graphs]

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

    print("...> writing molecules to lmdb")
    # print("number of molecules to write: ", len(molecule_ind_list))
    write_molecule_lmdb(
        graphs=dgl_graphs_serialized,
        lmdb_dir=lmdb_dir,
        lmdb_name="molecule.lmdb",
        global_values=global_dict,
        chunk=chunk,
    )


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
