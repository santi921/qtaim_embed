import os
import dgl
import lmdb
import pickle
import glob
import tempfile
import numpy as np 
from copy import deepcopy
from tqdm import tqdm

from qtaim_embed.utils.dataset import (
    clean,
    clean_op
)

from qtaim_embed.core.dataset import Subset
scalar = 1 / 1024

def TransformMol(data_object):
    serialized_graph = data_object['molecule_graph']
    # check if serialized_graph is DGL graph or if it is chunk 
    if isinstance(serialized_graph, dgl.DGLGraph):
        return data_object
    elif isinstance(serialized_graph, dgl.DGLHeteroGraph):
        return data_object

    dgl_graph = load_dgl_graph_from_serialized(serialized_graph)
    #data_object["molecule_graph"] = dgl_graph
    return dgl_graph


def serialize_dgl_graph(dgl_graph):
    # import pdb
    # pdb.set_trace()
    # Create a temporary file
    with tempfile.NamedTemporaryFile() as tmpfile:
        # Save the graph to the temporary file

        dgl.save_graphs(tmpfile.name, [dgl_graph])

        # Read the content of the file
        tmpfile.seek(0)
        serialized_data = tmpfile.read()

    return serialized_data


def load_dgl_graph_from_serialized(serialized_graph):
    with tempfile.NamedTemporaryFile(mode='wb', delete=True) as tmpfile:
        tmpfile.write(serialized_graph)
        tmpfile.flush()  # Ensure all data is written

        # Rewind the file to the beginning before reading
        tmpfile.seek(0)

        # Load the graph using the file handle
        graphs, _ = dgl.load_graphs(tmpfile.name)

    return graphs[0]  # Assuming there's only one graph


def write_molecule_lmdb(
    graphs,
    lmdb_dir,
    lmdb_name,
    global_values
):
    os.makedirs(lmdb_dir, exist_ok=True)

    key_template = ["molecule_graph"]
    dataset = [
        {k: v for k, v in zip(key_template, values)}
        for values in zip(graphs)
    ]

    db = lmdb.open(
        lmdb_dir + lmdb_name,
        map_size=int(1099511627776 * 2),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    #write samples
    for ind, sample in enumerate(dataset):
        #sample_index = sample["molecule_index"]
        sample_index = ind
        txn = db.begin(write=True)
        txn.put(
            #let index of molecule identical to index of sample
            f"{sample_index}".encode("ascii"),
            pickle.dumps(sample, protocol=-1),
        )
        txn.commit()

    #write properties.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(len(dataset), protocol=-1))
    txn.commit()

    for key, value in global_values.items():
        txn = db.begin(write=True)
        txn.put(key.encode("ascii"), pickle.dumps(value, protocol=-1))
        txn.commit()

    db.sync()
    db.close()


def construct_lmdb_and_save_dataset(dataset, lmdb_dir):
    """
    Converts dataset to lmdb and saves it to the specified directory
    Takes: 
        dataset: dataset object
        lmdb_dir: directory to save the lmdb
    Returns:
        None
    """
    
    if type(dataset) == Subset:
        feature_size = dataset.dataset.feature_size()
        feature_name = dataset.dataset.feature_names()
        element_set = dataset.dataset.element_set
        log_scale_features = dataset.dataset.log_scale_features
        allowed_charges = dataset.dataset.allowed_charges
        allowed_spins = dataset.dataset.allowed_spins
        allowed_ring_size = dataset.dataset.allowed_ring_size
        target_dict = dataset.dataset.target_dict
        extra_dataset_info = dataset.dataset.extra_dataset_info
        # List of Molecules
        dgl_graphs_serialized = [serialize_dgl_graph(dataset.dataset.graphs[ind]) for ind in dataset.indices]


    else:
        feature_size = dataset.feature_size()
        feature_name = dataset.feature_names()
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
        "feature_names": feature_name,
        "element_set": element_set,
        "ring_size_set": allowed_ring_size,
        "allowed_charges": allowed_charges,
        "allowed_spins": allowed_spins,
        "target_dict": target_dict,
        "extra_dataset_info": extra_dataset_info,

    }

    print("...> writing molecules to lmdb")
    #print("number of molecules to write: ", len(molecule_ind_list))
    write_molecule_lmdb(
        graphs=dgl_graphs_serialized,
        lmdb_dir=lmdb_dir,
        lmdb_name="molecule.lmdb",
        global_values=global_dict
    )
    print("...> writing reactions to lmdb")

