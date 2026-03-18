import logging
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import lmdb
from copy import deepcopy
from torch.utils.data import Dataset
from pathlib import Path
import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from qtaim_embed.utils.grapher import get_grapher
from qtaim_embed.data.molwrapper import mol_wrappers_from_df
from qtaim_embed.data.processing import (
    HeteroGraphStandardScaler,
    HeteroGraphLogMagnitudeScaler,
)

logger = logging.getLogger(__name__)


def _get_ndata(graph: HeteroData, key: str) -> Dict[str, torch.Tensor]:
    """
    Helper to get a feature dict from PyG HeteroData.
    Returns a dict mapping node type -> tensor for all node types that have the attribute.
    """
    return {
        nt: getattr(graph[nt], key)
        for nt in graph.node_types
        if hasattr(graph[nt], key)
    }


def _set_ndata(graph: HeteroData, key: str, data_dict: Dict[str, torch.Tensor]) -> None:
    """
    Helper to set a feature dict on PyG HeteroData.
    Sets the attribute on each node type present in data_dict.
    """
    for nt, tensor in data_dict.items():
        setattr(graph[nt], key, tensor)


class HeteroGraphNodeLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str,
        standard_scale_features: bool = True,
        log_scale_features: bool = True,
        log_scale_targets: bool = False,
        standard_scale_targets: bool = True,
        allowed_ring_size: List[int] = [3, 4, 5, 6, 7],
        element_set: List[str] = [],
        allowed_charges: Optional[List[int]] = None,
        allowed_spins: Optional[List[int]] = None,
        self_loop: bool = True,
        size: Optional[int] = None,
        filter_self_bonds: bool = False,
        debug: bool = False,
        extra_keys: Dict[str, List[str]] = {
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["extra_feat_bond_esp_total", "bond_length"],
            "global": [],
        },
        target_dict: Dict[str, List[str]] = {
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["extra_feat_bond_esp_total"],
        },
        bond_key: str = "bonds",
        map_key: Optional[str] = None,
        extra_dataset_info: Dict[str, Any] = {},
        num_workers: int = 1,
        verbose: bool = True,
        rbf_cutoff: float = 5.0,
    ):
        """
        Baseline dataset for hetero graph node label prediction. Includes global feautures.
        Args:
            file (string): path to data file
            standard_scale_features (bool): whether to scale features
            log_scale_features (bool): whether to log scale features
            allowed_ring_size (list): list of allowed ring sizes
            allowed_charges (list): list of allowed charges
            self_loop (bool): whether to add self loops to the graph
            debug (bool): whether to run in debug mode
            extra_keys (dict): dictionary of keys to grab from the data file
            target_dict (dict): dictionary of keys to use as labels
            extra_dataset_info (dict): dictionary of extra info to be stored in the dataset
        """
        # check if file ends in pkl
        if file[-3:] == "pkl":
            #try:
            df = pd.read_pickle(file)
            #except:
            #    import pickle5 as pickle

            #    with open(file, "rb") as fh:
            #        df = pickle.load(fh)

        elif file[-3:] == "json":
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file)

        if debug:
            logger.info("Running in debug mode")
            df = df.head(100)
        else:
            if size != None:
                # randomly sample size number of rows
                df = df.sample(size)

        for key_check in ["atom", "bond", "global"]:
            if key_check not in extra_keys.keys():
                extra_keys[key_check] = []

        mol_wrappers, element_set_ret = mol_wrappers_from_df(
            df=df,
            bond_key=bond_key,
            map_key=map_key,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            filter_self_bonds=filter_self_bonds,
        )
        del df

        if element_set == [] or element_set == None:
            self.element_set = sorted(element_set_ret)
        else:
            self.element_set = element_set

        grapher = get_grapher(
            element_set=self.element_set,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            allowed_ring_size=allowed_ring_size,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
            self_loop=self_loop,
            rbf_cutoff=rbf_cutoff,
        )

        # Build graphs - use parallel processing if num_workers > 1
        if num_workers > 1 and len(mol_wrappers) > 1:
            graph_list, names = self._build_graphs_parallel(
                mol_wrappers, grapher, num_workers, verbose
            )
        else:
            # Serial graph building (original behavior)
            graph_list = []
            logger.info("Building graphs and featurizing")
            for mol in tqdm(mol_wrappers):
                graph = grapher.build_graph(mol)
                graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
                graph_list.append(graph)

        # Free source data -- all information is now in the graphs.
        del mol_wrappers, grapher
        import gc
        gc.collect()
        # Force glibc to return freed pages to the OS.
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        self.standard_scale_features = standard_scale_features
        self.log_scale_features = log_scale_features
        self.log_scale_labels = log_scale_targets
        self.standard_scale_labels = standard_scale_targets
        self.feature_scalers = []
        self.label_scalers = []
        self.data = None
        self.feat_names = names
        self.graphs = graph_list
        self.allowed_spins = allowed_spins
        self.allowed_charges = allowed_charges
        self.allowed_ring_size = allowed_ring_size
        self.target_dict = target_dict
        self.extra_dataset_info = extra_dataset_info
        self.verbose = verbose
        self.self_loop = self_loop

        self.load()
        logger.info("LOADED DATASET")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Any:
        # print(idx)
        return self.graphs[idx]  # , self.labels[idx]

    def _build_graphs_parallel(
        self, mol_wrappers, grapher, num_workers, verbose
    ):
        """
        Build graphs in parallel using multiprocessing.Pool.

        Args:
            mol_wrappers: List of MoleculeWrapper objects
            grapher: HeteroCompleteGraphFromMolWrapper instance
            num_workers: Number of parallel workers
            verbose: Whether to print progress information

        Returns:
            Tuple of (graph_list, feature_names)
        """
        from multiprocessing import Pool, cpu_count
        from qtaim_embed.data.parallel_utils import build_and_featurize_graph_worker

        # Extract grapher config for workers to reconstruct in each process
        # NOTE: when adding new featurizer params, update all 3 _build_graphs_parallel sites
        grapher_config = {
            'element_set': grapher.atom_featurizer.element_set,
            'atom_keys': grapher.atom_featurizer.selected_keys,
            'bond_keys': grapher.bond_featurizer.selected_keys,
            'global_keys': grapher.global_featurizer.selected_keys,
            'allowed_ring_size': grapher.atom_featurizer.allowed_ring_size,
            'allowed_charges': grapher.global_featurizer.allowed_charges,
            'allowed_spins': grapher.global_featurizer.allowed_spins,
            'self_loop': grapher.self_loop,
            'rbf_cutoff': grapher.bond_featurizer.rbf_cutoff,
        }

        # Clamp workers to reasonable bounds
        actual_workers = min(num_workers, len(mol_wrappers), cpu_count())

        # Prepare arguments for workers
        args_list = [
            (mol, grapher_config, idx) for idx, mol in enumerate(mol_wrappers)
        ]

        # Pre-allocate results list
        graph_list = [None] * len(mol_wrappers)
        feat_names = None

        # Process with progress bar
        with Pool(processes=actual_workers) as pool:
            results = pool.imap_unordered(
                build_and_featurize_graph_worker,
                args_list,
                chunksize=max(1, len(args_list) // (actual_workers * 4))
            )

            if verbose:
                logger.info("Building graphs and featurizing (parallel, %d workers)", actual_workers)

            for idx, graph, names in tqdm(
                results,
                total=len(mol_wrappers),
                desc="Building graphs (parallel)",
                disable=not verbose
            ):
                if graph is not None:
                    graph_list[idx] = graph
                    if feat_names is None:  # Store feature names from first successful result
                        feat_names = names
                else:
                    # Log error but continue (error already printed by worker)
                    pass

        return graph_list, feat_names

    def get_include_exclude_indices(self) -> None:
        target_locs = {}
        # get locations of target features
        for node_type, value_list in self.target_dict.items():
            if node_type not in target_locs:
                target_locs[node_type] = []

            for value in value_list:
                target_locs[node_type].append(self.feat_names[node_type].index(value))

        # now partition features into feats in target_locs and feats not in target_locs
        include_locs = {}
        exclude_locs = {}
        include_names = {}
        exclude_names = {}

        for node_type, value_list in self.feat_names.items():
            for i, value in enumerate(value_list):
                if node_type in target_locs.keys():
                    if i in target_locs[node_type]:
                        if node_type not in include_names:
                            include_names[node_type] = []
                            include_locs[node_type] = []

                        include_locs[node_type].append(i)
                        include_names[node_type].append(value)
                    else:
                        if node_type not in exclude_names:
                            exclude_names[node_type] = []
                            exclude_locs[node_type] = []
                        exclude_locs[node_type].append(i)
                        exclude_names[node_type].append(value)
                else:
                    if node_type not in exclude_names:
                        exclude_names[node_type] = []
                        exclude_locs[node_type] = []
                    exclude_locs[node_type].append(i)
                    exclude_names[node_type].append(value)

        self.include_locs = include_locs
        self.exclude_locs = exclude_locs
        self.include_names = include_names
        self.exclude_names = exclude_names

        if self.verbose:
            logger.debug("included in labels: %s", self.include_names)
            logger.debug("included in graph features: %s", self.exclude_names)

    def load(self) -> None:
        self.get_include_exclude_indices()
        logger.info("Parsing labels and features in graphs")
        for graph in tqdm(self.graphs):
            labels = {}
            features_new = {}
            ndata_feat = _get_ndata(graph, "feat")
            for key, value in ndata_feat.items():
                if key in self.include_names.keys():
                    features_new[key] = ndata_feat[key][
                        :, self.exclude_locs[key]
                    ]
                    labels[key] = ndata_feat[key][:, self.include_locs[key]]
            _set_ndata(graph, "feat", features_new)
            _set_ndata(graph, "labels", labels)

        if self.verbose:
            logger.debug("original loader node types: %s", list(_get_ndata(graph, "feat").keys()))
            logger.debug("original loader label types: %s", list(_get_ndata(graph, "labels").keys()))

        if self.log_scale_features:
            logger.info("Log scaling features")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=True, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.feature_scalers.append(log_scaler)
            logger.info("Log scaling features complete")

        if self.standard_scale_features:
            logger.info("Scaling features")
            scaler = HeteroGraphStandardScaler()
            self.graphs = scaler(self.graphs)
            self.feature_scalers.append(scaler)
            logger.info("Scaling features complete")
            logger.debug("feature mean(s): %s", scaler.mean)
            logger.debug("feature std(s): %s", scaler.std)

        if self.log_scale_labels:
            logger.info("Log scaling targets")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=False, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.label_scalers.append(log_scaler)
            logger.info("Log scaling targets complete")

        if self.standard_scale_labels:
            logger.info("Scaling targets")
            scaler = HeteroGraphStandardScaler(features_tf=False)
            self.graphs = scaler(self.graphs)
            self.label_scalers.append(scaler)
            logger.info("Scaling targets complete")
            logger.debug("feature mean(s): %s", scaler.mean)
            logger.debug("feature std(s): %s", scaler.std)

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        return self.exclude_names

    @property
    def label_names(self) -> Dict[str, List[str]]:
        return self.include_names

    @property
    def feature_size(self) -> Dict[str, int]:
        len_dict = {}
        for key, value in self.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict

    def unscale_features(self, graphs: List[Any]) -> List[Any]:
        """
        Perform inverse standardization on the given graphs.
        Takes:
            graphs: list of PyG HeteroData graphs
        Returns:
            graphs: list of PyG HeteroData graphs with inverse standardized features
        """
        assert (
            self.log_scale_features or self.standard_scale_features
        ), "a scaler must be used to unscale features"
        graphs_ret = deepcopy(graphs)
        if self.standard_scale_features and self.log_scale_features:
            logger.debug("Unscaling features with log and standard scalers")
            graphs_ret = self.feature_scalers[1].inverse(graphs_ret)
            graphs_ret = self.feature_scalers[0].inverse(graphs_ret)

        elif self.log_scale_features or self.standard_scale_features:
            graphs_ret = self.feature_scalers[0].inverse(graphs_ret)

        return graphs_ret

    def unscale_targets(self, graphs: List[Any]) -> List[Any]:
        """
        Perform inverse standardization of targets on the given graphs.
        Takes:
            graphs: list of PyG HeteroData graphs
        Returns:
            graphs: list of PyG HeteroData graphs with inverse standardized targets
        """
        assert (
            self.log_scale_labels or self.standard_scale_labels
        ), "a scaler must be used to unscale targets"

        graphs_ret = deepcopy(graphs)

        if self.standard_scale_labels and self.log_scale_labels:
            logger.debug("Unscaling targets with log and standard scalers")
            graphs_ret = self.label_scalers[1].inverse(graphs_ret)
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        elif self.log_scale_labels or self.standard_scale_labels:
            if self.log_scale_labels:
                logger.debug("Unscaling targets with log scaler")
            if self.standard_scale_labels:
                logger.debug("Unscaling targets with standard scaler")
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        return graphs_ret


class HeteroGraphGraphLabelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str,
        standard_scale_features: bool = True,
        log_scale_features: bool = True,
        log_scale_targets: bool = False,
        standard_scale_targets: bool = True,
        allowed_ring_size: List[int] = [3, 4, 5, 6, 7],
        element_set: List[str] = [],
        allowed_charges: Optional[List[int]] = None,
        allowed_spins: Optional[List[int]] = None,
        self_loop: bool = True,
        debug: bool = False,
        filter_self_bonds: bool = True,
        size: Optional[int] = None,
        bond_key: str = "bonds",
        map_key: Optional[str] = "bonds",
        extra_keys: Dict[str, List[str]] = {
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["extra_feat_bond_esp_total", "bond_length"],
            "global": ["extra_feat_global_E1_CAM"],
        },
        target_list: List[str] = ["extra_feat_global_E1_CAM"],
        extra_dataset_info: Dict[str, Any] = {},
        num_workers: int = 1,
        verbose: bool = True,
        rbf_cutoff: float = 5.0,
    ):
        """
        Baseline dataset for hetero graph node label prediction. Includes global feautures.
        Args:
            file (string): path to data file
            standard_scale_features (bool): whether to scale features
            log_scale_features (bool): whether to log scale features
            allowed_ring_size (list): list of allowed ring sizes
            allowed_charges (list): list of allowed charges
            self_loop (bool): whether to add self loops to the graph
            debug (bool): whether to run in debug mode
            extra_keys (dict): dictionary of keys to grab from the data file
            target_list (list of strings): dictionary of global keys to use as labels
            extra_dataset_info (dict): dictionary of extra info to be stored in the dataset
        """
        # check if file exists
        if not Path(file).exists():
            raise FileNotFoundError(f"File {file} not found")

        # check if file ends in pkl
        if file[-3:] == "pkl":
            try:
                df = pd.read_pickle(file)
            except:
                import pickle5 as pickle

                with open(file, "rb") as fh:
                    df = pickle.load(fh)

        elif file[-3:] == "json":
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file)

        if debug:
            logger.info("Running in debug mode")
            df = df.head(100)
        else:
            if size != None:
                # randomly sample size number of rows
                df = df.sample(size)

        for key_check in ["atom", "bond", "global"]:
            if key_check not in extra_keys.keys():
                extra_keys[key_check] = []

        logger.debug("pandas df dims: %s", df.shape)

        mol_wrappers, element_set_ret = mol_wrappers_from_df(
            df=df,
            bond_key=bond_key,
            map_key=map_key,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            filter_self_bonds=filter_self_bonds,
        )
        del df

        # legacy used if element_set == None:
        # if element_set == None:
        if element_set == [] or element_set == None:
            self.element_set = sorted(element_set_ret)
        else:
            self.element_set = element_set
        logger.debug("element set: %s", self.element_set)

        grapher = get_grapher(
            element_set=self.element_set,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            allowed_ring_size=allowed_ring_size,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
            self_loop=self_loop,
            rbf_cutoff=rbf_cutoff,
        )

        # Build graphs - use parallel processing if num_workers > 1
        if num_workers > 1 and len(mol_wrappers) > 1:
            graph_list, names = self._build_graphs_parallel(
                mol_wrappers, grapher, num_workers, verbose
            )
        else:
            # Serial graph building (original behavior)
            graph_list = []
            logger.info("Building graphs and featurizing")
            for mol in tqdm(mol_wrappers):
                graph = grapher.build_graph(mol)
                graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
                graph_list.append(graph)

        # Free source data -- all information is now in the graphs.
        del mol_wrappers, grapher
        import gc
        gc.collect()
        # Force glibc to return freed pages to the OS.
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        self.standard_scale_features = standard_scale_features
        self.log_scale_features = log_scale_features
        self.log_scale_labels = log_scale_targets
        self.standard_scale_labels = standard_scale_targets
        self.feature_scalers = []
        self.label_scalers = []
        self.data = None
        self.feat_names = names
        self.allowed_spins = allowed_spins
        self.allowed_charges = allowed_charges
        self.allowed_ring_size = allowed_ring_size
        self.graphs = graph_list
        target_dict = {"global": target_list}
        self.target_dict = target_dict
        self.extra_dataset_info = extra_dataset_info
        self.verbose = verbose

        self.load()
        logger.info("LOADED DATASET")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Any:
        # print(idx)
        return self.graphs[idx]  # , self.labels[idx]

    def _build_graphs_parallel(
        self, mol_wrappers, grapher, num_workers, verbose
    ):
        """
        Build graphs in parallel using multiprocessing.Pool.

        Args:
            mol_wrappers: List of MoleculeWrapper objects
            grapher: HeteroCompleteGraphFromMolWrapper instance
            num_workers: Number of parallel workers
            verbose: Whether to print progress information

        Returns:
            Tuple of (graph_list, feature_names)
        """
        from multiprocessing import Pool
        import multiprocessing as mp
        from qtaim_embed.data.parallel_utils import build_and_featurize_graph_worker

        # Extract grapher config for workers to reconstruct in each process
        # NOTE: when adding new featurizer params, update all 3 _build_graphs_parallel sites
        grapher_config = {
            'element_set': grapher.atom_featurizer.element_set,
            'atom_keys': grapher.atom_featurizer.selected_keys,
            'bond_keys': grapher.bond_featurizer.selected_keys,
            'global_keys': grapher.global_featurizer.selected_keys,
            'allowed_ring_size': grapher.atom_featurizer.allowed_ring_size,
            'allowed_charges': grapher.global_featurizer.allowed_charges,
            'allowed_spins': grapher.global_featurizer.allowed_spins,
            'self_loop': grapher.self_loop,
            'rbf_cutoff': grapher.bond_featurizer.rbf_cutoff,
        }

        # Clamp workers to reasonable bounds
        actual_workers = min(num_workers, len(mol_wrappers), mp.cpu_count())

        # Prepare arguments for workers
        args_list = [
            (mol, grapher_config, idx) for idx, mol in enumerate(mol_wrappers)
        ]

        # Pre-allocate results list
        graph_list = [None] * len(mol_wrappers)
        feat_names = None

        # Process with progress bar
        with Pool(processes=actual_workers) as pool:
            results = pool.imap_unordered(
                build_and_featurize_graph_worker,
                args_list,
                chunksize=max(1, len(args_list) // (actual_workers * 4))
            )

            if verbose:
                logger.info("Building graphs and featurizing (parallel, %d workers)", actual_workers)

            for idx, graph, names in tqdm(
                results,
                total=len(mol_wrappers),
                desc="Building graphs (parallel)",
                disable=not verbose
            ):
                if graph is not None:
                    graph_list[idx] = graph
                    if feat_names is None:  # Store feature names from first successful result
                        feat_names = names
                else:
                    # Log error but continue (error already printed by worker)
                    pass

        if feat_names is None:
            raise RuntimeError(
                "No graphs were successfully built in parallel processing; "
                "feature names could not be determined."
            )

        return graph_list, feat_names


    def get_include_exclude_indices(self) -> None:
        target_locs = {}
        # get locations of target features
        for node_type, value_list in self.target_dict.items():
            if node_type not in target_locs:
                target_locs[node_type] = []

            for value in value_list:
                target_locs[node_type].append(self.feat_names[node_type].index(value))

        # now partition features into feats in target_locs and feats not in target_locs
        include_locs = {}
        exclude_locs = {}
        include_names = {}
        exclude_names = {}

        for node_type, value_list in self.feat_names.items():
            # if node_type not in include_locs:
            #    include_locs[node_type] = []
            #    exclude_locs[node_type] = []
            #    include_names[node_type] = []
            #    exclude_names[node_type] = []

            for i, value in enumerate(value_list):
                if node_type in target_locs.keys():
                    if i in target_locs[node_type]:
                        if node_type not in include_names:
                            include_names[node_type] = []
                            include_locs[node_type] = []

                        include_locs[node_type].append(i)
                        include_names[node_type].append(value)
                    else:
                        if node_type not in exclude_names:
                            exclude_names[node_type] = []
                            exclude_locs[node_type] = []
                        exclude_locs[node_type].append(i)
                        exclude_names[node_type].append(value)
                else:
                    if node_type not in exclude_names:
                        exclude_names[node_type] = []
                        exclude_locs[node_type] = []
                    exclude_locs[node_type].append(i)
                    exclude_names[node_type].append(value)

        self.include_locs = include_locs
        self.exclude_locs = exclude_locs
        self.include_names = include_names
        self.exclude_names = exclude_names
        if self.verbose:
            logger.debug("included in graph features: %s", self.exclude_names)

    def load(self) -> None:
        self.get_include_exclude_indices()
        logger.info("Parsing labels and features in graphs")
        for graph in tqdm(self.graphs):
            labels = {}
            features_new = {}
            ndata_feat = _get_ndata(graph, "feat")
            for key, value in ndata_feat.items():
                if key in self.include_names.keys():
                    features_new[key] = ndata_feat[key][
                        :, self.exclude_locs[key]
                    ]
                    labels[key] = ndata_feat[key][:, self.include_locs[key]]
            _set_ndata(graph, "feat", features_new)
            _set_ndata(graph, "labels", labels)

        if self.log_scale_features:
            logger.info("Log scaling features")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=True, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.feature_scalers.append(log_scaler)
            logger.info("Log scaling features complete")

        if self.standard_scale_features:
            logger.info("Scaling features")
            scaler = HeteroGraphStandardScaler()
            self.graphs = scaler(self.graphs)
            self.feature_scalers.append(scaler)

            if self.verbose:
                logger.debug("feature mean(s): %s", scaler.mean)
                logger.debug("feature std(s): %s", scaler.std)

        if self.log_scale_labels:
            logger.info("Log scaling targets")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=False, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.label_scalers.append(log_scaler)
            logger.info("Log scaling targets complete")

        if self.standard_scale_labels:
            logger.info("Scaling targets")
            scaler = HeteroGraphStandardScaler(features_tf=False)
            self.graphs = scaler(self.graphs)
            self.label_scalers.append(scaler)

            if self.verbose:
                logger.debug("feature mean(s): %s", scaler.mean)
                logger.debug("feature std(s): %s", scaler.std)

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        return self.exclude_names

    @property
    def label_names(self) -> Dict[str, List[str]]:
        return self.include_names

    @property
    def feature_size(self) -> Dict[str, int]:
        len_dict = {}
        for key, value in self.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict

    def unscale_features(self, graphs: List[Any]) -> List[Any]:
        """
        Perform inverse standardization on the given graphs.
        Takes:
            graphs: list of PyG HeteroData graphs
        Returns:
            graphs: list of PyG HeteroData graphs with inverse standardized features
        """
        assert (
            self.log_scale_features or self.standard_scale_features
        ), "a scaler must be used to unscale features"
        graphs_ret = deepcopy(graphs)
        if self.standard_scale_features and self.log_scale_features:
            logger.debug("Unscaling features with log and standard scalers")
            graphs_ret = self.feature_scalers[1].inverse(graphs_ret)
            graphs_ret = self.feature_scalers[0].inverse(graphs_ret)

        elif self.log_scale_features or self.standard_scale_features:
            graphs_ret = self.feature_scalers[0].inverse(graphs_ret)

        return graphs_ret

    def unscale_targets(self, graphs: List[Any]) -> List[Any]:
        """
        Perform inverse standardization of targets on the given graphs.
        Takes:
            graphs: list of PyG HeteroData graphs
        Returns:
            graphs: list of PyG HeteroData graphs with inverse standardized targets
        """
        assert (
            self.log_scale_labels or self.standard_scale_labels
        ), "a scaler must be used to unscale targets"

        graphs_ret = deepcopy(graphs)

        if self.standard_scale_labels and self.log_scale_labels:
            logger.debug("Unscaling targets with log and standard scalers")
            graphs_ret = self.label_scalers[1].inverse(graphs_ret)
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        elif self.log_scale_labels or self.standard_scale_labels:
            if self.log_scale_labels:
                logger.debug("Unscaling targets with log scaler")
            if self.standard_scale_labels:
                logger.debug("Unscaling targets with standard scaler")
            graphs_ret = self.label_scalers[0].inverse(graphs_ret)
        return graphs_ret


class HeteroGraphGraphLabelClassifierDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file: str,
        standard_scale_features: bool = True,
        log_scale_features: bool = True,
        allowed_ring_size: List[int] = [3, 4, 5, 6, 7],
        element_set: List[str] = [],
        allowed_charges: Optional[List[int]] = None,
        allowed_spins: Optional[List[int]] = None,
        self_loop: bool = True,
        debug: bool = False,
        size: Optional[int] = None,
        filter_self_bonds: bool = True,
        extra_keys: Dict[str, List[str]] = {
            "atom": ["extra_feat_atom_esp_total"],
            "bond": ["extra_feat_bond_esp_total", "bond_length"],
            "global": ["NR-AR"],
        },
        bond_key: str = "bonds",
        map_key: Optional[str] = None,
        target_list: List[str] = ["NR-AR"],
        extra_dataset_info: Dict[str, Any] = {},
        impute: bool = False,
        verbose: bool = True,
        num_workers: int = 1,
        rbf_cutoff: float = 5.0,
    ):
        """
        Baseline dataset for hetero graph node label prediction. Includes global feautures.

        Args:
            file (string): path to data file
            standard_scale_features (bool): whether to scale features
            log_scale_features (bool): whether to log scale features
            allowed_ring_size (list): list of allowed ring sizes
            allowed_charges (list): list of allowed charges
            self_loop (bool): whether to add self loops to the graph
            debug (bool): whether to run in debug mode
            extra_keys (dict): dictionary of keys to grab from the data file
            target_list (list of strings): dictionary of global keys to use as labels
            extra_dataset_info (dict): dictionary of extra info to be stored in the dataset
        """
        # check if file ends in pkl
        if file[-3:] == "pkl":
            try:
                df = pd.read_pickle(file)
            except:
                import pickle5 as pickle

                with open(file, "rb") as fh:
                    df = pickle.load(fh)

        elif file[-3:] == "json":
            df = pd.read_json(file)
        else:
            df = pd.read_csv(file)

        if debug:
            logger.info("Running in debug mode")
            df = df.head(100)
        else:
            if size != None:
                # randomly sample size number of rows
                df = df.sample(size)

        for key_check in ["atom", "bond", "global"]:
            if key_check not in extra_keys.keys():
                extra_keys[key_check] = []

        mol_wrappers, element_set_ret = mol_wrappers_from_df(
            df=df,
            bond_key=bond_key,
            map_key=map_key,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            filter_self_bonds=filter_self_bonds,
        )
        del df

        if element_set == [] or element_set == None:
            self.element_set = sorted(element_set_ret)
        else:
            self.element_set = element_set
        logger.debug("element set: %s", self.element_set)
        grapher = get_grapher(
            element_set=self.element_set,
            atom_keys=extra_keys["atom"],
            bond_keys=extra_keys["bond"],
            global_keys=extra_keys["global"],
            allowed_ring_size=allowed_ring_size,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
            self_loop=self_loop,
            rbf_cutoff=rbf_cutoff,
        )

        # Build graphs (parallel or serial)
        if num_workers > 1 and len(mol_wrappers) > 1:
            graph_list, names = self._build_graphs_parallel(
                mol_wrappers, grapher, num_workers, verbose
            )
        else:
            graph_list = []
            if verbose:
                logger.info("Building graphs and featurizing")
            for mol in tqdm(mol_wrappers, disable=not verbose):
                graph = grapher.build_graph(mol)
                graph, names = grapher.featurize(graph, mol, ret_feat_names=True)
                graph_list.append(graph)

        self.standard_scale_features = standard_scale_features
        self.log_scale_features = log_scale_features
        self.feature_scalers = []
        self.data = mol_wrappers  # to be filtered
        self.feat_names = names
        self.graphs = graph_list  # to be filtered
        target_dict = {"global": target_list}
        self.allowed_spins = allowed_spins
        self.allowed_charges = allowed_charges
        self.allowed_ring_size = allowed_ring_size
        self.target_dict = target_dict
        self.extra_dataset_info = extra_dataset_info
        self.impute = impute
        self.verbose = verbose

        self.load()
        logger.info("LOADED DATASET")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Any:
        # print(idx)
        return self.graphs[idx]  # , self.labels[idx]

    def _build_graphs_parallel(self, mol_wrappers, grapher, num_workers, verbose):
        """
        Build graphs in parallel using multiprocessing.Pool.

        Args:
            mol_wrappers: List of MoleculeWrapper objects
            grapher: HeteroCompleteGraphFromMolWrapper instance
            num_workers: Number of parallel workers
            verbose: Whether to print progress

        Returns:
            Tuple of (graph_list, feature_names)
        """
        from multiprocessing import Pool
        import multiprocessing as mp
        from qtaim_embed.data.parallel_utils import build_and_featurize_graph_worker

        # Extract grapher config for workers (avoid pickling complex objects)
        # NOTE: when adding new featurizer params, update all 3 _build_graphs_parallel sites
        grapher_config = {
            'element_set': grapher.atom_featurizer.element_set,
            'atom_keys': grapher.atom_featurizer.selected_keys,
            'bond_keys': grapher.bond_featurizer.selected_keys,
            'global_keys': grapher.global_featurizer.selected_keys,
            'allowed_ring_size': grapher.atom_featurizer.allowed_ring_size,
            'allowed_charges': grapher.global_featurizer.allowed_charges,
            'allowed_spins': grapher.global_featurizer.allowed_spins,
            'self_loop': grapher.self_loop,
            'rbf_cutoff': grapher.bond_featurizer.rbf_cutoff,
        }

        # Clamp workers to reasonable bounds
        actual_workers = min(num_workers, len(mol_wrappers), mp.cpu_count())

        # Prepare arguments
        args_list = [(mol, grapher_config, idx) for idx, mol in enumerate(mol_wrappers)]

        # Pre-allocate results list
        graph_list = [None] * len(mol_wrappers)
        feat_names = None

        # Process with progress bar
        with Pool(processes=actual_workers) as pool:
            results = pool.imap_unordered(
                build_and_featurize_graph_worker,
                args_list,
                chunksize=max(1, len(args_list) // (actual_workers * 4))
            )

            if verbose:
                logger.info("Building graphs and featurizing (parallel, %d workers)", actual_workers)

            for idx, graph, names in tqdm(
                results,
                total=len(mol_wrappers),
                desc="Building graphs (parallel)",
                disable=not verbose
            ):
                if graph is not None:
                    graph_list[idx] = graph
                    if feat_names is None:
                        feat_names = names
        
        if feat_names is None:  
            raise RuntimeError(  
                "Failed to build any graphs in parallel; no feature names were produced. "  
                "Check the input data and grapher configuration."  
            )  
        return graph_list, feat_names

    def get_include_exclude_indices(self) -> None:
        target_locs = {}
        # get locations of target features
        for node_type, value_list in self.target_dict.items():
            if node_type not in target_locs:
                target_locs[node_type] = []

            for value in value_list:
                target_locs[node_type].append(self.feat_names[node_type].index(value))

        # now partition features into feats in target_locs and feats not in target_locs
        include_locs = {}
        exclude_locs = {}
        include_names = {}
        exclude_names = {}

        for node_type, value_list in self.feat_names.items():
            for i, value in enumerate(value_list):
                if node_type in target_locs.keys():
                    if i in target_locs[node_type]:
                        if node_type not in include_names:
                            include_names[node_type] = []
                            include_locs[node_type] = []

                        include_locs[node_type].append(i)
                        include_names[node_type].append(value)
                    else:
                        if node_type not in exclude_names:
                            exclude_names[node_type] = []
                            exclude_locs[node_type] = []
                        exclude_locs[node_type].append(i)
                        exclude_names[node_type].append(value)
                else:
                    if node_type not in exclude_names:
                        exclude_names[node_type] = []
                        exclude_locs[node_type] = []
                    exclude_locs[node_type].append(i)
                    exclude_names[node_type].append(value)

        self.include_locs = include_locs
        self.exclude_locs = exclude_locs
        self.include_names = include_names
        self.exclude_names = exclude_names
        logger.debug("included in labels: %s", self.include_names)
        logger.debug("included in graph features: %s", self.exclude_names)

    def load(self) -> None:
        self.get_include_exclude_indices()
        logger.info("Parsing labels and features in graphs")
        filtered_graph_count = 0
        filter_ind = []
        categories_set_list = [set([]) for i in range(len(self.include_locs["global"]))]
        if self.impute:
            self.mode_dict = {}

        for ind, graph in tqdm(enumerate(self.graphs)):
            labels = {}
            features_new = {}
            ndata_feat = _get_ndata(graph, "feat")
            for key, value in ndata_feat.items():
                if key in self.include_names.keys():
                    features_new[key] = ndata_feat[key][
                        :, self.exclude_locs[key]
                    ]
                    labels_temp = ndata_feat[key][:, self.include_locs[key]]
                    # check if any of the labels are nan
                    if self.impute:
                        for ind_task, val in enumerate(labels_temp[0]):
                            if not torch.isnan(val):
                                if ind_task not in self.mode_dict:
                                    self.mode_dict[ind_task] = [0, 0]
                                self.mode_dict[ind_task][int(val.item())] += 1

                    if torch.isnan(labels_temp).any():
                        filtered_graph_count += 1
                        filter_ind.append(ind)
                        list_labels = labels_temp.tolist()[0]
                        for i, label in enumerate(list_labels):
                            if not np.isnan(label):
                                categories_set_list[i].add(int(label))

                    else:
                        list_labels = labels_temp.tolist()[0]

                        [
                            categories_set_list[i].add(int(list_labels[i]))
                            for i in range(len(self.include_locs["global"]))
                        ]  # update number of categories
                    labels[key] = labels_temp

            _set_ndata(graph, "feat", features_new)
            _set_ndata(graph, "labels", labels)
        if self.impute:
            # convert mode dict to mode
            for key, value in self.mode_dict.items():
                self.mode_dict[key] = value.index(max(value))

        self.categories_set_list = categories_set_list
        dict_distro = {}
        graphs_filtered = []
        mol_wrappers_filtered = []
        for ind, graph in enumerate(self.graphs):
            if self.impute == False:
                if ind not in filter_ind:
                    graphs_filtered.append(graph)
                    mol_wrappers_filtered.append(self.data[ind])
                    # convert labels to one hot
                    labels = []
                    labels_raw = graph["global"].labels

                    if len(categories_set_list) == 1:
                        labels = np.zeros((1, len(categories_set_list[0])))
                    else:
                        labels = np.zeros(
                            (len(categories_set_list), len(categories_set_list[0]))
                        )
                        labels_raw = labels_raw[0]

                    for j, category_set in enumerate(categories_set_list):
                        ind_hot = list(category_set).index(int(labels_raw[j]))
                        if j not in dict_distro:
                            dict_distro[j] = [0 for i in range(len(category_set))]
                        dict_distro[j][ind_hot] += 1
                        if len(categories_set_list) == 1:
                            labels[0, ind_hot] = 1
                        else:
                            labels[j, ind_hot] = 1

                    labels = torch.tensor(np.array([labels]), dtype=torch.int64)
                    _set_ndata(graph, "labels", {"global": labels})
            else:
                graphs_filtered.append(graph)
                mol_wrappers_filtered.append(self.data[ind])
                labels = []
                labels_raw = graph["global"].labels
                if len(categories_set_list) == 1:
                    labels = np.zeros((1, len(categories_set_list[0])))
                else:
                    labels = np.zeros(
                        (len(categories_set_list), len(categories_set_list[0]))
                    )
                    labels_raw = labels_raw[0]
                for j, category_set in enumerate(categories_set_list):
                    if torch.isnan(labels_raw[j]):
                        ind_hot = self.mode_dict[j]
                    else:
                        ind_hot = list(category_set).index(int(labels_raw[j]))

                    if j not in dict_distro:
                        dict_distro[j] = [0 for i in range(len(category_set))]
                    dict_distro[j][ind_hot] += 1
                    if len(categories_set_list) == 1:
                        labels[0, ind_hot] = 1
                    else:
                        labels[j, ind_hot] = 1
                labels = torch.tensor(np.array([labels]), dtype=torch.int64)
                _set_ndata(graph, "labels", {"global": labels})

        self.graphs = graphs_filtered
        self.data = mol_wrappers_filtered
        self.dict_distro = dict_distro
        logger.info("Number of categories for each label:")
        for i, category_set in enumerate(categories_set_list):
            logger.info(
                "label %s: %d with distribution: %s",
                self.include_names["global"][i],
                len(category_set),
                dict_distro[i],
            )
        logger.debug("original loader node types: %s", list(_get_ndata(graph, "feat").keys()))
        logger.debug("original loader label types: %s", list(_get_ndata(graph, "labels").keys()))

        if self.impute:
            logger.info("Number of graphs imputed: %d", filtered_graph_count)
        else:
            logger.info("Number of graphs filtered: %d", filtered_graph_count)

        if self.log_scale_features:
            logger.info("Log scaling features")
            log_scaler = HeteroGraphLogMagnitudeScaler(features_tf=True, shift=1)
            self.graphs = log_scaler(self.graphs)
            self.feature_scalers.append(log_scaler)
            logger.info("Log scaling features complete")

        if self.standard_scale_features:
            logger.info("Scaling features")
            scaler = HeteroGraphStandardScaler()
            self.graphs = scaler(self.graphs)
            self.feature_scalers.append(scaler)
            logger.info("Scaling features complete")
            if self.verbose:
                logger.debug("feature mean(s): %s", scaler.mean)
                logger.debug("feature std(s): %s", scaler.std)

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        return self.exclude_names

    @property
    def label_names(self) -> Dict[str, List[str]]:
        return self.include_names

    @property
    def feature_size(self) -> Dict[str, int]:
        len_dict = {}
        for key, value in self.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict

    def unscale_features(self, graphs: List[Any]) -> List[Any]:
        """
        Perform inverse standardization on the given graphs.
        Takes:
            graphs: list of PyG HeteroData graphs
        Returns:
            graphs: list of PyG HeteroData graphs with inverse standardized features
        """
        assert (
            self.log_scale_features or self.standard_scale_features
        ), "a scaler must be used to unscale features"
        graph_ret = deepcopy(graphs)
        if self.standard_scale_features and self.log_scale_features:
            logger.debug("Unscaling features with log and standard scalers")
            graph_ret = self.feature_scalers[1].inverse(graph_ret)
            graph_ret = self.feature_scalers[0].inverse(graph_ret)

        elif self.log_scale_features or self.standard_scale_features:
            graph_ret = self.feature_scalers[0].inverse(graph_ret)

        return graph_ret


class LMDBBaseDataset(Dataset):
    def __init__(self, config: Dict[str, Any], transform: Optional[Any] = None):
        super(LMDBBaseDataset, self).__init__()

        self.config = config
        self.path = Path(self.config["src"])
        self._worker_env = None
        self._worker_envs = None

        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"
            self._db_paths = list(db_paths)

            self._keys = []
            # Open envs in main process to read metadata; kept for property access.
            # __getitem__ lazily reopens per-worker to support num_workers > 0.
            self.envs = []
            for db_path in db_paths:
                cur_env = self.connect_db(db_path)
                self.envs.append(cur_env)

                # If "length" encoded as ascii is present, use that
                length_entry = cur_env.begin().get("length".encode("ascii"))
                if length_entry is not None:
                    num_entries = pickle.loads(length_entry)
                else:
                    # Get the number of stores data from the number of entries in the LMDB
                    num_entries = cur_env.stat()["entries"]

                # Append the keys (0->num_entries) as a list
                self._keys.append(list(range(num_entries)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)

        else:
            self._db_paths = None
            # Open env in main process for metadata; kept for property access.
            self.env = self.connect_db(self.path)

            # If "length" encoded as ascii is present, use that
            # If there are additional properties, there must be length.
            length_entry = self.env.begin().get("length".encode("ascii"))
            if length_entry is not None:
                num_entries = pickle.loads(length_entry)
            else:
                # Get the number of stores data from the number of entries
                # in the LMDB
                num_entries = self.env.stat()["entries"]

            self._keys = list(range(num_entries))
            self.num_samples = num_entries

        # Get portion of total dataset
        self.sharded = False
        if "shard" in self.config and "total_shards" in self.config:
            self.sharded = True
            self.indices = range(self.num_samples)
            # split all available indices into 'total_shards' bins
            self.shards = np.array_split(
                self.indices, self.config.get("total_shards", 1)
            )
            # limit each process to see a subset of data based off defined shard
            self.available_indices = self.shards[self.config.get("shard", 0)]
            self.num_samples = len(self.available_indices)

        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Any:
        # if sharding, remap idx to appropriate idx of the sharded set
        if self.sharded:
            idx = self.available_indices[idx]

        if not self.path.is_file():
            # Lazy per-worker init: each DataLoader worker opens its own env
            # after fork, avoiding shared-handle conflicts with max_readers > 1.
            if self._worker_envs is None:
                self._worker_envs = [self.connect_db(p) for p in self._db_paths]

            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self._worker_envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)

        else:
            # Lazy per-worker init: each DataLoader worker opens its own env.
            if self._worker_env is None:
                self._worker_env = self.connect_db(self.path)

            datapoint_pickled = self._worker_env.begin().get(
                f"{self._keys[idx]}".encode("ascii")
            )

            data_object = pickle.loads(datapoint_pickled)

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path: Optional[str] = None) -> lmdb.Environment:
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=126,
        )
        return env

    def close_db(self) -> None:
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

    def _close_worker_envs(self) -> None:
        """Close lazily-opened per-worker LMDB environments."""
        if self._worker_envs is not None:
            for env in self._worker_envs:
                env.close()
            self._worker_envs = None
        if self._worker_env is not None:
            self._worker_env.close()
            self._worker_env = None

    def __del__(self) -> None:
        self._close_worker_envs()

    def get_metadata(self, num_samples: int = 100) -> None:
        pass


class LMDBMoleculeDataset(LMDBBaseDataset):
    def __init__(self, config: Dict[str, Any], transform: Optional[Any] = None):
        super(LMDBMoleculeDataset, self).__init__(config=config, transform=transform)
        if not self.path.is_file():
            self.single_file = False

        elif self.path.is_file():
            # self.env = self.env
            self.single_file = True

    @property
    def allowed_charges(self) -> Any:
        if not self.single_file:
            return pickle.loads(
                self.envs[0].begin().get("allowed_charges".encode("ascii"))
            )
        return pickle.loads(self.env.begin().get("allowed_charges".encode("ascii")))

    @property
    def allowed_spins(self) -> Any:
        if not self.single_file:
            return pickle.loads(
                self.envs[0].begin().get("allowed_spins".encode("ascii"))
            )
        return pickle.loads(self.env.begin().get("allowed_spins".encode("ascii")))

    @property
    def ring_size_set(self) -> Any:
        if not self.single_file:
            return pickle.loads(
                self.envs[0].begin().get("ring_size_set".encode("ascii"))
            )
        return pickle.loads(self.env.begin().get("ring_size_set".encode("ascii")))

    @property
    def element_set(self) -> Any:
        if not self.single_file:
            return pickle.loads(self.envs[0].begin().get("element_set".encode("ascii")))
        return pickle.loads(self.env.begin().get("element_set".encode("ascii")))

    @property
    def feature_names(self) -> Any:
        if not self.single_file:
            feature_names = self.envs[0].begin().get("feature_names".encode("ascii"))
        else:
            feature_names = self.env.begin().get("feature_names".encode("ascii"))
        return pickle.loads(feature_names)

    @property
    def feature_size(self) -> Any:
        if not self.single_file:
            feature_size = self.envs[0].begin().get("feature_size".encode("ascii"))
        else:
            feature_size = self.env.begin().get("feature_size".encode("ascii"))
        return pickle.loads(feature_size)

    @property
    def target_dict(self) -> Any:
        if not self.single_file:
            target_dict = self.envs[0].begin().get("target_dict".encode("ascii"))
        else:
            target_dict = self.env.begin().get("target_dict".encode("ascii"))
        return pickle.loads(target_dict)


class Subset(Dataset):
    def __init__(self, dataset: Any, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def feature_names(self) -> Dict[str, List[str]]:
        return self.dataset.exclude_names

    @property
    def label_names(self) -> Dict[str, List[str]]:
        return self.dataset.include_names

    @property
    def feature_size(self) -> Dict[str, int]:
        len_dict = {}
        for key, value in self.dataset.exclude_names.items():
            len_dict[key] = len(value)
        return len_dict


class SubsetLMDB(Dataset):
    def __init__(self, dataset: Any, indices: Sequence[int]):
        self.dtype = dataset.dtype
        self.dataset = dataset
        self.indices = indices

        self.feature_size = dataset.feature_size
        self.feature_names = dataset.feature_names

        self.element_set = dataset.element_set
        self.log_scale_features = dataset.log_scale_features
        self.allowed_charges = dataset.allowed_charges
        self.allowed_spins = dataset.allowed_spins
        self.allowed_ring_size = dataset.allowed_ring_size
        self.target_dict = dataset.target_dict
        self.extra_dataset_info = dataset.extra_dataset_info

        self.graphs = dataset.graphs

    @property
    def feature_size(self) -> Any:
        return self._feature_size

    @property
    def feature_name(self) -> Any:
        return self._feature_name

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)
