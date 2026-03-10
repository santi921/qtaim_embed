"""
Parallel processing utilities for dataset preprocessing.

This module contains worker functions for multiprocessing-based parallelization
of graph building, featurization, and serialization operations.
"""

import logging

logger = logging.getLogger(__name__)


def build_and_featurize_graph_worker(args_tuple):
    """
    Worker function for parallel graph building and featurization.

    This function must be at module level to be pickleable for multiprocessing.
    It reconstructs the grapher from a configuration dictionary in each worker
    process to avoid pickling issues with complex featurizer objects.

    Args:
        args_tuple: Tuple containing:
            - mol: MoleculeWrapper instance
            - grapher_config: Dictionary with grapher initialization parameters
            - idx: Integer index for maintaining ordering

    Returns:
        Tuple of (idx, graph, feature_names) on success, or
        (idx, None, error_message) on failure
    """
    try:
        mol, grapher_config, idx = args_tuple

        # Import here to ensure workers have access
        from qtaim_embed.utils.grapher import get_grapher

        # Reconstruct grapher from config in worker
        grapher = get_grapher(**grapher_config)

        # Build and featurize graph
        graph = grapher.build_graph(mol)
        graph, names = grapher.featurize(graph, mol, ret_feat_names=True)

        return (idx, graph, names)

    except Exception as e:
        # Return error info without crashing the pool
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error processing molecule {idx}: {e}")
        return (idx, None, error_msg)


def serialize_graph_worker(args_tuple):
    """
    Worker function for parallel graph serialization (LMDB conversion only).

    This function serializes PyG graphs to bytes for storage in LMDB.
    Must be at module level for multiprocessing pickling.

    Args:
        args_tuple: Tuple containing:
            - idx: Integer index for maintaining ordering
            - graph: PyG HeteroData graph to serialize

    Returns:
        Tuple of (idx, serialized_bytes) on success, or
        (idx, None) on failure
    """
    try:
        idx, graph = args_tuple

        # Import here to ensure workers have access
        from qtaim_embed.data.lmdb import serialize_graph

        # Serialize the graph
        serialized = serialize_graph(graph)

        return (idx, serialized)

    except Exception as e:
        # Return None for this graph, log error
        import traceback
        logger.error(f"Error serializing graph {idx}: {e}\n{traceback.format_exc()}")
        return (idx, None)
