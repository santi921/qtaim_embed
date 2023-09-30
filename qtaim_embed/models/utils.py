import torch 


def _split_batched_output(graph, value, key):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.

    Returns:
        list of tensor.

    """
    n_nodes = graph.batch_num_nodes(key)
    #convert to tuple 
    n_nodes = tuple(n_nodes)
    return torch.split(value, n_nodes)