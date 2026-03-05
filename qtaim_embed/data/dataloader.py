import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.utils import negative_sampling
from qtaim_embed.data.transforms import hetero_to_homo


def _get_ndata(data, key):
    """Helper to get feature dict from a PyG HeteroData batch."""
    return {nt: getattr(data[nt], key) for nt in data.node_types if hasattr(data[nt], key)}


class DataLoaderMoleculeNodeTask(DataLoader):
    """
    Dataloader for node-level tasks. Labels are in the "labels" attribute of node data.
    This assumes a heterograph dataset.
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        kwargs.pop("collate_fn", None)
        self.transforms = transforms

        def collate(samples):
            graphs = samples
            batched_graphs = Batch.from_data_list(graphs)
            if self.transforms is not None:
                batched_graphs = self.transforms(batched_graphs)
            batched_labels = _get_ndata(batched_graphs, "labels")
            return batched_graphs, batched_labels

        super(DataLoaderMoleculeNodeTask, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


class DataLoaderMoleculeGraphTask(DataLoader):
    """
    Dataloader for graph-level tasks. Labels are in the "labels" attribute of node data.
    This assumes a heterograph dataset.
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        kwargs.pop("collate_fn", None)

        self.transforms = transforms

        def collate(samples):
            graphs = samples
            batched_graphs = Batch.from_data_list(graphs)
            if self.transforms is not None:
                batched_graphs = self.transforms(batched_graphs)
            batched_labels = _get_ndata(batched_graphs, "labels")
            return batched_graphs, batched_labels

        super(DataLoaderMoleculeGraphTask, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


class DataLoaderLMDB(DataLoader):
    """
    Dataloader for LMDB-backed datasets. Labels are in the "labels" attribute of node data.
    This assumes a heterograph dataset.
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        kwargs.pop("collate_fn", None)
        self.transforms = transforms

        def collate(samples):
            graphs = samples
            if self.transforms is not None:
                graphs = [self.transforms(graph) for graph in graphs]

            batched_graphs = Batch.from_data_list(graphs)
            batched_labels = _get_ndata(batched_graphs, "labels")
            return batched_graphs, batched_labels

        super(DataLoaderLMDB, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderLinkLMDB(DataLoader):
    """
    Dataloader for link prediction tasks from LMDB-backed datasets.
    Converts heterographs to homographs and generates negative samples.
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        kwargs.pop("collate_fn", None)
        self.transforms = transforms
        self.transformer = hetero_to_homo(concat_global=True)

        def collate(samples):
            graphs = samples
            if self.transforms is not None:
                graphs = [self.transforms(graph) for graph in graphs]
            # normal graphs
            graphs_hetero_to_homo = [self.transformer(i) for i in graphs]
            # negative graphs
            graphs_negative = [
                get_negative_graph(
                    graphs_hetero_to_homo[i], graphs_hetero_to_homo[i].edge_index.size(1)
                )
                for i in range(len(graphs_hetero_to_homo))
            ]

            batched_graphs = Batch.from_data_list(graphs_hetero_to_homo)
            batched_negative_graphs = Batch.from_data_list(graphs_negative)

            feat = batched_graphs.ft

            return batched_graphs, batched_negative_graphs, feat

        super(DataLoaderLinkLMDB, self).__init__(dataset, collate_fn=collate, **kwargs)


class DataLoaderLinkTaskHeterograph(DataLoader):
    """
    Dataloader for link tasks. Use normal datasets from qtaim-embed but they
    are converted to homographs.
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        print("DataLoaderLinkTaskHeterograph")
        kwargs.pop("collate_fn", None)
        self.transforms = transforms
        self.transformer = hetero_to_homo(concat_global=True)

        def collate(samples):
            graphs = samples
            if self.transforms is not None:
                batched_graphs = Batch.from_data_list(graphs)
                graphs = self.transforms(batched_graphs)

            # convert to homographs
            graphs_hetero_to_homo = [self.transformer(i) for i in graphs]

            # get negative samples
            graphs_negative = [
                get_negative_graph(
                    graphs_hetero_to_homo[i], graphs_hetero_to_homo[i].edge_index.size(1)
                )
                for i in range(len(graphs_hetero_to_homo))
            ]

            batched_graphs = Batch.from_data_list(graphs_hetero_to_homo)
            batched_negative_graphs = Batch.from_data_list(graphs_negative)

            feat = batched_graphs.ft

            return batched_graphs, batched_negative_graphs, feat

        super(DataLoaderLinkTaskHeterograph, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


def get_negative_graph(graph_pos, k):
    """
    Given a positive graph, generate a randomly sampled negative graph.

    Args:
        graph_pos: PyG Data object (homogeneous positive graph)
        k: number of negative edge samples

    Returns:
        PyG Data object representing the negative graph
    """
    num_nodes = graph_pos.num_nodes
    neg_edge_index = negative_sampling(
        edge_index=graph_pos.edge_index,
        num_nodes=num_nodes,
        num_neg_samples=k,
    )

    # If we didn't get enough, try with force=True
    if neg_edge_index.size(1) < k:
        neg_edge_index = negative_sampling(
            edge_index=graph_pos.edge_index,
            num_nodes=num_nodes,
            num_neg_samples=k,
            force_undirected=False,
        )

    return Data(edge_index=neg_edge_index, num_nodes=num_nodes)


def get_negative_graph_explicit(graph_pos):
    """
    Given a positive graph, get the negative graph (all non-existing edges).
    """
    edge_index = graph_pos.edge_index
    num_nodes = graph_pos.num_nodes
    num_edges = edge_index.size(1)

    source = edge_index[0]
    dest = edge_index[1]

    # get all possible edges
    all_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            all_edges.append((i, j))
    # remove positive edges
    for i in range(num_edges):
        all_edges.remove((int(source[i]), int(dest[i])))

    # get negative edges
    negative_edges = all_edges[:num_edges]
    neg_src = [e[0] for e in negative_edges]
    neg_dst = [e[1] for e in negative_edges]
    neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long)

    negative_graph = Data(edge_index=neg_edge_index, num_nodes=num_nodes)

    return negative_graph, negative_edges
