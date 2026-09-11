import logging
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.utils import negative_sampling
from qtaim_embed.data.transforms import hetero_to_homo

logger = logging.getLogger(__name__)


def _get_ndata(data, key):
    """Helper to get feature dict from a PyG HeteroData batch."""
    return {nt: getattr(data[nt], key) for nt in data.node_types if hasattr(data[nt], key)}


# node-store tensors concatenated by the direct collate; anything else on a
# store is ignored (the model reads feat/labels/pos/z and batch only)
_COLLATE_NODE_KEYS = ("feat", "labels", "pos", "z")


def collate_hetero_direct(graphs: Sequence[HeteroData]) -> HeteroData:
    """Batch HeteroData graphs of one schema without PyG's generic collate.

    Batch.from_data_list introspects every attribute of every store and
    records slice/increment bookkeeping for to_data_list; at batch 1024 that
    is 160 ms of worker CPU per batch on tm_react (E5), about the GPU step
    time. This concatenates the known node tensors, offsets edge_index by the
    per-type node pointers, and sets `batch`, `ptr`, and `num_graphs`, which
    is everything the models, pooling layers, and _split_batched_output read.
    Output is a HeteroData, not a Batch: to_data_list is not available.
    Falls back to Batch.from_data_list if the graphs disagree on schema.
    """
    try:
        return _collate_hetero_direct(graphs)
    except (KeyError, AttributeError, RuntimeError) as exc:
        logger.warning("direct collate failed (%s); falling back to Batch.from_data_list", exc)
        return Batch.from_data_list(graphs)


def _collate_hetero_direct(graphs: Sequence[HeteroData]) -> HeteroData:
    out = HeteroData()
    g0 = graphs[0]
    n_graphs = len(graphs)
    ptr = {}
    for nt in g0.node_types:
        stores = [g[nt] for g in graphs]
        counts = torch.tensor([s.num_nodes for s in stores])
        cum = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        ptr[nt] = cum
        present = tuple(k for k in _COLLATE_NODE_KEYS if k in stores[0])
        for st in stores[1:]:
            if tuple(k for k in _COLLATE_NODE_KEYS if k in st) != present:
                raise KeyError(f"node type {nt!r}: graphs disagree on {_COLLATE_NODE_KEYS}")
        for key in present:
            out[nt][key] = torch.cat([s[key] for s in stores], 0)
        out[nt].num_nodes = int(cum[-1])
        out[nt].batch = torch.repeat_interleave(torch.arange(n_graphs), counts)
        out[nt].ptr = cum
    for et in g0.edge_types:
        src_t, _, dst_t = et
        eis = [g[et].edge_index for g in graphs]
        n_e = torch.tensor([e.shape[1] for e in eis])
        off = torch.stack([ptr[src_t][:-1], ptr[dst_t][:-1]])
        out[et].edge_index = torch.cat(eis, 1) + torch.repeat_interleave(off, n_e, dim=1)
    out.num_graphs = n_graphs
    return out


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

    def __init__(self, dataset, transforms=None, dense_shape_of=None, with_labels=True, **kwargs):
        """dense_shape_of: optional callable (atom_counts, bond_counts) -> (N_b, B_b)
        stamped on each batch as `dense_shape` (BucketBatchSampler.batch_shape),
        so ResidualBlockDense pads every batch of a shape class identically.
        with_labels=False yields the batched HeteroData alone (bond classifier:
        candidates and labels are built in the model step from atom.pos / atom.z
        and the a2b connectivity)."""
        kwargs.pop("collate_fn", None)
        self.transforms = transforms
        self.dense_shape_of = dense_shape_of
        self.with_labels = with_labels

        def collate(samples):
            graphs = samples
            if self.transforms is not None:
                graphs = [self.transforms(graph) for graph in graphs]

            batched_graphs = collate_hetero_direct(graphs)
            if self.dense_shape_of is not None:
                batched_graphs.dense_shape = self.dense_shape_of(
                    [int(g["atom"].num_nodes) for g in graphs],
                    [int(g["bond"].num_nodes) for g in graphs],
                )
            if not self.with_labels:
                return batched_graphs
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
        logger.debug("DataLoaderLinkTaskHeterograph")
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
