from torch.utils.data import DataLoader
import dgl


class DataLoaderMoleculeNodeTask(DataLoader):
    """
    Dataloader for node-level tasks. labels are in the "label" key of node data
    This assumes a heterograph dataset
    """

    def __init__(self, dataset, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally', you need not to provide one"
            )

        def collate(samples):
            graphs = samples
            batched_graphs = dgl.batch(graphs)
            # batched_labels = [graph.ndata["labels"] for graph in graphs]
            batched_labels = batched_graphs.ndata["labels"]
            return batched_graphs, batched_labels

        super(DataLoaderMoleculeNodeTask, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )
