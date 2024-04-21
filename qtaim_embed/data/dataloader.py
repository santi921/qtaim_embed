from torch.utils.data import DataLoader
import dgl


class DataLoaderMoleculeNodeTask(DataLoader):
    """
    Dataloader for node-level tasks. labels are in the "label" key of node data
    This assumes a heterograph dataset
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally', you need not to provide one"
            )
        self.transforms = transforms

        def collate(samples):
            graphs = samples
            batched_graphs = dgl.batch(graphs)
            if self.transforms is not None:
                batched_graphs = self.transforms(batched_graphs)
            # batched_labels = [graph.ndata["labels"] for graph in graphs]
            batched_labels = batched_graphs.ndata["labels"]
            return batched_graphs, batched_labels

        super(DataLoaderMoleculeNodeTask, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


class DataLoaderMoleculeGraphTask(DataLoader):
    """
    Dataloader for node-level tasks. labels are in the "label" key of node data
    This assumes a heterograph dataset
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally', you need not to provide one"
            )
        
        self.transforms = transforms

        def collate(samples):
            graphs = samples
            batched_graphs = dgl.batch(graphs)
            if self.transforms is not None:
                batched_graphs = self.transforms(batched_graphs)
            # batched_labels = [graph.ndata["labels"] for graph in graphs]
            batched_labels = batched_graphs.ndata["labels"]
            return batched_graphs, batched_labels

        super(DataLoaderMoleculeGraphTask, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


class DataLoaderLMDB(DataLoader):
    """
    Dataloader for node-level tasks. labels are in the "label" key of node data
    This assumes a heterograph dataset
    """

    def __init__(self, dataset, transforms=None, **kwargs):
        if "collate_fn" in kwargs:
            raise ValueError(
                "'collate_fn' provided internally', you need not to provide one"
            )
        self.transforms = transforms

        def collate(samples):
            graphs = samples
            if self.transforms is not None:
                graphs = [self.transforms(graph) for graph in graphs]

            batched_graphs = dgl.batch(graphs)
            # batched_labels = [graph.ndata["labels"] for graph in graphs]
            batched_labels = batched_graphs.ndata["labels"]
            return batched_graphs, batched_labels

        super(DataLoaderLMDB, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )
