from torch.utils.data import DataLoader
import dgl
from dgl.sampling import global_uniform_negative_sampling
from qtaim_embed.data.transforms import DropBondHeterograph, hetero_to_homo, homo_to_hetero


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


class DataLoaderLinkTaskHeterograph(DataLoader):
    """
    Dataloader for link tasks. Use normal datasets from qtaim-embed but they are converted to homographs
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
                batched_graphs = dgl.batch(graphs)
                graphs = self.transforms(batched_graphs)

            #graphs = samples
            transformer = hetero_to_homo(concat_global=True)
            # convert to homographs
            graphs_hetero_to_homo = [transformer(i) for i in graphs]
            # get number of edges 
            
            # get negative samples
            graphs_negative = [get_negative_graph(graphs_hetero_to_homo[i], graphs_hetero_to_homo[i].num_edges()) for i in range(len(graphs_hetero_to_homo))]

            #source, dest = global_uniform_negative_sampling(g, num_samples = self.k)
             
            batched_graphs = dgl.batch(graphs_hetero_to_homo)
            batched_negative_graphs = dgl.batch(graphs_negative)
            feat = batched_graphs.ndata['ft']
            
            return batched_graphs, batched_negative_graphs, feat
            

        super(DataLoaderLinkTaskHeterograph, self).__init__(
            dataset, collate_fn=collate, **kwargs
        )


def get_negative_graph(graph_pos, k):
    """
    Given a positive graph, generate a randomly sampled negative graph.
    Takes:
        graph_pos(dgl.graph): positive graph
        k(int): number of negative samples
    Returns:
        dgl.graph: negative graph
    """
    source, dest = global_uniform_negative_sampling(
        graph_pos, num_samples = k, replace=False
    )
    
    if int(source.shape[0]) != k:
        source, dest = global_uniform_negative_sampling(
            graph_pos, num_samples = k, replace=False, redundancy=10.0
        )   
    return dgl.graph((source, dest), num_nodes=graph_pos.num_nodes())
    