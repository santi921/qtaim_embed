import torch 
from dgl.transforms import BaseTransform
from torch.distributions import Bernoulli


class DropBondHeterograph(BaseTransform):
    r"""Randomly drop edges, as described in
    `DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
    <https://arxiv.org/abs/1907.10903>`__ and `Graph Contrastive Learning with Augmentations
    <https://arxiv.org/abs/2010.13902>`__.

    Takes
    ----------
    p : float, optional
        Probability of an edge to be dropped.
    drop_node_type : list, optional
        list of node types to drop
    drop_edge_types : list, optional
        list of edge types to drop that correspond to the node types in drop_node_type

    """

    def __init__(self, p=0.5, drop_node_type=["bond"], drop_edge_types=["a2b", "b2a", "g2b", "b2g", "b2b"]):
        self.p = p
        self.dist = Bernoulli(p)
        self.drop_node_type = drop_node_type
        self.drop_edge_types = drop_edge_types

    def __call__(self, g):
        g = g.clone()
        # Fast path
        if self.p == 0:
            return g
        
        for c_etype in self.drop_node_type:
            samples = self.dist.sample(torch.Size([g.num_nodes(c_etype)]))
            node_ids_to_remove = g.nodes(ntype=c_etype)[samples.bool()]
            g.remove_nodes(node_ids_to_remove, ntype=c_etype)

        return g
