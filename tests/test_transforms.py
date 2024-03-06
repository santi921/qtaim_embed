import torch 
import dgl

from qtaim_embed.data.dataloader import DataLoaderMoleculeNodeTask
from qtaim_embed.data.transforms import DropBondHeterograph
from qtaim_embed.utils.tests import get_dataset_graph_level

def test_edge_dropout():

    drop_edge = DropBondHeterograph(p=0.8)

    dataset_graph_level = get_dataset_graph_level(
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )

    batch_loader = DataLoaderMoleculeNodeTask(
        dataset=dataset_graph_level,
        transforms=drop_edge,
        shuffle=False,
    )

    for ind, batch in enumerate(batch_loader):
        assert dataset_graph_level.graphs[ind].num_nodes(ntype='bond') > batch[0].num_nodes(ntype='bond')
    

test_edge_dropout()