import torch
import dgl

from qtaim_embed.data.dataloader import DataLoaderMoleculeNodeTask
from qtaim_embed.data.transforms import DropBondHeterograph
from qtaim_embed.utils.tests import get_dataset_graph_level
from qtaim_embed.data.transforms import (
    DropBondHeterograph,
    hetero_to_homo,
    homo_to_hetero,
)
from qtaim_embed.utils.grapher import compare_graphs


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
        assert dataset_graph_level.graphs[ind].num_nodes(ntype="bond") > batch[
            0
        ].num_nodes(ntype="bond")


def test_homo_conversion():
    dataset_graph_level = get_dataset_graph_level(
        log_scale_features=True,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )

    batch_loader = DataLoaderMoleculeNodeTask(
        dataset=dataset_graph_level,
        shuffle=False,
    )

    for _, batch in enumerate(batch_loader):
        g = batch[0]
        transformer = hetero_to_homo(concat_global=True)
        g_homo = transformer(g)
        transform_back = homo_to_hetero(transformer.global_feat_len)
        g_back = transform_back(g_homo)
        compare_graphs(g, g_back)
