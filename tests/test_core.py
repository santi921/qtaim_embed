import pandas as pd
import dgl
import networkx as nx
import numpy as np
import torch

from qtaim_embed.core.molwrapper import MoleculeWrapper
from qtaim_embed.utils.descriptors import get_atom_feats, get_bond_features
from qtaim_embed.data.grapher import HeteroCompleteGraphFromMolWrapper
from qtaim_embed.data.featurizer import (
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,
)
from qtaim_embed.core.datamodule import QTAIMNodeTaskDataModule

from qtaim_embed.core.dataset import HeteroGraphNodeLabelDataset


def test_molwrapper():
    # TODO
    pass


def test_dataset():
    # TODO
    pass


def test_node_datamodule():
    dm = QTAIMNodeTaskDataModule()
    print(dm.config)
    feature_size, feat_name = dm.prepare_data("fit")
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()


def test_graph_datamodule():
    dm = QTAIMNodeTaskDataModule()
    print(dm.config)
    feature_size, feat_name = dm.prepare_data("fit")
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
