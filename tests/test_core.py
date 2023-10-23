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


def test_datamodule():
    dm = QTAIMNodeTaskDataModule()
    print(dm.config)
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()


test_datamodule()
