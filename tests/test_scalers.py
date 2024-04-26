import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx
import torch
from qtaim_embed.utils.tests import get_dataset


def test_log_scaler():
    dataset_log = get_dataset(
        log_scale_features=True,
        log_scale_targets=True,
        standard_scale_features=False,
        standard_scale_targets=False,
    )
    # generate 10 values without replacement between 0 and len(dataset)
    idxs = np.random.choice(len(dataset_log), 10, replace=False)
    feat_scaler = dataset_log.feature_scalers[0]
    label_scaler = dataset_log.label_scalers[0]

    # get the graphs at those indices
    graphs_test = [dataset_log.graphs[idx] for idx in idxs]
    node_types_feats = list(graphs_test[0].ndata["feat"].keys())
    node_types_labels = list(graphs_test[0].ndata["labels"].keys())

    # get the node features at those indices
    scaled_node_feats = deepcopy([graph.ndata["feat"] for graph in graphs_test])
    # get the targets at those indices
    scaled_target_feats = deepcopy([graph.ndata["labels"] for graph in graphs_test])

    # graphs_unscaled = feat_scaler.inverse(graphs_test)
    # graphs_unscaled = label_scaler.inverse(graphs_unscaled)
    graphs_unscaled = dataset_log.unscale_features(graphs_test)
    graphs_unscaled = dataset_log.unscale_targets(graphs_unscaled)
    # get the node features at those indices
    unscaled_node_feats = deepcopy([graph.ndata["feat"] for graph in graphs_unscaled])
    # get the targets at those indices
    unscaled_target_feats = deepcopy(
        [graph.ndata["labels"] for graph in graphs_unscaled]
    )

    # test forward scaling - feats
    for feat_ind in range(len(unscaled_node_feats)):
        for nt in node_types_feats:
            feats = unscaled_node_feats[feat_ind][nt]
            test_scale_sign = torch.sign(feats)
            test_scale = torch.log(torch.abs(feats) + feat_scaler.shift)
            test_scale = test_scale * test_scale_sign
            assert torch.allclose(scaled_node_feats[feat_ind][nt], test_scale)
    # test forward scaling - targets
    for label_ind in range(len(unscaled_target_feats)):
        for nt in node_types_labels:
            feats = unscaled_target_feats[label_ind][nt]
            test_scale_sign = torch.sign(feats)
            test_scale = torch.log(torch.abs(feats) + feat_scaler.shift)
            test_scale = test_scale * test_scale_sign
            assert torch.allclose(scaled_target_feats[label_ind][nt], test_scale)

    # test inverse utility - feats
    for feat_ind in range(len(scaled_node_feats)):
        for nt in node_types_feats:
            # manually do unscale scaled features
            feats = scaled_node_feats[feat_ind][nt]
            sign_feats = torch.sign(feats)
            feats_abs = torch.abs(feats)
            test_scale = torch.exp(feats_abs) - feat_scaler.shift
            test_scale = test_scale * sign_feats
            torch.allclose(unscaled_node_feats[feat_ind][nt], test_scale)

    # test inverse utility - targts
    for label_ind in range(len(scaled_target_feats)):
        for nt in node_types_labels:
            feats = scaled_target_feats[label_ind][nt]
            sign_feats = torch.sign(feats)
            feats_abs = torch.abs(feats)
            test_scale = torch.exp(feats_abs) - feat_scaler.shift
            test_scale = test_scale * sign_feats
            torch.allclose(unscaled_target_feats[label_ind][nt], test_scale)

    print("log scaler test passed!")


def test_standard_scaler():
    dataset_standard = get_dataset(
        log_scale_features=False,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )

    # generate 10 values without replacement between 0 and len(dataset)
    idxs = np.random.choice(len(dataset_standard), 10, replace=False)
    # get the graphs at those indices
    graphs_test = [dataset_standard.graphs[idx] for idx in idxs]
    node_types_feats = list(graphs_test[0].ndata["feat"].keys())
    node_types_labels = list(graphs_test[0].ndata["labels"].keys())

    # get the node features at those indices
    scaled_node_feats = deepcopy([graph.ndata["feat"] for graph in graphs_test])
    # get the targets at those indices
    scaled_target_feats = deepcopy([graph.ndata["labels"] for graph in graphs_test])

    feat_scaler = dataset_standard.feature_scalers[0]
    label_scaler = dataset_standard.label_scalers[0]

    # graphs_unscaled = feat_scaler.inverse(graphs_test)
    graphs_unscaled = dataset_standard.unscale_features(graphs_test)
    # graphs_unscaled = label_scaler.inverse(graphs_unscaled)
    graphs_unscaled = dataset_standard.unscale_targets(graphs_unscaled)
    # get the node features at those indices
    unscaled_node_feats = deepcopy([graph.ndata["feat"] for graph in graphs_unscaled])
    # get the targets at those indices
    unscaled_target_feats = deepcopy(
        [graph.ndata["labels"] for graph in graphs_unscaled]
    )

    for feat_ind in range(len(unscaled_target_feats)):
        for nt in node_types_labels:
            feats = unscaled_target_feats[feat_ind][nt]
            test_scale = (feats - label_scaler._mean[nt]) / label_scaler._std[nt]
            assert torch.allclose(scaled_target_feats[feat_ind][nt], test_scale)

    for feat_ind in range(len(unscaled_node_feats)):
        for nt in node_types_feats:
            feats = unscaled_node_feats[feat_ind][nt]
            test_scale = (feats - feat_scaler._mean[nt]) / feat_scaler._std[nt]

            assert torch.allclose(
                scaled_node_feats[feat_ind][nt], test_scale, atol=1e-5
            )

    # now test inverse function
    for feat_ind in range(len(scaled_node_feats)):
        for nt in node_types_feats:
            # manually do unscale scaled features
            feats = scaled_node_feats[feat_ind][nt]
            test_scale = feats * feat_scaler._std[nt] + feat_scaler._mean[nt]
            torch.allclose(unscaled_node_feats[feat_ind][nt], test_scale)

    for feat_ind in range(len(scaled_target_feats)):
        for nt in node_types_labels:
            # manuall do unscale scaled labels
            feats = scaled_target_feats[feat_ind][nt]
            test_scale = feats * label_scaler._std[nt] + label_scaler._mean[nt]
            torch.allclose(unscaled_target_feats[feat_ind][nt], test_scale)

    print("standard scaler test passed!")


def test_standard_log_scaler():
    dataset_standard_log = get_dataset(
        log_scale_features=True,
        log_scale_targets=True,
        standard_scale_features=True,
        standard_scale_targets=True,
    )
    # generate 10 values without replacement between 0 and len(dataset)
    idxs = np.random.choice(len(dataset_standard_log), 10, replace=False)
    # get the graphs at those indices
    graphs_test = [dataset_standard_log.graphs[idx] for idx in idxs]
    node_types_feats = list(graphs_test[0].ndata["feat"].keys())
    node_types_labels = list(graphs_test[0].ndata["labels"].keys())

    # get the node features at those indices
    scaled_node_feats = deepcopy([graph.ndata["feat"] for graph in graphs_test])
    # get the targets at those indices
    scaled_target_feats = deepcopy([graph.ndata["labels"] for graph in graphs_test])

    feat_scalers = dataset_standard_log.feature_scalers
    label_scalers = dataset_standard_log.label_scalers

    graphs_unscaled = dataset_standard_log.unscale_features(graphs_test)
    graphs_unscaled = dataset_standard_log.unscale_targets(graphs_unscaled)
    # graphs_unscaled = feat_scalers[1].inverse(graphs_unscaled)
    # graphs_unscaled = label_scalers[0].inverse(graphs_unscaled)
    # graphs_unscaled = label_scalers[1].inverse(graphs_unscaled)

    # get the node features at those indices
    unscaled_node_feats = deepcopy([graph.ndata["feat"] for graph in graphs_unscaled])
    # get the targets at those indices
    unscaled_target_feats = deepcopy(
        [graph.ndata["labels"] for graph in graphs_unscaled]
    )

    for feat_ind in range(len(unscaled_node_feats)):
        for nt in node_types_feats:
            feats = unscaled_node_feats[feat_ind][nt]
            # log scale
            test_scale_sign = torch.sign(feats)
            test_scale = torch.log(torch.abs(feats) + feat_scalers[0].shift)
            test_scale = test_scale * test_scale_sign
            # standard scale
            test_scale = (test_scale - feat_scalers[1]._mean[nt]) / feat_scalers[
                1
            ]._std[nt]

            assert torch.allclose(
                scaled_node_feats[feat_ind][nt], test_scale, atol=1e-5
            )

    # test forward scaling - targets
    for label_ind in range(len(unscaled_target_feats)):
        for nt in node_types_labels:
            feats = unscaled_target_feats[label_ind][nt]
            test_scale_sign = torch.sign(feats)
            test_scale = torch.log(torch.abs(feats) + feat_scalers[0].shift)
            test_scale = test_scale * test_scale_sign
            test_scale = (test_scale - label_scalers[1]._mean[nt]) / label_scalers[
                1
            ]._std[nt]
            # print max erro
            assert torch.allclose(
                scaled_target_feats[label_ind][nt], test_scale, atol=1e-5
            )
