import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx
import torch
from qtaim_embed.utils.tests import get_dataset
from qtaim_embed.data.processing import HeteroGraphStandardScalerIterative, merge_scalers


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


def test_iterative_standard_scaler():
    nt_feats = ["atom", "bond", "global"]
    nt_labels = ["global"]

    dataset_standard = get_dataset(
        log_scale_features=False,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )

    dataset_raw = get_dataset(
        log_scale_features=False,
        log_scale_targets=False,
        standard_scale_features=False,
        standard_scale_targets=False,
    )

    dataset_raw_for_iterative = get_dataset(
        log_scale_features=False,
        log_scale_targets=False,
        standard_scale_features=False,
        standard_scale_targets=False,
    )
    # HeteroGraphStandardScalerIterative

    # compute in one go
    # print(dataset_raw.label_scalers, dataset_raw.feature_scalers)

    feature_scaler = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    feature_scaler.update(dataset_raw.graphs)
    feature_scaler.finalize()

    label_scaler = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )
    label_scaler.update(dataset_raw.graphs)
    label_scaler.finalize()

    # assert that label mean, std are equal - for each node type
    for nt in nt_labels:
        assert torch.allclose(
            label_scaler._mean[nt], dataset_standard.label_scalers[0]._mean[nt].to(torch.float64)
        ), "label mean not equal"
        assert torch.allclose(
            label_scaler._std[nt], dataset_standard.label_scalers[0]._std[nt].to(torch.float64)
        ), "label std not equal"

    # assert that feature mean, std are equal
    for nt in nt_feats:
        assert torch.allclose(
            feature_scaler._mean[nt], dataset_standard.feature_scalers[0]._mean[nt].to(torch.float64)
        ), "feature mean not equal"
        assert torch.allclose(
            feature_scaler._std[nt],
            dataset_standard.feature_scalers[0]._std[nt].to(torch.float64),
            atol=0.001,
        ), "feature std not equal"

    #print("iterative standard scaler!")
    # compute iteratively through list of graphs

    feature_scaler_iterative = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )

    label_scaler_iterative = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )

    #print("dataset_raw_for_iterative.graphs", len(dataset_raw_for_iterative.graphs))
    test_global_labels = []
    # read in 4 graphs at a time
    for i in range(0, len(dataset_raw_for_iterative.graphs), 25):
        feature_scaler_iterative.update(dataset_raw_for_iterative.graphs[i : i + 25])
        label_scaler_iterative.update(dataset_raw_for_iterative.graphs[i : i + 25])
    feature_scaler_iterative.finalize()    
    label_scaler_iterative.finalize()

    for i in range(0, len(dataset_raw_for_iterative.graphs), 25):
        label_scaler_iterative.update(dataset_raw_for_iterative.graphs[i : i + 25])
    label_scaler_iterative.finalize()

    # assert that label mean, std are equal - for each node type
    for nt in nt_labels:
        
        assert torch.allclose(
            label_scaler_iterative._mean[nt],
            dataset_standard.label_scalers[0]._mean[nt].to(torch.float64),
        ), "label mean not equal"
        
        assert torch.allclose(
            label_scaler_iterative._std[nt], dataset_standard.label_scalers[0]._std[nt].to(torch.float64),
            atol=0.001
        ), "label std not equal"

    # assert that feature mean, std are equal
    for nt in nt_feats:
        #print(f"Testing label scaler for node type: {nt}")
        #print("feature_scaler_iterative._mean[nt]", feature_scaler_iterative._mean[nt])
        #print("dataset_standard.feature_scalers[0]._mean[nt]", dataset_standard.feature_scalers[0]._mean[nt])
        
        assert torch.allclose(
            feature_scaler_iterative._mean[nt],
            dataset_standard.feature_scalers[0]._mean[nt].to(torch.float64),
        ), "feature mean not equal"
        assert torch.allclose(
            feature_scaler_iterative._std[nt],
            dataset_standard.feature_scalers[0]._std[nt].to(torch.float64),
            atol=0.001,
        ), "feature std not equal"


    # test saving 
    # save the scalers
    label_scaler_iterative.save_scaler(
        "data/scalers/test_scalers_label_scaler_iterative.json"
    )
    feature_scaler_iterative.save_scaler(
        "data/scalers/test_scalers_feature_scaler_iterative.json"
    )
    # load the scalers
    label_scaler_iterative_loaded = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}, load=True, load_path="data/scalers/test_scalers_label_scaler_iterative.json"
    )
    
    feature_scaler_iterative_loaded = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}, load=True, load_path="data/scalers/test_scalers_feature_scaler_iterative.json"
    )

    # assert that label mean, std are equal - for each node type
    for nt in nt_labels:
        assert torch.allclose(
            label_scaler_iterative_loaded._mean[nt],
            dataset_standard.label_scalers[0]._mean[nt].to(torch.float64),
        ), "label mean not equal"
        assert torch.allclose(
            label_scaler_iterative_loaded._std[nt],
            dataset_standard.label_scalers[0]._std[nt].to(torch.float64),
        ), "label std not equal"


    # assert that feature mean, std are equal
    for nt in nt_feats:
        assert torch.allclose(
            feature_scaler_iterative_loaded._mean[nt],
            dataset_standard.feature_scalers[0]._mean[nt].to(torch.float64),
        ), "feature mean not equal"
        assert torch.allclose(
            feature_scaler_iterative_loaded._std[nt],
            dataset_standard.feature_scalers[0]._std[nt].to(torch.float64),
            atol=0.001,
        ), "feature std not equal"


def test_iterative_standard_scaler_merge(): 
    nt_feats = ["atom", "bond", "global"]
    nt_labels = ["global"]

    dataset_standard = get_dataset(
        log_scale_features=False,
        log_scale_targets=False,
        standard_scale_features=True,
        standard_scale_targets=True,
    )

    dataset_raw_for_iterative = get_dataset(
        log_scale_features=False,
        log_scale_targets=False,
        standard_scale_features=False,
        standard_scale_targets=False,
    )

    # compute iteratively through list of graphs
    feature_scaler_iterative = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    label_scaler_iterative = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )
    # TO MERGE
    feature_scaler_iterative_1 = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    label_scaler_iterative_1 = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )
    feature_scaler_iterative_2 = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}
    )
    label_scaler_iterative_2 = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}
    )

    # read in 4 graphs at a time
    for i in range(0, len(dataset_raw_for_iterative.graphs), 25):
        feature_scaler_iterative.update(dataset_raw_for_iterative.graphs[i : i + 25])
        label_scaler_iterative.update(dataset_raw_for_iterative.graphs[i : i + 25])

    feature_scaler_iterative_1.update(
        dataset_raw_for_iterative.graphs[0 : len(dataset_raw_for_iterative.graphs) // 2]
    )
    label_scaler_iterative_1.update(
        dataset_raw_for_iterative.graphs[0 : len(dataset_raw_for_iterative.graphs) // 2]
    )
    feature_scaler_iterative_2.update(
        dataset_raw_for_iterative.graphs[len(dataset_raw_for_iterative.graphs) // 2 :]
    )
    label_scaler_iterative_2.update(
        dataset_raw_for_iterative.graphs[len(dataset_raw_for_iterative.graphs) // 2 :]
    )

    feature_scaler_iterative.finalize()
    label_scaler_iterative.finalize()
    feature_scaler_iterative_1.finalize()
    label_scaler_iterative_1.finalize()
    feature_scaler_iterative_2.finalize()
    label_scaler_iterative_2.finalize()

    feature_scaler_merged = merge_scalers(
        [feature_scaler_iterative_1, feature_scaler_iterative_2]
    )

    label_scaler_merged = merge_scalers(
        [label_scaler_iterative_1, label_scaler_iterative_2]
    )

    # assert that label mean, std are equal - for each node type
    for nt in nt_labels:
        assert torch.allclose(
            label_scaler_merged._mean[nt],
            dataset_standard.label_scalers[0]._mean[nt].to(torch.float64),
        ), "label mean not equal"
        
        #print("label_scaler_merged._std[nt]", label_scaler_merged._std[nt])
        #print("dataset_standard.label_scalers[0]._std[nt]", dataset_standard.label_scalers[0]._std[nt])
        
        assert torch.allclose(
            label_scaler_merged._std[nt], 
            dataset_standard.label_scalers[0]._std[nt].to(torch.float64),
            atol=0.01,
        ), "label std not equal"

    # assert that feature mean, std are equal
    for nt in nt_feats:
        assert torch.allclose(
            feature_scaler_merged._mean[nt],
            dataset_standard.feature_scalers[0]._mean[nt].to(torch.float64),
        ), "feature mean not equal"
        
        #print("feature_scaler_merged._std[nt]", feature_scaler_merged._std[nt])
        #print("dataset_standard.feature_scalers[0]._std[nt]", dataset_standard.feature_scalers[0]._std[nt])
        print(feature_scaler_merged._std[nt] - dataset_standard.feature_scalers[0]._std[nt].to(torch.float64))
        #print(feature_scaler_merged._std[nt].shape, dataset_standard.feature_scalers[0]._std[nt].to(torch.float64).shape)
        assert torch.allclose(
            feature_scaler_merged._std[nt],
            dataset_standard.feature_scalers[0]._std[nt].to(torch.float64),
            atol=0.01,
            rtol=0.01,
        ), "feature std not equal"
    
    
    # test saving 
    # save the scalers
    label_scaler_iterative_1.save_scaler(
        "data/scalers/test_scalers_label_scaler_iterative_1.json"
    )
    feature_scaler_iterative_1.save_scaler(
        "data/scalers/test_scalers_feature_scaler_iterative_1.json"
    )

    label_scaler_iterative_2.save_scaler(
        "data/scalers/test_scalers_label_scaler_iterative_2.json"
    )
    feature_scaler_iterative_2.save_scaler(
        "data/scalers/test_scalers_feature_scaler_iterative_2.json"
    )

    # load the scalers
    label_scaler_iterative_loaded_1 = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}, load=True, load_path="data/scalers/test_scalers_label_scaler_iterative_1.json"
    )
    
    feature_scaler_iterative_loaded_1 = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}, load=True, load_path="data/scalers/test_scalers_feature_scaler_iterative_1.json"
    )

    feature_scaler_iterative_loaded_2 = HeteroGraphStandardScalerIterative(
        features_tf=True, mean={}, std={}, load=True, load_path="data/scalers/test_scalers_feature_scaler_iterative_2.json"
    )

    label_scaler_iterative_loaded_2 = HeteroGraphStandardScalerIterative(
        features_tf=False, mean={}, std={}, load=True, load_path="data/scalers/test_scalers_label_scaler_iterative_2.json"
    )

    merged_feature_scaler = merge_scalers([feature_scaler_iterative_loaded_1, feature_scaler_iterative_loaded_2])
    merged_label_scaler = merge_scalers([label_scaler_iterative_loaded_1, label_scaler_iterative_loaded_2])


    # assert that label mean, std are equal - for each node type
    for nt in nt_labels:
        assert torch.allclose(
            merged_label_scaler._mean[nt],
            dataset_standard.label_scalers[0]._mean[nt].to(torch.float64),
        ), "label mean not equal"
        assert torch.allclose(
            merged_label_scaler._std[nt], 
            dataset_standard.label_scalers[0]._std[nt].to(torch.float64),
            atol=0.001,
        ), "label std not equal"

    # assert that feature mean, std are equal
    for nt in nt_feats:
        assert torch.allclose(
            merged_feature_scaler._mean[nt],
            dataset_standard.feature_scalers[0]._mean[nt].to(torch.float64),
        ), "feature mean not equal"
        assert torch.allclose(
            merged_feature_scaler._std[nt],
            dataset_standard.feature_scalers[0]._std[nt].to(torch.float64),
            atol=0.01,
            rtol=0.01,
        ), "feature std not equal"
    
#test_iterative_standard_scaler()
#test_iterative_standard_scaler_merge()