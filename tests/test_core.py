import pandas as pd
import numpy as np
import torch

from qtaim_embed.core.datamodule import QTAIMNodeTaskDataModule
from qtaim_embed.utils.tests import get_datasets_graph_level_classifier


def test_molwrapper():
    # TODO
    pass


def test_classifier_dataset():
    # test single class
    dataset_single, dataset_multi = get_datasets_graph_level_classifier(
        log_scale_features=True, standard_scale_features=True
    )

    target_single = ["NR-AR"]
    full_set = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]
    target_multi = ["NR-AR", "SR-p53"]

    df = pd.read_pickle("./data/test_classifier_labelled.pkl")
    list_target_single = df[target_single].values.tolist()
    list_target_multi = df[target_multi].values.tolist()
    # find the number of classes
    # find the number of nan values
    nan_count_single = np.sum(np.isnan(list_target_single))
    nan_count_multi = 0

    for target in list_target_multi:
        if np.any(np.isnan(target)):
            nan_count_multi += 1
    # print(nan_count_single)  # 2
    # print(nan_count_multi)  # 9

    len_dataset_single = len(dataset_single)
    len_dataset_multi = len(dataset_multi)

    assert len_dataset_single == 100 - nan_count_single, "Dataset size mismatch"
    assert len_dataset_multi == 100 - nan_count_multi, "Dataset size mismatch"
    ind_check_shift = 0
    for ind, graph in enumerate(dataset_single.graphs):
        label_single = graph.ndata["labels"]["global"]
        check_val = list_target_single[ind + ind_check_shift]
        un_one_hot = torch.argmax(label_single[0], dim=1)
        if not np.isnan(check_val[0]):
            assert int(un_one_hot.tolist()[0]) == int(
                check_val[0]
            ), "One hot encoding mismatch"
        else:
            ind_check_shift += 1

    graph_ind = 0
    for ind, check_vals_raw in enumerate(list_target_multi):
        label_multi = dataset_multi.graphs[graph_ind].ndata["labels"]["global"]
        check_vals = [i for i in check_vals_raw]
        if not np.isnan(np.array(check_vals)).any():
            check_vals = [int(i) for i in check_vals_raw]

            un_one_hot = torch.argmax(label_multi[0].T, dim=1).tolist()
            un_one_hot = [int(i) for i in un_one_hot]
            assert un_one_hot == check_vals, "One hot encoding mismatch"
            graph_ind += 1


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
