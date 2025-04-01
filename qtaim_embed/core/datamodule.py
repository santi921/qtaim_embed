import os
import pytorch_lightning as pl
from qtaim_embed.data.dataloader import (
    DataLoaderMoleculeNodeTask,
    DataLoaderLinkTaskHeterograph,
    DataLoaderMoleculeGraphTask,
    DataLoaderLMDB,
    DataLoaderLinkLMDB,
)

from qtaim_embed.utils.data import (
    get_default_node_level_config,
    get_default_link_level_config,
    get_default_graph_level_config,
)
from qtaim_embed.core.dataset import (
    HeteroGraphNodeLabelDataset,
    HeteroGraphGraphLabelDataset,
    HeteroGraphGraphLabelClassifierDataset,
    LMDBMoleculeDataset,
)
from qtaim_embed.utils.data import train_validation_test_split
from qtaim_embed.data.transforms import DropBondHeterograph
from qtaim_embed.data.lmdb import TransformMol


class QTAIMLinkTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()
        self.prepare_tf = False
        if config == None:
            self.config = get_default_node_level_config()
        else:
            self.config = config

        if "edge_dropout" not in self.config["dataset"].keys():
            print("... > no edge dropout on datamodule")
            self.transforms = None
        elif type(self.config["dataset"]["edge_dropout"]) != float:
            print("... > no edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                print("... > using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

        self.node_len = None

    def setup(self, stage: str):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def prepare_data(self, stage=None):
        if self.prepare_tf == False:
            if stage == "fit" or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.full_dataset = HeteroGraphNodeLabelDataset(
                    file=self.config["dataset"]["train_dataset_loc"],
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    allowed_spins=self.config["dataset"]["allowed_spins"],
                    self_loop=self.config["dataset"]["self_loop"],
                    element_set=self.config["dataset"]["element_set"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_dict=self.config["dataset"]["target_dict"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    log_scale_targets=self.config["dataset"]["log_scale_targets"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    standard_scale_targets=self.config["dataset"][
                        "standard_scale_targets"
                    ],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                    verbose=self.config["dataset"]["verbose"],
                )
                validation = self.config["dataset"]["val_prop"]
                test_size = self.config["dataset"]["test_prop"]

                if test_size > 0.0:
                    (
                        self.train_dataset,
                        self.val_dataset,
                        self.test_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=test_size,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )
                else:
                    print("... > no test set in datamodule")
                    (
                        self.train_dataset,
                        self.val_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=0.0,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )

                self.prepare_tf = True

                if self.node_len is None:
                    feat_dict = self.train_dataset.feature_size
                    self.node_len = feat_dict["atom"] + feat_dict["global"]

                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )

            if stage == "test" or stage == "predict":
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"
                self.test_dataset = HeteroGraphNodeLabelDataset(
                    file=self.test_dataset_loc,
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    allowed_spins=self.config["dataset"]["allowed_spins"],
                    self_loop=self.config["dataset"]["self_loop"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_dict=self.config["dataset"]["target_dict"],
                    element_set=self.config["dataset"]["element_set"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    log_scale_targets=self.config["dataset"]["log_scale_targets"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                    standard_scale_targets=self.config["dataset"][
                        "standard_scale_targets"
                    ],
                    verbose=self.config["dataset"]["verbose"],
                )
                self.prepare_tf = True
                if self.node_len is None:
                    feat_dict = self.test_dataset.feature_size
                    self.node_len = feat_dict["atom"] + feat_dict["global"]

                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

        else:
            if stage == "fit" or stage is None:
                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )
            else:
                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

    def train_dataloader(self):
        dl = DataLoaderLinkTaskHeterograph(
            self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            transforms=self.transforms,
        )
        if self.node_len is None:
            _, _, ft = next(iter(dl))
            self.node_len = ft.shape[1]
        return dl

    def val_dataloader(self):
        dl = DataLoaderLinkTaskHeterograph(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            transforms=self.transforms,
        )

        if self.node_len is None:
            _, _, ft = next(iter(dl))
            self.node_len = ft.shape[1]
        return dl

    def test_dataloader(self):
        dl = DataLoaderLinkTaskHeterograph(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            transforms=self.transforms,
        )

        if self.node_len is None:
            _, _, ft = next(iter(dl))
            self.node_len = ft.shape[1]
        return dl


class QTAIMNodeTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()
        self.prepare_tf = False
        if config == None:
            self.config = get_default_node_level_config()
        else:
            self.config = config

        if "edge_dropout" not in self.config["dataset"].keys():
            print("... > no edge dropout on datamodule")
            self.transforms = None
        elif type(self.config["dataset"]["edge_dropout"]) != float:
            print("... > no edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                print("... > using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

    def setup(self, stage: str):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def prepare_data(self, stage=None):
        if self.prepare_tf == False:
            if stage == "fit" or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.full_dataset = HeteroGraphNodeLabelDataset(
                    file=self.config["dataset"]["train_dataset_loc"],
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    allowed_spins=self.config["dataset"]["allowed_spins"],
                    self_loop=self.config["dataset"]["self_loop"],
                    element_set=self.config["dataset"]["element_set"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_dict=self.config["dataset"]["target_dict"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    log_scale_targets=self.config["dataset"]["log_scale_targets"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    standard_scale_targets=self.config["dataset"][
                        "standard_scale_targets"
                    ],
                    verbose=self.config["dataset"]["verbose"],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                )

                validation = self.config["dataset"]["val_prop"]
                test_size = self.config["dataset"]["test_prop"]

                if test_size > 0.0:
                    (
                        self.train_dataset,
                        self.val_dataset,
                        self.test_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=test_size,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )
                else:
                    print("... > no test set in datamodule")
                    (
                        self.train_dataset,
                        self.val_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=0.0,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )

                self.prepare_tf = True
                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )

            if stage == "test" or stage == "predict":
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"
                self.test_dataset = HeteroGraphNodeLabelDataset(
                    file=self.test_dataset_loc,
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    allowed_spins=self.config["dataset"]["allowed_spins"],
                    self_loop=self.config["dataset"]["self_loop"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_dict=self.config["dataset"]["target_dict"],
                    element_set=self.config["dataset"]["element_set"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    log_scale_targets=self.config["dataset"]["log_scale_targets"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    standard_scale_targets=self.config["dataset"][
                        "standard_scale_targets"
                    ],
                    verbose=self.config["dataset"]["verbose"],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                )
                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

        else:
            if stage == "fit" or stage is None:
                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )
            else:
                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

    def train_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            transforms=self.transforms,
        )

    def val_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            transforms=self.transforms,
        )

    def test_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            transforms=self.transforms,
        )


class QTAIMGraphTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()
        self.prepare_tf = False
        if config == None:
            print("no config passed - using default on data module")
            self.config = get_default_graph_level_config()
        else:
            self.config = config

        if "edge_dropout" not in self.config["dataset"].keys():
            print("... > no edge dropout on datamodule")
            self.transforms = None
        elif type(self.config["dataset"]["edge_dropout"]) != float:
            print("... > no edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                print("... > using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

    def setup(self, stage: str):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def prepare_data(self, stage=None):
        if self.prepare_tf == False:
            if stage == "fit" or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.full_dataset = HeteroGraphGraphLabelDataset(
                    file=self.config["dataset"]["train_dataset_loc"],
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    allowed_spins=self.config["dataset"]["allowed_spins"],
                    self_loop=self.config["dataset"]["self_loop"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    element_set=self.config["dataset"]["element_set"],
                    target_list=self.config["dataset"]["target_list"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    log_scale_targets=self.config["dataset"]["log_scale_targets"],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    standard_scale_targets=self.config["dataset"][
                        "standard_scale_targets"
                    ],
                    verbose=self.config["dataset"]["verbose"],
                )

                validation = self.config["dataset"]["val_prop"]
                test_size = self.config["dataset"]["test_prop"]

                if test_size > 0.0:
                    (
                        self.train_dataset,
                        self.val_dataset,
                        self.test_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=test_size,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )
                    print("training set size: ", len(self.train_dataset))
                    print("validation set size: ", len(self.val_dataset))
                    print("test set size: ", len(self.test_dataset))

                else:
                    print("... > no test set in datamodule")
                    (
                        self.train_dataset,
                        self.val_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=0.0,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )
                    print("training set size: ", len(self.train_dataset))
                    print("validation set size: ", len(self.val_dataset))

                self.prepare_tf = True
                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )

            if stage == "test" or stage == "predict":
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"

                self.test_dataset = HeteroGraphGraphLabelDataset(
                    file=self.config["dataset"]["test_dataset_loc"],
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    allowed_spins=self.config["dataset"]["allowed_spins"],
                    self_loop=self.config["dataset"]["self_loop"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_list=self.config["dataset"]["target_list"],
                    element_set=self.config["dataset"]["element_set"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    log_scale_targets=self.config["dataset"]["log_scale_targets"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    standard_scale_targets=self.config["dataset"][
                        "standard_scale_targets"
                    ],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                    verbose=self.config["dataset"]["verbose"],
                )

                print("test set size: ", len(self.test_dataset))
                self.test_dataset.feature_size
                self.test_dataset.feature_names

                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

        else:
            if stage == "fit" or stage is None:
                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )
            else:
                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

    def train_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=True,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def val_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def test_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )


class QTAIMGraphTaskClassifyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()
        self.prepare_tf = False
        if config == None:
            print("no config passed - using default on data module")
            self.config = get_default_graph_level_config()
        else:
            self.config = config
        # print(type(self.config["dataset"]["edge_dropout"]))

        if "edge_dropout" not in self.config["dataset"].keys():
            print("... > no edge dropout on datamodule")
            self.transforms = None
        elif type(self.config["dataset"]["edge_dropout"]) != float:
            print("... > no edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                print("... > using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    dropout=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

    def setup(self, stage: str):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def prepare_data(self, stage=None):
        if self.prepare_tf == False:
            if stage == "fit" or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.full_dataset = HeteroGraphGraphLabelClassifierDataset(
                    file=self.config["dataset"]["train_dataset_loc"],
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    self_loop=self.config["dataset"]["self_loop"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_list=self.config["dataset"]["target_list"],
                    element_set=self.config["dataset"]["element_set"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    impute=self.config["dataset"]["impute"],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                    verbose=self.config["dataset"]["verbose"],
                )

                validation = self.config["dataset"]["val_prop"]
                test_size = self.config["dataset"]["test_prop"]

                if test_size > 0.0:
                    (
                        self.train_dataset,
                        self.val_dataset,
                        self.test_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=test_size,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )
                    print("training set size: ", len(self.train_dataset))
                    print("validation set size: ", len(self.val_dataset))
                    print("test set size: ", len(self.test_dataset))

                else:
                    print("... > no test set in datamodule")
                    (
                        self.train_dataset,
                        self.val_dataset,
                    ) = train_validation_test_split(
                        self.full_dataset,
                        test=0.0,
                        validation=validation,
                        random_seed=self.config["dataset"]["seed"],
                    )
                    print("training set size: ", len(self.train_dataset))
                    print("validation set size: ", len(self.val_dataset))

                self.prepare_tf = True

                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )

            if stage == "test" or stage == "predict":
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"

                self.test_dataset = HeteroGraphGraphLabelDataset(
                    file=self.test_dataset_loc,
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    self_loop=self.config["dataset"]["self_loop"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_list=self.config["dataset"]["target_list"],
                    element_set=self.config["dataset"]["element_set"],
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                    bond_key=self.config["dataset"]["bond_key"],
                    map_key=self.config["dataset"]["map_key"],
                    verbose=self.config["dataset"]["verbose"],
                )
                print("test set size: ", len(self.test_dataset))

                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

        else:
            if stage == "fit" or stage is None:
                return (
                    self.train_dataset.feature_names,
                    self.train_dataset.feature_size,
                )
            else:
                return (
                    self.test_dataset.feature_names,
                    self.test_dataset.feature_size,
                )

    def train_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=True,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def val_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def test_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )


class LMDBDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.train_lmdb_loc = config["dataset"]["train_lmdb"]

        if "val_lmdb" in self.config["dataset"]:
            self.val_lmdb_loc = config["dataset"]["val_lmdb"]

        if "test_lmdb" in self.config["dataset"]:
            self.test_lmdb_loc = config["dataset"]["test_lmdb"]

        self.prepare_tf = False

        if "edge_dropout" not in self.config["dataset"].keys():
            print("... > no edge dropout on datamodule")
            self.transforms = None
        elif type(self.config["dataset"]["edge_dropout"]) != float:
            print("... > no edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                print("... > using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

    def prepare_data(self, stage=None):
        
        if "test_lmdb" in self.config["dataset"]:
            # check if there is a single lmdb, if so use it, else leave the folder
            check_file = os.path.join(self.test_lmdb_loc, "molecule.lmdb")
            if not os.path.exists(check_file):
                check_file = self.test_lmdb_loc

            self.test_dataset = LMDBMoleculeDataset(
                config={"src": check_file},
                transform=TransformMol,
            )

        if "val_lmdb" in self.config["dataset"]:
            check_file = os.path.join(self.val_lmdb_loc, "molecule.lmdb")
            if not os.path.exists(check_file):
                check_file = self.val_lmdb_loc

            self.val_dataset = LMDBMoleculeDataset(
                config={"src": check_file},
                transform=TransformMol,
            )
        
        check_file = os.path.join(self.train_lmdb_loc, "molecule.lmdb")
        if not os.path.exists(check_file):
            check_file = self.train_lmdb_loc

        self.train_dataset = LMDBMoleculeDataset(
            config={"src": check_file},
            transform=TransformMol,
        )

        return self.train_dataset.feature_names, self.train_dataset.feature_size

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def train_dataloader(self):
        return DataLoaderLMDB(
            dataset=self.train_ds,
            batch_size=self.config["optim"]["train_batch_size"],
            shuffle=True,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=self.config["optim"]["pin_memory"],
            persistent_workers=self.config["optim"]["persistent_workers"],
        )

    def test_dataloader(self):
        return DataLoaderLMDB(
            dataset=self.test_ds,
            batch_size=len(self.test_ds),
            shuffle=False,
            num_workers=self.config["optim"]["num_workers"],
        )

    def val_dataloader(self):
        return DataLoaderLMDB(
            dataset=self.val_ds,
            batch_size=len(self.val_ds),
            shuffle=False,
            num_workers=self.config["optim"]["num_workers"],
        )


class LMDBLinkDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.train_lmdb_loc = config["dataset"]["train_lmdb"]

        if "val_lmdb" in self.config["dataset"]:
            self.val_lmdb_loc = config["dataset"]["val_lmdb"]

        if "test_lmdb" in self.config["dataset"]:
            self.test_lmdb_loc = config["dataset"]["test_lmdb"]

        self.prepare_tf = False

        if "edge_dropout" not in self.config["dataset"].keys():
            print("... > no edge dropout on datamodule")
            self.transforms = None
        elif type(self.config["dataset"]["edge_dropout"]) != float:
            print("... > no edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                print("... > using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

    def prepare_data(self, stage=None):
        if "test_lmdb" in self.config["dataset"]:
            # check if there is a single lmdb, if so use it, else leave the folder
            check_file = os.path.join(self.test_lmdb_loc, "molecule.lmdb")
            if not os.path.exists(check_file):
                check_file = self.test_lmdb_loc

            self.test_dataset = LMDBMoleculeDataset(
                config={"src": check_file},
                transform=TransformMol,
            )

        if "val_lmdb" in self.config["dataset"]:
            check_file = os.path.join(self.val_lmdb_loc, "molecule.lmdb")
            if not os.path.exists(check_file):
                check_file = self.val_lmdb_loc

            self.val_dataset = LMDBMoleculeDataset(
                config={"src": check_file},
                transform=TransformMol,
            )
        
        check_file = os.path.join(self.train_lmdb_loc, "molecule.lmdb")
        if not os.path.exists(check_file):
            check_file = self.train_lmdb_loc

        self.train_dataset = LMDBMoleculeDataset(
            config={"src": check_file},
            transform=TransformMol,
        )

        return self.train_dataset.feature_names, self.train_dataset.feature_size

    def setup(self, stage):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

            self.train_dl = DataLoaderLinkLMDB(
                dataset=self.train_ds,
                batch_size=self.config["optim"]["train_batch_size"],
                shuffle=True,
                num_workers=self.config["optim"]["num_workers"],
                pin_memory=self.config["optim"]["pin_memory"],
                persistent_workers=self.config["optim"]["persistent_workers"],
            )

            self.val_dl = DataLoaderLinkLMDB(
                dataset=self.val_ds,
                batch_size=len(self.val_ds),
                shuffle=False,
                num_workers=self.config["optim"]["num_workers"],
            )

            _, _, ft = next(iter(self.train_dl))
            self.node_len = ft.shape[1]

        if stage in ("test", "predict"):
            self.test_dl = DataLoaderLinkLMDB(
                dataset=self.test_ds,
                batch_size=len(self.test_ds),
                shuffle=False,
                num_workers=self.config["optim"]["num_workers"],
            )

            _, _, ft = next(iter(self.test_dl))
            self.node_len = ft.shape[1]

    def train_dataloader(self):
        return self.train_dl

    def test_dataloader(self):
        return self.test_dl

    def val_dataloader(self):
        return self.val_dl
