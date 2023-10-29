import pytorch_lightning as pl
from torch.utils.data import random_split
from torch import Generator
from qtaim_embed.data.dataloader import (
    DataLoaderMoleculeNodeTask,
    DataLoaderMoleculeGraphTask,
)
from qtaim_embed.utils.data import (
    get_default_node_level_config,
    get_default_graph_level_config,
)
from qtaim_embed.core.dataset import (
    HeteroGraphNodeLabelDataset,
    HeteroGraphGraphLabelDataset,
    HeteroGraphGraphLabelClassifierDataset,
)
from qtaim_embed.utils.data import train_validation_test_split


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

    def setup(self, stage: str):
        if stage in (None, "fit", "validate"):
            self.train_ds = self.train_dataset
            self.val_ds = self.val_dataset

        if stage in ("test", "predict"):
            self.test_ds = self.test_dataset

    def prepare_data(self, stage):
        if self.prepare_tf == False:
            if stage == "fit" or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.full_dataset = HeteroGraphNodeLabelDataset(
                    file=self.config["dataset"]["train_dataset_loc"],
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    self_loop=self.config["dataset"]["self_loop"],
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
                )

                validation = self.config["dataset"]["val_prop"]
                test_size = self.config["dataset"]["test_prop"]

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

                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names(),
                    self.train_dataset.feature_size(),
                )

            if stage == "test" or stage == "predict":
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"
                self.test_dataset = HeteroGraphNodeLabelDataset(
                    file=self.test_dataset_loc,
                    allowed_ring_size=self.config["dataset"]["allowed_ring_size"],
                    allowed_charges=self.config["dataset"]["allowed_charges"],
                    self_loop=self.config["dataset"]["self_loop"],
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
                )
                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names(),
                    self.test_dataset.feature_size(),
                )

        else:
            if stage == "fit":
                return (
                    self.train_dataset.feature_names(),
                    self.train_dataset.feature_size(),
                )
            else:
                return (
                    self.test_dataset.feature_names(),
                    self.test_dataset.feature_size(),
                )

    def train_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.train_dataset, batch_size=self.config["dataset"]["train_batch_size"]
        )

    def val_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.val_dataset, batch_size=len(self.val_dataset)
        )

    def test_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.test_dataset, batch_size=len(self.test_dataset)
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
                    self_loop=self.config["dataset"]["self_loop"],
                    extra_keys=self.config["dataset"]["extra_keys"],
                    target_list=self.config["dataset"]["target_list"],
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
                )

                validation = self.config["dataset"]["val_prop"]
                test_size = self.config["dataset"]["test_prop"]

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
                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names(),
                    self.train_dataset.feature_size(),
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
                )
                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names(),
                    self.train_dataset.feature_size(),
                )

        else:
            if stage == "fit":
                return (
                    self.train_dataset.feature_names(),
                    self.train_dataset.feature_size(),
                )
            else:
                return (
                    self.test_dataset.feature_names(),
                    self.test_dataset.feature_size(),
                )

    def train_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=True,
            num_workers=self.config["dataset"]["num_workers"],
        )

    def val_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
        )

    def test_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
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
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                )

                validation = self.config["dataset"]["val_prop"]
                test_size = self.config["dataset"]["test_prop"]

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
                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names(),
                    self.train_dataset.feature_size(),
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
                    extra_dataset_info=self.config["dataset"]["extra_dataset_info"],
                    debug=self.config["dataset"]["debug"],
                    log_scale_features=self.config["dataset"]["log_scale_features"],
                    standard_scale_features=self.config["dataset"][
                        "standard_scale_features"
                    ],
                )
                self.prepare_tf = True
                return (
                    self.test_dataset.feature_names(),
                    self.train_dataset.feature_size(),
                )

        else:
            if stage == "fit":
                return (
                    self.train_dataset.feature_names(),
                    self.train_dataset.feature_size(),
                )
            else:
                return (
                    self.test_dataset.feature_names(),
                    self.test_dataset.feature_size(),
                )

    def train_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=True,
            num_workers=self.config["dataset"]["num_workers"],
        )

    def val_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
        )

    def test_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.test_dataset,
            batch_size=len(self.test_dataset),
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
        )
