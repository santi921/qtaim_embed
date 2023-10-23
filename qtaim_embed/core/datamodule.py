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
)
from qtaim_embed.utils.data import train_validation_test_split


class QTAIMNodeTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()
        self.setup_tf = False
        if config == None:
            self.config = get_default_node_level_config()
        else:
            self.config = config

    def setup(self, stage: str):
        if self.setup_tf == False:
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
                self.setup_tf = True
                return self.train_dataset.feature_size()

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
                self.setup_tf = True
                return self.test_dataset.feature_size()

        else:
            if stage == "fit":
                return self.train_dataset.feature_size()
            else:
                return self.test_dataset.feature_size()

    def train_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.train_dataset, batch_size=self.config["train_batch_size"]
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
        self.setup_tf = False
        if config == None:
            self.config = get_default_graph_level_config()
        else:
            self.config = config

    def setup(self, stage: str):
        if self.setup_tf == False:
            if stage == "fit" or stage is None:
                # Assign train/val datasets for use in dataloaders
                self.full_dataset = HeteroGraphGraphLabelDataset(
                    file=self.config["train_dataset_loc"],
                    allowed_ring_size=self.config["allowed_ring_size"],
                    allowed_charges=self.config["allowed_charges"],
                    self_loop=self.config["self_loop"],
                    extra_keys=self.config["extra_keys"],
                    target_list=self.config["target_list"],
                    extra_dataset_info=self.config["extra_dataset_info"],
                    debug=self.config["debug"],
                    log_scale_features=self.config["log_scale_features"],
                    log_scale_targets=self.config["log_scale_targets"],
                    standard_scale_features=self.config["standard_scale_features"],
                    standard_scale_targets=self.config["standard_scale_targets"],
                )

                # train test split
                # train_size = int(
                #    (1 - self.config["val_prop"] - self.config["test_prop"])
                #    * len(full_dataset)
                # )
                validation = self.config["val_prop"]
                test_size = self.config["test_prop"]

                (
                    self.train_dataset,
                    self.val_dataset,
                    self.test_dataset,
                ) = train_validation_test_split(
                    self.full_dataset,
                    test=test_size,
                    validation=validation,
                    random_seed=self.config["seed"],
                )
                self.setup_tf = True
                return self.train_dataset.feature_size()

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
                self.setup_tf = True
                return self.test_dataset.feature_size()

        else:
            if stage == "fit":
                return self.train_dataset.feature_size()
            else:
                return self.test_dataset.feature_size()

    def train_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            self.val_dataset, batch_size=len(self.val_dataset), shuffle=False
        )

    def test_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            self.test_dataset, batch_size=len(self.test_dataset), shuffle=False
        )
