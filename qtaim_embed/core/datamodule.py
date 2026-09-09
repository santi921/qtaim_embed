import logging
import os
from pathlib import Path
from functools import partial
import pytorch_lightning as pl
from qtaim_embed.data.dataloader import (
    DataLoaderMoleculeNodeTask,
    DataLoaderLinkTaskHeterograph,
    DataLoaderMoleculeGraphTask,
    DataLoaderLMDB,
    DataLoaderLinkLMDB,
    DataLoaderBondLMDB,
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

logger = logging.getLogger(__name__)


class QTAIMLinkTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()
        if config == None:
            self.config = get_default_node_level_config()
        else:
            self.config = config

        if "edge_dropout" not in self.config["dataset"].keys():
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        elif not isinstance(self.config["dataset"]["edge_dropout"], float):
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                logger.info("Using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

        self.node_len = None
        self._fit_setup_done = False
        self._test_setup_done = False

    def prepare_data(self, stage=None):
        # No-op: in DDP, prepare_data runs only on rank 0.
        # Dataset creation is done in setup() which runs on all ranks.
        pass

    def setup(self, stage=None):
        if stage in (None, "fit", "validate") and not self._fit_setup_done:
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
                num_workers=self.config["dataset"].get("num_workers", 1),
                rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
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
                logger.info("No test set in datamodule")
                (
                    self.train_dataset,
                    self.val_dataset,
                ) = train_validation_test_split(
                    self.full_dataset,
                    test=0.0,
                    validation=validation,
                    random_seed=self.config["dataset"]["seed"],
                )

            if self.node_len is None:
                feat_dict = self.train_dataset.feature_size
                self.node_len = feat_dict["atom"] + feat_dict["global"]

            self._fit_setup_done = True

        if stage in ("test", "predict") and not self._test_setup_done:
            if not hasattr(self, "test_dataset"):
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"
                self.test_dataset = HeteroGraphNodeLabelDataset(
                    file=self.config["dataset"]["test_dataset_loc"],
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
                    num_workers=self.config["dataset"].get("num_workers", 1),
                    rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
                )
            if self.node_len is None:
                feat_dict = self.test_dataset.feature_size
                self.node_len = feat_dict["atom"] + feat_dict["global"]

            self._test_setup_done = True

    def train_dataloader(self):
        dl = DataLoaderLinkTaskHeterograph(
            self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )
        if self.node_len is None:
            _, _, ft = next(iter(dl))
            self.node_len = ft.shape[1]
        return dl

    def val_dataloader(self):
        dl = DataLoaderLinkTaskHeterograph(
            self.val_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

        if self.node_len is None:
            _, _, ft = next(iter(dl))
            self.node_len = ft.shape[1]
        return dl

    def test_dataloader(self):
        dl = DataLoaderLinkTaskHeterograph(
            self.test_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
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
        if config == None:
            self.config = get_default_node_level_config()
        else:
            self.config = config

        if "edge_dropout" not in self.config["dataset"].keys():
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        elif not isinstance(self.config["dataset"]["edge_dropout"], float):
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                logger.info("Using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

        self._fit_setup_done = False
        self._test_setup_done = False

    def prepare_data(self, stage=None):
        # No-op: in DDP, prepare_data runs only on rank 0.
        # Dataset creation is done in setup() which runs on all ranks.
        pass

    def setup(self, stage=None):
        if stage in (None, "fit", "validate") and not self._fit_setup_done:
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
                num_workers=self.config["dataset"].get("num_workers", 1),
                rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
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
                logger.info("No test set in datamodule")
                (
                    self.train_dataset,
                    self.val_dataset,
                ) = train_validation_test_split(
                    self.full_dataset,
                    test=0.0,
                    validation=validation,
                    random_seed=self.config["dataset"]["seed"],
                )

            self._fit_setup_done = True

        if stage in ("test", "predict") and not self._test_setup_done:
            if not hasattr(self, "test_dataset"):
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"
                self.test_dataset = HeteroGraphNodeLabelDataset(
                    file=self.config["dataset"]["test_dataset_loc"],
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
                    num_workers=self.config["dataset"].get("num_workers", 1),
                    rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
                )
            self._test_setup_done = True

    def train_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.train_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def val_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.val_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def test_dataloader(self):
        return DataLoaderMoleculeNodeTask(
            self.test_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )


class QTAIMGraphTaskDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: dict = None,
    ):
        super().__init__()
        if config == None:
            logger.warning("No config passed - using default on data module")
            self.config = get_default_graph_level_config()
        else:
            self.config = config

        if "edge_dropout" not in self.config["dataset"].keys():
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        elif not isinstance(self.config["dataset"]["edge_dropout"], float):
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                logger.info("Using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

        self._fit_setup_done = False
        self._test_setup_done = False

    def prepare_data(self, stage=None):
        # No-op: in DDP, prepare_data runs only on rank 0.
        # Dataset creation is done in setup() which runs on all ranks.
        pass

    def setup(self, stage=None):
        if stage in (None, "fit", "validate") and not self._fit_setup_done:
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
                num_workers=self.config["dataset"].get("num_workers", 1),
                rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
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
                logger.info(f"Training set size: {len(self.train_dataset)}")
                logger.info(f"Validation set size: {len(self.val_dataset)}")
                logger.info(f"Test set size: {len(self.test_dataset)}")

            else:
                logger.info("No test set in datamodule")
                (
                    self.train_dataset,
                    self.val_dataset,
                ) = train_validation_test_split(
                    self.full_dataset,
                    test=0.0,
                    validation=validation,
                    random_seed=self.config["dataset"]["seed"],
                )
                logger.info(f"Training set size: {len(self.train_dataset)}")
                logger.info(f"Validation set size: {len(self.val_dataset)}")

            self._fit_setup_done = True

        if stage in ("test", "predict") and not self._test_setup_done:
            if not hasattr(self, "test_dataset"):
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
                    num_workers=self.config["dataset"].get("num_workers", 1),
                    rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
                )

            logger.info(f"Test set size: {len(self.test_dataset)}")
            self._test_setup_done = True

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
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def test_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.test_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
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
        if config == None:
            logger.warning("No config passed - using default on data module")
            self.config = get_default_graph_level_config()
        else:
            self.config = config

        if "edge_dropout" not in self.config["dataset"].keys():
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        elif not isinstance(self.config["dataset"]["edge_dropout"], float):
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                logger.info("Using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    dropout=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

        self._fit_setup_done = False
        self._test_setup_done = False

    def prepare_data(self, stage=None):
        # No-op: in DDP, prepare_data runs only on rank 0.
        # Dataset creation is done in setup() which runs on all ranks.
        pass

    def setup(self, stage=None):
        if stage in (None, "fit", "validate") and not self._fit_setup_done:
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
                num_workers=self.config["dataset"].get("num_workers", 1),
                rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
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
                logger.info(f"Training set size: {len(self.train_dataset)}")
                logger.info(f"Validation set size: {len(self.val_dataset)}")
                logger.info(f"Test set size: {len(self.test_dataset)}")

            else:
                logger.info("No test set in datamodule")
                (
                    self.train_dataset,
                    self.val_dataset,
                ) = train_validation_test_split(
                    self.full_dataset,
                    test=0.0,
                    validation=validation,
                    random_seed=self.config["dataset"]["seed"],
                )
                logger.info(f"Training set size: {len(self.train_dataset)}")
                logger.info(f"Validation set size: {len(self.val_dataset)}")

            self._fit_setup_done = True

        if stage in ("test", "predict") and not self._test_setup_done:
            if not hasattr(self, "test_dataset"):
                assert (
                    self.config["dataset"]["test_dataset_loc"] is not None
                ), "test_dataset_loc is None"

                self.test_dataset = HeteroGraphGraphLabelClassifierDataset(
                    file=self.config["dataset"]["test_dataset_loc"],
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
                    num_workers=self.config["dataset"].get("num_workers", 1),
                    rbf_cutoff=self.config["dataset"].get("rbf_cutoff", 5.0),
                )
            logger.info(f"Test set size: {len(self.test_dataset)}")
            self._test_setup_done = True

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
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )

    def test_dataloader(self):
        return DataLoaderMoleculeGraphTask(
            dataset=self.test_dataset,
            batch_size=self.config["dataset"]["train_batch_size"],
            shuffle=False,
            num_workers=self.config["dataset"]["num_workers"],
            transforms=self.transforms,
        )


def _resolve_lmdb_path(base_path: str) -> str:
    """Resolve LMDB path, checking for molecule.lmdb inside directory."""
    check_file = os.path.join(base_path, "molecule.lmdb")
    if os.path.exists(check_file):
        return check_file
    return base_path


class LMDBDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.train_lmdb_loc = config["dataset"]["train_lmdb"]

        if "val_lmdb" in self.config["dataset"]:
            self.val_lmdb_loc = config["dataset"]["val_lmdb"]

        if "test_lmdb" in self.config["dataset"]:
            self.test_lmdb_loc = config["dataset"]["test_lmdb"]

        if "edge_dropout" not in self.config["dataset"].keys():
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        elif not isinstance(self.config["dataset"]["edge_dropout"], float):
            logger.info("No edge dropout on datamodule")
            self.transforms = None
        else:
            if self.config["dataset"]["edge_dropout"] > 0.0:
                logger.info("Using edge dropout on datamodule")
                self.transforms = DropBondHeterograph(
                    p=config["dataset"]["edge_dropout"]
                )
            else:
                self.transforms = None

        # Graph feature dtype at load. Scaled LMDBs carry float64 features
        # (scaler mean/std are float64); default float32 keeps them compatible
        # with float32/bf16 weights. Set config["dataset"]["dtype"] to override.
        self._feature_dtype = self.config["dataset"].get("dtype", "float32")

        self._setup_done = False

    def prepare_data(self, stage=None):
        # No-op: in DDP, prepare_data runs only on rank 0.
        # LMDB files already exist on disk; dataset creation is in setup().
        pass

    def setup(self, stage=None):
        if self._setup_done:
            return

        transform = partial(TransformMol, dtype=self._feature_dtype)

        if "test_lmdb" in self.config["dataset"]:
            self.test_dataset = LMDBMoleculeDataset(
                config={"src": _resolve_lmdb_path(self.test_lmdb_loc)},
                transform=transform,
            )

        if "val_lmdb" in self.config["dataset"]:
            self.val_dataset = LMDBMoleculeDataset(
                config={"src": _resolve_lmdb_path(self.val_lmdb_loc)},
                transform=transform,
            )

        self.train_dataset = LMDBMoleculeDataset(
            config={"src": _resolve_lmdb_path(self.train_lmdb_loc)},
            transform=transform,
        )

        self._setup_done = True

    def _bucket_sampler(self, dataset, lmdb_loc, shuffle):
        """BucketBatchSampler for `dataset` when config["dataset"]["bucketing"] is set.

        Groups graphs by padded (atoms, bonds) shape so ResidualBlockDense sees a
        few static shapes per epoch (performance plan A3). Sizes are read once
        and cached next to the LMDB (or in config["dataset"]["bucket_cache_dir"]).
        """
        from qtaim_embed.data.bucketing import BucketBatchSampler, graph_sizes

        grid = int(self.config["dataset"].get("bucket_grid", self.config["model"].get("dense_grid", 16)))
        cache_dir = self.config["dataset"].get("bucket_cache_dir")
        base = Path(lmdb_loc)
        base = base.parent if base.is_file() else base
        cache = (Path(cache_dir) if cache_dir else base) / f".qtaim_sizes_{base.name}.npz"
        atoms, bonds = graph_sizes(dataset, cache_path=str(cache),
                                   num_workers=self.config["optim"].get("num_workers", 0))
        sampler = BucketBatchSampler(
            atoms, bonds, batch_size=self.config["optim"]["train_batch_size"], grid=grid,
            shuffle=shuffle, seed=self.config["dataset"].get("seed", 0),
        )
        logger.info("bucketing %s: %d shape classes, padding waste %.1f %%, %d batches/epoch",
                    base.name, len(sampler.classes), 100 * sampler.padding_waste(), len(sampler))
        return sampler

    def _loader(self, dataset, lmdb_loc, shuffle, **extra):
        if self.config["dataset"].get("bucketing", False):
            return DataLoaderLMDB(
                dataset=dataset,
                batch_sampler=self._bucket_sampler(dataset, lmdb_loc, shuffle),
                num_workers=self.config["optim"]["num_workers"],
                **extra,
            )
        return DataLoaderLMDB(
            dataset=dataset,
            batch_size=self.config["optim"]["train_batch_size"],
            shuffle=shuffle,
            num_workers=self.config["optim"]["num_workers"],
            **extra,
        )

    def train_dataloader(self):
        return self._loader(
            self.train_dataset, self.train_lmdb_loc, shuffle=True,
            pin_memory=self.config["optim"]["pin_memory"],
            persistent_workers=self.config["optim"]["persistent_workers"],
        )

    def test_dataloader(self):
        return self._loader(self.test_dataset, self.test_lmdb_loc, shuffle=False)

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.val_lmdb_loc, shuffle=False)


class LMDBLinkDataModule(LMDBDataModule):
    """LMDB data module for GCNLinkPred: same dataset construction as
    LMDBDataModule, homograph conversion plus negative sampling in the collate."""

    def __init__(self, config):
        super().__init__(config)
        self.node_len = None

    def _get_node_len(self, dl):
        """Lazily compute node_len from a DataLoader batch if not yet known."""
        if self.node_len is None:
            _, _, ft = next(iter(dl))
            self.node_len = ft.shape[1]

    def _loader(self, dataset, shuffle):
        optim = self.config["optim"]
        return DataLoaderLinkLMDB(
            dataset=dataset,
            transforms=self.transforms,
            batch_size=optim["train_batch_size"],
            shuffle=shuffle,
            num_workers=optim["num_workers"],
            pin_memory=optim.get("pin_memory", False),
            persistent_workers=bool(optim.get("persistent_workers", False)) and optim["num_workers"] > 0,
        )

    def train_dataloader(self):
        dl = self._loader(self.train_dataset, shuffle=True)
        self._get_node_len(dl)
        return dl

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)


class LMDBBondDataModule(LMDBDataModule):
    """LMDB data module for GCNBondPred.

    Same config shape and dataset construction as LMDBDataModule (train/val/test
    LMDB paths, dtype, optim.*); differs only in the collate, which returns the
    batched HeteroData alone since candidates and labels are built in the model
    step. dataset.val_lmdb is required: the decision threshold is calibrated on
    validation.
    """

    def __init__(self, config):
        assert "val_lmdb" in config["dataset"] and config["dataset"]["val_lmdb"] is not None, (
            "LMDBBondDataModule needs dataset.val_lmdb: the decision threshold is calibrated on validation"
        )
        super().__init__(config)

    def _loader(self, dataset, shuffle):
        optim = self.config["optim"]
        num_workers = optim.get("num_workers", 0)
        return DataLoaderBondLMDB(
            dataset=dataset,
            transforms=self.transforms,
            batch_size=optim["train_batch_size"],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=optim.get("pin_memory", False),
            persistent_workers=bool(optim.get("persistent_workers", False)) and num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            raise ValueError("dataset.test_lmdb is not set; nothing to test on")
        return self._loader(self.test_dataset, shuffle=False)
