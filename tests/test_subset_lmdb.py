"""SubsetLMDB and the LMDBDataModule subset_frac hook."""

import pytest

from qtaim_embed.core.dataset import LMDBMoleculeDataset, SubsetLMDB
from qtaim_embed.core.datamodule import LMDBDataModule

TRAIN_LMDB = "./data/lmdb_node/train/molecule.lmdb"
VAL_LMDB = "./data/lmdb_node/val/molecule.lmdb"


@pytest.fixture
def base_dataset():
    return LMDBMoleculeDataset(config={"src": TRAIN_LMDB})


def _dm(**dataset_overrides):
    config = {
        "dataset": {"train_lmdb": TRAIN_LMDB, "val_lmdb": VAL_LMDB, **dataset_overrides},
        "optim": {
            "num_workers": 0,
            "train_batch_size": 4,
            "pin_memory": False,
            "persistent_workers": False,
        },
    }
    dm = LMDBDataModule(config=config)
    dm.setup()
    return dm


class TestSubsetLMDB:
    def test_length_and_indices_sorted(self, base_dataset):
        sub = SubsetLMDB.random(base_dataset, frac=0.2, seed=0)
        assert len(sub) == round(len(base_dataset) * 0.2)
        assert sub.indices == sorted(sub.indices)
        assert len(set(sub.indices)) == len(sub.indices)
        assert max(sub.indices) < len(base_dataset)

    def test_seed_reproducible_and_distinct(self, base_dataset):
        a = SubsetLMDB.random(base_dataset, frac=0.3, seed=7)
        b = SubsetLMDB.random(base_dataset, frac=0.3, seed=7)
        c = SubsetLMDB.random(base_dataset, frac=0.3, seed=8)
        assert a.indices == b.indices
        assert a.indices != c.indices

    def test_items_match_base(self, base_dataset):
        sub = SubsetLMDB.random(base_dataset, frac=0.2, seed=1)
        for i in range(len(sub)):
            assert str(sub[i]) == str(base_dataset[sub.indices[i]])

    def test_metadata_delegated(self, base_dataset):
        sub = SubsetLMDB.random(base_dataset, frac=0.2, seed=0)
        assert sub.feature_size == base_dataset.feature_size
        assert sub.feature_names == base_dataset.feature_names
        assert sub.target_dict == base_dataset.target_dict

    def test_unknown_attribute_raises(self, base_dataset):
        sub = SubsetLMDB.random(base_dataset, frac=0.2, seed=0)
        with pytest.raises(AttributeError):
            sub.definitely_not_an_attribute

    def test_n_instead_of_frac(self, base_dataset):
        assert len(SubsetLMDB.random(base_dataset, n=5, seed=0)) == 5

    def test_n_clamped_to_dataset(self, base_dataset):
        big = len(base_dataset) * 10
        assert len(SubsetLMDB.random(base_dataset, n=big, seed=0)) == len(base_dataset)

    @pytest.mark.parametrize("kwargs", [{}, {"frac": 0.5, "n": 5}])
    def test_requires_exactly_one_of_frac_or_n(self, base_dataset, kwargs):
        with pytest.raises(ValueError):
            SubsetLMDB.random(base_dataset, **kwargs)

    @pytest.mark.parametrize("frac", [0.0, -0.1, 1.5])
    def test_bad_frac_rejected(self, base_dataset, frac):
        with pytest.raises(ValueError):
            SubsetLMDB.random(base_dataset, frac=frac)

    def test_frac_one_keeps_everything(self, base_dataset):
        sub = SubsetLMDB.random(base_dataset, frac=1.0, seed=0)
        assert sub.indices == list(range(len(base_dataset)))


class TestDataModuleSubset:
    def test_subset_frac_shrinks_train_and_val(self):
        full = _dm()
        sub = _dm(subset_frac=0.2, subset_seed=0)
        assert len(sub.train_dataset) == round(len(full.train_dataset) * 0.2)
        assert len(sub.val_dataset) == round(len(full.val_dataset) * 0.2)

    def test_test_split_is_never_subset(self):
        config = {
            "dataset": {
                "train_lmdb": TRAIN_LMDB,
                "val_lmdb": VAL_LMDB,
                "test_lmdb": "./data/lmdb_node/test/molecule.lmdb",
                "subset_frac": 0.2,
            },
            "optim": {
                "num_workers": 0,
                "train_batch_size": 4,
                "pin_memory": False,
                "persistent_workers": False,
            },
        }
        dm = LMDBDataModule(config=config)
        dm.setup()
        full_test = LMDBMoleculeDataset(config={"src": "./data/lmdb_node/test/molecule.lmdb"})
        assert len(dm.test_dataset) == len(full_test)
        assert not isinstance(dm.test_dataset, SubsetLMDB)

    def test_no_subset_by_default(self):
        dm = _dm()
        assert not isinstance(dm.train_dataset, SubsetLMDB)

    def test_frac_one_is_a_noop(self):
        dm = _dm(subset_frac=1.0)
        assert not isinstance(dm.train_dataset, SubsetLMDB)

    def test_seed_is_honoured_across_datamodules(self):
        a = _dm(subset_frac=0.3, subset_seed=2)
        b = _dm(subset_frac=0.3, subset_seed=2)
        c = _dm(subset_frac=0.3, subset_seed=3)
        assert a.train_dataset.indices == b.train_dataset.indices
        assert a.train_dataset.indices != c.train_dataset.indices

    def test_feature_size_still_readable_by_train_scripts(self):
        dm = _dm(subset_frac=0.2)
        assert set(dm.train_dataset.feature_size) == {"atom", "bond", "global"}

    def test_dataloader_yields_batches(self):
        dm = _dm(subset_frac=0.4)
        batch = next(iter(dm.train_dataloader()))
        assert batch is not None
        n_batches = sum(1 for _ in dm.train_dataloader())
        assert n_batches == -(-len(dm.train_dataset) // 4)
