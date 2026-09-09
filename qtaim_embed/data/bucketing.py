"""Static-shape bucketing for padded (dense) training (performance plan A3).

`BucketBatchSampler` groups graphs whose (ceil(atoms / grid), ceil(bonds / grid))
match into fixed-size batches, so every batch of a shape class pads to one
static (N_b, B_b) block in `models.layers_dense.to_dense_hetero` and the
compiled dense conv stack sees a small, fixed set of shapes (CUDA graphs are
recorded once per shape). E4 in docs/research/2026-09-track-a-measurements.md:
grid 16 gives 11 % padding waste on tm_react with 23 shape classes, against
43 % for random batches padded to their own maximum.

Shape classes smaller than `min_class_fraction * batch_size` graphs are merged
into the next larger class so the epoch is not dominated by ragged
mini-batches; a batch drawn from a merged class pads to the maximum of the
graphs it contains, which is at most the elementwise maximum of the merged
shapes.

Distributed training: pass `rank` and `world_size` and every rank yields a
disjoint, equally sized slice of the epoch's batches (Lightning must then run
with `use_distributed_sampler=False`, which the train scripts set whenever
`dataset.bucketing` is on). With shuffling the tail batches that do not divide
evenly are dropped; without shuffling (eval) they are repeated so no sample is
lost.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

logger = logging.getLogger(__name__)


def shape_class(
    atoms: Sequence[int], bonds: Sequence[int], grid: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Static (N_b, B_b) a graph pads to; vectorized over arrays."""
    ca = np.maximum(grid, (np.ceil(np.asarray(atoms) / grid) * grid).astype(np.int64))
    cb = np.maximum(grid, (np.ceil(np.asarray(bonds) / grid) * grid).astype(np.int64))
    return ca, cb


class BucketBatchSampler(Sampler[List[int]]):
    """Yield fixed-size batches of graphs that share a padded (atoms, bonds) shape.

    Iteration order is class-major, so with `shuffle=False` it is NOT dataset
    order; use `indices()` to align per-sample outputs with the dataset.
    `rank` / `world_size` shard the batches for distributed training.
    """

    def __init__(
        self,
        atom_counts: Sequence[int],
        bond_counts: Sequence[int],
        batch_size: int,
        grid: int = 16,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        min_class_fraction: float = 0.5,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.atom_counts = np.asarray(atom_counts, dtype=np.int64)
        self.bond_counts = np.asarray(bond_counts, dtype=np.int64)
        if self.atom_counts.shape != self.bond_counts.shape:
            raise ValueError("atom_counts and bond_counts must have the same length")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.batch_size = int(batch_size)
        self.grid = int(grid)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.classes, self._class_of_shape = self._build_classes(min_class_fraction)

    def _build_classes(
        self, min_class_fraction: float
    ) -> Tuple[List[Tuple[Tuple[int, int], np.ndarray]], Dict[Tuple[int, int], Tuple[int, int]]]:
        ca, cb = shape_class(self.atom_counts, self.bond_counts, self.grid)
        keys = ca * (1 << 32) + cb
        order = np.argsort(keys, kind="stable")
        uniq, starts = np.unique(keys[order], return_index=True)
        groups = np.split(order, starts[1:])
        shapes = [(int(k >> 32), int(k & ((1 << 32) - 1))) for k in uniq]
        # merge small classes upward (classes are sorted by atoms then bonds)
        min_size = max(1, int(min_class_fraction * self.batch_size))
        merged: List[Tuple[Tuple[int, int], np.ndarray]] = []
        members: List[List[Tuple[int, int]]] = []
        carry_idx, carry_shape, carry_members = [], (0, 0), []
        for shape, idx in zip(shapes, groups):
            carry_idx.append(idx)
            carry_members.append(shape)
            carry_shape = (max(carry_shape[0], shape[0]), max(carry_shape[1], shape[1]))
            if sum(len(i) for i in carry_idx) >= min_size:
                merged.append((carry_shape, np.concatenate(carry_idx)))
                members.append(carry_members)
                carry_idx, carry_shape, carry_members = [], (0, 0), []
        if carry_idx:
            if merged:
                shape, idx = merged.pop()
                carry_members = members.pop() + carry_members
                carry_shape = (max(carry_shape[0], shape[0]), max(carry_shape[1], shape[1]))
                carry_idx.append(idx)
            merged.append((carry_shape, np.concatenate(carry_idx)))
            members.append(carry_members)
        lookup = {m: s for (s, _), ms in zip(merged, members) for m in ms}
        return merged, lookup

    def batch_shape(self, atom_counts: Sequence[int], bond_counts: Sequence[int]) -> Tuple[int, int]:
        """Static (N_b, B_b) the batch containing these graphs pads to.

        Graphs of one (merged) class map to that class's shape, so every batch
        drawn from the class gets the same answer regardless of which graphs it
        holds. Graphs outside the table (not from this dataset) fall back to
        their own rounded shape. Called from the collate (DataLoaderLMDB).
        """
        ca, cb = shape_class(atom_counts, bond_counts, self.grid)
        na = nb = 0
        for a, b in zip(ca.tolist(), cb.tolist()):
            sa, sb = self._class_of_shape.get((a, b), (a, b))
            na, nb = max(na, sa), max(nb, sb)
        return na, nb

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        return [s for s, _ in self.classes]

    def padding_waste(self) -> float:
        """Fraction of padded atom+bond rows that are padding under this scheme."""
        actual = padded = 0
        for (na, nb), idx in self.classes:
            actual += int(self.atom_counts[idx].sum() + self.bond_counts[idx].sum())
            padded += len(idx) * (na + nb)
        return 1.0 - actual / padded

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _total_batches(self) -> int:
        n = 0
        for _, idx in self.classes:
            q, r = divmod(len(idx), self.batch_size)
            n += q + (0 if (self.drop_last or r == 0) else 1)
        return n

    def _shard(self, batches: List[List[int]]) -> List[List[int]]:
        if self.world_size == 1:
            return batches
        n = len(batches)
        rem = n % self.world_size
        if rem:
            if self.shuffle:
                batches = batches[: n - rem]
            else:
                batches = batches + batches[: self.world_size - rem]
        return batches[self.rank :: self.world_size]

    def _batches(self) -> List[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch) if self.shuffle else None
        batches = []
        for _, idx in self.classes:
            idx = rng.permutation(idx) if rng is not None else idx
            for s in range(0, len(idx), self.batch_size):
                chunk = idx[s:s + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk.tolist())
        if rng is not None:
            rng.shuffle(batches)
        return self._shard(batches)

    def indices(self) -> List[int]:
        """Dataset indices in the order this rank's batches yield them this epoch.

        With `shuffle=False` this is the permutation to apply to per-sample
        outputs collected from the loader to get them back into dataset order.
        """
        return [i for b in self._batches() for i in b]

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._batches()
        if self.shuffle:
            self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        n = self._total_batches()
        if self.world_size == 1:
            return n
        rem = n % self.world_size
        if rem:
            n = n - rem if self.shuffle else n + self.world_size - rem
        return n // self.world_size


def _sizes_collate(samples):
    return (
        torch.tensor([int(g["atom"].num_nodes) for g in samples]),
        torch.tensor([int(g["bond"].num_nodes) for g in samples]),
    )


def graph_sizes(
    dataset: Sequence,
    cache_path: Optional[str] = None,
    num_workers: int = 0,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-graph atom and bond counts for a map-style graph dataset.

    Reads every record once (about 0.5 ms each for LMDB shards; use
    num_workers to parallelize) and caches the result as an .npz keyed by the
    dataset length when cache_path is given.
    """
    n = len(dataset)
    if cache_path is not None and os.path.exists(cache_path):
        z = np.load(cache_path)
        if int(z["n"]) == n:
            return z["atoms"], z["bonds"]
        logger.warning("size cache %s has %d entries for a dataset of %d; rebuilding", cache_path, int(z["n"]), n)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=_sizes_collate)
    atoms, bonds = [], []
    for a, b in loader:
        atoms.append(a)
        bonds.append(b)
    atoms = torch.cat(atoms).numpy()
    bonds = torch.cat(bonds).numpy()
    if cache_path is not None:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, n=n, atoms=atoms, bonds=bonds)
        except OSError as exc:
            logger.warning("could not write size cache %s: %s", cache_path, exc)
    return atoms, bonds
