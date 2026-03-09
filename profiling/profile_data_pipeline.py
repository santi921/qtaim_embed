#!/usr/bin/env python3
"""
Profile the data loading pipeline to identify bottlenecks.

This script measures timing for each stage of data loading:
1. LMDB read
2. Deserialization (pickle)
3. Graph construction
4. Feature extraction
5. Collation/batching
6. Transfer to GPU

Usage:
    python profiling/profile_data_pipeline.py
"""

import time
import pickle
import lmdb
import torch
from pathlib import Path
from contextlib import contextmanager
from statistics import mean, stdev

from qtaim_embed.core.datamodule import QTAIMLinkTaskDataModule, LMDBLinkDataModule
from qtaim_embed.utils.data import get_default_link_level_config


@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {name}: {elapsed*1000:.2f}ms")


def profile_lmdb_reads(lmdb_path: str, num_samples: int = 100):
    """Profile raw LMDB read performance."""
    print("\n1. LMDB Read Performance")
    print("-" * 60)

    # Handle both directory and .lmdb file paths
    path_obj = Path(lmdb_path)
    if path_obj.name == "molecule.lmdb":
        actual_path = str(path_obj)
    else:
        actual_path = str(path_obj / "molecule.lmdb")

    env = lmdb.open(
        actual_path,
        subdir=False,  # LMDB file, not directory
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    times = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= num_samples:
                break

            start = time.perf_counter()
            _ = txn.get(key)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

    env.close()

    print(f"  Samples read: {len(times)}")
    print(f"  Mean read time: {mean(times):.3f}ms")
    print(f"  Std dev: {stdev(times):.3f}ms")
    print(f"  Min: {min(times):.3f}ms, Max: {max(times):.3f}ms")
    print(f"  Total for {num_samples} samples: {sum(times):.2f}ms")


def profile_deserialization(lmdb_path: str, num_samples: int = 100):
    """Profile pickle deserialization."""
    print("\n2. Deserialization Performance")
    print("-" * 60)

    # Handle both directory and .lmdb file paths
    path_obj = Path(lmdb_path)
    if path_obj.name == "molecule.lmdb":
        actual_path = str(path_obj)
    else:
        actual_path = str(path_obj / "molecule.lmdb")

    env = lmdb.open(
        actual_path,
        subdir=False,  # LMDB file, not directory
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    times = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= num_samples:
                break

            start = time.perf_counter()
            _ = pickle.loads(value)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms

    env.close()

    print(f"  Samples deserialized: {len(times)}")
    print(f"  Mean deserialize time: {mean(times):.3f}ms")
    print(f"  Std dev: {stdev(times):.3f}ms")
    print(f"  Total for {num_samples} samples: {sum(times):.2f}ms")


def profile_dataloader(config: dict, num_batches: int = 20):
    """Profile the full DataLoader pipeline."""
    print("\n3. DataLoader Pipeline Performance")
    print("-" * 60)

    # Create datamodule - use LMDB version
    dm = LMDBLinkDataModule(config=config)
    dm.prepare_data("fit")
    dm.setup("fit")

    # Get train loader
    train_loader = dm.train_dataloader()

    print(f"  Batch size: {config['optim']['train_batch_size']}")
    print(f"  Num workers: {config['optim']['num_workers']}")
    print(f"  Pin memory: {config['optim']['pin_memory']}")

    # Time first batch (includes worker startup)
    print("\n  First batch (includes worker startup):")
    start = time.perf_counter()
    first_batch = next(iter(train_loader))
    first_time = time.perf_counter() - start
    print(f"    Time: {first_time*1000:.2f}ms")

    # Time subsequent batches (steady state)
    print("\n  Subsequent batches (steady state):")
    batch_times = []
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        start = time.perf_counter()
        _ = batch  # Just iterate, don't process
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed * 1000)

    print(f"    Batches timed: {len(batch_times)}")

    if len(batch_times) < 2:
        print(f"    WARNING: Not enough batches for statistics (need at least 2)")
        if len(batch_times) == 1:
            print(f"    Single batch time: {batch_times[0]:.2f}ms")
    else:
        print(f"    Mean batch time: {mean(batch_times):.2f}ms")
        print(f"    Std dev: {stdev(batch_times):.2f}ms")
        print(f"    Throughput: {1000/mean(batch_times):.2f} batches/sec")

        # Estimate samples/sec
        samples_per_sec = (config['optim']['train_batch_size'] * 1000) / mean(batch_times)
        print(f"    Samples/sec: {samples_per_sec:.1f}")


def profile_gpu_transfer(config: dict, num_batches: int = 20):
    """Profile CPU->GPU transfer time."""
    print("\n4. GPU Transfer Performance")
    print("-" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping GPU transfer profiling")
        return

    dm = LMDBLinkDataModule(config=config)
    dm.prepare_data("fit")
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    device = torch.device("cuda")

    # Skip first batch
    _ = next(iter(train_loader))

    # Time GPU transfers
    transfer_times = []
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        start = time.perf_counter()
        batch = batch.to(device)
        torch.cuda.synchronize()  # Wait for transfer to complete
        elapsed = time.perf_counter() - start
        transfer_times.append(elapsed * 1000)

    print(f"  Batches transferred: {len(transfer_times)}")
    print(f"  Mean transfer time: {mean(transfer_times):.2f}ms")
    print(f"  Std dev: {stdev(transfer_times):.2f}ms")


def profile_optimizer_step(config: dict, num_steps: int = 100):
    """Profile optimizer step overhead."""
    print("\n5. Optimizer Step Performance")
    print("-" * 60)

    from qtaim_embed.models.utils import load_link_level_model_from_config

    # Create model
    model = load_link_level_model_from_config(config["model"])

    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create dummy batch
    dm = LMDBLinkDataModule(config=config)
    dm.prepare_data("fit")
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    batch = batch.to(device)

    # Test different optimizers
    optimizers = {
        "Adam": torch.optim.Adam(model.parameters(), lr=0.01),
        "AdamW": torch.optim.AdamW(model.parameters(), lr=0.01),
        "SGD": torch.optim.SGD(model.parameters(), lr=0.01),
    }

    # Try fused Adam if available
    try:
        optimizers["Adam (fused)"] = torch.optim.Adam(
            model.parameters(), lr=0.01, fused=True
        )
    except:
        print("  Note: Fused Adam not available on this PyTorch version")

    for name, optimizer in optimizers.items():
        times = []

        for _ in range(num_steps):
            # Forward pass
            optimizer.zero_grad()
            output = model(batch)
            loss = output.mean()  # Dummy loss

            # Backward pass
            loss.backward()

            # Time optimizer step
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        print(f"\n  {name}:")
        print(f"    Mean step time: {mean(times):.3f}ms")
        print(f"    Std dev: {stdev(times):.3f}ms")
        print(f"    Steps/sec: {1000/mean(times):.1f}")


def profile_callbacks_overhead(config: dict, num_iterations: int = 50):
    """Profile PyTorch Lightning callback overhead."""
    print("\n6. Callback Overhead")
    print("-" * 60)

    from pytorch_lightning import Trainer
    from qtaim_embed.models.utils import load_link_level_model_from_config
    from qtaim_embed.core.datamodule import QTAIMLinkTaskDataModule

    # Create model and datamodule
    model = load_link_level_model_from_config(config["model"])
    dm = LMDBLinkDataModule(config=config)
    dm.setup("fit")

    # Test with minimal callbacks
    print("\n  Minimal callbacks:")
    start = time.perf_counter()
    trainer_min = Trainer(
        max_epochs=1,
        limit_train_batches=num_iterations,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    elapsed_min = time.perf_counter() - start
    print(f"    Trainer setup: {elapsed_min*1000:.2f}ms")

    # Note: Full training would require more setup
    print("\n  Note: For full callback overhead, compare training runs with/without:")
    print("    - WandbLogger")
    print("    - ModelCheckpoint")
    print("    - EarlyStopping")
    print("    - Custom callbacks (LogParameters)")


def main():
    """Run all profiling tests."""
    print("=" * 60)
    print("Data Pipeline Profiling")
    print("=" * 60)

    # Setup config
    config = get_default_link_level_config()
    config["dataset"]["train_lmdb"] = "tests/data/lmdb_link_large/train/"
    config["dataset"]["val_lmdb"] = "tests/data/lmdb_link_large/val/"
    config["optim"]["train_batch_size"] = 128  # Now using full LMDB dataset (1511 samples)

    # Check if LMDB exists
    lmdb_dir = Path(config["dataset"]["train_lmdb"])
    lmdb_path = lmdb_dir / "molecule.lmdb"
    if not lmdb_path.exists():
        print(f"\nError: LMDB not found at {lmdb_path}")
        print("Please run data preparation first.")
        return

    # Run profiling tests
    profile_lmdb_reads(str(lmdb_dir))
    profile_deserialization(str(lmdb_dir))

    # Test with different num_workers
    print("\n" + "=" * 60)
    print("Testing num_workers impact")
    print("=" * 60)

    for num_workers in [0, 2, 4]:
        config["optim"]["num_workers"] = num_workers
        config["optim"]["persistent_workers"] = num_workers > 0
        print(f"\n--- num_workers={num_workers} ---")
        profile_dataloader(config, num_batches=20)

    # GPU transfer profiling
    config["optim"]["num_workers"] = 4
    profile_gpu_transfer(config, num_batches=20)

    # Optimizer profiling
    profile_optimizer_step(config, num_steps=100)

    # Callback overhead
    profile_callbacks_overhead(config)

    print("\n" + "=" * 60)
    print("Profiling Complete!")
    print("=" * 60)

    print("\nKey areas to investigate based on results:")
    print("1. If LMDB reads are slow: Consider SSD, increase cache")
    print("2. If deserialization is slow: Consider different serialization format")
    print("3. If dataloader is slow: Increase num_workers or optimize collate")
    print("4. If GPU transfer is slow: Check pin_memory, consider pre-loading")
    print("5. If optimizer is slow: Try fused optimizer or different optimizer")


if __name__ == "__main__":
    main()
