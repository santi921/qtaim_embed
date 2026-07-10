#!/usr/bin/env python3
"""
Standalone PyG graph-level profiling script.

Bypasses wandb, uses torch profiler for detailed GPU/CPU breakdown.
Measures: throughput (it/s, samples/s), GPU utilization, memory usage,
parameter counts, and per-operation time breakdown.

Usage:
    python profiling/profile_pyg_graph.py -config profiling/graph_configs/graph_baseline_w0.json
    python profiling/profile_pyg_graph.py -config profiling/graph_configs/graph_gat_large_w4.json
"""

import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
    Callback,
)
from pytorch_lightning.loggers import CSVLogger

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

PROFILING_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Timing / profiling callback
# ---------------------------------------------------------------------------
class ProfilingCallback(Callback):
    """Captures per-epoch timing, GPU memory, and throughput."""

    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.epoch_start = None
        self.train_batch_count = 0
        self.train_sample_count = 0
        self.gpu_mem_peak = 0.0
        self.gpu_mem_allocated = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.perf_counter()
        self.train_batch_count = 0
        self.train_sample_count = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_batch_count += 1
        # Count samples in this batch
        batch_graph, _ = batch
        if hasattr(batch_graph, "num_graphs"):
            self.train_sample_count += batch_graph.num_graphs
        elif hasattr(batch_graph, "batch") or hasattr(batch_graph["atom"], "batch"):
            # HeteroData batched -- count unique batch indices
            batch_vec = batch_graph["atom"].batch
            self.train_sample_count += int(batch_vec.max().item()) + 1

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.perf_counter() - self.epoch_start
        self.epoch_times.append({
            "epoch": trainer.current_epoch,
            "time_s": elapsed,
            "batches": self.train_batch_count,
            "samples": self.train_sample_count,
            "it_per_s": self.train_batch_count / elapsed if elapsed > 0 else 0,
            "samples_per_s": self.train_sample_count / elapsed if elapsed > 0 else 0,
        })
        if torch.cuda.is_available():
            self.gpu_mem_peak = torch.cuda.max_memory_allocated() / 1e9
            self.gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_torch_profiler(model, dm, config, num_warmup=3, num_active=5):
    """
    Run torch.profiler on a few training steps to get detailed breakdown.
    Returns the profiler key_averages table.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Get a dataloader
    dm.setup(stage="fit")
    loader = dm.train_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["lr"])
    scaler = torch.amp.GradScaler("cuda") if "16" in str(config["optim"]["precision"]) or "bf16" in str(config["optim"]["precision"]) else None
    use_amp = scaler is not None
    amp_dtype = torch.bfloat16 if "bf16" in str(config["optim"]["precision"]) else torch.float16

    # Warmup
    batch_iter = iter(loader)
    for i in range(num_warmup):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        batch_graph, batch_label = batch
        batch_graph = batch_graph.to(device)
        for key in batch_label:
            if hasattr(batch_label[key], "to"):
                batch_label[key] = batch_label[key].to(device)

        feat_dict = {
            nt: batch_graph[nt].feat
            for nt in batch_graph.node_types
            if hasattr(batch_graph[nt], "feat")
        }

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(batch_graph, feat_dict)
            labels = batch_label["global"]
            loss = torch.nn.functional.mse_loss(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # Profile
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for i in range(num_active):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(loader)
                batch = next(batch_iter)

            batch_graph, batch_label = batch
            batch_graph = batch_graph.to(device)
            for key in batch_label:
                if hasattr(batch_label[key], "to"):
                    batch_label[key] = batch_label[key].to(device)

            feat_dict = {
                nt: batch_graph[nt].feat
                for nt in batch_graph.node_types
                if hasattr(batch_graph[nt], "feat")
            }

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_graph, feat_dict)
                labels = batch_label["global"]
                loss = torch.nn.functional.mse_loss(logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            prof.step()

    model = model.cpu()
    return prof


def main():
    parser = argparse.ArgumentParser(description="PyG graph-level profiling")
    parser.add_argument("-config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--skip_profiler", action="store_true",
                        help="Skip torch.profiler (faster, just get throughput)")
    parser.add_argument("--trace_dir", type=str, default=None,
                        help="Directory to save chrome trace (optional)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    exp_name = Path(args.config).stem

    print("=" * 70)
    print(f"PyG Graph-Level Profiling: {exp_name}")
    print("=" * 70)
    print(f"  Model:        {config['model']['conv_fn']}")
    print(f"  Hidden:       {config['model']['hidden_size']}")
    print(f"  Embedding:    {config['model']['embedding_size']}")
    print(f"  Conv layers:  {config['model']['n_conv_layers']}")
    print(f"  FC size:      {config['model']['fc_hidden_size_1']} x {config['model']['fc_num_layers']}")
    print(f"  Compiled:     {config['model']['compiled']}")
    print(f"  Batch size:   {config['optim']['train_batch_size']}")
    print(f"  Workers:      {config['optim']['num_workers']}")
    print(f"  Precision:    {config['optim']['precision']}")
    print(f"  Epochs:       {config['model']['max_epochs']}")
    print()

    # -----------------------------------------------------------------------
    # Data module
    # -----------------------------------------------------------------------
    from qtaim_embed.core.datamodule import LMDBDataModule
    from qtaim_embed.models.utils import load_graph_level_model_from_config

    config["model"]["target_dict"]["global"] = config["dataset"]["target_list"]

    dm = LMDBDataModule(config=config)
    dm.setup(stage="fit")

    # Infer feature sizes from the first graph in the train dataset
    sample = dm.train_dataset[0]
    feature_size = {
        "atom":   sample["atom"]["feat"].shape[-1],
        "bond":   sample["bond"]["feat"].shape[-1],
        "global": sample["global"]["feat"].shape[-1],
    }
    feature_names = {"atom": [], "bond": [], "global": []}

    print(f"  Feature sizes: atom={feature_size['atom']}, bond={feature_size['bond']}, global={feature_size['global']}")

    config["model"]["atom_feature_size"] = feature_size["atom"]
    config["model"]["bond_feature_size"] = feature_size["bond"]
    config["model"]["global_feature_size"] = feature_size["global"]

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = load_graph_level_model_from_config(config["model"])

    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters:   {total_params:,} total ({trainable_params:,} trainable)")
    param_mb = total_params * 4 / 1e6  # fp32
    print(f"  Param memory: {param_mb:.1f} MB (fp32)")
    print()

    # -----------------------------------------------------------------------
    # Phase 1: Lightning training (throughput measurement)
    # -----------------------------------------------------------------------
    print("-" * 70)
    print("Phase 1: Training throughput (Lightning)")
    print("-" * 70)

    profiling_cb = ProfilingCallback()
    log_dir = f"./profiling/logs/{exp_name}"

    trainer = pl.Trainer(
        max_epochs=config["model"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config["optim"]["num_devices"],
        gradient_clip_val=config["optim"]["gradient_clip_val"],
        accumulate_grad_batches=config["optim"]["accumulate_grad_batches"],
        enable_progress_bar=True,
        callbacks=[profiling_cb],
        enable_checkpointing=False,
        strategy=config["optim"]["strategy"],
        default_root_dir=log_dir,
        logger=CSVLogger(log_dir, name="profiling"),
        precision=config["optim"]["precision"],
    )

    t_start = time.perf_counter()
    trainer.fit(model, dm)
    t_total = time.perf_counter() - t_start

    print()
    print("Epoch-level results:")
    print(f"  {'Epoch':<8} {'Time(s)':<10} {'it/s':<10} {'samples/s':<12} {'Batches':<10} {'Samples':<10}")
    for et in profiling_cb.epoch_times:
        print(f"  {et['epoch']:<8} {et['time_s']:<10.2f} {et['it_per_s']:<10.2f} {et['samples_per_s']:<12.1f} {et['batches']:<10} {et['samples']:<10}")

    # Steady-state = skip epoch 0 (warmup)
    steady = profiling_cb.epoch_times[1:] if len(profiling_cb.epoch_times) > 1 else profiling_cb.epoch_times
    avg_its = sum(e["it_per_s"] for e in steady) / len(steady) if steady else 0
    avg_sps = sum(e["samples_per_s"] for e in steady) / len(steady) if steady else 0

    print()
    print(f"  Steady-state avg: {avg_its:.2f} it/s, {avg_sps:.1f} samples/s")
    print(f"  Total training time: {t_total:.1f}s")
    print(f"  GPU peak memory: {profiling_cb.gpu_mem_peak:.2f} GB")
    print()

    # -----------------------------------------------------------------------
    # Phase 2: Torch profiler (detailed breakdown)
    # -----------------------------------------------------------------------
    profiler_results = None
    if not args.skip_profiler:
        print("-" * 70)
        print("Phase 2: Torch Profiler (detailed breakdown)")
        print("-" * 70)

        # Need a fresh model for profiling (training may have moved state)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_prof = load_graph_level_model_from_config(config["model"])
        prof = run_torch_profiler(model_prof, dm, config)

        # Print top operations by CPU time
        print()
        print("Top 20 operations by CPU time (total):")
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20,
        ))

        if torch.cuda.is_available():
            print()
            print("Top 20 operations by CUDA time (total):")
            print(prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20,
            ))

        # Save chrome trace if requested
        if args.trace_dir:
            trace_path = os.path.join(args.trace_dir, f"{exp_name}_trace.json")
            prof.export_chrome_trace(trace_path)
            print(f"\nChrome trace saved to: {trace_path}")

        # Extract key metrics from profiler
        ka = prof.key_averages()
        total_cpu_us = sum(e.cpu_time_total for e in ka)
        # FunctionEventAvg uses device_time_total (not cuda_time_total)
        total_cuda_us = sum(getattr(e, "device_time_total", getattr(e, "self_device_time_total", 0)) for e in ka) if torch.cuda.is_available() else 0

        profiler_results = {
            "total_cpu_time_ms": total_cpu_us / 1000,
            "total_cuda_time_ms": total_cuda_us / 1000,
            "cuda_pct": total_cuda_us / total_cpu_us * 100 if total_cpu_us > 0 else 0,
        }

        # Categorize operations
        categories = {
            "conv/linear": 0,
            "activation": 0,
            "norm": 0,
            "optimizer": 0,
            "loss": 0,
            "data_transfer": 0,
            "other": 0,
        }
        for e in ka:
            name = e.key.lower()
            cpu_ms = e.cpu_time_total / 1000
            if any(k in name for k in ["linear", "conv", "matmul", "mm", "addmm", "gemm"]):
                categories["conv/linear"] += cpu_ms
            elif any(k in name for k in ["relu", "gelu", "silu", "elu", "sigmoid", "tanh", "softmax"]):
                categories["activation"] += cpu_ms
            elif any(k in name for k in ["batch_norm", "layer_norm", "norm"]):
                categories["norm"] += cpu_ms
            elif any(k in name for k in ["adam", "sgd", "optimizer", "step"]):
                categories["optimizer"] += cpu_ms
            elif any(k in name for k in ["mse", "cross_entropy", "loss", "nll"]):
                categories["loss"] += cpu_ms
            elif any(k in name for k in ["to", "copy", "pin_memory", "memcpy"]):
                categories["data_transfer"] += cpu_ms
            else:
                categories["other"] += cpu_ms

        total_cat_ms = sum(categories.values())
        print()
        print("Operation category breakdown (CPU time):")
        for cat, ms in sorted(categories.items(), key=lambda x: -x[1]):
            pct = ms / total_cat_ms * 100 if total_cat_ms > 0 else 0
            print(f"  {cat:<20} {ms:>8.1f} ms ({pct:>5.1f}%)")

        profiler_results["categories"] = categories
        del model_prof
        gc.collect()

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "experiment": exp_name,
        "config": {
            "conv_fn": config["model"]["conv_fn"],
            "hidden_size": config["model"]["hidden_size"],
            "embedding_size": config["model"]["embedding_size"],
            "n_conv_layers": config["model"]["n_conv_layers"],
            "fc_hidden_size_1": config["model"]["fc_hidden_size_1"],
            "fc_num_layers": config["model"]["fc_num_layers"],
            "compiled": config["model"]["compiled"],
            "batch_size": config["optim"]["train_batch_size"],
            "num_workers": config["optim"]["num_workers"],
            "precision": config["optim"]["precision"],
        },
        "model": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_mb": param_mb,
        },
        "throughput": {
            "avg_it_per_s": avg_its,
            "avg_samples_per_s": avg_sps,
            "total_time_s": t_total,
            "gpu_peak_memory_gb": profiling_cb.gpu_mem_peak,
            "epoch_details": profiling_cb.epoch_times,
        },
    }
    if profiler_results:
        results["profiler"] = profiler_results

    results_dir = PROFILING_DIR / "graph_results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / f"{exp_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print()
    print(f"Results saved to: {results_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"SUMMARY: {exp_name}")
    print("=" * 70)
    print(f"  Model:          {config['model']['conv_fn']} (hidden={config['model']['hidden_size']}, embed={config['model']['embedding_size']})")
    print(f"  Parameters:     {total_params:,}")
    print(f"  Throughput:     {avg_its:.2f} it/s | {avg_sps:.1f} samples/s")
    print(f"  GPU peak mem:   {profiling_cb.gpu_mem_peak:.2f} GB")
    if profiler_results:
        print(f"  CUDA time pct:  {profiler_results['cuda_pct']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
