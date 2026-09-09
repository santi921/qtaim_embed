#!/usr/bin/env python3
"""qtaim-embed-bench: measure training throughput of a config on one GPU.

Two modes:
  raw        plain fwd/bwd/step loop over the real DataLoader; reports step
             time, data-wait fraction, sampled GPU utilization, CUDA kernels
             per step, device syncs per step, and device-time shares for
             gather/scatter, matmul, encoder, and neighbor build.
  lightning  pl.Trainer with limit_train_batches; reports it/s and samples/s
             including Lightning overhead.

The config JSON is merged over the task's default config (node or graph), so
bench configs only list what differs. `--set a.b.c=value` overrides on top.

Examples:
  qtaim-embed-bench --config profiling/bench_configs/tm_react_node.json
  qtaim-embed-bench --config profiling/bench_configs/tm_react_node.json \
      --set optim.train_batch_size=1024 --set model.encoder_fn=schnet
"""

import argparse
import copy
import json
import os
import statistics
import subprocess
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import torch

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore", message=".*self.log().*")
warnings.filterwarnings("ignore", message=".*Profiler clears events.*")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "profiling" / "bench_results"

# Exact profiler keys of leaf ops. key_averages() device time is inclusive of
# children, so parents (aten::matmul, aten::batch_norm, Optimizer.step) must
# not be listed next to the ops they dispatch to.
_GATHER_SCATTER = (
    "aten::index_select",
    "aten::index_add_",
    "aten::scatter_add_",
    "aten::scatter_",
    "aten::gather",
    "aten::index_put_",
    "aten::index",
)
_MATMUL = ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm")
_BATCH_NORM = (
    "aten::cudnn_batch_norm",
    "aten::native_batch_norm",
    "aten::cudnn_batch_norm_backward",
    "aten::native_batch_norm_backward",
)
_OPTIMIZER = ("aten::_fused_adam_", "aten::_foreach_addcdiv_", "aten::_foreach_addcmul_",
              "aten::_foreach_lerp_", "aten::_foreach_sqrt", "aten::_foreach_div_")
_SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize")
_MEMCPY = ("cudaMemcpyAsync", "Memcpy DtoH (Device -> Pageable)", "Memcpy DtoH (Device -> Pinned)")


# dict-valued config entries that are whole values, not nested sections
_REPLACE_KEYS = {"target_dict", "extra_keys", "extra_dataset_info", "predictor_param_dict"}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in _REPLACE_KEYS:
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, sets: list) -> dict:
    for item in sets:
        key, _, raw = item.partition("=")
        if not _ or not key:
            raise ValueError(f"--set expects key.path=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


class GpuSampler(threading.Thread):
    """Poll nvidia-smi for the GPU torch is using; pynvml is not installed."""

    def __init__(self, interval_s: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval_s
        self.util, self.mem_mib = [], []
        self._stop_evt = threading.Event()
        self._gpu = self._resolve_gpu()

    @staticmethod
    def _resolve_gpu():
        if not torch.cuda.is_available():
            return None
        cur = torch.cuda.current_device()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis:
            parts = [p.strip() for p in vis.split(",") if p.strip()]
            return parts[cur] if cur < len(parts) else str(cur)
        return str(cur)

    def run(self):
        if self._gpu is None:
            return
        cmd = [
            "nvidia-smi",
            "-i",
            self._gpu,
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop_evt.is_set():
            try:
                line = subprocess.check_output(cmd, text=True, timeout=2).strip()
                u, m = line.split(",")
                self.util.append(float(u))
                self.mem_mib.append(float(m))
            except Exception:
                pass
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)

    def summary(self) -> dict:
        if not self.util:
            return {"gpu_util_mean": None, "gpu_util_std": None, "gpu_util_max": None,
                    "gpu_mem_used_max_gb": None, "gpu_util_samples": 0}
        return {
            "gpu_util_mean": statistics.fmean(self.util),
            "gpu_util_std": statistics.pstdev(self.util),
            "gpu_util_max": max(self.util),
            "gpu_mem_used_max_gb": max(self.mem_mib) / 1024,
            "gpu_util_samples": len(self.util),
        }


def _amp_settings(precision):
    p = str(precision)
    if p.startswith("bf16"):
        return torch.bfloat16, False
    if p in ("16", "16-mixed", "16-true"):
        return torch.float16, True
    return None, False


def _num_graphs(batch_graph) -> int:
    if hasattr(batch_graph, "num_graphs"):
        return int(batch_graph.num_graphs)
    return int(batch_graph["atom"].batch.max().item()) + 1


def _count_edges(batch_graph) -> int:
    return int(sum(batch_graph[et].edge_index.shape[1] for et in batch_graph.edge_types))


def build(cfg: dict):
    from qtaim_embed.core.datamodule import LMDBDataModule
    from qtaim_embed.utils.data import (
        get_default_graph_level_config,
        get_default_node_level_config,
    )

    task = cfg.get("task", "node")
    default = (
        get_default_graph_level_config() if task == "graph" else get_default_node_level_config()
    )
    cfg = deep_update(default, cfg)

    dm = LMDBDataModule(config=cfg)
    dm.setup(stage="fit")
    sample = dm.train_dataset[0]
    for nt in ("atom", "bond", "global"):
        cfg["model"][f"{nt}_feature_size"] = int(sample[nt].feat.shape[-1])

    if task == "graph":
        from qtaim_embed.models.utils import load_graph_level_model_from_config

        if "target_list" in cfg["dataset"]:
            cfg["model"].setdefault("target_dict", {})
            cfg["model"]["target_dict"]["global"] = list(cfg["dataset"]["target_list"])
        model = load_graph_level_model_from_config(cfg["model"])
    else:
        from qtaim_embed.models.utils import load_node_level_model_from_config

        if "target_dict" not in cfg["model"]:
            cfg["model"]["target_dict"] = cfg["dataset"]["target_dict"]
        for nt, names in cfg["model"]["target_dict"].items():
            if names and names != [None] and "labels" in sample[nt]:
                got = int(sample[nt].labels.shape[-1])
                assert got == len(names), (
                    f"{nt} labels have {got} columns but target_dict lists {len(names)}"
                )
        model = load_node_level_model_from_config(cfg["model"])

    return cfg, dm, model, task


def _to_device(batch, device, non_blocking):
    graph, labels = batch
    graph = graph.to(device, non_blocking=non_blocking)
    labels = {k: v.to(device, non_blocking=non_blocking) for k, v in labels.items()}
    return graph, labels


def _profile_window(model, opt, step_fn, next_fn, n_steps: int) -> dict:
    from torch.profiler import DeviceType, ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_steps):
            batch, _ = next_fn()
            step_fn(batch)
        torch.cuda.synchronize()

    events = prof.events()
    kernels = [e for e in events if e.device_type == DeviceType.CUDA]
    kernel_ms = sum(e.self_device_time_total for e in kernels) / n_steps / 1e3
    ka = {e.key: e for e in prof.key_averages()}

    def count(names):
        return sum(ka[n].count for n in names if n in ka) / n_steps

    def dev_ms(names):
        return sum(
            getattr(ka[n], "device_time_total", 0.0) for n in names if n in ka
        ) / n_steps / 1e3

    return {
        "profiler_steps": n_steps,
        "kernels_per_step": len(kernels) / n_steps,
        "kernel_busy_ms_per_step": kernel_ms,
        "nonzero_per_step": count(("aten::nonzero",)),
        "sync_per_step": count(_SYNC),
        "memcpy_per_step": count(_MEMCPY),
        "gather_scatter_ms": dev_ms(_GATHER_SCATTER),
        "matmul_ms": dev_ms(_MATMUL),
        "encoder_ms": dev_ms(("encoder",)),
        "neighbor_radius_ms": dev_ms(("neighbors.radius",)),
        "neighbor_triplets_ms": dev_ms(("neighbors.triplets",)),
        "batch_norm_ms": dev_ms(_BATCH_NORM),
        "optimizer_ms": dev_ms(_OPTIMIZER),
    }


def run_raw(cfg, dm, model, args) -> dict:
    device = torch.device("cuda")
    model = model.to(device)
    model.train()
    opt = model.configure_optimizers()
    opt = opt[0][0] if isinstance(opt, (tuple, list)) else opt
    amp_dtype, use_scaler = _amp_settings(cfg["optim"]["precision"])
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    clip = float(cfg["optim"].get("gradient_clip_val", 0.0) or 0.0)
    non_blocking = bool(cfg["optim"].get("pin_memory", False))

    loader = dm.train_dataloader()
    state = {"it": iter(loader)}

    def next_batch():
        t0 = time.perf_counter()
        try:
            batch = next(state["it"])
        except StopIteration:
            state["it"] = iter(loader)
            batch = next(state["it"])
        batch = _to_device(batch, device, non_blocking)
        return batch, time.perf_counter() - t0

    def step(batch):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            loss = model.training_step(batch, 0)
        if scaler is not None:
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        return loss

    n_warmup = args.warmup_steps
    if args.warmup_epoch:
        # one full pass so every bucket shape has been compiled / recorded
        n_warmup = max(n_warmup, len(loader))
    for _ in range(n_warmup):
        batch, _ = next_batch()
        step(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    sampler = GpuSampler()
    sampler.start()
    events, waits, samples, atoms, edges = [], [], 0, 0, 0
    t_start = time.perf_counter()
    for _ in range(args.steps):
        batch, wait = next_batch()
        waits.append(wait)
        samples += _num_graphs(batch[0])
        atoms += int(batch[0]["atom"].num_nodes)
        edges += _count_edges(batch[0])
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step(batch)
        e.record()
        events.append((s, e))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    sampler.stop()

    stream_ms = [s.elapsed_time(e) for s, e in events]
    res = {
        "steps": args.steps,
        "warmup_steps": n_warmup,
        "wall_s": wall,
        "it_per_s": args.steps / wall,
        "samples_per_s": samples / wall,
        "atoms_per_s": atoms / wall,
        "step_ms_mean": wall / args.steps * 1e3,
        "stream_step_ms_mean": statistics.fmean(stream_ms),
        "stream_step_ms_std": statistics.pstdev(stream_ms),
        "data_wait_frac": sum(waits) / wall,
        "data_wait_ms_mean": statistics.fmean(waits) * 1e3,
        "atoms_per_batch": atoms / args.steps,
        "edges_per_batch": edges / args.steps,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    res.update(sampler.summary())
    if not args.skip_profiler:
        res.update(_profile_window(model, opt, step, next_batch, args.profiler_steps))
    return res


def run_lightning(cfg, dm, model, args) -> dict:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback

    class BenchCallback(Callback):
        def __init__(self, warmup_steps, warmup_epochs):
            self.warmup_steps = warmup_steps
            self.warmup_epochs = warmup_epochs
            self.step_times, self.step_samples = [], []
            self.epoch_stats = []
            self._t = None
            self._epoch_t = None
            self._epoch_batches = 0
            self._epoch_samples = 0

        def on_train_epoch_start(self, trainer, pl_module):
            self._epoch_t = time.perf_counter()
            self._epoch_batches = 0
            self._epoch_samples = 0

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self._t = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            n = _num_graphs(batch[0])
            self._epoch_batches += 1
            self._epoch_samples += n
            if trainer.global_step > self.warmup_steps and trainer.current_epoch >= self.warmup_epochs:
                self.step_times.append(time.perf_counter() - self._t)
                self.step_samples.append(n)

        def on_train_epoch_end(self, trainer, pl_module):
            dt = time.perf_counter() - self._epoch_t
            self.epoch_stats.append({
                "epoch": trainer.current_epoch,
                "time_s": dt,
                "batches": self._epoch_batches,
                "samples": self._epoch_samples,
                "it_per_s": self._epoch_batches / dt,
                "samples_per_s": self._epoch_samples / dt,
            })

    if args.epochs:
        max_epochs = args.warmup_epochs + args.epochs
        limit = 1.0
        warmup_steps = 0
    else:
        max_epochs = 1
        limit = args.warmup_steps + args.steps
        warmup_steps = args.warmup_steps

    cb = BenchCallback(warmup_steps, args.warmup_epochs if args.epochs else 0)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=limit,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        precision=cfg["optim"]["precision"],
        gradient_clip_val=cfg["optim"].get("gradient_clip_val", 0.0),
        accumulate_grad_batches=cfg["optim"].get("accumulate_grad_batches", 1),
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
    )
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler()
    sampler.start()
    t0 = time.perf_counter()
    trainer.fit(model, dm)
    wall = time.perf_counter() - t0
    sampler.stop()

    res = {"wall_s": wall, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    if args.epochs:
        steady = cb.epoch_stats[args.warmup_epochs:]
        its = [e["it_per_s"] for e in steady]
        sps = [e["samples_per_s"] for e in steady]
        res.update({
            "epochs_measured": len(steady),
            "it_per_s": statistics.fmean(its),
            "it_per_s_std": statistics.pstdev(its) if len(its) > 1 else 0.0,
            "samples_per_s": statistics.fmean(sps),
            "samples_per_s_std": statistics.pstdev(sps) if len(sps) > 1 else 0.0,
            "epoch_details": cb.epoch_stats,
        })
    else:
        total = sum(cb.step_times)
        res.update({
            "steps": len(cb.step_times),
            "it_per_s": len(cb.step_times) / total if total else 0.0,
            "samples_per_s": sum(cb.step_samples) / total if total else 0.0,
            "step_ms_mean": statistics.fmean(cb.step_times) * 1e3 if cb.step_times else 0.0,
            "step_ms_std": statistics.pstdev(cb.step_times) * 1e3 if len(cb.step_times) > 1 else 0.0,
        })
    res.update(sampler.summary())
    return res


def measure_data_path(dm, batch_size: int, n_batches: int = 10, n_reads: int = 200) -> dict:
    from torch_geometric.data import Batch

    ds = dm.train_dataset
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), n_reads, replace=False)
    t0 = time.perf_counter()
    graphs = [ds[int(i)] for i in idx]
    read_ms = (time.perf_counter() - t0) / n_reads * 1e3

    reps = max(1, batch_size // n_reads + 1)
    pool = (graphs * reps)[:batch_size]
    # time the loader's real collate (direct hetero collate for LMDB loaders)
    # and PyG's generic Batch.from_data_list for reference
    collate_fn = getattr(dm.train_dataloader(), "collate_fn", None) or Batch.from_data_list
    times, times_pyg = [], []
    for _ in range(n_batches):
        t0 = time.perf_counter()
        collate_fn(pool)
        times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        Batch.from_data_list(pool)
        times_pyg.append(time.perf_counter() - t0)
    return {
        "read_deserialize_ms_per_graph": read_ms,
        "collate_ms_per_batch": statistics.fmean(times) * 1e3,
        "collate_pyg_ms_per_batch": statistics.fmean(times_pyg) * 1e3,
        "collate_batch_size": batch_size,
    }


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="bench config JSON (merged over task defaults)")
    parser.add_argument("--mode", choices=["raw", "lightning"], default="raw")
    parser.add_argument("--set", action="append", default=[], help="override, e.g. optim.train_batch_size=512")
    parser.add_argument("--name", default=None, help="output stem (default: config stem plus overrides)")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--warmup_epoch", action="store_true",
                        help="raw mode: warm up for at least one full epoch (compiled models with bucketing)")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=0, help="lightning mode: measured epochs (0 = step budget)")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--profiler_steps", type=int, default=5)
    parser.add_argument("--skip_profiler", action="store_true")
    parser.add_argument("--skip_data_path", action="store_true")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = json.load(f)
    cfg = apply_overrides(cfg, args.set)

    name = args.name or "__".join(
        [Path(args.config).stem] + [s.replace("=", "-").replace(".", "_") for s in args.set] + [args.mode]
    )

    cfg, dm, model, task = build(cfg)
    n_params = count_parameters(model)
    print(f"[bench] {name}: task={task} params={n_params:,} batch={cfg['optim']['train_batch_size']} "
          f"workers={cfg['optim']['num_workers']} precision={cfg['optim']['precision']} "
          f"conv_fn={cfg['model']['conv_fn']} hidden={cfg['model']['hidden_size']} "
          f"encoder={cfg['model'].get('encoder_fn', 'none')}", flush=True)

    metrics = run_raw(cfg, dm, model, args) if args.mode == "raw" else run_lightning(cfg, dm, model, args)
    if not args.skip_data_path:
        metrics.update(measure_data_path(dm, cfg["optim"]["train_batch_size"]))

    result = {
        "name": name,
        "mode": args.mode,
        "task": task,
        "git_sha": git_sha(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "n_params": n_params,
        "summary": {
            "batch_size": cfg["optim"]["train_batch_size"],
            "num_workers": cfg["optim"]["num_workers"],
            "precision": str(cfg["optim"]["precision"]),
            "conv_fn": cfg["model"]["conv_fn"],
            "hidden_size": cfg["model"]["hidden_size"],
            "n_conv_layers": cfg["model"]["n_conv_layers"],
            "encoder_fn": cfg["model"].get("encoder_fn", "none"),
            "encoder_cutoff": cfg["model"].get("encoder_cutoff"),
            "encoder_max_neighbors": cfg["model"].get("encoder_max_neighbors"),
            "encoder_lmax": cfg["model"].get("encoder_lmax"),
            "train_lmdb": cfg["dataset"]["train_lmdb"],
        },
        "metrics": metrics,
        "config": cfg,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=str)

    keys = ["it_per_s", "samples_per_s", "atoms_per_s", "step_ms_mean", "data_wait_frac",
            "gpu_util_mean", "peak_mem_gb", "kernels_per_step", "sync_per_step",
            "gather_scatter_ms", "matmul_ms", "encoder_ms", "neighbor_radius_ms",
            "collate_ms_per_batch", "read_deserialize_ms_per_graph"]
    print("[bench] " + " ".join(
        f"{k}={metrics[k]:.3g}" for k in keys if metrics.get(k) is not None
    ))
    print(f"[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
