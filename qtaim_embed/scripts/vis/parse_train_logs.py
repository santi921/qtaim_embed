"""Parse Lightning RichProgressBar stdout dumps (SLURM job logs) into tables.

The training scripts write a Rich live display to stdout. Redirected to a file
that becomes hundreds of MB of ANSI escape codes and redraws of the same bar,
several ranks interleaved. This module streams such a file once and recovers:

- the config dicts the script logs at startup (dataset / model / optim)
- one row per (stream, epoch): batches seen, epoch wall time, it/s, and the
  progress-bar metrics (val_loss, val_r2, val_mse, train_loss, train_mse)
- every ``trainer.test`` results table
- NCCL / Python errors, wandb offline run ids

Usage::

    python -m qtaim_embed.scripts.vis.parse_train_logs data/logs_0914
    python -m qtaim_embed.scripts.vis.parse_train_logs run.log --csv out/

or from Python::

    from qtaim_embed.scripts.vis.parse_train_logs import parse_log, summarize_dir
    run = parse_log("data/logs_0914/gcdb_h2_emb128.log")
    run.epochs            # per-epoch DataFrame (rank-0 stream)
    run.test              # test tables, one row per table
    summarize_dir("data/logs_0914")

Metric semantics (Lightning progress bar): during epoch N the bar shows the
val_* values from the end of epoch N-1 until validation of epoch N finishes.
``epochs`` therefore reports, for epoch N, the val_* values seen in the first
render of epoch N+1 (or the last render of N for the final epoch, see the
``val_source`` column). ``train_loss`` is the last step loss of the epoch;
``train_mse`` is the epoch-level torchmetrics value.

Streams: renders are attributed to the ``v_num`` shown in the bar. Under a
multi-task SLURM launch only the process holding the wandb run shows an id;
every other process shows ``None`` and shares that one stream, so the ``None``
table can blend several processes. ``RunLog.epochs`` uses the named stream.
"""

from __future__ import annotations

import ast
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
# header only; elapsed / eta / rate / metrics follow in a window that Rich may
# wrap over several lines when the terminal is narrow
EPOCH_RE = re.compile(r"Epoch (?P<epoch>\d+)/(?P<max_epoch>\d+)\s+\S+\s+(?P<step>\d+)/(?P<total>\d+)")
TIME_RE = re.compile(r"(?<![\d.])\d+:\d\d(?::\d\d)?(?![\d.])|-:--:--")
RATE_RE = re.compile(r"([\d.]+)it/s")
KV_RE = re.compile(r"(?P<key>v_num|(?:train|val|test)_[A-Za-z0-9_]+):\s*(?P<val>\S+)")
V_NUM_RE = re.compile(r"[A-Za-z0-9]+")
WINDOW_STOPS = ("Validation", "Sanity", "Testing", "INFO -", "[rank", "wandb:")
TEST_ROW_RE = re.compile(r"(test_[A-Za-z0-9_]+)[\s│|]+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)")
CONFIG_RE = re.compile(r"^INFO - \S+ - (dataset|model|optim)\s+(\{.*\})\s*$", re.M)
WANDB_RUN_RE = re.compile(r"offline-run-\d{8}_\d{6}-[a-z0-9]+")
NCCL_TIMEOUT_RE = re.compile(
    r"\[Rank (\d+)\] Watchdog caught collective operation timeout: "
    r"WorkNCCL\(SeqNum=(\d+), OpType=(\w+)"
)
ERROR_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "DUE TO TIME LIMIT",
    "CANCELLED",
    "Killed",
)
PROGRESS_KEYS = ("val_loss", "val_r2", "val_mse", "val_mae", "train_loss", "train_mse")

CHUNK_BYTES = 32 * 1024 * 1024
CARRY_CHARS = 4096
METRIC_WINDOW = 800


def _to_float(token: str) -> float:
    token = token.strip().rstrip(",")
    if token in ("None", "nan", "NaN"):
        return math.nan
    if "…" in token:  # Rich truncated an over-wide value
        return math.nan
    try:
        return float(token)
    except ValueError:
        return math.nan


def _hms_to_seconds(text: str) -> float:
    parts = text.split(":")
    if not all(p.isdigit() for p in parts):
        return math.nan
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return float(total)


def iter_clean_text(path: Path, chunk_bytes: int = CHUNK_BYTES) -> Iterator[str]:
    """Yield ANSI-stripped text chunks with carriage returns turned into newlines."""
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(chunk_bytes)
            if not raw:
                return
            text = raw.decode("utf-8", errors="replace")
            text = ANSI_RE.sub("", text).replace("\r", "\n")
            yield text


@dataclass
class Render:
    stream: str
    epoch: int
    max_epoch: int
    step: int
    total: int
    elapsed_s: float
    rate: float
    metrics: Dict[str, float]


@dataclass
class RunLog:
    path: Path
    config: Dict[str, dict] = field(default_factory=dict)
    renders: List[Render] = field(default_factory=list)
    test_tables: List[Dict[str, float]] = field(default_factory=list)
    wandb_runs: List[str] = field(default_factory=list)
    nccl_timeouts: List[Dict[str, object]] = field(default_factory=list)
    error_lines: List[str] = field(default_factory=list)
    markers: Dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def streams(self) -> List[str]:
        return sorted({r.stream for r in self.renders})

    @property
    def rank0_stream(self) -> Optional[str]:
        """The stream whose progress bar shows a logger version (rank 0 under DDP)."""
        named = [s for s in self.streams if s != "None"]
        if len(named) == 1:
            return named[0]
        if not named and self.streams:
            return self.streams[0]
        return named[0] if named else None

    def epoch_table(self, stream: Optional[str] = None) -> pd.DataFrame:
        stream = stream or self.rank0_stream
        rows = [r for r in self.renders if r.stream == stream]
        if not rows:
            return pd.DataFrame()
        by_epoch: Dict[int, List[Render]] = {}
        for r in rows:
            by_epoch.setdefault(r.epoch, []).append(r)
        epochs = sorted(by_epoch)
        out = []
        for i, ep in enumerate(epochs):
            rs = by_epoch[ep]
            first, last = rs[0], rs[-1]
            done = [r for r in rs if r.step >= r.total]
            nxt = by_epoch[epochs[i + 1]][0] if i + 1 < len(epochs) else None
            src = nxt if nxt is not None else last
            row = {
                "epoch": ep,
                "max_epochs": first.max_epoch + 1,
                "batches_seen": max(r.step for r in rs),
                "batches_per_epoch": first.total,
                "completed": bool(done),
                "epoch_time_s": done[0].elapsed_s if done else math.nan,
                "it_per_s": max(r.rate for r in rs),
                "val_source": "next_epoch" if nxt is not None else "last_render",
            }
            for k in ("val_loss", "val_r2", "val_mse", "val_mae", "train_mse"):
                row[k] = src.metrics.get(k, math.nan)
            row["train_loss"] = last.metrics.get("train_loss", math.nan)
            out.append(row)
        df = pd.DataFrame(out)
        df.insert(0, "run", self.name)
        df.insert(1, "stream", stream)
        return df

    @property
    def epochs(self) -> pd.DataFrame:
        return self.epoch_table()

    @property
    def test(self) -> pd.DataFrame:
        if not self.test_tables:
            return pd.DataFrame()
        df = pd.DataFrame(self.test_tables)
        df.insert(0, "run", self.name)
        df.insert(1, "table", range(len(df)))
        return df

    @property
    def status(self) -> str:
        if self.nccl_timeouts:
            base = "nccl_timeout"
        elif self.error_lines:
            base = "error"
        else:
            base = "ok"
        if self.test_tables:
            return f"tested ({base})" if base != "ok" else "tested"
        if self.markers.get("Model fitted, testing"):
            return f"fitted ({base})"
        return f"unfinished ({base})" if base != "ok" else "unfinished"

    def summary(self) -> Dict[str, object]:
        ep = self.epochs
        model = self.config.get("model", {})
        optim = self.config.get("optim", {})
        dataset = self.config.get("dataset", {})
        row: Dict[str, object] = {
            "run": self.name,
            "status": self.status,
            "conv_fn": model.get("conv_fn"),
            "n_conv_layers": model.get("n_conv_layers"),
            "embedding_size": model.get("embedding_size"),
            "hidden_size": model.get("hidden_size"),
            "lr": model.get("lr"),
            "batch_size": optim.get("train_batch_size", dataset.get("train_batch_size")),
            "precision": optim.get("precision"),
            "n_targets": len(model.get("target_dict", {}).get("atom", [])),
            "streams": len(self.streams),
            "wandb_runs": len(self.wandb_runs),
            "nccl_timeouts": len(self.nccl_timeouts),
        }
        if ep.empty:
            return row
        row["epochs_seen"] = int(ep["epoch"].max()) + 1
        row["epochs_completed"] = int(ep["completed"].sum())
        row["batches_per_epoch"] = int(ep["batches_per_epoch"].iloc[0])
        row["median_epoch_time_min"] = float(ep["epoch_time_s"].median() / 60)
        valid = ep.dropna(subset=["val_loss"])
        if not valid.empty:
            best = valid.loc[valid["val_loss"].idxmin()]
            row["best_val_loss"] = float(best["val_loss"])
            row["best_val_epoch"] = int(best["epoch"])
            row["best_val_r2"] = float(valid["val_r2"].max())
            row["last_val_loss"] = float(valid["val_loss"].iloc[-1])
            row["last_val_r2"] = float(valid["val_r2"].iloc[-1])
        row["last_train_mse"] = float(ep["train_mse"].dropna().iloc[-1]) if ep["train_mse"].notna().any() else math.nan
        if self.test_tables:
            t = self.test_tables[0]
            row["test_mae"] = t.get("test_mae")
            row["test_r2"] = t.get("test_r2")
            row["test_mse"] = t.get("test_mse")
        return row


def _parse_renders(text: str, stream_hint: List[str]) -> List[Render]:
    renders = []
    matches = list(EPOCH_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        window = text[m.end(): min(end, m.end() + METRIC_WINDOW)]
        for stop in WINDOW_STOPS:
            cut = window.find(stop)
            if cut >= 0:
                window = window[:cut]
        times = TIME_RE.findall(window)
        elapsed = _hms_to_seconds(times[0]) if times else math.nan
        rate_m = RATE_RE.search(window)
        rate = _to_float(rate_m.group(1)) if rate_m else math.nan
        cleaned = RATE_RE.sub(" ", TIME_RE.sub(" ", window)).replace("•", " ")
        metrics: Dict[str, float] = {}
        stream = stream_hint[0]
        for kv in KV_RE.finditer(cleaned):
            key, val = kv.group("key"), kv.group("val")
            if key == "v_num":
                vm = V_NUM_RE.match(val)
                stream = vm.group(0) if vm else "None"
                stream_hint[0] = stream
            elif key.startswith("test_"):
                continue
            else:
                metrics[key] = _to_float(val)
        renders.append(
            Render(
                stream=stream,
                epoch=int(m.group("epoch")),
                max_epoch=int(m.group("max_epoch")),
                step=int(m.group("step")),
                total=int(m.group("total")),
                elapsed_s=elapsed,
                rate=rate,
                metrics=metrics,
            )
        )
    return renders


def _parse_test_tables(text: str) -> List[Dict[str, float]]:
    tables = []
    for block in text.split("Test metric")[1:]:
        block = block.split("└", 1)[0]
        rows = {k: float(v) for k, v in TEST_ROW_RE.findall(block)}
        if rows:
            tables.append(rows)
    return tables


def parse_log(path: str | Path) -> RunLog:
    """Stream one log file and return a RunLog with renders, tests, config, errors."""
    path = Path(path)
    run = RunLog(path=path)
    carry = ""
    stream_hint = ["None"]
    seen_wandb: set = set()
    for chunk in iter_clean_text(path):
        text = carry + chunk
        # process up to the last render start so a render split across chunks is kept whole
        cut = max(len(text) - CARRY_CHARS, 0)
        last_epoch = text.rfind("Epoch ", 0, len(text))
        if last_epoch > cut:
            cut = last_epoch
        head, carry = text[:cut], text[cut:]
        _consume(head, run, stream_hint, seen_wandb)
    _consume(carry, run, stream_hint, seen_wandb)
    return run


def _consume(text: str, run: RunLog, stream_hint: List[str], seen_wandb: set) -> None:
    if not text:
        return
    run.renders.extend(_parse_renders(text, stream_hint))
    run.test_tables.extend(_parse_test_tables(text))
    for m in CONFIG_RE.finditer(text):
        section = m.group(1)
        if section not in run.config:
            try:
                run.config[section] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                pass
    for m in WANDB_RUN_RE.finditer(text):
        if m.group(0) not in seen_wandb:
            seen_wandb.add(m.group(0))
            run.wandb_runs.append(m.group(0))
    for m in NCCL_TIMEOUT_RE.finditer(text):
        run.nccl_timeouts.append({"rank": int(m.group(1)), "seq": int(m.group(2)), "op": m.group(3)})
    for pat in ERROR_PATTERNS:
        n = text.count(pat)
        if n:
            run.error_lines.extend([pat] * n)
    for marker in ("Fitting model", "Model fitted, testing", "syncing is set to `offline`"):
        n = text.count(marker)
        if n:
            run.markers[marker] = run.markers.get(marker, 0) + n


def summarize_dir(directory: str | Path, pattern: str = "*.log") -> pd.DataFrame:
    """One summary row per log file in a directory."""
    runs = [parse_log(p) for p in sorted(Path(directory).glob(pattern))]
    return pd.DataFrame([r.summary() for r in runs])


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", help="log files or directories")
    parser.add_argument("--csv", default=None, help="directory to write <run>_epochs.csv / summary.csv")
    parser.add_argument("--all-streams", action="store_true", help="print every rank stream, not just rank 0")
    args = parser.parse_args(argv)

    files: List[Path] = []
    for p in args.paths:
        p = Path(p)
        files.extend(sorted(p.glob("*.log")) if p.is_dir() else [p])

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    summaries = []
    for f in files:
        run = parse_log(f)
        summaries.append(run.summary())
        print(f"\n=== {run.name}  [{run.status}]  streams={run.streams}  wandb_runs={len(run.wandb_runs)}")
        if run.nccl_timeouts:
            ops = sorted({(d["rank"], d["op"]) for d in run.nccl_timeouts})
            print(f"NCCL collective timeouts: {ops}")
        streams = run.streams if args.all_streams else [run.rank0_stream]
        for s in streams:
            df = run.epoch_table(s)
            if df.empty:
                continue
            cols = ["stream", "epoch", "batches_seen", "batches_per_epoch", "epoch_time_s",
                    "val_loss", "val_r2", "val_mse", "train_loss", "train_mse", "val_source"]
            print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
            if args.csv:
                Path(args.csv).mkdir(parents=True, exist_ok=True)
                df.to_csv(Path(args.csv) / f"{run.name}_{s}_epochs.csv", index=False)
        if run.test_tables:
            print("test tables (one per printing rank):")
            print(run.test.T.to_string(float_format=lambda x: f"{x:.4f}"))
            if args.csv:
                run.test.to_csv(Path(args.csv) / f"{run.name}_test.csv", index=False)
    summary = pd.DataFrame(summaries)
    print("\n=== summary")
    print(summary.T.to_string())
    if args.csv:
        summary.to_csv(Path(args.csv) / "summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
