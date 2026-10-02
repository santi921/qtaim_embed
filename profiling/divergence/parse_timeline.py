"""Q1: per-epoch timeline from train.log progress lines and from the TensorBoard
event file (which also has lr-Adam from LearningRateMonitor)."""
import bisect
import collections
import re
import sys

from common import OUT, ROOT

LOG = ROOT + "/profiling/train_runs/b128_lr0.001/train.log"
TB = ROOT + "/profiling/train_runs/b128_lr0.001/test_logs/version_0"

# ---- train.log: keep, per epoch, the last progress line at 100% that carries val_* ----
pat = re.compile(r"Epoch (\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+).*?v_num=\S+, (.*?)\]")
per_epoch = {}
raw = open(LOG, "rb").read().decode("utf-8", errors="replace").replace("\r", "\n")
for line in raw.split("\n"):
    m = pat.search(line)
    if not m:
        continue
    ep, pct, it, tot, kv = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    d = dict(p.split("=") for p in kv.split(", ") if "=" in p)
    # the val_* on a 100% line of epoch e are the metrics computed at the end of epoch e
    if it == tot:
        per_epoch[ep] = d

lines = []
lines.append("## Q1a. Timeline parsed from train.log (last 100% progress line of each epoch)")
lines.append("")
lines.append("| epoch | val_loss | val_r2 | val_mse (RMSE) | train_loss | train_mse (RMSE) |")
lines.append("|---|---|---|---|---|---|")
for ep in sorted(per_epoch):
    d = per_epoch[ep]
    lines.append(f"| {ep} | {d.get('val_loss')} | {d.get('val_r2')} | {d.get('val_mse')} | {d.get('train_loss')} | {d.get('train_mse')} |")
lines.append("")
lines.append("Note: the progress bar for epoch e shows val metrics from the end of epoch e once the bar reaches 100%; "
             "epoch 27 has no 100% line (run killed by SIGTERM mid-epoch).")

# ---- TensorBoard events ----
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(TB, size_guidance={"scalars": 0})
ea.Reload()
rows = collections.defaultdict(dict)
for t in ["val_loss", "val_r2", "val_mae", "val_mse", "train_loss", "train_r2", "train_mae", "train_mse", "epoch"]:
    for e in ea.Scalars(t):
        if t == "val_mae" and e.value == 1e7:
            continue  # on_train_epoch_end logs val_mae=1e7 at epoch 0 (placeholder)
        rows[e.step][t] = e.value
lr = ea.Scalars("lr-Adam")
lr_steps = [e.step for e in lr]
lines.append("")
lines.append("## Q1b. Timeline from TensorBoard events (includes val_mae, the monitored metric, and lr-Adam)")
lines.append("")
lines.append("| step | epoch | val_loss | val_r2 | val_mae | val_mse (RMSE) | train_loss | train_r2 | train_mae | train_mse (RMSE) | lr |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for s in sorted(rows):
    r = rows[s]
    if "val_loss" not in r:
        continue
    i = bisect.bisect_right(lr_steps, s) - 1
    vals = [f"{r.get(k, float('nan')):.4g}" for k in ["val_loss", "val_r2", "val_mae", "val_mse", "train_loss", "train_r2", "train_mae", "train_mse"]]
    lines.append(f"| {s} | {int(r.get('epoch', -1))} | " + " | ".join(vals) + f" | {lr[i].value:.3g} |")
lines.append("")
prev = None
changes = []
for e in lr:
    if e.value != prev:
        changes.append((e.step, e.value))
        prev = e.value
lines.append("lr-Adam changes (step, value): " + ", ".join(f"({s}, {v:.3g})" for s, v in changes))
lines.append("930 optimizer steps per epoch, so the drops at steps 12099 and 22349 are the ReduceLROnPlateau "
             "(factor 0.6, patience 10, monitor val_mae) firing after epochs 12 and 23: 1e-3 -> 6e-4 -> 3.6e-4.")

# per-target val r2 at the first few epochs
lines.append("")
lines.append("Per-target val R2 (epochs 0,1,2,3):")
lines.append("")
lines.append("| target | ep0 | ep1 | ep2 | ep3 |")
lines.append("|---|---|---|---|---|")
for t in ["charge_adch", "charge_hirshfeld", "charge_cm5", "charge_becke", "charge_mulliken_orca", "charge_loewdin_orca"]:
    ev = ea.Scalars(f"val_r2_atom_{t}")
    lines.append(f"| {t} | " + " | ".join(f"{ev[i].value:.4g}" for i in range(4)) + " |")

txt = "\n".join(lines)
open(OUT + "/timeline.md", "w").write(txt)
print(txt)
