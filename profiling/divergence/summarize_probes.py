"""Build markdown tables from probe_<variant>.jsonl files."""
import glob
import json
import os
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
order = ["baseline", "bn_mom001", "mean_g", "mean_all", "sym_all", "clip1", "lr3e-4", "bn_eps1e-2"]
runs = {}
for f in glob.glob(f"{OUT}/probe_*.jsonl"):
    v = os.path.basename(f)[6:-6]
    runs[v] = [json.loads(l) for l in open(f)]

lines = []
P = lambda s="": lines.append(s)


def fmt(x):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:.4g}"


def gmax(d, suffix):
    vals = [v for k, v in d.items() if k.endswith(suffix) and not k.endswith("nonfinite")]
    return max(vals) if vals else float("nan")


# ---- Q2 detailed table for baseline ----
if "baseline" in runs:
    P("### Baseline continuation from the epoch-1 checkpoint (step 0 = end of epoch 1; 930 steps per epoch)")
    P()
    P("| step | train loss (sum of 6 MSE) | grad norm pre-clip | eval-mode val MSE | eval fp32 | batch-stat val MSE | max eval act atom/bond/global | max batch-stat act | max train act | BN dead ch (rv<1e-3) | global rv max | global rm max |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs["baseline"]:
        bn = r["bn"]
        dead = sum(v["dead"] for v in bn.values())
        g_rv = max(v["rv_max"] for k, v in bn.items() if k.endswith("global"))
        g_rm = max(v["rm_max"] for k, v in bn.items() if k.endswith("global"))
        ea = r["eval_acts"]
        P(f"| {r['step']} | {fmt(r['train_loss_sum'])} | {fmt(r['grad_norm_preclip'])} | {fmt(r['eval_mse'])} | {fmt(r['eval32_mse'])} | {fmt(r['bstat_mse'])} | "
          f"{fmt(gmax(ea,'atom'))} / {fmt(gmax(ea,'bond'))} / {fmt(gmax(ea,'global'))} | {fmt(gmax(r['bstat_acts'],''))} | {fmt(gmax(r['train_acts'],'')) if r['train_acts'] else '-'} | {dead} | {fmt(g_rv)} | {fmt(g_rm)} |")
    P()
    P("Per-block max |activation| in eval mode on the val subset (baseline):")
    P()
    keys = sorted({k for r in runs["baseline"] for k in r["eval_acts"] if not k.endswith("nonfinite")})
    P("| step | " + " | ".join(keys) + " |")
    P("|---|" + "---|" * len(keys))
    for r in runs["baseline"]:
        P(f"| {r['step']} | " + " | ".join(fmt(r["eval_acts"].get(k)) for k in keys) + " |")
    P()
    P("BN dead channels (running_var < 1e-3) per block and node type (baseline):")
    P()
    bkeys = sorted(runs["baseline"][0]["bn"].keys())
    P("| step | " + " | ".join(bkeys) + " |")
    P("|---|" + "---|" * len(bkeys))
    for r in runs["baseline"]:
        P(f"| {r['step']} | " + " | ".join(str(r["bn"][k]["dead"]) for k in bkeys) + " |")
    P()
    P("BN running_var min per block and node type (baseline):")
    P()
    P("| step | " + " | ".join(bkeys) + " |")
    P("|---|" + "---|" * len(bkeys))
    for r in runs["baseline"]:
        P(f"| {r['step']} | " + " | ".join(fmt(r["bn"][k]["rv_min"]) for k in bkeys) + " |")
    P()

# ---- Q4 comparison ----
P("### Hypothesis tests: eval-mode val MSE (fixed 50-batch val subset) per probe step")
P()
vs = [v for v in order if v in runs]
steps = sorted({r["step"] for v in vs for r in runs[v]})
P("| step | " + " | ".join(vs) + " |")
P("|---|" + "---|" * len(vs))
for s in steps:
    row = []
    for v in vs:
        m = [r for r in runs[v] if r["step"] == s]
        row.append(fmt(m[0]["eval_mse"]) if m else "")
    P(f"| {s} | " + " | ".join(row) + " |")
P()
P("Batch-stat BN val MSE per probe step (same runs):")
P()
P("| step | " + " | ".join(vs) + " |")
P("|---|" + "---|" * len(vs))
for s in steps:
    row = []
    for v in vs:
        m = [r for r in runs[v] if r["step"] == s]
        row.append(fmt(m[0]["bstat_mse"]) if m else "")
    P(f"| {s} | " + " | ".join(row) + " |")
P()
P("Train loss (running mean over the 150 steps before the probe) and grad norm before clipping:")
P()
P("| step | " + " | ".join(f"{v} loss / gnorm" for v in vs) + " |")
P("|---|" + "---|" * len(vs))
for s in steps:
    row = []
    for v in vs:
        m = [r for r in runs[v] if r["step"] == s]
        row.append(f"{fmt(m[0]['train_loss_sum'])} / {fmt(m[0]['grad_norm_preclip'])}" if m else "")
    P(f"| {s} | " + " | ".join(row) + " |")
P()
P("Dead BN channels (running_var < 1e-3, all 72 BN modules) and max eval-mode activation:")
P()
P("| step | " + " | ".join(f"{v} dead / act" for v in vs) + " |")
P("|---|" + "---|" * len(vs))
for s in steps:
    row = []
    for v in vs:
        m = [r for r in runs[v] if r["step"] == s]
        if m:
            dead = sum(x["dead"] for x in m[0]["bn"].values())
            row.append(f"{dead} / {fmt(gmax(m[0]['eval_acts'], ''))}")
        else:
            row.append("")
    P(f"| {s} | " + " | ".join(row) + " |")
P()
P("Summary per variant: max eval-mode val MSE over probes at steps 150..930, and whether it stayed below 1:")
P()
P("| variant | max eval MSE (steps 150-930) | min | final (930) | stayed < 1 | max batch-stat MSE |")
P("|---|---|---|---|---|---|")
for v in vs:
    rs = [r for r in runs[v] if 150 <= r["step"] <= 930]
    if not rs:
        continue
    mx = max(r["eval_mse"] for r in rs)
    mn = min(r["eval_mse"] for r in rs)
    fin = [r for r in rs if r["step"] == 930]
    P(f"| {v} | {fmt(mx)} | {fmt(mn)} | {fmt(fin[0]['eval_mse']) if fin else '-'} | {'yes' if mx < 1 else 'no'} | {fmt(max(r['bstat_mse'] for r in rs))} |")

txt = "\n".join(lines)
open(f"{OUT}/probe_summary.md", "w").write(txt)
print(txt)
