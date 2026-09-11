#!/usr/bin/env python3
"""Render Track A experiment tables (markdown) from profiling/bench_results.

Usage: python profiling/report_track_a.py [--results profiling/bench_results]
Prints one markdown table per experiment: E1 batch sweep, E2 encoder table,
A5 before/after neighbor build, E3 layer formulations, E4 bucketing, E5
collate/deser, E6 neighbor build.
"""
import argparse
import json
from pathlib import Path


def f(v, nd=3):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.{nd}g}"
    return str(v)


def table(cols, rows):
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f(getter(r)) for _, getter in cols) + " |")
    return "\n".join(out)


def load_bench(results):
    rows = {}
    for p in sorted(results.glob("*.json")):
        d = json.load(open(p))
        if isinstance(d, dict) and "name" in d and "metrics" in d:
            r = dict(d["summary"]); r.update(d["metrics"]); r["name"] = d["name"]; r["mode"] = d["mode"]
            rows[d["name"]] = r
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="profiling/bench_results")
    a = ap.parse_args(argv)
    R = Path(a.results)
    B = load_bench(R)
    m = lambda k: (lambda r: r.get(k))

    print("## E1 batch, precision, workers (tm_react node, hidden 128/256, encoder none, raw mode)\n")
    e1 = [r for n, r in B.items() if n.startswith("tm_react__none")]
    e1.sort(key=lambda r: (r["hidden_size"], r["batch_size"], r["precision"], r["num_workers"], r["name"]))
    print(table([("run", m("name")), ("batch", m("batch_size")), ("workers", m("num_workers")), ("precision", m("precision")),
                 ("samples/s", m("samples_per_s")), ("atoms/s", m("atoms_per_s")), ("step ms", m("step_ms_mean")),
                 ("data wait", m("data_wait_frac")), ("GPU util %", m("gpu_util_mean")), ("peak GB", m("peak_mem_gb")),
                 ("kernels/step", m("kernels_per_step")), ("gather+scatter ms", m("gather_scatter_ms")), ("matmul ms", m("matmul_ms")),
                 ("collate ms", m("collate_ms_per_batch"))], e1))
    tm = [r for n, r in B.items() if n.startswith("tmqm_graph")]
    tm.sort(key=lambda r: (r["mode"], r["batch_size"]))
    print("\n## E1 TMQM graph-level continuity (baseline 25.6 it/s at batch 128)\n")
    print(table([("run", m("name")), ("mode", m("mode")), ("batch", m("batch_size")), ("it/s", m("it_per_s")), ("samples/s", m("samples_per_s")),
                 ("data wait", m("data_wait_frac")), ("GPU util %", m("gpu_util_mean")), ("peak GB", m("peak_mem_gb"))], tm))

    print("\n## E2 encoder table (batch 128/512 tm_react, 16/64 H7; before the dense neighbor build)\n")
    e2 = [r for n, r in B.items() if ("schnet" in n or "dimenetpp" in n or "equivariant" in n or n.startswith("h7__none")) and not n.endswith(("__dense_nb", "__cw"))]
    e2.sort(key=lambda r: (r["train_lmdb"], r["encoder_fn"], r["name"]))
    enc_cols = [("run", m("name")), ("batch", m("batch_size")), ("samples/s", m("samples_per_s")), ("atoms/s", m("atoms_per_s")),
                ("step ms", m("step_ms_mean")), ("peak GB", m("peak_mem_gb")), ("GPU util %", m("gpu_util_mean")), ("syncs/step", m("sync_per_step")),
                ("encoder ms", m("encoder_ms")), ("neighbor ms", m("neighbor_radius_ms")), ("gather+scatter ms", m("gather_scatter_ms")), ("matmul ms", m("matmul_ms"))]
    print(table(enc_cols, e2))

    print("\n## A5-1 dense per-molecule neighbor build: before vs after (same config)\n")
    rows = []
    for n, r in sorted(B.items()):
        if n.endswith("__dense_nb"):
            base = B.get(n[:-len("__dense_nb")]) or B.get(n[:-len("__dense_nb")].replace("__w8", ""))
            rows.append({"name": n[:-len("__dense_nb")], "before": base["samples_per_s"] if base else None, "after": r["samples_per_s"],
                         "gain": (r["samples_per_s"] / base["samples_per_s"]) if base else None,
                         "nb_before": base["neighbor_radius_ms"] if base else None, "nb_after": r["neighbor_radius_ms"],
                         "sync_before": base["sync_per_step"] if base else None, "sync_after": r["sync_per_step"],
                         "mem_before": base["peak_mem_gb"] if base else None, "mem_after": r["peak_mem_gb"]})
    print(table([("run", m("name")), ("samples/s before", m("before")), ("samples/s after", m("after")), ("gain", m("gain")),
                 ("neighbor ms before", m("nb_before")), ("neighbor ms after", m("nb_after")), ("syncs before", m("sync_before")), ("syncs after", m("sync_after")),
                 ("peak GB before", m("mem_before")), ("peak GB after", m("mem_after"))], rows))

    print("\n## A4 direct collate in the real step (__direct_collate) vs Batch.from_data_list\n")
    rows = []
    for n, r in sorted(B.items()):
        if n.endswith("__direct_collate"):
            base = B.get(n[:-len("__direct_collate")])
            rows.append({"name": n[:-len("__direct_collate")], "before": base["samples_per_s"] if base else None, "after": r["samples_per_s"],
                         "gain": (r["samples_per_s"] / base["samples_per_s"]) if base else None,
                         "wait_before": base["data_wait_frac"] if base else None, "wait_after": r["data_wait_frac"],
                         "util_before": base["gpu_util_mean"] if base else None, "util_after": r["gpu_util_mean"]})
    print(table([("run", m("name")), ("samples/s before", m("before")), ("samples/s after", m("after")), ("gain", m("gain")),
                 ("data wait before", m("wait_before")), ("data wait after", m("wait_after")), ("GPU util before", m("util_before")), ("GPU util after", m("util_after"))], rows))

    print("\n## A3 end-to-end (batch_norm on): ResidualBlock vs ResidualBlockDense with bucketing, eager and compiled (CUDA graphs)\n")
    rows = [r for n, r in sorted(B.items()) if "__bn__" in n]
    print(table([("run", m("name")), ("conv_fn", m("conv_fn")), ("batch", m("batch_size")), ("hidden", m("hidden_size")), ("encoder", m("encoder_fn")),
                 ("samples/s", m("samples_per_s")), ("atoms/s", m("atoms_per_s")), ("step ms", m("step_ms_mean")), ("data wait", m("data_wait_frac")),
                 ("GPU util %", m("gpu_util_mean")), ("peak GB", m("peak_mem_gb")), ("kernels/step", m("kernels_per_step")), ("gather+scatter ms", m("gather_scatter_ms")),
                 ("matmul ms", m("matmul_ms"))], rows))

    print("\n## A5-2 channel-wise equivariant tensor product (__cw) vs fully connected\n")
    rows = [r for n, r in sorted(B.items()) if "equivariant" in n and (n.endswith("__cw") or not n.endswith("__dense_nb"))]
    print(table([("run", m("name")), ("batch", m("batch_size")), ("samples/s", m("samples_per_s")), ("step ms", m("step_ms_mean")), ("peak GB", m("peak_mem_gb")),
                 ("encoder ms", m("encoder_ms")), ("matmul ms", m("matmul_ms"))], rows))

    p = R / "layers_e3.json"
    if p.exists():
        rows = json.load(open(p))
        ref = {(r["dataset"], r["batch_size"], r["hidden"], r["dtype"]): r["fwd_bwd_ms"] for r in rows if r["impl"] == "ref"}
        for r in rows:
            base = ref.get((r["dataset"], r["batch_size"], r["hidden"], r["dtype"]))
            r["speedup"] = base / r["fwd_bwd_ms"] if base and r["fwd_bwd_ms"] == r["fwd_bwd_ms"] else None
            r["shape"] = f"{r['dense_shape']['N_b']}x{r['dense_shape']['B_b']}"
            r["waste"] = r["dense_shape"]["pad_waste"]
        rows.sort(key=lambda r: (r["dataset"], r["batch_size"], r["hidden"], r["dtype"], ["ref", "fused", "dense", "dense_compile", "dense_cudagraph"].index(r["impl"])))
        print("\n## E3 conv-stack formulations, fwd+bwd of 8 hetero conv layers (same weights, parity asserted at fp32)\n")
        print(table([("dataset", m("dataset")), ("batch", m("batch_size")), ("hidden", m("hidden")), ("dtype", m("dtype")), ("impl", m("impl")),
                     ("fwd+bwd ms", m("fwd_bwd_ms")), ("speedup vs ref", m("speedup")), ("peak GB", m("peak_gb")), ("kernels/step", m("kernels_per_step")),
                     ("dense shape", m("shape")), ("pad waste", m("waste")), ("error", m("error"))], rows))

    print("\n## E4 padding waste of static-shape schemes (fraction of padded atom+bond rows that are padding)\n")
    for p in sorted(R.glob("bucketing_*.json")):
        d = json.load(open(p))
        print(f"\n{d['dataset']}: n_total {d['n_total']}, atoms mean {d['atoms']['mean']:.1f} max {d['atoms']['max']}, bonds mean {d['bonds']['mean']:.1f} max {d['bonds']['max']}\n")
        rows = d["buckets"]
        for r in rows:
            r["shapes"] = r.get("n_shapes", r["k"] if isinstance(r["k"], int) else None)
        print(table([("scheme", lambda r: f"quantile k={r['k']}" if isinstance(r["k"], int) else f"round-up {r['k'][4:]}"), ("static shapes", m("shapes")),
                     ("waste atoms", m("waste_atoms")), ("waste bonds", m("waste_bonds")), ("waste total", m("waste_total")),
                     ("ragged b256", m("ragged_frac_b256")), ("ragged b512", m("ragged_frac_b512")), ("ragged b1024", m("ragged_frac_b1024"))], rows))
        fl = d["floors"]
        print("\nfloors: " + ", ".join(f"{k} {v:.3f}" for k, v in fl.items()))

    p = R / "collate_e5.json"
    if p.exists():
        d = json.load(open(p))
        print(f"\n## E5 data path on one CPU thread ({d['dataset']})\n")
        print(table([("batch", m("batch_size")), ("Batch.from_data_list ms", m("pyg_ms")), ("direct collate ms", m("direct_ms")), ("speedup", m("speedup"))], d["collate"]))
        de = d["deser"]
        print(f"\ndeserialization per record ({de['n_records']} records): pickled HeteroData torch.load {de['heterodata_ms_per_record']:.3f} ms ({de['heterodata_kb']:.0f} KB); "
              f"flat tensor dict with weights_only=True {de['tensor_dict_ms_per_record']:.3f} ms ({de['tensor_dict_kb']:.0f} KB); ratio {de['speedup']:.2f}x")

    p = R / "neighbors_e6.json"
    if p.exists():
        rows = json.load(open(p))
        print("\n## E6 radius-graph build (edge sets asserted identical)\n")
        print(table([("dataset", m("dataset")), ("batch", m("batch_size")), ("cutoff", m("cutoff")), ("atoms", m("atoms")), ("edges", m("edges")), ("mean deg", m("mean_degree")),
                     ("chunked GPU ms", m("chunked_gpu_ms")), ("dense GPU ms", m("dense_gpu_ms")), ("speedup", lambda r: r["chunked_gpu_ms"] / r["dense_gpu_ms"]),
                     ("nonzero calls chunked", m("chunked_nonzero_calls")), ("nonzero calls dense", m("dense_nonzero_calls")),
                     ("chunked CPU1 ms", m("chunked_cpu1_ms")), ("dense CPU1 ms", m("dense_cpu1_ms")), ("dense peak GB", m("dense_gpu_peak_gb")),
                     ("triplets maxn16", m("triplets_maxn16_count")), ("triplets maxn32", m("triplets_maxn32_count")), ("triplets maxn32 ms", m("triplets_maxn32_ms"))], rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
