"""Aggregate checkpoint metrics across seeds using population standard deviations."""

import os
import json
import argparse
import numpy as np
import torch

METRICS = [
    ("severity_acc", "↑", "random .333"),
    ("detect_acc", "↑", "random .333"),
    ("side_acc", "↑", "random .500"),
    ("count_mae", "↓", "naive=mean"),
    ("nobj_mae", "↓", "naive=mean"),
    ("nsmall_mae", "↓", "naive=mean"),
]


def collect_uwm(runs, name, seeds):
    """Return {metric: [values across seeds]} for the UWM model."""
    acc = {m[0]: [] for m in METRICS}
    found = 0
    for s in seeds:
        ck = os.path.join(runs, f"{name}_seed{s}", "best_uwm.pt")
        if not os.path.exists(ck):
            print(f"  [warn] missing UWM ckpt for seed {s}: {ck}")
            continue
        vm = torch.load(ck, map_location="cpu",
                        weights_only=False)["val_metrics"]
        for m in METRICS:
            if m[0] in vm:
                acc[m[0]].append(float(vm[m[0]]))
        found += 1
    return acc, found


def collect_baseline(label_dir, seeds, which):
    """which = 'onehot' or 'naive'."""
    acc = {m[0]: [] for m in METRICS}
    found = 0
    for s in seeds:
        p = os.path.join(label_dir, f"baseline_comparison_seed{s}.json")
        if not os.path.exists(p):
            print(f"  [warn] missing baseline file for seed {s}: {p}")
            continue
        with open(p) as f:
            d = json.load(f)
        sub = d.get(which)
        if sub is None:
            continue
        for m in METRICS:
            if m[0] in sub and sub[m[0]] is not None:
                acc[m[0]].append(float(sub[m[0]]))
        found += 1
    return acc, found


def fmt(vals):
    if not vals:
        return "    n/a"
    a = np.array(vals)
    if len(a) == 1:
        return f"{a[0]:.3f}"
    return f"{a.mean():.3f}±{a.std():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/weatherfilm_runs")
    ap.add_argument("--uwm-name", default="uwm_main")
    ap.add_argument("--label-dir",
                    default="data/understanding_labels")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print("=" * 70)
    print(f"Multi-seed aggregation over seeds {args.seeds}")
    print("=" * 70)

    uwm, n_uwm = collect_uwm(args.runs, args.uwm_name, args.seeds)
    oh, n_oh = collect_baseline(args.label_dir, args.seeds, "onehot")
    nv, n_nv = collect_baseline(args.label_dir, args.seeds, "naive")
    print(f"  collected: UWM={n_uwm} seeds, one-hot={n_oh}, naive={n_nv}\n")

    print(f"  {'metric':14s} {'UWM(VLM)':>16s} {'one-hot':>16s} "
          f"{'naive':>16s}   note")
    print(f"  {'-'*74}")
    for key, arrow, note in METRICS:
        print(f"  {key:14s} {fmt(uwm[key]):>16s} {fmt(oh[key]):>16s} "
              f"{fmt(nv[key]):>16s}   {arrow} {note}")

    # Save per-seed values.
    raw = {"seeds": args.seeds,
           "uwm": {k: uwm[k] for k, _, _ in METRICS},
           "onehot": {k: oh[k] for k, _, _ in METRICS},
           "naive": {k: nv[k] for k, _, _ in METRICS}}
    out = os.path.join(args.label_dir, "multiseed_summary.json")
    with open(out, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n  saved raw per-seed values -> {out}")
    print("  (table cells are mean±std; report these in the paper)")


if __name__ == "__main__":
    main()
