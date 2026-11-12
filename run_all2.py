#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run multi-seed Byzantine, BDD-IW and partial-participation experiments."""

import os
import sys
import json
import time
import argparse
import numpy as np
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FED = os.path.join(HERE, "fed_padu.py")

_LOG = []


def W(s=""):
    s = str(s)
    _LOG.append(s)
    print(s, flush=True)


def dump(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG))


def hms(sec):
    sec = int(sec)
    return f"{sec//3600}h{(sec%3600)//60:02d}m{sec%60:02d}s"


def human(n):
    n = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def cell(vals, nd):
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f}±{np.std(vals):.{nd}f}"



# stages

def stages(args):
    seeds = ["0"] if args.fast else ["0", "1", "2"]
    br = "25" if args.fast else "30"
    rounds = "40" if args.fast else "60"
    bdd_seeds = ["0"]

    bdd = ["--seeds", *bdd_seeds, "--exp", "0,A,B",
           "--tok-root", args.bdd_tok_root,
           "--label-dir", args.bdd_label_dir,
           "--client-split", "density", "--n-shards", "6",
           "--mode", "ram", "--rounds", rounds]

    return [
        ("byz_majority_ms", ["--seeds", *seeds, "--exp", "C",
                             "--byz-rounds", br]),
        ("byz_minority_ms", ["--seeds", *seeds, "--exp", "C",
                             "--byz-rounds", br, "--byz-ids", "2,3"]),
        ("bdd_main", bdd),
        ("participation", ["--seeds", *seeds, "--exp", "P",
                           "--rounds", rounds,
                           "--fracs", "1.0,0.5,0.3"]),
    ]


def preflight(args):
    """Check required input paths before starting experiments."""
    ok = True
    W("\nPre-flight checks")
    if not os.path.exists(FED):
        W("  MISSING fed_padu.py next to run_all2.py")
        return False
    W(f"  driver   {FED}  OK")
    for sp in ["train", "val", "test"]:
        for pat, root in [(f"{sp}_tokens.npy", args.bdd_tok_root),
                          (f"{sp}_meta.npz", args.bdd_tok_root),
                          (f"{sp}_understanding_labels.json", args.bdd_label_dir),
                          (f"{sp}_detectability_labels.json", args.bdd_label_dir)]:
            p = os.path.join(root, pat)
            if not os.path.exists(p):
                W(f"  MISSING {p}")
                ok = False
    if ok:
        W(f"  BDD-IW   tokens + labels present  OK")
        tot = sum(os.path.getsize(os.path.join(args.bdd_tok_root,
                                               f"{s}_tokens.npy"))
                  for s in ["train", "val", "test"])
        W(f"  BDD-IW   token store {human(tot)} (loaded into RAM, not VRAM)")
    else:
        W("  -> the bdd_main stage will be skipped; the rest can still run")
    return True


def run_stage(name, extra, root, args):
    out = os.path.join(root, name)
    done = os.path.join(out, "fed_results.json")
    if args.resume and os.path.exists(done):
        W(f"  [skip] {name}: already finished")
        return True
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, FED, "--out-dir", out] + extra
    W(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "8")
    r = subprocess.run(cmd, env=env)
    W(f"  [{name}] exit {r.returncode} in {hms(time.time()-t0)}")
    return r.returncode == 0


def load(root, name):
    p = os.path.join(root, name, "fed_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)



# aggregation

ATTACKS = ["none", "label_flip", "sign_flip", "gauss"]


def table_byz_multiseed(blobs, tex):
    W("\n" + "=" * 78)
    W("TABLE 6  Byzantine robustness, mean ± std over seeds")
    W("=" * 78)
    W("  Final-round global model on the XWOD test set. This replaces the "
      "single-seed\n  table from the first batch: the seed spread is what "
      "tells you whether a cell\n  is a mechanism or an accident.")

    for name, label in [
            ("byz_majority_ms",
             "adversaries hold the two largest shards (67% of the data)"),
            ("byz_minority_ms",
             "adversaries hold two mid-size shards (~20% of the data)")]:
        blob = blobs.get(name)
        if not blob:
            W(f"\n  [{name}] missing")
            continue
        seeds = sorted(blob.get("seeds", {}))
        first = blob["seeds"][seeds[0]]
        setup = first.get("expC_setup", {})
        C0 = first.get("expC", {})
        if not C0:
            W(f"\n  [{name}] no expC block")
            continue
        rules = list(next(iter(C0.values())).keys())
        W(f"\n  --- {label} ---")
        W(f"  {len(seeds)} seed(s); adversaries: {setup.get('byz_clients')}, "
          f"{setup.get('byz_data_share', 0)*100:.1f}% of the training data")

        for metric, mlabel in [("detect_acc", "Detectability accuracy"),
                               ("detect_bal", "Detectability balanced acc"),
                               ("side_acc", "Spatial (side) accuracy")]:
            W(f"\n  {mlabel}")
            W(f"  {'attack':<13}" + "".join(f"{r:>17}" for r in rules))
            for atk in ATTACKS:
                cells = ""
                for r in rules:
                    vals = []
                    for sd in seeds:
                        C = blob["seeds"][sd].get("expC", {})
                        g = (C.get(atk, {}).get(r, {}).get("global_final")
                             or C.get(atk, {}).get(r, {}).get("global"))
                        if g and g.get(metric) is not None:
                            vals.append(float(g[metric]))
                    cells += f"{cell(vals, 3):>17}"
                if cells.strip():
                    W(f"  {atk:<13}{cells}")

        # Flag large variation across seeds.
        W("\n  Seed spread of detectability accuracy (max - min across seeds):")
        W(f"  {'attack':<13}" + "".join(f"{r:>17}" for r in rules))
        for atk in ATTACKS:
            cells = ""
            for r in rules:
                vals = []
                for sd in seeds:
                    C = blob["seeds"][sd].get("expC", {})
                    g = (C.get(atk, {}).get(r, {}).get("global_final")
                         or C.get(atk, {}).get(r, {}).get("global"))
                    if g:
                        vals.append(float(g["detect_acc"]))
                cells += (f"{max(vals)-min(vals):>17.3f}" if len(vals) > 1
                          else f"{'--':>17}")
            if cells.strip():
                W(f"  {atk:<13}{cells}")
        W("  A large spread means that cell is seed noise, not a property of "
          "the aggregation rule.")

    W("\n  Compare each attack with the clean ('none') row under the same corruption setting.")


def table_bdd(blob, tex):
    W("\n" + "=" * 78)
    W("TABLE 7  Federated PaDU on BDD-IW (agents split by object density)")
    W("=" * 78)
    if not blob:
        W("  [bdd_main] missing — stage skipped or failed")
        return
    W("  BDD-IW has no weather labels, so agents are formed from object-density\n"
      "  bands: a stand-in for fleets working in different traffic regimes.\n"
      "  This checks that the federated conclusions are not an artifact of the\n"
      "  extreme-weather setting.")
    seeds = sorted(blob.get("seeds", {}))
    W(f"  {len(seeds)} seed(s)")

    r0 = blob["seeds"][seeds[0]]
    cl = r0.get("clients", [])
    if cl:
        tot = sum(c["n"] for c in cl)
        W(f"\n  {'agent':<14}{'n':>8}{'share':>9}{'obj/img':>10}{'JS(det)':>10}")
        for c in cl:
            W(f"  {c['client']:<14}{c['n']:>8}{c['n']/tot*100:>8.1f}%"
              f"{c['mean_objects']:>10.1f}{c['js_detect']:>10.4f}")
        sizes = [c["n"] for c in cl]
        objs = [c["mean_objects"] for c in cl]
        W(f"  size imbalance {max(sizes)/max(min(sizes),1):.1f}x, "
          f"objects/image {min(objs):.1f} to {max(objs):.1f} "
          f"({max(objs)/max(min(objs),1e-9):.1f}x)")

    def grab(key, path):
        vals = {}
        for sd in seeds:
            r = blob["seeds"][sd]
            m = path(r)
            if not m:
                continue
            for k in ["severity_acc", "detect_acc", "detect_bal", "side_acc",
                      "count_mae"]:
                if m.get(k) is not None:
                    vals.setdefault(k, []).append(float(m[k]))
        return vals

    rows = [
        ("Centralized (upper bound)", lambda r: r.get("exp0_centralized")),
        ("FedAvg", lambda r: (r.get("expA", {}).get("fedavg") or {}).get("global")),
        ("FedProx", lambda r: (r.get("expA", {}).get("fedprox") or {}).get("global")),
        ("FedAvg + fp16", lambda r: (r.get("expA", {}).get("fedavg_fp16") or {}).get("global")),
        ("FedAvg + int8", lambda r: (r.get("expA", {}).get("fedavg_int8") or {}).get("global")),
    ]
    W(f"\n  {'method':<28}{'SevAcc':>13}{'DetAcc':>13}{'DetBal':>13}"
      f"{'SideAcc':>13}{'CntMAE':>13}")
    base = None
    tex_rows = []
    for label, fn in rows:
        v = grab(label, fn)
        if not v:
            continue
        if base is None:
            base = v
        W(f"  {label:<28}{cell(v.get('severity_acc',[]),3):>13}"
          f"{cell(v.get('detect_acc',[]),3):>13}"
          f"{cell(v.get('detect_bal',[]),3):>13}"
          f"{cell(v.get('side_acc',[]),3):>13}"
          f"{cell(v.get('count_mae',[]),2):>13}")
        tex_rows.append((label, v))

    W(f"\n  {'method':<28}{'SevAcc':>13}{'DetAcc':>13}{'DetBal':>13}"
      f"{'SideAcc':>13}{'CntMAE':>13}   (personalized)")
    for label, fn in [
            ("Local-only",
             lambda r: ((r.get("expA", {}).get("local") or {})
                        .get("personalized", {}) or {}).get("pooled")),
            ("FedPer",
             lambda r: ((r.get("expA", {}).get("fedper") or {})
                        .get("personalized", {}) or {}).get("pooled"))]:
        v = grab(label, fn)
        if not v:
            continue
        W(f"  {label:<28}{cell(v.get('severity_acc',[]),3):>13}"
          f"{cell(v.get('detect_acc',[]),3):>13}"
          f"{cell(v.get('detect_bal',[]),3):>13}"
          f"{cell(v.get('side_acc',[]),3):>13}"
          f"{cell(v.get('count_mae',[]),2):>13}")

    if base:
        W("\n  Gap to the centralized upper bound on BDD-IW:")
        for label, fn in rows[1:]:
            v = grab(label, fn)
            parts = []
            for k in ["severity_acc", "detect_acc", "side_acc"]:
                if v.get(k) and base.get(k):
                    parts.append(f"{k.split('_')[0]} "
                                 f"{np.mean(v[k])-np.mean(base[k]):+.3f}")
            if parts:
                W(f"    {label:<26}{'   '.join(parts)}")
        W("\n  Compare these gaps with the XWOD gaps in TABLE 1. If they have "
          "the same sign and\n  rough magnitude, the federated result "
          "generalizes across domain, taxonomy and\n  a fourfold difference in "
          "object density.")


def table_participation(blob, tex):
    W("\n" + "=" * 78)
    W("TABLE 8  Dynamic participation (XWOD)")
    W("=" * 78)
    if not blob:
        W("  [participation] missing")
        return
    seeds = sorted(blob.get("seeds", {}))
    W(f"  Only a fraction C of the fleet uploads each round, as vehicles join\n"
      f"  and leave. {len(seeds)} seed(s).")
    keys = sorted({k for sd in seeds
                   for k in blob["seeds"][sd].get("expP", {})})
    W(f"\n  {'setting':<24}{'uplink total':>15}{'SevAcc':>13}{'DetAcc':>13}"
      f"{'DetBal':>13}{'SideAcc':>13}{'CntMAE':>13}")
    for k in keys:
        vals, up = {}, []
        for sd in seeds:
            r = blob["seeds"][sd].get("expP", {}).get(k)
            if not r:
                continue
            up.append(r.get("uplink_bytes_total", 0))
            g = r.get("global")
            if not g:
                continue
            for m in ["severity_acc", "detect_acc", "detect_bal", "side_acc",
                      "count_mae"]:
                if g.get(m) is not None:
                    vals.setdefault(m, []).append(float(g[m]))
        if not vals:
            continue
        W(f"  {k:<24}{human(np.mean(up)):>15}"
          f"{cell(vals.get('severity_acc',[]),3):>13}"
          f"{cell(vals.get('detect_acc',[]),3):>13}"
          f"{cell(vals.get('detect_bal',[]),3):>13}"
          f"{cell(vals.get('side_acc',[]),3):>13}"
          f"{cell(vals.get('count_mae',[]),2):>13}")
    W("\n  Sparse participation lowers total traffic proportionally. Report "
      "accuracy per byte,\n  not accuracy alone: if C=0.5 loses little but "
      "halves the uplink, that is the result.")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/fed_results")
    ap.add_argument("--bdd-tok-root", default="data/bdd_patch_tokens")
    ap.add_argument("--bdd-label-dir",
                    default="data/bdd_understanding_labels")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    t0 = time.time()
    W("=" * 78)
    W("run_all2.py — second batch of federated-PaDU experiments")
    W("=" * 78)
    W(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    W(f"root    : {args.root}")

    st = stages(args)
    if not args.aggregate_only:
        if not preflight(args):
            sys.exit(1)
        bdd_ok = os.path.exists(os.path.join(args.bdd_tok_root,
                                             "train_tokens.npy"))
        if args.only:
            want = {x.strip() for x in args.only.split(",")}
            st = [x for x in st if x[0] in want]
        if not bdd_ok:
            st = [x for x in st if x[0] != "bdd_main"]
        W(f"\nstages  : {[x[0] for x in st]}")
        W(f"estimate: {'~2 h' if args.fast else '~7 h'} on one RTX 4090\n")
        for i, (name, extra) in enumerate(st, 1):
            W(f"\n{'#'*78}\n# STAGE {i}/{len(st)}: {name}\n{'#'*78}")
            if not run_stage(name, extra, args.root, args):
                W(f"  !! stage {name} failed; continuing")
            dump(os.path.join(args.root, "RUN_ALL2.log"))

    W(f"\n\n{'#'*78}\n# AGGREGATED RESULTS (batch 2)\n{'#'*78}")
    blobs = {n: load(args.root, n) for n in
             ["byz_majority_ms", "byz_minority_ms", "bdd_main", "participation"]}
    tex = []
    table_byz_multiseed(blobs, tex)
    table_bdd(blobs.get("bdd_main"), tex)
    table_participation(blobs.get("participation"), tex)

    W("\n" + "=" * 78)
    W(f"TOTAL WALL TIME  {hms(time.time()-t0)}")
    W("=" * 78)
    W("  Batch 1 tables stay valid: FINAL_TABLES.txt is untouched.")

    final = os.path.join(args.root, "FINAL_TABLES2.txt")
    dump(os.path.join(args.root, "RUN_ALL2.log"))
    with open(final, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG))
    with open(os.path.join(args.root, "tables2.tex"), "w",
              encoding="utf-8") as f:
        f.write("% auto-generated by run_all2.py\n\n" + "\n\n".join(tex) + "\n")
    print(f"\n  tables : {final}")
    print(f"  log    : {os.path.join(args.root, 'RUN_ALL2.log')}")


if __name__ == "__main__":
    main()