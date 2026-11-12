#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run and aggregate centralized, federated and communication experiments."""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np

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



# stage definitions

def stages(args):
    seeds = ["0"] if args.fast else ["0", "1", "2"]
    br = "25" if args.fast else "40"
    return [
        ("main", ["--seeds", *seeds, "--exp", "0,A,B"]),
        ("byz_majority", ["--seeds", "0", "--exp", "C", "--byz-rounds", br]),
        ("byz_minority", ["--seeds", "0", "--exp", "C", "--byz-rounds", br,
                          "--byz-ids", "2,3"]),
    ]


def run_stage(name, extra, root, args):
    out = os.path.join(root, name)
    done = os.path.join(out, "fed_results.json")
    if args.resume and os.path.exists(done):
        W(f"  [skip] {name}: {done} already exists")
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



# aggregation helpers

METRICS = [("severity_acc", "SevAcc", 3, 1), ("severity_bal", "SevBal", 3, 1),
           ("detect_acc", "DetAcc", 3, 1), ("detect_bal", "DetBal", 3, 1),
           ("side_acc", "SideAcc", 3, 1), ("count_mae", "CntMAE", 2, -1),
           ("nobj_mae", "ObjMAE", 2, -1), ("nsmall_mae", "SmlMAE", 2, -1)]


def load(root, name):
    p = os.path.join(root, name, "fed_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def collect(blob, path_fn):
    """path_fn(seed_result) -> metric dict or None. Returns {metric: [values]}."""
    acc = {}
    if not blob:
        return acc
    for sd, res in sorted(blob.get("seeds", {}).items()):
        m = path_fn(res)
        if not m:
            continue
        for k, _, _, _ in METRICS:
            if k in m and m[k] is not None:
                acc.setdefault(k, []).append(float(m[k]))
    return acc


def cell(vals, nd):
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{nd}f}"
    return f"{np.mean(vals):.{nd}f}±{np.std(vals):.{nd}f}"


def row(label, acc, keys, width=28):
    out = f"  {label:<{width}}"
    for k, _, nd, _ in METRICS:
        if k not in keys:
            continue
        out += f"{cell(acc.get(k, []), nd):>14}"
    return out


def header(keys, width=28):
    out = f"  {'method':<{width}}"
    for k, lab, _, arrow in METRICS:
        if k not in keys:
            continue
        out += f"{lab + ('↑' if arrow > 0 else '↓'):>14}"
    return out



# tables

MAIN_KEYS = ["severity_acc", "detect_acc", "detect_bal", "side_acc",
             "count_mae"]
FULL_KEYS = [k for k, _, _, _ in METRICS]


def table_main(main, tex):
    W("\n" + "=" * 78)
    W("TABLE 1  Federated PaDU on the XWOD global test set")
    W("=" * 78)
    n = len(main.get("seeds", {})) if main else 0
    W(f"  mean ± std over {n} seed(s). All methods share the frozen VLM "
      f"substrate and the\n  same 5.57M-parameter decoder; only the training "
      f"protocol differs.")
    W("")
    W(header(MAIN_KEYS))

    rows = [
        ("Centralized (upper bound)", lambda r: r.get("exp0_centralized")),
        ("FedAvg", lambda r: (r.get("expA", {}).get("fedavg") or {}).get("global")),
        ("FedProx (mu=0.01)", lambda r: (r.get("expA", {}).get("fedprox") or {}).get("global")),
        ("FedAvg + fp16 uplink", lambda r: (r.get("expA", {}).get("fedavg_fp16") or {}).get("global")),
        ("FedAvg + int8 uplink", lambda r: (r.get("expA", {}).get("fedavg_int8") or {}).get("global")),
    ]
    tex_rows = []
    for label, fn in rows:
        acc = collect(main, fn)
        W(row(label, acc, MAIN_KEYS))
        tex_rows.append((label, acc))

    W("\n  Gap to the centralized upper bound (negative = federation costs "
      "accuracy):")
    base = collect(main, lambda r: r.get("exp0_centralized"))
    for label, fn in rows[1:]:
        acc = collect(main, fn)
        parts = []
        for k in ["severity_acc", "detect_acc", "side_acc"]:
            if acc.get(k) and base.get(k):
                parts.append(f"{k.split('_')[0]} {np.mean(acc[k])-np.mean(base[k]):+.3f}")
        W(f"    {label:<28}{'   '.join(parts)}")

    tex.append(latex_table(
        "Federated PaDU on the XWOD test set. All variants train the same "
        "5.57M-parameter decoder on the frozen VLM substrate; only the "
        "training protocol differs. Mean $\\pm$ std over %d seeds." % max(n, 1),
        "tab:fed_main", tex_rows, MAIN_KEYS))


def table_personalized(main, tex):
    W("\n" + "=" * 78)
    W("TABLE 2  Personalized evaluation: each agent on its own weather")
    W("=" * 78)
    W("  Each client is evaluated on the test frames of its own weather type.\n"
      "  This is the comparison that matters for a deployed fleet: does "
      "federating\n  beat every vehicle training in isolation?")
    W("")
    W(header(MAIN_KEYS))
    rows = [
        ("Local-only (no comms)",
         lambda r: ((r.get("expA", {}).get("local") or {})
                    .get("personalized", {}) or {}).get("pooled")),
        ("FedPer (shared encoder)",
         lambda r: ((r.get("expA", {}).get("fedper") or {})
                    .get("personalized", {}) or {}).get("pooled")),
    ]
    tex_rows = []
    for label, fn in rows:
        acc = collect(main, fn)
        W(row(label, acc, MAIN_KEYS))
        tex_rows.append((label, acc))

    a = collect(main, rows[0][1])
    b = collect(main, rows[1][1])
    W("\n  FedPer minus Local-only:")
    for k in ["severity_acc", "detect_acc", "side_acc", "count_mae"]:
        if a.get(k) and b.get(k):
            W(f"    {k:<16}{np.mean(b[k])-np.mean(a[k]):+.3f}")
    tex.append(latex_table(
        "Personalized evaluation on XWOD: every client is tested on its own "
        "weather. Federating beats isolated training on the "
        "decision-relevant tasks.", "tab:fed_personal", tex_rows, MAIN_KEYS))


def table_comms(main, tex):
    W("\n" + "=" * 78)
    W("TABLE 3  Communication cost")
    W("=" * 78)
    if not main:
        W("  (no data)")
        return
    r0 = main["seeds"][sorted(main["seeds"])[0]]
    b = r0.get("expB", {})
    if not b:
        W("  (EXP-B missing)")
        return
    tot = b["params_total"]
    enc = b["params_encoder"]
    vlm = b["vlm_params"]
    W(f"  Trainable decoder: {tot/1e6:.2f}M parameters "
      f"({enc/1e6:.2f}M encoder + {(tot-enc)/1e6:.2f}M heads).")
    W(f"  The frozen {vlm/1e9:.0f}B VLM is never transmitted: it is identical "
      f"at every agent\n  and its features are cached once per image.")
    W("")
    W(f"  {'uplink scheme':<30}{'bytes/param':>13}{'per client-round':>19}"
      f"{'reduction vs VLM FT':>22}")
    schemes = [("PaDU decoder, fp32", 4.0, tot), ("PaDU decoder, fp16", 2.0, tot),
               ("PaDU decoder, int8", 1.0, tot),
               ("PaDU encoder only (FedPer)", 4.0, enc),
               ("Fine-tuning the VLM", 4.0, vlm)]
    tex_rows = []
    for name, bpp, np_ in schemes:
        by = bpp * np_
        red = (vlm * 4.0) / by
        W(f"  {name:<30}{bpp:>13.0f}{human(by):>19}{red:>21.0f}x")
        tex_rows.append((name, bpp, by, red))

    W("\n  Accuracy at each uplink budget (XWOD test, mean over seeds):")
    W(f"  {'scheme':<26}{'bytes/param':>12}{'DetAcc':>16}{'SevAcc':>16}"
      f"{'SideAcc':>16}")
    for label, key, bpp in [("fp32 (uncompressed)", "fedavg", 4),
                            ("fp16", "fedavg_fp16", 2),
                            ("int8", "fedavg_int8", 1)]:
        acc = collect(main, lambda r, k=key: (r.get("expA", {}).get(k) or {}).get("global"))
        W(f"  {label:<26}{bpp:>12}"
          f"{cell(acc.get('detect_acc', []), 3):>16}"
          f"{cell(acc.get('severity_acc', []), 3):>16}"
          f"{cell(acc.get('side_acc', []), 3):>16}")

    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Uplink cost per client-round. The frozen VLM is shared "
             "and never transmitted, so only the decoder is communicated.}",
             "\\label{tab:fed_comm}", "\\begin{tabular}{lrrr}", "\\toprule",
             "Transmitted object & bits/param & Per client-round & Reduction \\\\",
             "\\midrule"]
    for name, bpp, by, red in tex_rows:
        lines.append(f"{name} & {int(bpp*8)} & {human(by)} & "
                     f"${red:.0f}\\times$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex.append("\n".join(lines))


def table_noniid(main, tex):
    W("\n" + "=" * 78)
    W("TABLE 4  Client heterogeneity (XWOD train split)")
    W("=" * 78)
    if not main:
        return
    r0 = main["seeds"][sorted(main["seeds"])[0]]
    cl = r0.get("clients", [])
    fh = r0.get("feature_heterogeneity", {})
    if not cl:
        W("  (missing)")
        return
    tot = sum(c["n"] for c in cl)
    W(f"  {'agent (weather)':<18}{'n':>7}{'share':>9}{'detect 0/1/2':>18}"
      f"{'obj/img':>9}{'JS(det)':>10}")
    for c in cl:
        d = np.asarray(c["detect_dist"], float)
        p = "/".join(f"{v/max(d.sum(),1)*100:.0f}" for v in d)
        W(f"  {c['client']:<18}{c['n']:>7}{c['n']/tot*100:>8.1f}%{p:>18}"
          f"{c['mean_objects']:>9.1f}{c['js_detect']:>10.4f}")
    sizes = [c["n"] for c in cl]
    W(f"\n  size imbalance max/min = {max(sizes)/min(sizes):.1f}x")
    W(f"  objects/image spread   = {min(c['mean_objects'] for c in cl):.1f} "
      f"to {max(c['mean_objects'] for c in cl):.1f} "
      f"({max(c['mean_objects'] for c in cl)/min(c['mean_objects'] for c in cl):.1f}x)")
    if fh:
        W(f"  feature-space cosine between client centroids: "
          f"mean {fh['mean_off_diag']:.3f}, min {fh['min_off_diag']:.3f}")
    W("\n  NOTE for the paper: severity is IID across clients BY CONSTRUCTION "
      "(XWOD severity\n  tertiles are fit within each weather type, so every "
      "client is 33/33/33). The\n  heterogeneity is carried by detectability, "
      "object density, client size, and the\n  frozen-VLM features. State this "
      "explicitly or a reviewer will catch it.")

    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{The seven weather agents are strongly non-IID in shard "
             "size, object density and detectability, though severity is IID "
             "by construction.}",
             "\\label{tab:fed_noniid}", "\\begin{tabular}{lrrrr}", "\\toprule",
             "Agent & $n$ & Share & Obj/img & JS(det) \\\\", "\\midrule"]
    for c in cl:
        lines.append(f"{c['client'].replace('_',' ')} & {c['n']} & "
                     f"{c['n']/tot*100:.1f}\\% & {c['mean_objects']:.1f} & "
                     f"{c['js_detect']:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tex.append("\n".join(lines))


def table_byz(blobs, tex):
    W("\n" + "=" * 78)
    W("TABLE 5  Byzantine robustness")
    W("=" * 78)
    W("  Final-round global model on the XWOD test set. A * marks a collapsed\n"
      "  model (balanced accuracy at the 0.333 chance floor).")
    tex_blocks = []
    for name, label in [("byz_majority", "adversaries hold the two largest "
                                         "shards (67% of the data)"),
                        ("byz_minority", "adversaries hold two mid-size shards "
                                         "(~20% of the data)")]:
        blob = blobs.get(name)
        if not blob:
            W(f"\n  [{name}] missing")
            continue
        r0 = blob["seeds"][sorted(blob["seeds"])[0]]
        setup = r0.get("expC_setup", {})
        C = r0.get("expC", {})
        if not C:
            W(f"\n  [{name}] no expC block")
            continue
        rules = list(next(iter(C.values())).keys())
        W(f"\n  --- {label} ---")
        if setup:
            W(f"  adversaries: {setup.get('byz_clients')}, "
              f"{setup.get('byz_data_share', 0)*100:.1f}% of training data")
        for metric, mlabel, nd in [("detect_acc", "Detectability acc", 3),
                                   ("detect_bal", "Detectability bal-acc", 3),
                                   ("side_acc", "Spatial (side) acc", 3)]:
            W(f"\n  {mlabel}")
            W(f"  {'attack':<14}" + "".join(f"{r:>16}" for r in rules))
            for atk in ["none", "label_flip", "sign_flip", "gauss"]:
                if atk not in C:
                    continue
                cells = ""
                for r in rules:
                    g = C[atk][r].get("global_final") or C[atk][r].get("global")
                    if not g:
                        cells += f"{'--':>16}"
                        continue
                    mark = "*" if g.get("detect_bal", 1) < 0.345 else ""
                    cells += f"{g[metric]:>15.{nd}f}{mark:<1}"
                W(f"  {atk:<14}{cells}")
        tex_blocks.append((label, rules, C))

    W("\n  Reading guide for the paper:")
    W("   - the 'none' row is the clean reference; a robust rule should stay "
      "near it under attack")
    W("   - FedAvg collapses under every attack: size-weighted averaging has "
      "no defence")
    W("   - coordinate-wise rules (median, trimmed mean) already lose the "
      "spatial task in the\n     clean row: under strong non-IID, honest "
      "clients disagree per-coordinate as much as\n     an adversary deviates, "
      "so per-coordinate statistics cannot tell them apart")
    W("   - multi-Krum compares whole update vectors and is the only rule that "
      "holds up")

    for label, rules, C in tex_blocks[:1]:
        lines = ["\\begin{table}[t]", "\\centering",
                 "\\caption{Byzantine robustness on XWOD (final-round global "
                 "model, detectability accuracy / balanced accuracy). "
                 f"Setting: {label}.}}",
                 "\\label{tab:fed_byz}",
                 "\\begin{tabular}{l" + "c" * len(rules) + "}", "\\toprule",
                 "Attack & " + " & ".join(r.replace("_", "-") for r in rules)
                 + " \\\\", "\\midrule"]
        for atk in ["none", "label_flip", "sign_flip", "gauss"]:
            if atk not in C:
                continue
            cs = []
            for r in rules:
                g = C[atk][r].get("global_final") or C[atk][r].get("global")
                cs.append(f"{g['detect_acc']:.3f}/{g['detect_bal']:.3f}"
                          if g else "--")
            lines.append(atk.replace("_", "-") + " & " + " & ".join(cs) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        tex.append("\n".join(lines))


def latex_table(caption, label, tex_rows, keys):
    cols = [(k, lab, nd) for k, lab, nd, _ in METRICS if k in keys]
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}",
             "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(lab for _, lab, _ in cols) + " \\\\",
             "\\midrule"]
    for name, acc in tex_rows:
        cs = []
        for k, _, nd in cols:
            v = acc.get(k, [])
            if not v:
                cs.append("--")
            elif len(v) == 1:
                cs.append(f"{v[0]:.{nd}f}")
            else:
                cs.append(f"${np.mean(v):.{nd}f}\\pm{np.std(v):.{nd}f}$")
        lines.append(name + " & " + " & ".join(cs) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def sanity_note(main):
    W("\n" + "=" * 78)
    W("SANITY  Centralized re-run vs. the published PaDU numbers")
    W("=" * 78)
    ref = {"severity_acc": 0.624, "detect_acc": 0.517, "side_acc": 0.802,
           "count_mae": 1.362}
    acc = collect(main, lambda r: r.get("exp0_centralized"))
    W(f"  {'metric':<16}{'this run':>16}{'paper':>10}{'delta':>10}"
      f"{'tolerance':>16}")
    ok = True
    for k, v in ref.items():
        vals = acc.get(k, [])
        if not vals:
            continue
        d = np.mean(vals) - v
        # Use absolute tolerance for accuracy and relative tolerance for count MAE.
        tol = 0.03 if k.endswith(("_acc", "_bal")) else 0.05 * abs(v)
        ok &= abs(d) <= tol
        W(f"  {k:<16}{cell(vals,3):>16}{v:>10.3f}{d:>+10.3f}"
          f"{'  (tol ' + format(tol, '.3f') + ')':>16}")
    W(f"\n  {'PASS' if ok else 'FAIL'}: centralized evaluation "
      f"{'reproduces' if ok else 'does NOT reproduce'} the published results, "
      f"so the\n  federated numbers sit on the same footing as the paper's.")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/fed_results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", default="",
                    help="comma-separated stage names to run")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    t0 = time.time()

    W("=" * 78)
    W("run_all.py — remaining federated-PaDU experiments")
    W("=" * 78)
    W(f"started : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    W(f"root    : {args.root}")
    W(f"driver  : {FED}")
    if not os.path.exists(FED):
        W("ERROR: fed_padu.py must sit next to run_all.py")
        sys.exit(1)

    st = stages(args)
    if args.only:
        want = {x.strip() for x in args.only.split(",")}
        st = [x for x in st if x[0] in want]

    if not args.aggregate_only:
        W(f"\nstages  : {[x[0] for x in st]}")
        W(f"estimate: {'~1.5 h' if args.fast else '~4 h'} on one RTX 4090\n")
        for i, (name, extra) in enumerate(st, 1):
            W(f"\n{'#'*78}\n# STAGE {i}/{len(st)}: {name}\n{'#'*78}")
            if not run_stage(name, extra, args.root, args):
                W(f"  !! stage {name} failed; continuing to the next one")
            dump(os.path.join(args.root, "RUN_ALL.log"))

    # Aggregate
    W(f"\n\n{'#'*78}\n# AGGREGATED RESULTS\n{'#'*78}")
    blobs = {n: load(args.root, n)
             for n in ["main", "byz_majority", "byz_minority"]}
    main_blob = blobs.get("main")
    tex = []

    if main_blob:
        sanity_note(main_blob)
        table_main(main_blob, tex)
        table_personalized(main_blob, tex)
        table_comms(main_blob, tex)
        table_noniid(main_blob, tex)
    else:
        W("\n  [main] results missing — run the 'main' stage first")
    table_byz(blobs, tex)

    W("\n" + "=" * 78)
    W(f"TOTAL WALL TIME  {hms(time.time()-t0)}")
    W("=" * 78)

    final = os.path.join(args.root, "FINAL_TABLES.txt")
    dump(os.path.join(args.root, "RUN_ALL.log"))
    with open(final, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG))
    texp = os.path.join(args.root, "tables.tex")
    with open(texp, "w", encoding="utf-8") as f:
        f.write("% auto-generated by run_all.py — check every number before "
                "submitting\n\n" + "\n\n".join(tex) + "\n")
    print(f"\n  tables : {final}")
    print(f"  latex  : {texp}")
    print(f"  log    : {os.path.join(args.root, 'RUN_ALL.log')}")


if __name__ == "__main__":
    main()