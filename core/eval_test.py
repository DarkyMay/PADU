"""Evaluate validation-selected PaDU checkpoints on the XWOD test split."""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_model import (UnifiedWeatherModel, WeatherUnderstandingDataset,
                           collate)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0
    sev_c = det_c = side_c = side_t = 0
    cnt_abs = 0.0
    dens_abs = np.zeros(2)
    sev_cm = np.zeros((3, 3), dtype=int)
    det_cm = np.zeros((3, 3), dtype=int)
    for b in loader:
        out = model(b["tokens"].to(device))
        B = b["tokens"].shape[0]; n += B

        sp = out["severity"].argmax(1).cpu().numpy(); st = b["severity"].numpy()
        sev_c += (sp == st).sum()
        for t, p in zip(st, sp):
            sev_cm[t, p] += 1

        dp = out["detect"].argmax(1).cpu().numpy(); dt = b["detect"].numpy()
        det_c += (dp == dt).sum()
        for t, p in zip(dt, dp):
            det_cm[t, p] += 1

        cp = torch.expm1(out["count"]).clamp(min=0).cpu().numpy()
        cnt_abs += np.abs(cp - b["count"].numpy()).sum()
        de = torch.expm1(out["density"]).clamp(min=0).cpu().numpy()
        dens_abs += np.abs(de - b["density"].numpy()).sum(axis=0)

        sm = b["side"].numpy(); mask = sm >= 0
        if mask.sum():
            spd = out["side"].argmax(1).cpu().numpy()
            side_c += (spd[mask] == sm[mask]).sum(); side_t += mask.sum()

    return ({
        "severity_acc": float(sev_c / n), "detect_acc": float(det_c / n),
        "side_acc": float(side_c / side_t) if side_t else 0.0,
        "count_mae": float(cnt_abs / n), "nobj_mae": float(dens_abs[0] / n),
        "nsmall_mae": float(dens_abs[1] / n),
    }, sev_cm, det_cm)


def show_cm(name, cm, labels):
    print(f"\n  {name} confusion matrix (summed over seeds; rows=true):")
    print("        " + "".join(f"{l:>9s}" for l in labels))
    tot = cm.sum()
    for i, row in enumerate(cm):
        rsum = row.sum()
        cells = "".join(f"{v:>9d}" for v in row)
        rec = (row[i] / rsum) if rsum else 0.0
        print(f"  {labels[i]:>6s}{cells}   (recall={rec:.3f})")
    diag = sum(cm[i][i] for i in range(len(cm)))
    print(f"  -> accuracy = {diag/tot:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/weatherfilm_runs")
    ap.add_argument("--name", default="uwm_main")
    ap.add_argument("--tok-root",
                    default="data/weather_patch_tokens")
    ap.add_argument("--label-dir",
                    default="data/understanding_labels")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print(f"TEST-set evaluation over seeds {args.seeds}")
    print("=" * 64)

    test_ds = WeatherUnderstandingDataset("test", args.tok_root,
                                          args.label_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, collate_fn=collate)

    metric_keys = ["severity_acc", "detect_acc", "side_acc",
                   "count_mae", "nobj_mae", "nsmall_mae"]
    acc = {k: [] for k in metric_keys}
    sev_cm_sum = np.zeros((3, 3), dtype=int)
    det_cm_sum = np.zeros((3, 3), dtype=int)

    for s in args.seeds:
        ck_path = os.path.join(args.runs, f"{args.name}_seed{s}",
                               "best_uwm.pt")
        if not os.path.exists(ck_path):
            print(f"  [warn] missing ckpt seed {s}: {ck_path}")
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        margs = ck.get("args", {})
        model = UnifiedWeatherModel(
            d_model=margs.get("d_model", 384),
            n_layers=margs.get("n_layers", 3),
            n_heads=margs.get("n_heads", 6),
            dropout=margs.get("dropout", 0.1)).to(device)
        model.load_state_dict(ck["model_state"])
        m, sev_cm, det_cm = evaluate(model, test_loader, device)
        sev_cm_sum += sev_cm; det_cm_sum += det_cm
        for k in metric_keys:
            acc[k].append(m[k])
        print(f"  seed {s}: sevAcc {m['severity_acc']:.3f} "
              f"detAcc {m['detect_acc']:.3f} sideAcc {m['side_acc']:.3f} "
              f"cntMAE {m['count_mae']:.2f}")

    # Mean ± std table
    print(f"\n  {'metric':14s} {'TEST (mean±std)':>18s}   note")
    print(f"  {'-'*40}")
    notes = {"severity_acc": "↑ random .333", "detect_acc": "↑ random .333",
             "side_acc": "↑ random .500", "count_mae": "↓", "nobj_mae": "↓",
             "nsmall_mae": "↓"}
    for k in metric_keys:
        a = np.array(acc[k])
        cell = (f"{a.mean():.3f}±{a.std():.3f}" if len(a) > 1
                else (f"{a[0]:.3f}" if len(a) else "n/a"))
        print(f"  {k:14s} {cell:>18s}   {notes[k]}")

    show_cm("severity", sev_cm_sum, ["mild", "moder", "sever"])
    show_cm("detect", det_cm_sum, ["easy", "med", "hard"])

    print("\n  These TEST numbers (mean±std) are the paper's headline results.")


if __name__ == "__main__":
    main()
