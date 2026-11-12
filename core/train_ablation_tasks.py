"""Train selected PaDU task heads for auxiliary-task ablation."""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_model import (UnifiedWeatherModel, WeatherUnderstandingDataset,
                           collate)

ALL_TASKS = ["severity", "detect", "side", "density", "count"]


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device, tasks):
    """Evaluate only the active tasks; return their metrics + a combined score
    defined over the active classification accuracies (or neg-MAE if no cls)."""
    model.eval()
    n = 0
    sev_c = det_c = side_c = side_t = 0
    cnt_abs = 0.0; dens_abs = np.zeros(2)
    sev_cm = np.zeros((3, 3), dtype=int); det_cm = np.zeros((3, 3), dtype=int)
    for b in loader:
        out = model(b["tokens"].to(device)); n += b["tokens"].shape[0]
        if "severity" in tasks:
            sp = out["severity"].argmax(1).cpu().numpy(); st = b["severity"].numpy()
            sev_c += (sp == st).sum()
            for t, p in zip(st, sp):
                sev_cm[t, p] += 1
        if "detect" in tasks:
            dp = out["detect"].argmax(1).cpu().numpy(); dt = b["detect"].numpy()
            det_c += (dp == dt).sum()
            for t, p in zip(dt, dp):
                det_cm[t, p] += 1
        if "count" in tasks:
            cp = torch.expm1(out["count"]).clamp(min=0).cpu().numpy()
            cnt_abs += np.abs(cp - b["count"].numpy()).sum()
        if "density" in tasks:
            de = torch.expm1(out["density"]).clamp(min=0).cpu().numpy()
            dens_abs += np.abs(de - b["density"].numpy()).sum(axis=0)
        if "side" in tasks:
            sm = b["side"].numpy(); mask = sm >= 0
            if mask.sum():
                spd = out["side"].argmax(1).cpu().numpy()
                side_c += (spd[mask] == sm[mask]).sum(); side_t += mask.sum()

    def bal(cm):
        r = [(cm[i][i]/cm[i].sum()) if cm[i].sum() else 0.0
             for i in range(len(cm))]
        return float(np.mean(r))

    m = {}
    if "severity" in tasks:
        m["severity_acc"] = float(sev_c/n); m["severity_bal_acc"] = bal(sev_cm)
    if "detect" in tasks:
        m["detect_acc"] = float(det_c/n); m["detect_bal_acc"] = bal(det_cm)
    if "side" in tasks:
        m["side_acc"] = float(side_c/side_t) if side_t else 0.0
    if "count" in tasks:
        m["count_mae"] = float(cnt_abs/n)
    if "density" in tasks:
        m["nobj_mae"] = float(dens_abs[0]/n); m["nsmall_mae"] = float(dens_abs[1]/n)

    # combined score for early stopping over ACTIVE tasks:
    # mean of active accuracies, minus small penalty on active MAEs
    accs = [m[k] for k in ("severity_acc", "detect_acc", "side_acc") if k in m]
    maes = [m[k] for k in ("count_mae", "nobj_mae", "nsmall_mae") if k in m]
    score = (np.mean(accs) if accs else 0.0) - 0.02 * (np.mean(maes) if maes else 0.0)
    m["combined"] = float(score)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-root",
                    default="data/weather_patch_tokens")
    ap.add_argument("--label-dir",
                    default="data/understanding_labels")
    ap.add_argument("--out-root",
                    default="runs/weatherfilm_runs")
    ap.add_argument("--name", default="abl")
    ap.add_argument("--tasks", default="all",
                    help="'all' for multi-task, or one of "
                         "severity/detect/side/density/count for single-task")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-sev", type=float, default=1.0)
    ap.add_argument("--w-det", type=float, default=1.0)
    ap.add_argument("--w-side", type=float, default=1.0)
    ap.add_argument("--w-dens", type=float, default=1.0)
    ap.add_argument("--w-cnt", type=float, default=1.0)
    args = ap.parse_args()

    tasks = ALL_TASKS if args.tasks == "all" else [args.tasks]
    for t in tasks:
        assert t in ALL_TASKS, f"unknown task {t}"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tasks
    out_dir = os.path.join(args.out_root, f"{args.name}_{tag}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"Ablation: tasks={tasks} | seed={args.seed}")
    print("=" * 64)

    train_ds = WeatherUnderstandingDataset("train", args.tok_root,
                                           args.label_dir)
    val_ds = WeatherUnderstandingDataset("val", args.tok_root, args.label_dir)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate)

    model = UnifiedWeatherModel().to(device)
    ce = nn.CrossEntropyLoss(); ce_s = nn.CrossEntropyLoss(ignore_index=-1)
    sl1 = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    best = -1e9; bad = 0

    for ep in range(args.epochs):
        model.train()
        for b in tl:
            out = model(b["tokens"].to(device))
            loss = 0.0
            if "severity" in tasks:
                loss = loss + args.w_sev * ce(
                    out["severity"], b["severity"].to(device))
            if "detect" in tasks:
                loss = loss + args.w_det * ce(
                    out["detect"], b["detect"].to(device))
            if "side" in tasks:
                loss = loss + args.w_side * ce_s(
                    out["side"], b["side"].to(device))
            if "density" in tasks:
                loss = loss + args.w_dens * sl1(
                    out["density"], torch.log1p(b["density"].to(device)))
            if "count" in tasks:
                loss = loss + args.w_cnt * sl1(
                    out["count"], torch.log1p(b["count"].to(device)))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        m = evaluate(model, vl, device, tasks)
        if m["combined"] > best + 1e-4:
            best = m["combined"]; bad = 0
            torch.save({"epoch": ep+1, "model_state": model.state_dict(),
                        "args": vars(args), "tasks": tasks,
                        "val_metrics": m}, os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {ep+1}")
                break
        if (ep + 1) % 10 == 0 or ep == 0:
            msg = " ".join(f"{k}={v:.3f}" for k, v in m.items()
                           if k != "combined")
            print(f"ep {ep+1:3d} | {msg} | comb {m['combined']:.4f}")

    # report best val metrics
    best_ck = torch.load(os.path.join(out_dir, "best.pt"),
                         map_location="cpu", weights_only=False)
    print(f"\n  [tasks={tag}] best val metrics (epoch {best_ck['epoch']}):")
    for k, v in best_ck["val_metrics"].items():
        if k != "combined":
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
