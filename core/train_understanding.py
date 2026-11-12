"""Train the PaDU decoder on cached visual tokens.

Use cross-entropy for severity, detectability and side, with Smooth L1
on log1p density and count targets. The --lam option weights regression
losses. Select the checkpoint by the combined validation metric."""

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

from unified_model import (UnifiedWeatherModel, WeatherUnderstandingDataset,
                           collate)


def evaluate(model, loader, device):
    model.eval()
    n = 0
    sev_correct = det_correct = side_correct = side_total = 0
    cnt_abs = 0.0
    dens_abs = np.zeros(2)
    # confusion matrices for the two 3-way cls tasks
    sev_cm = np.zeros((3, 3), dtype=int)
    det_cm = np.zeros((3, 3), dtype=int)
    with torch.no_grad():
        for b in loader:
            tok = b["tokens"].to(device)
            out = model(tok)
            B = tok.shape[0]; n += B

            sp = out["severity"].argmax(1).cpu().numpy()
            st = b["severity"].numpy()
            sev_correct += (sp == st).sum()
            for t, p in zip(st, sp):
                sev_cm[t, p] += 1

            dp = out["detect"].argmax(1).cpu().numpy()
            dt = b["detect"].numpy()
            det_correct += (dp == dt).sum()
            for t, p in zip(dt, dp):
                det_cm[t, p] += 1

            # count regression: pred is log1p space -> expm1
            cp = torch.expm1(out["count"]).clamp(min=0).cpu().numpy()
            ct = b["count"].numpy()
            cnt_abs += np.abs(cp - ct).sum()

            de = torch.expm1(out["density"]).clamp(min=0).cpu().numpy()
            dgt = b["density"].numpy()
            dens_abs += np.abs(de - dgt).sum(axis=0)

            sm = b["side"].numpy()
            mask = sm >= 0
            if mask.sum() > 0:
                spd = out["side"].argmax(1).cpu().numpy()
                side_correct += (spd[mask] == sm[mask]).sum()
                side_total += mask.sum()

    metrics = {
        "severity_acc": float(sev_correct / n),
        "detect_acc": float(det_correct / n),
        "side_acc": float((side_correct / side_total) if side_total else 0.0),
        "count_mae": float(cnt_abs / n),
        "nobj_mae": float(dens_abs[0] / n),
        "nsmall_mae": float(dens_abs[1] / n),
        "severity_cm": sev_cm.tolist(),
        "detect_cm": det_cm.tolist(),
    }
    # Select by mean classification accuracy minus 0.02 times count MAE.
    metrics["combined"] = float(
        (metrics["severity_acc"] + metrics["detect_acc"]
         + metrics["side_acc"]) / 3.0
        - 0.02 * metrics["count_mae"])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-root",
                    default="data/weather_patch_tokens")
    ap.add_argument("--label-dir",
                    default="data/understanding_labels")
    ap.add_argument("--out-root",
                    default="runs/weatherfilm_runs")
    ap.add_argument("--name", default="uwm_main")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--lam", type=float, default=3.0,
                    help="regression (density/count) loss weight")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_root, f"{args.name}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"Unified weather-understanding training | name={args.name}")
    print("=" * 64)

    train_ds = WeatherUnderstandingDataset("train", args.tok_root,
                                           args.label_dir)
    val_ds = WeatherUnderstandingDataset("val", args.tok_root,
                                         args.label_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=collate)

    model = UnifiedWeatherModel(
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, dropout=args.dropout).to(device)

    ce = nn.CrossEntropyLoss()
    ce_side = nn.CrossEntropyLoss(ignore_index=-1)
    sl1 = nn.SmoothL1Loss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    history = []
    best = -1e9
    bad = 0

    for epoch in range(args.epochs):
        model.train()
        agg = {"sev": 0.0, "det": 0.0, "side": 0.0,
               "dens": 0.0, "cnt": 0.0, "total": 0.0}
        nb = 0
        for b in train_loader:
            tok = b["tokens"].to(device)
            out = model(tok)

            l_sev = ce(out["severity"], b["severity"].to(device))
            l_det = ce(out["detect"], b["detect"].to(device))
            l_side = ce_side(out["side"], b["side"].to(device))
            # log1p regression targets
            dens_t = torch.log1p(b["density"].to(device))
            cnt_t = torch.log1p(b["count"].to(device))
            l_dens = sl1(out["density"], dens_t)
            l_cnt = sl1(out["count"], cnt_t)

            loss = l_sev + l_det + l_side + args.lam * l_dens + args.lam * l_cnt

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            agg["sev"] += l_sev.item(); agg["det"] += l_det.item()
            agg["side"] += l_side.item(); agg["dens"] += l_dens.item()
            agg["cnt"] += l_cnt.item(); agg["total"] += loss.item()
            nb += 1
        sched.step()
        for k in agg:
            agg[k] /= max(nb, 1)

        m = evaluate(model, val_loader, device)
        print(f"ep {epoch+1:3d}/{args.epochs} | loss {agg['total']:.3f} "
              f"(sev {agg['sev']:.3f} det {agg['det']:.3f} "
              f"side {agg['side']:.3f} dens {agg['dens']:.3f} "
              f"cnt {agg['cnt']:.3f}) | "
              f"val sevAcc {m['severity_acc']:.3f} detAcc {m['detect_acc']:.3f} "
              f"sideAcc {m['side_acc']:.3f} cntMAE {m['count_mae']:.2f} "
              f"| comb {m['combined']:.4f}")
        history.append({"epoch": epoch + 1, "train": agg, "val": m})

        if m["combined"] > best + 1e-4:
            best = m["combined"]
            bad = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "args": vars(args),
                "val_metrics": m,
            }, os.path.join(out_dir, "best_uwm.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch+1}")
                break

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best val combined={best:.4f}")
    print(f"Saved: {os.path.join(out_dir, 'best_uwm.pt')}")
    print("Best val metrics:")
    bm = max(history, key=lambda h: h["val"]["combined"])["val"]
    print(f"  severity_acc={bm['severity_acc']:.3f}  "
          f"detect_acc={bm['detect_acc']:.3f}  "
          f"side_acc={bm['side_acc']:.3f}")
    print(f"  count_mae={bm['count_mae']:.2f}  nobj_mae={bm['nobj_mae']:.2f}  "
          f"nsmall_mae={bm['nsmall_mae']:.2f}")
    print(f"  severity confusion matrix (rows=true 0/1/2): "
          f"{bm['severity_cm']}")
    print(f"  detect confusion matrix (rows=true 0/1/2): {bm['detect_cm']}")


if __name__ == "__main__":
    main()
