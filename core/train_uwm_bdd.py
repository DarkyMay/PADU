"""Train the PaDU decoder on cached BDD-IW tokens and report held-out metrics."""

import os
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_model import UnifiedWeatherModel


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class BddDataset(Dataset):
    """Reads BDD-IW patch tokens + programmatic labels, aligned by filename."""
    def __init__(self, split, tok_root, label_dir):
        toks = np.load(os.path.join(tok_root, f"{split}_tokens.npy"),
                       mmap_mode="r")
        meta = np.load(os.path.join(tok_root, f"{split}_meta.npz"),
                       allow_pickle=True)
        fnames = [str(f) for f in meta["filenames"]]
        valid = {f: bool(v) for f, v in zip(fnames, meta["valid"])}
        self.toks = toks
        self.idx_of = {f: i for i, f in enumerate(fnames)}

        with open(os.path.join(label_dir,
                               f"{split}_understanding_labels.json")) as f:
            self.u = json.load(f)
        with open(os.path.join(label_dir,
                               f"{split}_detectability_labels.json")) as f:
            self.d = json.load(f)
        self.items = [f for f in fnames
                      if valid.get(f, False) and f in self.u and f in self.d]
        print(f"  [{split}] usable: {len(self.items)} / {len(fnames)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn = self.items[i]
        z = torch.from_numpy(
            np.asarray(self.toks[self.idx_of[fn]], dtype=np.float32))
        u, d = self.u[fn], self.d[fn]
        side = u["vqa"]["largest_side"]
        side_lbl = 0 if side == "left" else (1 if side == "right" else -1)
        return {
            "tokens": z,
            "severity": torch.tensor(int(u["severity"])),
            "density": torch.tensor([float(u["density"]["n_objects"]),
                                     float(u["density"]["n_small"])]),
            "count": torch.tensor(float(u["vqa"]["count_total"])),
            "side": torch.tensor(side_lbl),
            "detect": torch.tensor(int(d["difficulty"])),
        }


def collate(b):
    return {
        "tokens": torch.stack([x["tokens"] for x in b]),
        "severity": torch.stack([x["severity"] for x in b]),
        "density": torch.stack([x["density"] for x in b]),
        "count": torch.stack([x["count"] for x in b]),
        "side": torch.stack([x["side"] for x in b]),
        "detect": torch.stack([x["detect"] for x in b]),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0; sev_c = det_c = side_c = side_t = 0
    cnt_abs = 0.0; dens_abs = np.zeros(2)
    sev_cm = np.zeros((3, 3), dtype=int); det_cm = np.zeros((3, 3), dtype=int)
    for b in loader:
        out = model(b["tokens"].to(device)); n += b["tokens"].shape[0]
        sp = out["severity"].argmax(1).cpu().numpy(); st = b["severity"].numpy()
        for t, p in zip(st, sp):
            sev_cm[t, p] += 1
        dp = out["detect"].argmax(1).cpu().numpy(); dt = b["detect"].numpy()
        for t, p in zip(dt, dp):
            det_cm[t, p] += 1
        sev_c += (sp == st).sum(); det_c += (dp == dt).sum()
        cp = torch.expm1(out["count"]).clamp(min=0).cpu().numpy()
        cnt_abs += np.abs(cp - b["count"].numpy()).sum()
        de = torch.expm1(out["density"]).clamp(min=0).cpu().numpy()
        dens_abs += np.abs(de - b["density"].numpy()).sum(axis=0)
        sm = b["side"].numpy(); mask = sm >= 0
        if mask.sum():
            spd = out["side"].argmax(1).cpu().numpy()
            side_c += (spd[mask] == sm[mask]).sum(); side_t += mask.sum()

    def bal(cm):
        r = [(cm[i][i]/cm[i].sum()) if cm[i].sum() else 0.0
             for i in range(len(cm))]
        return float(np.mean(r))
    return {
        "severity_acc": float(sev_c/n), "severity_bal_acc": bal(sev_cm),
        "detect_acc": float(det_c/n), "detect_bal_acc": bal(det_cm),
        "side_acc": float(side_c/side_t) if side_t else 0.0,
        "count_mae": float(cnt_abs/n), "nobj_mae": float(dens_abs[0]/n),
        "nsmall_mae": float(dens_abs[1]/n),
        "severity_cm": sev_cm.tolist(), "detect_cm": det_cm.tolist(),
    }


def naive_baseline(train_ds, eval_ds):
    """Train-majority class / train-mean, evaluated on eval_ds."""
    sev = [train_ds[i]["severity"].item() for i in range(len(train_ds))]
    det = [train_ds[i]["detect"].item() for i in range(len(train_ds))]
    cnt = [train_ds[i]["count"].item() for i in range(len(train_ds))]
    nobj = [train_ds[i]["density"][0].item() for i in range(len(train_ds))]
    nsmall = [train_ds[i]["density"][1].item() for i in range(len(train_ds))]
    maj_sev = max(set(sev), key=sev.count)
    maj_det = max(set(det), key=det.count)
    mean_cnt = float(np.mean(cnt)); mean_nobj = float(np.mean(nobj))
    mean_nsmall = float(np.mean(nsmall))

    es = [eval_ds[i]["severity"].item() for i in range(len(eval_ds))]
    ed = [eval_ds[i]["detect"].item() for i in range(len(eval_ds))]
    ec = [eval_ds[i]["count"].item() for i in range(len(eval_ds))]
    eo = [eval_ds[i]["density"][0].item() for i in range(len(eval_ds))]
    en = [eval_ds[i]["density"][1].item() for i in range(len(eval_ds))]
    return {
        "severity_acc": float(np.mean([s == maj_sev for s in es])),
        "severity_bal_acc": 1/3,
        "detect_acc": float(np.mean([d == maj_det for d in ed])),
        "detect_bal_acc": 1/3,
        "side_acc": 0.5,
        "count_mae": float(np.mean(np.abs(np.array(ec) - mean_cnt))),
        "nobj_mae": float(np.mean(np.abs(np.array(eo) - mean_nobj))),
        "nsmall_mae": float(np.mean(np.abs(np.array(en) - mean_nsmall))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-root", default="data/bdd_patch_tokens")
    ap.add_argument("--label-dir",
                    default="data/bdd_understanding_labels")
    ap.add_argument("--out-root", default="runs/weatherfilm_runs")
    ap.add_argument("--name", default="bdd_uwm")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lam", type=float, default=3.0,
                    help="regression loss weight (XWOD ablation: 3.0)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_root, f"{args.name}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"BDD-IW UWM training (seed={args.seed}, lambda={args.lam})")
    print("=" * 64)

    train_ds = BddDataset("train", args.tok_root, args.label_dir)
    val_ds = BddDataset("val", args.tok_root, args.label_dir)
    test_ds = BddDataset("test", args.tok_root, args.label_dir)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate)
    el = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate)

    model = UnifiedWeatherModel().to(device)
    ce = nn.CrossEntropyLoss(); ce_s = nn.CrossEntropyLoss(ignore_index=-1)
    sl1 = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    best = -1e9; best_state = None; bad = 0

    for ep in range(args.epochs):
        model.train()
        for b in tl:
            out = model(b["tokens"].to(device))
            loss = (ce(out["severity"], b["severity"].to(device))
                    + ce(out["detect"], b["detect"].to(device))
                    + ce_s(out["side"], b["side"].to(device))
                    + args.lam * sl1(out["density"],
                                     torch.log1p(b["density"].to(device)))
                    + args.lam * sl1(out["count"],
                                     torch.log1p(b["count"].to(device))))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        m = evaluate(model, vl, device)
        comb = (m["severity_acc"] + m["detect_acc"] + m["side_acc"]) / 3 \
            - 0.02 * m["count_mae"]
        if comb > best + 1e-4:
            best = comb; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {ep+1}")
                break
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep {ep+1:3d} | sev {m['severity_acc']:.3f} "
                  f"det {m['detect_acc']:.3f} side {m['side_acc']:.3f} "
                  f"cnt {m['count_mae']:.2f}")

    # final test eval with val-selected best
    model.load_state_dict(best_state)
    test_m = evaluate(model, el, device)
    torch.save({"model_state": best_state, "args": vars(args),
                "test_metrics": test_m}, os.path.join(out_dir, "best_uwm.pt"))

    print(f"\n  === BDD UWM TEST metrics (seed {args.seed}) ===")
    for k, v in test_m.items():
        if not k.endswith("_cm"):
            print(f"    {k}: {v:.4f}")

    # Evaluate the training-set constant baseline.
    nv = naive_baseline(train_ds, test_ds)
    print(f"\n  === BDD naive baseline (test) ===")
    for k, v in nv.items():
        print(f"    {k}: {v:.4f}")

    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump({"uwm": {k: v for k, v in test_m.items()
                           if not k.endswith("_cm")},
                   "uwm_cm": {"severity": test_m["severity_cm"],
                              "detect": test_m["detect_cm"]},
                   "naive": nv, "seed": args.seed}, f, indent=2)
    print(f"\n  saved -> {os.path.join(out_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
