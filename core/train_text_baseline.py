"""Train MLP task heads on frozen text embeddings.

Select checkpoints by validation performance and evaluate the requested split."""

import os
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PREFIX_MAP = {
    "heavy_rain": "rain", "flooding": "flooding", "tornado": "tornado",
    "snow": "snow", "wildfire": "wildfire", "dust": "haze_sand_dust",
    "fog": "fog",
}


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def infer_weather(fn):
    f = fn.lower()
    for p, w in PREFIX_MAP.items():
        if f.startswith(p):
            return w
    return "unknown"


class TextEmbDataset(Dataset):
    def __init__(self, split, emb_root, label_dir):
        emb = np.load(os.path.join(emb_root, f"{split}_text_emb.npy"))
        meta = np.load(os.path.join(emb_root, f"{split}_text_meta.npz"),
                       allow_pickle=True)
        fnames = list(meta["filenames"])
        valid = {f: bool(v) for f, v in zip(fnames, meta["valid"])}
        f2i = {f: i for i, f in enumerate(fnames)}
        self.emb = emb

        with open(os.path.join(label_dir,
                               f"{split}_understanding_labels.json")) as f:
            self.u = json.load(f)
        with open(os.path.join(label_dir,
                               f"{split}_detectability_labels.json")) as f:
            self.d = json.load(f)
        self.items = [f for f in fnames
                      if valid.get(f, False) and f in self.u and f in self.d]
        self.f2i = f2i
        print(f"  [{split}] usable: {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn = self.items[idx]
        x = torch.from_numpy(
            np.asarray(self.emb[self.f2i[fn]], dtype=np.float32))
        u, d = self.u[fn], self.d[fn]
        side = u["vqa"]["largest_side"]
        side_lbl = 0 if side == "left" else (1 if side == "right" else -1)
        return {
            "x": x,
            "severity": torch.tensor(int(u["severity"])),
            "density": torch.tensor([float(u["density"]["n_objects"]),
                                     float(u["density"]["n_small"])]),
            "count": torch.tensor(float(u["vqa"]["count_total"])),
            "side": torch.tensor(side_lbl),
            "detect": torch.tensor(int(d["difficulty"])),
        }


def collate(b):
    return {
        "x": torch.stack([x["x"] for x in b]),
        "severity": torch.stack([x["severity"] for x in b]),
        "density": torch.stack([x["density"] for x in b]),
        "count": torch.stack([x["count"] for x in b]),
        "side": torch.stack([x["side"] for x in b]),
        "detect": torch.stack([x["detect"] for x in b]),
    }


class TextHeadModel(nn.Module):
    """Map a pooled 4096-dimensional text embedding to four task heads with an MLP."""
    def __init__(self, d_in=4096, d=384, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout))

        def head(o):
            return nn.Sequential(nn.Linear(d, d // 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d // 2, o))
        self.h_sev = head(3); self.h_dens = head(2)
        self.h_cnt = head(1); self.h_side = head(2); self.h_det = head(3)
        n = sum(p.numel() for p in self.parameters())
        print(f"  TextHeadModel: {n/1e6:.2f}M params")

    def forward(self, x):
        z = self.trunk(x)
        return {"severity": self.h_sev(z), "density": self.h_dens(z),
                "count": self.h_cnt(z).squeeze(-1),
                "side": self.h_side(z), "detect": self.h_det(z)}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0; sev_c = det_c = side_c = side_t = 0
    cnt_abs = 0.0; dens_abs = np.zeros(2)
    sev_cm = np.zeros((3, 3), dtype=int); det_cm = np.zeros((3, 3), dtype=int)
    for b in loader:
        out = model(b["x"].to(device)); n += b["x"].shape[0]
        sp = out["severity"].argmax(1).cpu().numpy(); st = b["severity"].numpy()
        for t, p in zip(st, sp):
            sev_cm[t, p] += 1
        dpr = out["detect"].argmax(1).cpu().numpy(); dtt = b["detect"].numpy()
        for t, p in zip(dtt, dpr):
            det_cm[t, p] += 1
        sev_c += (sp == st).sum(); det_c += (dpr == dtt).sum()
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
    return {"severity_acc": float(sev_c/n), "detect_acc": float(det_c/n),
            "side_acc": float(side_c/side_t) if side_t else 0.0,
            "count_mae": float(cnt_abs/n), "nobj_mae": float(dens_abs[0]/n),
            "nsmall_mae": float(dens_abs[1]/n),
            "severity_bal_acc": bal(sev_cm), "detect_bal_acc": bal(det_cm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-root", default="data/text_embeddings")
    ap.add_argument("--label-dir",
                    default="data/understanding_labels")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print(f"VLM TEXT-OUTPUT baseline (seed={args.seed}, "
          f"eval={args.eval_split})")
    print("=" * 64)

    train_ds = TextEmbDataset("train", args.emb_root, args.label_dir)
    val_ds = TextEmbDataset("val", args.emb_root, args.label_dir)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate,
                    drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, collate_fn=collate)

    model = TextHeadModel().to(device)
    ce = nn.CrossEntropyLoss(); ce_s = nn.CrossEntropyLoss(ignore_index=-1)
    sl1 = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1e9; best_state = None; bad = 0
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            out = model(b["x"].to(device))
            loss = (ce(out["severity"], b["severity"].to(device))
                    + ce(out["detect"], b["detect"].to(device))
                    + ce_s(out["side"], b["side"].to(device))
                    + sl1(out["density"], torch.log1p(b["density"].to(device)))
                    + sl1(out["count"], torch.log1p(b["count"].to(device))))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m = evaluate(model, vl, device)
        comb = (m["severity_acc"] + m["detect_acc"] + m["side_acc"]) / 3
        if comb > best + 1e-4:
            best = comb; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best_state)
    eval_ds = (val_ds if args.eval_split == "val"
               else TextEmbDataset(args.eval_split, args.emb_root,
                                   args.label_dir))
    el = DataLoader(eval_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, collate_fn=collate)
    fm = evaluate(model, el, device)

    print(f"\n  TEXT baseline ({args.eval_split}) metrics:")
    for k, v in fm.items():
        print(f"    {k}: {v:.4f}")

    out_path = os.path.join(
        args.label_dir,
        f"text_baseline_{args.eval_split}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump({"seed": args.seed, "eval_split": args.eval_split,
                   "text": fm}, f, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
