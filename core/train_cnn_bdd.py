"""Train a CNN scene-understanding baseline on the BDD-IW split manifest."""

import os
import json
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class BddImageDataset(Dataset):
    """Reads BDD-IW raw images via manifest; test comes from physical train."""
    def __init__(self, split, bdd_root, manifest, label_dir, img_size=224,
                 augment=False):
        phys = manifest[split]["phys_split"]
        self.img_dir = os.path.join(bdd_root, phys, "images")
        self.img_size = img_size
        self.augment = augment
        with open(os.path.join(label_dir,
                               f"{split}_understanding_labels.json")) as f:
            self.u = json.load(f)
        with open(os.path.join(label_dir,
                               f"{split}_detectability_labels.json")) as f:
            self.d = json.load(f)
        # restrict to this split's manifest files that have both labels + image
        man_files = set(manifest[split]["files"])
        self.items = [f for f in manifest[split]["files"]
                      if f in self.u and f in self.d
                      and os.path.exists(os.path.join(self.img_dir, f))]
        print(f"  [{split}] usable: {len(self.items)} / {len(man_files)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn = self.items[idx]
        img = cv2.imread(os.path.join(self.img_dir, fn))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        if self.augment:
            # Horizontal flips would invalidate the side target.
            if random.random() < 0.5:
                img = img * random.uniform(0.8, 1.2)
                img = np.clip(img, 0, 1)
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        img = (img - mean) / std
        img = torch.from_numpy(img.transpose(2, 0, 1))

        u, d = self.u[fn], self.d[fn]
        side = u["vqa"]["largest_side"]
        side_lbl = 0 if side == "left" else (1 if side == "right" else -1)
        return {
            "img": img,
            "severity": torch.tensor(int(u["severity"])),
            "density": torch.tensor([float(u["density"]["n_objects"]),
                                     float(u["density"]["n_small"])]),
            "count": torch.tensor(float(u["vqa"]["count_total"])),
            "side": torch.tensor(side_lbl),
            "detect": torch.tensor(int(d["difficulty"])),
        }


def collate(batch):
    return {
        "img": torch.stack([b["img"] for b in batch]),
        "severity": torch.stack([b["severity"] for b in batch]),
        "density": torch.stack([b["density"] for b in batch]),
        "count": torch.stack([b["count"] for b in batch]),
        "side": torch.stack([b["side"] for b in batch]),
        "detect": torch.stack([b["detect"] for b in batch]),
    }


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CNNBaseline(nn.Module):
    """Identical to the XWOD from-scratch CNN baseline."""
    def __init__(self, d=384, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 160), ConvBlock(160, 320),
            ConvBlock(320, 512), ConvBlock(512, d))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(d)

        def head(o):
            return nn.Sequential(
                nn.Linear(d, d // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d // 2, o))
        self.h_sev = head(3); self.h_dens = head(2)
        self.h_cnt = head(1); self.h_side = head(2); self.h_det = head(3)
        n = sum(p.numel() for p in self.parameters())
        print(f"  CNNBaseline: {n/1e6:.2f}M params")

    def forward(self, x):
        z = self.pool(self.stem(x)).flatten(1)
        z = self.norm(z)
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
        out = model(b["img"].to(device))
        B = b["img"].shape[0]; n += B
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
        r = [(cm[i][i] / cm[i].sum()) if cm[i].sum() else 0.0
             for i in range(len(cm))]
        return float(np.mean(r))
    return {"severity_acc": float(sev_c / n), "detect_acc": float(det_c / n),
            "side_acc": float(side_c / side_t) if side_t else 0.0,
            "count_mae": float(cnt_abs / n), "nobj_mae": float(dens_abs[0]/n),
            "nsmall_mae": float(dens_abs[1] / n),
            "severity_bal_acc": bal(sev_cm), "detect_bal_acc": bal(det_cm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd-root", default="data/BDD-IW")
    ap.add_argument("--manifest",
                    default="data/bdd_understanding/splits.json")
    ap.add_argument("--label-dir",
                    default="data/bdd_understanding_labels")
    ap.add_argument("--out-root", default="runs/weatherfilm_runs")
    ap.add_argument("--name", default="bdd_cnn")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_root, f"{args.name}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.manifest) as f:
        manifest = json.load(f)

    print("=" * 64)
    print(f"BDD-IW from-scratch CNN baseline | seed={args.seed} "
          f"lambda={args.lam}")
    print("=" * 64)

    train_ds = BddImageDataset("train", args.bdd_root, manifest,
                               args.label_dir, args.img_size, augment=True)
    val_ds = BddImageDataset("val", args.bdd_root, manifest,
                             args.label_dir, args.img_size, augment=False)
    test_ds = BddImageDataset("test", args.bdd_root, manifest,
                              args.label_dir, args.img_size, augment=False)
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate)
    el = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True,
                    collate_fn=collate)

    model = CNNBaseline().to(device)
    ce = nn.CrossEntropyLoss(); ce_s = nn.CrossEntropyLoss(ignore_index=-1)
    sl1 = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best = -1e9; best_state = None; bad = 0
    for ep in range(args.epochs):
        model.train()
        for b in tl:
            img = b["img"].to(device)
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = (ce(out["severity"], b["severity"].to(device))
                        + ce(out["detect"], b["detect"].to(device))
                        + ce_s(out["side"], b["side"].to(device))
                        + args.lam * sl1(out["density"],
                                         torch.log1p(b["density"].to(device)))
                        + args.lam * sl1(out["count"],
                                         torch.log1p(b["count"].to(device))))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        m = evaluate(model, vl, device)
        comb = (m["severity_acc"] + m["detect_acc"] + m["side_acc"]) / 3 \
            - 0.02 * m["count_mae"]
        if comb > best + 1e-4:
            best = comb; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {ep+1}"); break
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"ep {ep+1:3d} | val sev {m['severity_acc']:.3f} "
                  f"det {m['detect_acc']:.3f} side {m['side_acc']:.3f} "
                  f"cnt {m['count_mae']:.2f}")

    # final test eval with val-selected best
    model.load_state_dict(best_state)
    test_m = evaluate(model, el, device)
    torch.save({"model_state": best_state, "args": vars(args),
                "test_metrics": test_m}, os.path.join(out_dir, "best_cnn.pt"))

    print(f"\n  === BDD CNN TEST metrics (seed {args.seed}) ===")
    for k, v in test_m.items():
        print(f"    {k}: {v:.4f}")

    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump({"cnn": test_m, "seed": args.seed}, f, indent=2)
    print(f"\n  saved -> {os.path.join(out_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
