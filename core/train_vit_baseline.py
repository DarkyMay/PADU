"""Train a multi-task decoder on frozen ImageNet ViT-B/16 patch tokens.

Project 196 tokens with 768 channels into a 384-dimensional decoder."""
import os, json, copy, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import timm


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class ImageDataset(Dataset):
    """Raw XWOD images + programmatic labels (same labels as PaDU)."""
    def __init__(self, split, xwod_root, label_dir, img_size=224):
        self.img_dir = os.path.join(
            xwod_root, "valid/images" if split == "val" else f"{split}/images")
        self.img_size = img_size
        with open(os.path.join(label_dir,
                               f"{split}_understanding_labels.json")) as f:
            self.u = json.load(f)
        with open(os.path.join(label_dir,
                               f"{split}_detectability_labels.json")) as f:
            self.d = json.load(f)
        self.items = [f for f in self.u if f in self.d
                      and os.path.exists(os.path.join(self.img_dir, f))]
        print(f"  [{split}] usable: {len(self.items)}")
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fn = self.items[i]
        img = cv2.imread(os.path.join(self.img_dir, fn))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
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


def collate(b):
    return {k: torch.stack([x[k] for x in b]) for k in
            ["img", "severity", "density", "count", "side", "detect"]}


class ViTDecoder(nn.Module):
    """Frozen ViT-B/16 -> same decoder as PaDU (d=384, M=3, 6 heads)."""
    def __init__(self, d=384, n_layers=3, n_heads=6, dropout=0.1):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True,
                                     num_classes=0)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        d_v = self.vit.embed_dim  # 768
        self.in_proj = nn.Linear(d_v, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        # positional emb sized to ViT's patch count (196) + cls
        self.pos = nn.Parameter(torch.zeros(1, 197, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=d*4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d)

        def head(o):
            return nn.Sequential(nn.Linear(d, d//2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d//2, o))
        self.h_sev = head(3); self.h_dens = head(2)
        self.h_cnt = head(1); self.h_side = head(2); self.h_det = head(3)
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  ViTDecoder trainable: {n/1e6:.2f}M (ViT frozen)")

    @torch.no_grad()
    def extract(self, img):
        # ViT patch tokens WITHOUT the cls token: [B, 196, 768]
        feats = self.vit.forward_features(img)
        if feats.shape[1] == 197:        # has cls at position 0
            feats = feats[:, 1:, :]
        return feats

    def forward(self, img):
        z = self.extract(img)                       # [B,196,768]
        z = self.in_proj(z)                          # [B,196,384]
        B = z.shape[0]
        c = self.cls.expand(B, -1, -1)
        u = torch.cat([c, z], dim=1) + self.pos[:, :z.shape[1]+1]
        u = self.encoder(u)
        zc = self.norm(u[:, 0])
        return {"severity": self.h_sev(zc), "density": self.h_dens(zc),
                "count": self.h_cnt(zc).squeeze(-1),
                "side": self.h_side(zc), "detect": self.h_det(zc)}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = 0; sev_c = det_c = side_c = side_t = 0
    cnt_abs = 0.0; dens_abs = np.zeros(2)
    sev_cm = np.zeros((3, 3), int); det_cm = np.zeros((3, 3), int)
    for b in loader:
        out = model(b["img"].to(device)); n += b["img"].shape[0]
        sp = out["severity"].argmax(1).cpu().numpy(); st = b["severity"].numpy()
        for t, p in zip(st, sp): sev_cm[t, p] += 1
        dp = out["detect"].argmax(1).cpu().numpy(); dt = b["detect"].numpy()
        for t, p in zip(dt, dp): det_cm[t, p] += 1
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
        return float(np.mean([(cm[i][i]/cm[i].sum()) if cm[i].sum() else 0.0
                              for i in range(len(cm))]))
    return {"severity_acc": float(sev_c/n), "severity_bal_acc": bal(sev_cm),
            "detect_acc": float(det_c/n), "detect_bal_acc": bal(det_cm),
            "side_acc": float(side_c/side_t) if side_t else 0.0,
            "count_mae": float(cnt_abs/n), "nobj_mae": float(dens_abs[0]/n),
            "nsmall_mae": float(dens_abs[1]/n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xwod-root", default="data/xwod")
    ap.add_argument("--label-dir",
                    default="data/understanding_labels")
    ap.add_argument("--out-root", default="runs/weatherfilm_runs")
    ap.add_argument("--name", default="vit_baseline")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lam", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.out_root, f"{args.name}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"ImageNet ViT-B/16 baseline | seed={args.seed} lambda={args.lam}")
    print("=" * 64)

    tr = ImageDataset("train", args.xwod_root, args.label_dir)
    va = ImageDataset("val", args.xwod_root, args.label_dir)
    te = ImageDataset("test", args.xwod_root, args.label_dir)
    tl = DataLoader(tr, args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate, drop_last=True)
    vl = DataLoader(va, args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate)
    el = DataLoader(te, args.batch, shuffle=False, num_workers=args.workers,
                    pin_memory=True, collate_fn=collate)

    model = ViTDecoder().to(device)
    ce = nn.CrossEntropyLoss(); ce_s = nn.CrossEntropyLoss(ignore_index=-1)
    sl1 = nn.SmoothL1Loss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    best = -1e9; best_state = None; bad = 0

    for ep in range(args.epochs):
        model.train(); model.vit.eval()  # keep ViT frozen/eval
        for b in tl:
            img = b["img"].to(device)
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = (ce(out["severity"], b["severity"].to(device))
                        + ce(out["detect"], b["detect"].to(device))
                        + ce_s(out["side"], b["side"].to(device))
                        + args.lam*sl1(out["density"],
                                       torch.log1p(b["density"].to(device)))
                        + args.lam*sl1(out["count"],
                                       torch.log1p(b["count"].to(device))))
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        m = evaluate(model, vl, device)
        comb = (m["severity_acc"]+m["detect_acc"]+m["side_acc"])/3 \
            - 0.02*m["count_mae"]
        if comb > best + 1e-4:
            best = comb; best_state = copy.deepcopy(
                {k: v for k, v in model.state_dict().items()
                 if not k.startswith("vit.")}); bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {ep+1}"); break
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"ep {ep+1:3d} | sev {m['severity_acc']:.3f} "
                  f"det {m['detect_acc']:.3f} side {m['side_acc']:.3f} "
                  f"cnt {m['count_mae']:.2f}")

    # restore best (non-vit) weights and eval test
    sd = model.state_dict(); sd.update(best_state); model.load_state_dict(sd)
    test_m = evaluate(model, el, device)
    print(f"\n  === ViT-B/16 baseline TEST (seed {args.seed}) ===")
    for k, v in test_m.items():
        print(f"    {k}: {v:.4f}")
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump({"vit": test_m, "seed": args.seed}, f, indent=2)
    print(f"  saved -> {os.path.join(out_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
