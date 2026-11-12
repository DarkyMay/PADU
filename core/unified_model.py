"""PaDU dataset and multi-task transformer decoder.

Map frozen 192 by 4096 visual tokens to severity and detectability classes,
log-count regressions, and the largest-object side through shared features."""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


CLASS_NAMES = ["person", "car", "truck", "motorcycle", "bus", "bike"]



# Dataset: patch tokens + merged 4-task labels

class WeatherUnderstandingDataset(Dataset):
    def __init__(self, split, tok_root, label_dir, grid_hw=(12, 16)):
        self.split = split
        self.TH, self.TW = grid_hw

        # patch tokens [N, 192, 4096] fp16 (memory-mapped)
        self.tokens = np.load(os.path.join(tok_root, f"{split}_tokens.npy"),
                              mmap_mode="r")
        meta = np.load(os.path.join(tok_root, f"{split}_meta.npz"),
                       allow_pickle=True)
        self.fnames_tok = list(meta["filenames"])
        self.valid = {f: bool(v) for f, v in zip(self.fnames_tok,
                                                 meta["valid"])}
        self.f2tok = {f: i for i, f in enumerate(self.fnames_tok)}

        # understanding labels (severity/density/vqa)
        with open(os.path.join(label_dir,
                               f"{split}_understanding_labels.json")) as f:
            self.u = json.load(f)
        # detectability labels (separate file; some images skipped if no GT)
        with open(os.path.join(label_dir,
                               f"{split}_detectability_labels.json")) as f:
            self.d = json.load(f)

        # usable = has token + valid token + has understanding label + has
        # detectability label (so all 4 tasks have a target)
        self.items = [f for f in self.fnames_tok
                      if self.valid.get(f, False)
                      and f in self.u and f in self.d]
        print(f"  [{split}] usable images: {len(self.items)} "
              f"(of {len(self.fnames_tok)} tokens)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn = self.items[idx]
        tok = np.asarray(self.tokens[self.f2tok[fn]], dtype=np.float32)
        tok = torch.from_numpy(tok)                  # [192, 4096]

        u = self.u[fn]
        d = self.d[fn]
        # targets
        severity = int(u["severity"])                # 0/1/2
        n_obj = float(u["density"]["n_objects"])
        n_small = float(u["density"]["n_small"])
        count = float(u["vqa"]["count_total"])
        side = u["vqa"]["largest_side"]
        side_lbl = 0 if side == "left" else (1 if side == "right" else -1)
        detect = int(d["difficulty"])                # 0/1/2

        return {
            "tokens": tok,
            "severity": torch.tensor(severity, dtype=torch.long),
            "density": torch.tensor([n_obj, n_small], dtype=torch.float32),
            "count": torch.tensor(count, dtype=torch.float32),
            "side": torch.tensor(side_lbl, dtype=torch.long),
            "detect": torch.tensor(detect, dtype=torch.long),
            "fname": fn,
        }


def collate(batch):
    return {
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "severity": torch.stack([b["severity"] for b in batch]),
        "density": torch.stack([b["density"] for b in batch]),
        "count": torch.stack([b["count"] for b in batch]),
        "side": torch.stack([b["side"] for b in batch]),
        "detect": torch.stack([b["detect"] for b in batch]),
        "fnames": [b["fname"] for b in batch],
    }



# Model

class TransformerEncoder(nn.Module):
    """Lightweight transformer over 192 VLM tokens with a [CLS] aggregator."""

    def __init__(self, emb_dim=4096, d_model=384, n_layers=3, n_heads=6,
                 grid_hw=(12, 16), dropout=0.1):
        super().__init__()
        self.TH, self.TW = grid_hw
        K = self.TH * self.TW
        self.in_proj = nn.Linear(emb_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, K, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, tokens):
        # tokens: [B, K, emb_dim]
        B = tokens.shape[0]
        x = self.in_proj(tokens) + self.pos          # [B, K, d]
        cls = self.cls.expand(B, -1, -1)             # [B, 1, d]
        x = torch.cat([cls, x], dim=1)               # [B, K+1, d]
        x = self.encoder(x)
        z = self.norm(x[:, 0])                       # [B, d] (CLS)
        return z


class UnifiedWeatherModel(nn.Module):
    def __init__(self, emb_dim=4096, d_model=384, n_layers=3, n_heads=6,
                 grid_hw=(12, 16), dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            emb_dim, d_model, n_layers, n_heads, grid_hw, dropout)
        d = d_model

        def head(out):
            return nn.Sequential(
                nn.Linear(d, d // 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(d // 2, out))

        self.head_severity = head(3)        # 3-way cls
        self.head_density = head(2)         # regress (n_obj, n_small) log1p
        self.head_count = head(1)           # regress count log1p
        self.head_side = head(2)            # left/right
        self.head_detect = head(3)          # 3-way cls

        n = sum(p.numel() for p in self.parameters())
        print(f"  UnifiedWeatherModel: {n/1e6:.2f}M params "
              f"(d_model={d_model}, layers={n_layers})")

    def forward(self, tokens):
        z = self.encoder(tokens)
        return {
            "severity": self.head_severity(z),
            "density": self.head_density(z),
            "count": self.head_count(z).squeeze(-1),
            "side": self.head_side(z),
            "detect": self.head_detect(z),
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    m = UnifiedWeatherModel()
    x = torch.randn(4, 192, 4096)
    out = m(x)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print("  OK: forward produces all 5 task outputs.")
