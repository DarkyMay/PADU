"""Evaluate validation-selected CNN checkpoints on the XWOD test split."""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cnn_baseline import ImageDataset, CNNBaseline, collate, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/weatherfilm_runs")
    ap.add_argument("--name", default="cnn_baseline")
    ap.add_argument("--xwod-root", default="data/xwod")
    ap.add_argument("--label-dir",
                    default="data/understanding_labels")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print(f"CNN baseline TEST-set evaluation over seeds {args.seeds}")
    print("=" * 64)

    test_ds = ImageDataset("test", args.xwod_root, args.label_dir,
                           args.img_size, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, collate_fn=collate)

    keys = ["severity_acc", "severity_bal_acc", "detect_acc",
            "detect_bal_acc", "side_acc", "count_mae", "nobj_mae",
            "nsmall_mae"]
    acc = {k: [] for k in keys}

    for s in args.seeds:
        ck_path = os.path.join(args.runs, f"{args.name}_seed{s}",
                               "best_cnn.pt")
        if not os.path.exists(ck_path):
            print(f"  [warn] missing ckpt seed {s}: {ck_path}")
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = CNNBaseline().to(device)
        model.load_state_dict(ck["model_state"])
        m = evaluate(model, test_loader, device)
        for k in keys:
            acc[k].append(m[k])
        print(f"  seed {s}: sevAcc {m['severity_acc']:.3f} "
              f"detAcc {m['detect_acc']:.3f} sideAcc {m['side_acc']:.3f} "
              f"cntMAE {m['count_mae']:.2f}")

    print(f"\n  {'metric':18s} {'CNN TEST (mean±std)':>20s}")
    print(f"  {'-'*40}")
    for k in keys:
        a = np.array(acc[k])
        cell = (f"{a.mean():.3f}±{a.std():.3f}" if len(a) > 1
                else (f"{a[0]:.3f}" if len(a) else "n/a"))
        print(f"  {k:18s} {cell:>20s}")

    print("\n  Compare these to eval_test.py (UWM) test numbers.")


if __name__ == "__main__":
    main()
