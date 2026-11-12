"""Group XWOD frames by the fixed PaDU risk rule and report detector quality."""

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_model import (UnifiedWeatherModel, WeatherUnderstandingDataset,
                           collate)

ACTIONS = {0: "normal operation",
           1: "raise detection threshold / reduce speed",
           2: "request human takeover"}
RISK_NAME = {0: "low", 1: "medium", 2: "high"}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    sev_p, det_p = [], []
    for b in loader:
        o = model(b["tokens"].to(device))
        sev_p.append(o["severity"].argmax(1).cpu().numpy())
        det_p.append(o["detect"].argmax(1).cpu().numpy())
    return np.concatenate(sev_p), np.concatenate(det_p)


def risk_rule(sev, det, w_det=1.0, w_sev=0.5):
    """Map severity and detectability predictions to low, medium or high risk."""
    raw = w_det * det + w_sev * sev          # in [0, 3]
    # Risk thresholds: <= 0.5 low, <= 1.75 medium, otherwise high.
    return np.where(raw <= 0.5, 0, np.where(raw <= 1.75, 1, 2))


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
    print("Safety-decision demo: predicted risk vs. TRUE detector score")
    print("=" * 64)

    test_ds = WeatherUnderstandingDataset("test", args.tok_root,
                                          args.label_dir)
    loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)
    items = test_ds.items

    # true detector score per image, from detectability labels
    with open(os.path.join(args.label_dir,
                           "test_detectability_labels.json")) as f:
        det_lab = json.load(f)
    true_score = np.array([det_lab[f]["score"] if f in det_lab else np.nan
                           for f in items])
    true_recall = np.array([det_lab[f]["recall"] if f in det_lab else np.nan
                            for f in items])

    # Collect per-seed task predictions for majority voting.
    sev_votes, det_votes = [], []
    for s in args.seeds:
        ck_path = os.path.join(args.runs, f"{args.name}_seed{s}", "best_uwm.pt")
        if not os.path.exists(ck_path):
            print(f"  [warn] missing ckpt seed {s}"); continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        ma = ck.get("args", {})
        model = UnifiedWeatherModel(
            d_model=ma.get("d_model", 384), n_layers=ma.get("n_layers", 3),
            n_heads=ma.get("n_heads", 6), dropout=ma.get("dropout", 0.1)
        ).to(device)
        model.load_state_dict(ck["model_state"])
        sp, dp = predict(model, loader, device)
        sev_votes.append(sp); det_votes.append(dp)

    sev_votes = np.stack(sev_votes); det_votes = np.stack(det_votes)
    # majority vote across seeds
    sev_pred = np.array([np.bincount(sev_votes[:, i], minlength=3).argmax()
                         for i in range(sev_votes.shape[1])])
    det_pred = np.array([np.bincount(det_votes[:, i], minlength=3).argmax()
                         for i in range(det_votes.shape[1])])

    risk = risk_rule(sev_pred, det_pred)

    # Validation: true detector score grouped by predicted risk
    valid = ~np.isnan(true_score)
    print("\n  Predicted-risk groups vs. TRUE detector performance:")
    print(f"  {'risk':8s} {'n':>5s} {'true score':>12s} "
          f"{'true recall':>12s}")
    print("  " + "-" * 40)
    rows = []
    for r in [0, 1, 2]:
        m = (risk == r) & valid
        if m.sum() == 0:
            continue
        ts = float(true_score[m].mean()); tr = float(true_recall[m].mean())
        print(f"  {RISK_NAME[r]:8s} {int(m.sum()):>5d} "
              f"{ts:>12.3f} {tr:>12.3f}")
        rows.append((RISK_NAME[r], int(m.sum()), ts, tr))

    # Compare detector quality across predicted risk groups.
    if len(rows) >= 2:
        scores_by_risk = [r[2] for r in rows]
        mono = all(scores_by_risk[i] >= scores_by_risk[i+1]
                   for i in range(len(scores_by_risk)-1))
        print(f"\n  True detector score decreases with predicted risk: "
              f"{'YES' if mono else 'NO'}")
        print(f"  (low-risk true score {scores_by_risk[0]:.3f} vs "
              f"high-risk {scores_by_risk[-1]:.3f})")

    # A few qualitative cases (one per risk level)
    print("\n  Qualitative cases (filename | sev_pred det_pred | "
          "true_score | action):")
    for r in [0, 1, 2]:
        idxs = np.where((risk == r) & valid)[0]
        if len(idxs) == 0:
            continue
        # pick the case closest to that group's mean true score
        gm = true_score[idxs].mean()
        pick = idxs[np.argmin(np.abs(true_score[idxs] - gm))]
        print(f"    [{RISK_NAME[r]:6s}] {items[pick]:35s} "
              f"sev={sev_pred[pick]} det={det_pred[pick]} "
              f"score={true_score[pick]:.3f} -> {ACTIONS[r]}")

    # Save the selected examples.
    out = {"groups": [{"risk": r[0], "n": r[1], "true_score": r[2],
                       "true_recall": r[3]} for r in rows],
           "case_files": items}
    with open("./decision_demo_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  saved -> ./decision_demo_results.json")


if __name__ == "__main__":
    main()
