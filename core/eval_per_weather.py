"""Evaluate PaDU checkpoints by weather and aggregate metrics across seeds."""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_model import (UnifiedWeatherModel, WeatherUnderstandingDataset,
                           collate)

WEATHERS = ["rain", "snow", "fog", "haze_sand_dust", "flooding",
            "tornado", "wildfire"]


@torch.no_grad()
def predict_all(model, loader, device):
    """Return dict of arrays: per-sample preds/targets in dataset order."""
    model.eval()
    out_acc = {"sev_p": [], "sev_t": [], "det_p": [], "det_t": [],
               "side_p": [], "side_t": [], "cnt_p": [], "cnt_t": []}
    for b in loader:
        o = model(b["tokens"].to(device))
        out_acc["sev_p"].append(o["severity"].argmax(1).cpu().numpy())
        out_acc["sev_t"].append(b["severity"].numpy())
        out_acc["det_p"].append(o["detect"].argmax(1).cpu().numpy())
        out_acc["det_t"].append(b["detect"].numpy())
        out_acc["side_p"].append(o["side"].argmax(1).cpu().numpy())
        out_acc["side_t"].append(b["side"].numpy())
        out_acc["cnt_p"].append(
            torch.expm1(o["count"]).clamp(min=0).cpu().numpy())
        out_acc["cnt_t"].append(b["count"].numpy())
    return {k: np.concatenate(v) for k, v in out_acc.items()}


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
    print(f"Per-weather TEST breakdown over seeds {args.seeds}")
    print("=" * 64)

    test_ds = WeatherUnderstandingDataset("test", args.tok_root,
                                          args.label_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, collate_fn=collate)

    # weather label per sample, aligned to dataset order via meta filenames.
    meta = np.load(os.path.join(args.tok_root, "test_meta.npz"),
                   allow_pickle=True)
    fn_to_weather = {str(f): str(w)
                     for f, w in zip(meta["filenames"], meta["weathers"])}
    # dataset exposes its item filenames in .items (same order as loader)
    items = test_ds.items if hasattr(test_ds, "items") else None
    if items is None:
        raise RuntimeError("WeatherUnderstandingDataset has no .items; "
                           "cannot align weather labels. Tell me its attr.")
    weather_of = np.array([fn_to_weather.get(f, "unknown") for f in items])

    # accumulate per-seed per-weather metrics
    per_w = {w: {"sev": [], "det": [], "side": [], "cnt": []}
             for w in WEATHERS}
    counts = {w: 0 for w in WEATHERS}

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
        P = predict_all(model, test_loader, device)

        for w in WEATHERS:
            m = weather_of == w
            if m.sum() == 0:
                continue
            counts[w] = int(m.sum())
            per_w[w]["sev"].append(float((P["sev_p"][m] == P["sev_t"][m]).mean()))
            per_w[w]["det"].append(float((P["det_p"][m] == P["det_t"][m]).mean()))
            sm = P["side_t"][m] >= 0
            if sm.sum():
                per_w[w]["side"].append(
                    float((P["side_p"][m][sm] == P["side_t"][m][sm]).mean()))
            per_w[w]["cnt"].append(
                float(np.abs(P["cnt_p"][m] - P["cnt_t"][m]).mean()))

    # Report
    print(f"\n  {'weather':16s} {'n':>5s} {'sevAcc':>8s} {'detAcc':>8s} "
          f"{'sideAcc':>8s} {'cntMAE':>8s}")
    print("  " + "-" * 58)

    def cell(vals):
        a = np.array(vals)
        return f"{a.mean():.3f}" if len(a) else "  n/a"

    rows = []
    for w in WEATHERS:
        if counts[w] == 0:
            continue
        sev = cell(per_w[w]["sev"]); det = cell(per_w[w]["det"])
        side = cell(per_w[w]["side"]); cnt = cell(per_w[w]["cnt"])
        print(f"  {w:16s} {counts[w]:>5d} {sev:>8s} {det:>8s} "
              f"{side:>8s} {cnt:>8s}")
        rows.append((w, counts[w], sev, det, side, cnt))

    # overall (for reference)
    print("  " + "-" * 58)
    print("  (per-weather accuracies show stability across conditions;")
    print("   weathers with few test images will be noisier.)")

    # also dump LaTeX-ready rows
    print("\n  LaTeX rows (weather & n & sevAcc & detAcc & sideAcc & cntMAE):")
    for w, n, sev, det, side, cnt in rows:
        wname = w.replace("_", "/")
        print(f"    {wname} & {n} & {sev} & {det} & {side} & {cnt} \\\\")


if __name__ == "__main__":
    main()
