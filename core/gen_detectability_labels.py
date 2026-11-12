"""Generate detector-specific detectability labels.

Match predictions to ground-truth boxes by class at IoU >= 0.5. The score
is recall times mean matched IoU. Fit tertiles on train and reuse them
for validation and test."""

import os
import json
import argparse
from collections import defaultdict, Counter

import numpy as np

PREFIX_MAP = {
    "heavy_rain": "rain", "flooding": "flooding", "tornado": "tornado",
    "snow": "snow", "wildfire": "wildfire", "dust": "haze_sand_dust",
    "fog": "fog",
}


def infer_weather(fn):
    f = fn.lower()
    for p, w in PREFIX_MAP.items():
        if f.startswith(p):
            return w
    return "unknown"


def load_gt(label_path):
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)
    try:
        return np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
    except Exception:
        return np.zeros((0, 5), dtype=np.float32)


def xywhn_to_xyxy(box, W, H):
    cx, cy, w, h = box
    return (
        (cx - w / 2) * W, (cy - h / 2) * H,
        (cx + w / 2) * W, (cy + h / 2) * H,
    )


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def score_image(gt_arr, pred_boxes, pred_cls, W, H, iou_thr=0.5):
    """Greedy class-aware matching. Returns (recall, loc_qual, score)."""
    n_gt = len(gt_arr)
    if n_gt == 0:
        return None
    gt_xyxy = [xywhn_to_xyxy(g[1:5], W, H) for g in gt_arr]
    gt_cls = [int(g[0]) for g in gt_arr]
    used_pred = set()
    matched_ious = []
    for gi in range(n_gt):
        best_iou, best_pi = 0.0, -1
        for pi in range(len(pred_boxes)):
            if pi in used_pred:
                continue
            if pred_cls[pi] != gt_cls[gi]:
                continue
            iou = iou_xyxy(gt_xyxy[gi], pred_boxes[pi])
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_pi >= 0 and best_iou >= iou_thr:
            used_pred.add(best_pi)
            matched_ious.append(best_iou)
    matched = len(matched_ious)
    recall = matched / n_gt
    loc_qual = float(np.mean(matched_ious)) if matched_ious else 0.0
    return recall, loc_qual, recall * loc_qual


def process_split(split, xwod_root, weights, out_dir, device, batch, imgsz):
    from ultralytics import YOLO
    img_dir = os.path.join(xwod_root,
                           "valid/images" if split == "val"
                           else f"{split}/images")
    lbl_dir = os.path.join(xwod_root,
                           "valid/labels" if split == "val"
                           else f"{split}/labels")
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    print(f"\n[{split}] {len(files)} images — running detector predictions")

    model = YOLO(weights)
    # Run prediction in batches via the stream API.
    scores = {}
    raw = {}
    import cv2
    BATCH = batch
    for start in range(0, len(files), BATCH):
        chunk = files[start:start + BATCH]
        paths = [os.path.join(img_dir, f) for f in chunk]
        results = model.predict(paths, imgsz=imgsz, device=device,
                                verbose=False, conf=0.25)
        for fn, res in zip(chunk, results):
            H, W = res.orig_shape
            if res.boxes is not None and len(res.boxes) > 0:
                pb = res.boxes.xyxy.cpu().numpy().tolist()
                pc = res.boxes.cls.cpu().numpy().astype(int).tolist()
            else:
                pb, pc = [], []
            gt = load_gt(os.path.join(lbl_dir, fn.replace(".jpg", ".txt")))
            s = score_image(gt, pb, pc, W, H)
            if s is None:
                continue  # skip no-GT images for scoring
            recall, loc, sc = s
            scores[fn] = sc
            raw[fn] = {"weather": infer_weather(fn), "recall": recall,
                       "loc_qual": loc, "score": sc, "n_gt": len(gt),
                       "n_pred": len(pb)}
        if (start + BATCH) % (BATCH * 10) == 0:
            print(f"  processed ~{min(start+BATCH, len(files))}/{len(files)}")

    # Fit score tertiles on training images and reuse them on validation and test.
    thr_path = os.path.join(out_dir, "detectability_thresholds.json")
    vals = np.array(list(scores.values()))
    if split == "train":
        q1, q2 = np.quantile(vals, [1/3, 2/3])
        with open(thr_path, "w") as f:
            json.dump({"q1": float(q1), "q2": float(q2)}, f, indent=2)
        print(f"  fit & saved detectability thresholds -> {thr_path}")
    else:
        if not os.path.exists(thr_path):
            raise RuntimeError(
                f"{thr_path} not found. Run --split train first to fit "
                f"detectability thresholds before val/test.")
        with open(thr_path) as f:
            t = json.load(f)
        q1, q2 = t["q1"], t["q2"]

    for fn in raw:
        sc = scores[fn]
        # low score -> hard(2); high score -> easy(0)
        lvl = 2 if sc <= q1 else (1 if sc <= q2 else 0)
        raw[fn]["difficulty"] = int(lvl)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}_detectability_labels.json")
    with open(out_path, "w") as f:
        json.dump(raw, f)
    print(f"  saved {out_path}  ({len(raw)} scored images)")

    # Report
    print(f"\n  === [{split}] detectability report ===")
    print(f"  score: mean={vals.mean():.3f} median={np.median(vals):.3f} "
          f"min={vals.min():.3f} max={vals.max():.3f}")
    print(f"  tertile thresholds: q1(hard<=)= {q1:.3f}, q2(med<=)= {q2:.3f}")
    diff = Counter(r["difficulty"] for r in raw.values())
    print(f"  difficulty: easy={diff[0]} medium={diff[1]} hard={diff[2]}")
    # Report weather composition within each difficulty level.
    print(f"  weather composition per difficulty level:")
    for lvl, name in [(0, "easy"), (1, "medium"), (2, "hard")]:
        wc = Counter(r["weather"] for r in raw.values()
                     if r["difficulty"] == lvl)
        top = wc.most_common(4)
        print(f"    {name:6s}: {dict(top)}")
    # Check the dominant-weather fraction in each difficulty level.
    print("  (if a level is ~entirely one weather, difficulty leaks weather; "
          "the VLM task would be trivial — flag for discussion.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xwod-root", default="data/xwod")
    ap.add_argument("--weights",
                    default="runs/weatherfilm_runs/"
                            "baseline_lr001/weights/best.pt")
    ap.add_argument("--out-dir",
                    default="data/understanding_labels")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    splits = (["train", "val", "test"] if args.split == "all"
              else [args.split])
    for s in splits:
        process_split(s, args.xwod_root, args.weights, args.out_dir,
                      args.device, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
