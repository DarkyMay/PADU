"""Generate BDD-IW detectability labels using a detector trained on the carved training split."""

import os
import json
import argparse
from collections import Counter

import numpy as np


def load_gt(label_path):
    if not os.path.exists(label_path):
        return np.zeros((0, 5), dtype=np.float32)
    try:
        return np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
    except Exception:
        return np.zeros((0, 5), dtype=np.float32)


def xywhn_to_xyxy(box, W, H):
    cx, cy, w, h = box
    return ((cx - w / 2) * W, (cy - h / 2) * H,
            (cx + w / 2) * W, (cy + h / 2) * H)


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def score_image(gt_arr, pred_boxes, pred_cls, W, H, iou_thr=0.5):
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


def process_split(split, bdd_root, manifest, weights, out_dir,
                  device, batch, imgsz):
    from ultralytics import YOLO
    phys = manifest[split]["phys_split"]
    files = sorted(manifest[split]["files"])
    img_dir = os.path.join(bdd_root, phys, "images")
    lbl_dir = os.path.join(bdd_root, phys, "labels")
    print(f"\n[{split}] {len(files)} images (physical {phys}/) "
          f"— running detector")

    model = YOLO(weights)
    scores, raw = {}, {}
    for start in range(0, len(files), batch):
        chunk = files[start:start + batch]
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
                continue
            recall, loc, sc = s
            scores[fn] = sc
            raw[fn] = {"weather": "unknown", "recall": recall,
                       "loc_qual": loc, "score": sc, "n_gt": len(gt),
                       "n_pred": len(pb)}
        if (start + batch) % (batch * 20) == 0:
            print(f"  processed ~{min(start+batch, len(files))}/{len(files)}")

    thr_path = os.path.join(out_dir, "detectability_thresholds.json")
    os.makedirs(out_dir, exist_ok=True)
    vals = np.array(list(scores.values()))
    if split == "train":
        q1, q2 = np.quantile(vals, [1/3, 2/3])
        with open(thr_path, "w") as f:
            json.dump({"q1": float(q1), "q2": float(q2)}, f, indent=2)
        print(f"  fit & saved detectability thresholds -> {thr_path}")
    else:
        if not os.path.exists(thr_path):
            raise RuntimeError(
                f"{thr_path} not found. Run --split train first.")
        with open(thr_path) as f:
            t = json.load(f)
        q1, q2 = t["q1"], t["q2"]

    for fn in raw:
        sc = scores[fn]
        lvl = 2 if sc <= q1 else (1 if sc <= q2 else 0)
        raw[fn]["difficulty"] = int(lvl)

    out_path = os.path.join(out_dir, f"{split}_detectability_labels.json")
    with open(out_path, "w") as f:
        json.dump(raw, f)
    print(f"  saved {out_path}  ({len(raw)} scored images)")

    print(f"\n  === [{split}] detectability report ===")
    print(f"  score: mean={vals.mean():.3f} median={np.median(vals):.3f} "
          f"min={vals.min():.3f} max={vals.max():.3f}")
    print(f"  tertile thresholds: q1(hard<=)= {q1:.3f}, q2(med<=)= {q2:.3f}")
    diff = Counter(r["difficulty"] for r in raw.values())
    print(f"  difficulty: easy={diff[0]} medium={diff[1]} hard={diff[2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd-root", default="data/BDD-IW")
    ap.add_argument("--manifest",
                    default="data/bdd_understanding/splits.json")
    ap.add_argument("--weights",
                    default="runs/weatherfilm_runs/"
                            "bdd_baseline-2/weights/best.pt")
    ap.add_argument("--out-dir",
                    default="data/bdd_understanding_labels")
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    splits = (["train", "val", "test"] if args.split == "all"
              else [args.split])
    for s in splits:
        process_split(s, args.bdd_root, manifest, args.weights,
                      args.out_dir, args.device, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
