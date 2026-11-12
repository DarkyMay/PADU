"""Generate BDD-IW scene labels with global training-set severity thresholds."""

import os
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import cv2

# BDD-IW class names in annotation index order.
CLASS_NAMES = ["person", "car", "bus", "truck", "bike",
               "traffic_light", "traffic_sign"]
SMALL_AREA_FRAC = 0.005


# Image quality proxies (identical to xwod version)
def dark_channel_mean(img, patch=15):
    min_c = img.min(axis=2)
    kernel = np.ones((patch, patch), np.uint8)
    dark = cv2.erode(min_c, kernel)
    return float(dark.mean()) / 255.0


def rms_contrast(gray):
    g = gray.astype(np.float32) / 255.0
    return float(g.std())


def image_quality_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "dark_channel": dark_channel_mean(img_bgr),
        "rms_contrast": rms_contrast(gray),
        "brightness": float(gray.mean()) / 255.0,
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


# Box geometry (identical logic)
def load_boxes(label_path):
    if not os.path.exists(label_path):
        return []
    out = []
    try:
        arr = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
        for row in arr:
            out.append(tuple(float(v) for v in row))
    except Exception:
        pass
    return out


def iou_xywh(a, b):
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[3] * a[4] + b[3] * b[4] - inter
    return inter / ua if ua > 0 else 0.0


def density_features(boxes):
    n = len(boxes)
    small = sum(1 for b in boxes if b[3] * b[4] < SMALL_AREA_FRAC)
    occ = 0.0
    if n > 1:
        per = []
        for i in range(n):
            best = 0.0
            for j in range(n):
                if i != j:
                    best = max(best, iou_xywh(boxes[i], boxes[j]))
            per.append(best)
        occ = float(np.mean(per))
    return {"n_objects": n, "n_small": small,
            "small_frac": (small / n if n else 0.0),
            "occlusion": occ}


def vqa_pairs(boxes):
    n = len(boxes)
    cls_counter = Counter(int(b[0]) for b in boxes)
    qa = {"count_total": n}
    if cls_counter:
        dom = cls_counter.most_common(1)[0][0]
        qa["dominant_class"] = (CLASS_NAMES[dom] if dom < len(CLASS_NAMES)
                                else str(dom))
        largest = max(boxes, key=lambda b: b[3] * b[4])
        lc = int(largest[0])
        qa["largest_class"] = (CLASS_NAMES[lc] if lc < len(CLASS_NAMES)
                               else str(lc))
        qa["largest_side"] = "left" if largest[1] < 0.5 else "right"
        qa["per_class_count"] = {CLASS_NAMES[c]: cnt
                                 for c, cnt in cls_counter.items()
                                 if c < len(CLASS_NAMES)}
    else:
        qa["dominant_class"] = "none"
        qa["largest_class"] = "none"
        qa["largest_side"] = "none"
        qa["per_class_count"] = {}
    return qa


def process_split(split, bdd_root, manifest, out_dir):
    phys = manifest[split]["phys_split"]
    files = sorted(manifest[split]["files"])
    img_dir = os.path.join(bdd_root, phys, "images")
    lbl_dir = os.path.join(bdd_root, phys, "labels")
    print(f"\n[{split}] {len(files)} images (physical {phys}/)")

    records = {}
    raw_q = {}
    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, fn))
        if img is None:
            continue
        q = image_quality_features(img)
        raw_q[fn] = q
        boxes = load_boxes(os.path.join(lbl_dir, fn.replace(".jpg", ".txt")))
        records[fn] = {
            "weather": "unknown",   # Weather labels are unavailable in these task inputs.
            "quality": q,
            "density": density_features(boxes),
            "vqa": vqa_pairs(boxes),
        }
        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(files)}")

    # Task1 severity: global grouping (not per-weather)
    # degradation score = z(dark) - z(contrast) - z(log1p sharpness),
    # Fit normalization and tertiles on training images; reuse them on validation and test.
    thr_path = os.path.join(out_dir, "severity_thresholds.json")
    fit_mode = (split == "train")
    os.makedirs(out_dir, exist_ok=True)

    fns = list(records.keys())
    dc = np.array([raw_q[fn]["dark_channel"] for fn in fns])
    ct = np.array([raw_q[fn]["rms_contrast"] for fn in fns])
    sh = np.array([raw_q[fn]["sharpness"] for fn in fns])
    logsh = np.log1p(sh)

    if fit_mode:
        p = {
            "dc_m": float(dc.mean()), "dc_s": float(dc.std() + 1e-6),
            "ct_m": float(ct.mean()), "ct_s": float(ct.std() + 1e-6),
            "sh_m": float(logsh.mean()), "sh_s": float(logsh.std() + 1e-6),
        }
        score = ((dc - p["dc_m"]) / p["dc_s"]
                 - (ct - p["ct_m"]) / p["ct_s"]
                 - (logsh - p["sh_m"]) / p["sh_s"])
        q1, q2 = np.quantile(score, [1/3, 2/3])
        p["q1"], p["q2"] = float(q1), float(q2)
        with open(thr_path, "w") as f:
            json.dump(p, f, indent=2)
        print(f"  fit & saved GLOBAL severity thresholds -> {thr_path}")
    else:
        if not os.path.exists(thr_path):
            raise RuntimeError(
                f"{thr_path} not found. Run --split train first.")
        with open(thr_path) as f:
            p = json.load(f)
        score = ((dc - p["dc_m"]) / p["dc_s"]
                 - (ct - p["ct_m"]) / p["ct_s"]
                 - (logsh - p["sh_m"]) / p["sh_s"])
        q1, q2 = p["q1"], p["q2"]

    for fn, sc in zip(fns, score):
        lvl = 0 if sc <= q1 else (1 if sc <= q2 else 2)
        records[fn]["severity"] = int(lvl)

    out_path = os.path.join(out_dir, f"{split}_understanding_labels.json")
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"  saved {out_path}")

    # Report
    print(f"\n  === [{split}] task distribution report ===")
    sev = Counter(r["severity"] for r in records.values())
    print(f"  Task1 severity (GLOBAL tertiles): "
          f"mild={sev[0]} moderate={sev[1]} severe={sev[2]}")
    nobj = [r["density"]["n_objects"] for r in records.values()]
    nsmall = [r["density"]["n_small"] for r in records.values()]
    print(f"  Task2 density: n_objects mean={np.mean(nobj):.1f} "
          f"median={np.median(nobj):.0f} max={max(nobj)}; "
          f"small mean={np.mean(nsmall):.1f}")
    counts = [r["vqa"]["count_total"] for r in records.values()]
    sides = Counter(r["vqa"]["largest_side"] for r in records.values())
    print(f"  Task3 VQA: count mean={np.mean(counts):.1f}; "
          f"largest_side={dict(sides)}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd-root", default="data/BDD-IW")
    ap.add_argument("--manifest",
                    default="data/bdd_understanding/splits.json")
    ap.add_argument("--out-dir",
                    default="data/bdd_understanding_labels")
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all"])
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    splits = (["train", "val", "test"] if args.split == "all"
              else [args.split])
    for s in splits:
        process_split(s, args.bdd_root, manifest, args.out_dir)


if __name__ == "__main__":
    main()
