"""Create a deterministic image-level BDD-IW split manifest.

Hold out 15 percent of the original training images with seed 0 and retain
the original validation split. Report annotation class counts."""

import os
import json
import random
import argparse
from collections import Counter


def list_split(root, split):
    img_dir = os.path.join(root, split, "images")
    return sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))


def class_distribution(root, split, files):
    lbl_dir = os.path.join(root, split, "labels")
    c = Counter()
    boxes_per_img = []
    for f in files:
        lp = os.path.join(lbl_dir, f.replace(".jpg", ".txt"))
        n = 0
        if os.path.exists(lp):
            with open(lp) as fh:
                for line in fh:
                    parts = line.split()
                    if parts:
                        c[int(float(parts[0]))] += 1
                        n += 1
        boxes_per_img.append(n)
    return c, boxes_per_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/BDD-IW")
    ap.add_argument("--test-frac", type=float, default=0.15,
                    help="fraction of original train to hold out as test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir",
                    default="data/bdd_understanding")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    train_files = list_split(args.root, "train")
    val_files = list_split(args.root, "val")
    print(f"original: train={len(train_files)}, val={len(val_files)}")

    # carve test out of train
    shuffled = train_files[:]
    random.shuffle(shuffled)
    n_test = int(len(shuffled) * args.test_frac)
    test_files = sorted(shuffled[:n_test])
    new_train = sorted(shuffled[n_test:])
    print(f"new split: train={len(new_train)}, val={len(val_files)} "
          f"(unchanged), test={len(test_files)}")

    # manifests: which physical dir each split's files live in.
    # train/test both come from the physical "train" dir; val from "val".
    manifest = {
        "train": {"phys_split": "train", "files": new_train},
        "val":   {"phys_split": "val",   "files": val_files},
        "test":  {"phys_split": "train", "files": test_files},
    }
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump(manifest, f)
    print(f"  saved manifest -> {os.path.join(args.out_dir, 'splits.json')}")

    # class distribution over the (new) train
    c, bpi = class_distribution(args.root, "train", new_train)
    total = sum(c.values())
    print(f"\nclass-id distribution over new train "
          f"({total} boxes, {sum(bpi)/len(bpi):.1f} per image):")
    # Report counts by class index.
    for cid, cnt in sorted(c.items()):
        print(f"  id {cid}: {cnt:7d}  ({100*cnt/total:.1f}%)")

    print('\n  Check class names against the dataset annotation metadata.')
    print('  Class counts above are indexed by annotation class ID.')
    print('  Geometry-based tasks use numeric class IDs. Check annotation metadata for class names.')

    # Use prep_bdd_yolo.py to create detector split directories from the manifest.
    print('\n  Create detector training views with core/prep_bdd_yolo.py to exclude held-out test images.')


if __name__ == "__main__":
    main()
