"""Create symlinked YOLO training and validation views from the BDD-IW split manifest."""

import os
import json
import argparse

CLASS_NAMES = ["person", "car", "bus", "truck", "bike",
               "traffic_light", "traffic_sign"]


def link(src, dst):
    if not os.path.exists(dst):
        try:
            os.symlink(os.path.abspath(src), dst)
        except FileExistsError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/BDD-IW")
    ap.add_argument("--manifest",
                    default="data/bdd_understanding/splits.json")
    ap.add_argument("--work", default="data/bdd_yolo")
    args = ap.parse_args()

    with open(args.manifest) as f:
        man = json.load(f)

    for split in ["train", "val"]:
        img_o = os.path.join(args.work, split, "images")
        lbl_o = os.path.join(args.work, split, "labels")
        os.makedirs(img_o, exist_ok=True)
        os.makedirs(lbl_o, exist_ok=True)
        phys = man[split]["phys_split"]
        files = man[split]["files"]
        src_img = os.path.join(args.root, phys, "images")
        src_lbl = os.path.join(args.root, phys, "labels")
        n = 0
        for fn in files:
            link(os.path.join(src_img, fn), os.path.join(img_o, fn))
            tn = fn.replace(".jpg", ".txt")
            sp = os.path.join(src_lbl, tn)
            if os.path.exists(sp):
                link(sp, os.path.join(lbl_o, tn))
            n += 1
        print(f"  {split}: linked {n} images")

    yaml_path = os.path.join(args.work, "bdd.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.work)}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
    print(f"  wrote {yaml_path}")

    print("\n  Train the baseline with:")
    print(f"  yolo detect train data={yaml_path} model=yolov8m.pt "
          f"epochs=50 imgsz=640 batch=16 "
          f"project=runs/weatherfilm_runs name=bdd_baseline")


if __name__ == "__main__":
    main()
