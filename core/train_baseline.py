"""Train and evaluate a configurable YOLOv8m baseline on XWOD.

Use configs/detectors/ for the detector recipes used in the manuscript."""

import argparse
import os
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--data", default="data/xwod/data_fixed.yaml")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=-1)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/weatherfilm_runs")
    p.add_argument("--name", default="baseline_yolov8m")
    args = p.parse_args()

    
    # Phase 1: Training
    
    print("=" * 60)
    print("Phase 1: Training Baseline YOLO")
    print("=" * 60)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        label_smoothing=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,
        workers=4,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )

    
    # Phase 2: Test Evaluation
    
    print("\n" + "=" * 60)
    print("Phase 2: Evaluating on Test Set")
    print("=" * 60)

    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_path):
        model = YOLO(best_path)
        print(f"  Loaded best weights: {best_path}")
    else:
        print(f"  best.pt not found, using last trained model")

    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=True,
    )

    print(f"\n{'=' * 60}")
    print(f"BASELINE RESULTS (Test Set)")
    print(f"{'=' * 60}")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")

    class_names = ["person", "car", "truck", "motorcycle", "bus", "bike"]
    print(f"\n  Per-class mAP50:")
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        for i, name in enumerate(class_names):
            if i < len(metrics.box.ap50):
                print(f"    {name:12s}: {metrics.box.ap50[i]:.4f}")

    print(f"\n  Weights: {best_path}")


if __name__ == "__main__":
    main()
