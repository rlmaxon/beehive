"""Fine-tune a YOLOv8 model on your labeled bee dataset.

Defaults below are tuned for CPU-only training (no NVIDIA GPU): smaller image
size and batch than the GPU-typical 640/16, since CPU memory bandwidth -- not
compute -- is usually the bottleneck. If you train on a cloud GPU instead
(recommended for speed -- see TRAINING_PLAN.md), pass --device 0 --imgsz 640
--batch 16 to use settings more typical for GPU training.

Usage:
  python train/train.py --data train/dataset.yaml --base yolov8n.pt --epochs 60
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train/dataset.yaml")
    parser.add_argument("--base", default="yolov8n.pt", help="starting checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device", default="cpu", help="'cpu', '0' for first GPU, or 'mps' on Apple Silicon"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="cache images in RAM for faster epochs -- only enable if dataset fits comfortably in memory",
    )
    parser.add_argument("--workers", type=int, default=4, help="dataloader worker processes (CPU cores)")
    parser.add_argument("--project", default="train/runs")
    parser.add_argument("--name", default="bee_detector")
    args = parser.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        cache=args.cache,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=20,        # early stop if val mAP plateaus
        augment=True,
    )

    metrics = model.val()
    print(metrics)
    print(
        f"\nBest weights: {args.project}/{args.name}/weights/best.pt\n"
        "Copy that file to beehive_monitor/models/ and point config.yaml's "
        "model_path at it."
    )


if __name__ == "__main__":
    main()
