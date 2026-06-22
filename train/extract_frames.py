"""Sample frames from the live RTSP feed (or a recorded clip) for labeling.

You need frames from YOUR camera, at YOUR angle, in YOUR lighting — generic
public bee datasets help bootstrap a model but won't generalize perfectly to
your specific entrance setup. This script grabs evenly-spaced frames you can
hand-label.

Usage:
  python train/extract_frames.py --source rtsp://... --out train/raw_frames --every 2 --max 300
  python train/extract_frames.py --source recorded_clip.mp4 --out train/raw_frames --every 2
"""
from __future__ import annotations

import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="RTSP URL or path to a video file")
    parser.add_argument("--out", default="train/raw_frames")
    parser.add_argument("--every", type=float, default=2.0, help="seconds between saved frames")
    parser.add_argument("--max", type=int, default=500, help="max frames to save")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_interval = max(1, int(fps * args.every))

    saved = 0
    frame_idx = 0
    while saved < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(args.out, f"frame_{saved:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {args.out}")
    print(
        "Next: label these in Roboflow (web) or CVAT/Label Studio (self-hosted), "
        "export in YOLO format, then point train/dataset.yaml at the export."
    )


if __name__ == "__main__":
    main()
