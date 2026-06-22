"""One-off helper: grab a frame from the RTSP stream and click two points to
define the counting line. Prints coordinates to paste into config.yaml.
"""
from __future__ import annotations

import argparse

import cv2
import yaml

_points: list[tuple[int, int]] = []


def _on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_points) < 2:
        _points.append((x, y))
        print(f"Point {len(_points)}: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cap = cv2.VideoCapture(cfg["rtsp_url"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(
            "Could not read a frame from the RTSP stream. Check rtsp_url, "
            "network reachability, and that another process isn't already "
            "holding the stream open."
        )

    window = "Click two points across the entrance, then press 'q'"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, _on_click)

    while True:
        display = frame.copy()
        for p in _points:
            cv2.circle(display, p, 5, (0, 0, 255), -1)
        if len(_points) == 2:
            cv2.line(display, _points[0], _points[1], (0, 255, 0), 2)
        cv2.imshow(window, display)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(_points) == 2:
        print("\nPaste into config.yaml:")
        print("line:")
        print(f"  point_a: [{_points[0][0]}, {_points[0][1]}]")
        print(f"  point_b: [{_points[1][0]}, {_points[1][1]}]")
    else:
        print("Need exactly 2 points; nothing saved.")


if __name__ == "__main__":
    main()
