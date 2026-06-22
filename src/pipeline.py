"""Main pipeline: RTSP -> YOLO detect+track -> line-crossing count -> CSV log.

Run: python -m src.pipeline --config config.yaml [--display] [--save-video out.mp4]
"""
from __future__ import annotations

import argparse
import time

import cv2
import yaml
from ultralytics import YOLO

from src.counter import LineCounter
from src.logger import TrafficLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg_device: str) -> str:
    if cfg_device != "auto":
        return cfg_device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run(cfg: dict, display: bool, save_video_path: str | None):
    device = resolve_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    model = YOLO(cfg["model_path"])
    class_names = {int(k): v for k, v in cfg.get("class_names", {}).items()}
    classes_of_interest = cfg.get("classes_of_interest") or list(class_names.keys())

    counter = LineCounter(
        line_a=tuple(cfg["line"]["point_a"]),
        line_b=tuple(cfg["line"]["point_b"]),
        out_direction=cfg["out_direction"],
        class_names=class_names,
    )
    logger = TrafficLogger(cfg["log_path"])

    video_writer = None
    last_log_time = time.monotonic()
    log_interval = cfg.get("log_interval_seconds", 60)
    reconnect_delay = cfg.get("reconnect_delay_seconds", 5)

    # model.track(stream=True, ...) handles RTSP reconnects poorly on its own,
    # so we wrap it in a retry loop at the outer level.
    while True:
        try:
            results_gen = model.track(
                source=cfg["rtsp_url"],
                stream=True,
                persist=True,
                tracker=cfg.get("tracker", "bytetrack.yaml"),
                conf=cfg.get("confidence_threshold", 0.35),
                classes=classes_of_interest,
                device=device,
                vid_stride=cfg.get("vid_stride", 1),
                verbose=False,
            )

            for result in results_gen:
                frame = result.orig_img
                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    track_ids = boxes.id.cpu().numpy().astype(int)
                    cls_ids = boxes.cls.cpu().numpy().astype(int)

                    for (x1, y1, x2, y2), tid, cid in zip(xyxy, track_ids, cls_ids):
                        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                        counter.update(int(tid), centroid, int(cid))

                now = time.monotonic()
                if now - last_log_time >= log_interval:
                    logger.log(
                        counter.count_in,
                        counter.count_out,
                        dict(counter.count_in_by_class),
                        dict(counter.count_out_by_class),
                    )
                    print(
                        f"[{time.strftime('%H:%M:%S')}] in={counter.count_in} "
                        f"out={counter.count_out} net={counter.count_in - counter.count_out}"
                    )
                    counter.reset()
                    last_log_time = now

                if display or save_video_path:
                    annotated = result.plot()
                    cv2.line(
                        annotated,
                        tuple(cfg["line"]["point_a"]),
                        tuple(cfg["line"]["point_b"]),
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"in={counter.count_in} out={counter.count_out}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

                    if save_video_path:
                        if video_writer is None:
                            h, w = annotated.shape[:2]
                            video_writer = cv2.VideoWriter(
                                save_video_path,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                15,
                                (w, h),
                            )
                        video_writer.write(annotated)

                    if display:
                        cv2.imshow("beehive monitor", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            return

        except Exception as exc:  # noqa: BLE001 - want to survive and retry RTSP hiccups
            print(f"Stream error: {exc!r}. Reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
        finally:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            if display:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--save-video", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, display=args.display, save_video_path=args