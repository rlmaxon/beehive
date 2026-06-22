"""Append-only CSV logger for periodic traffic snapshots."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone


class TrafficLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._header_written = os.path.exists(path) and os.path.getsize(path) > 0

    def log(self, count_in: int, count_out: int, in_by_class: dict, out_by_class: dict):
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "count_in": count_in,
            "count_out": count_out,
            "net": count_in - count_out,
        }
        for cls, n in in_by_class.items():
            row[f"in_{cls}"] = n
        for cls, n in out_by_class.items():
            row[f"out_{cls}"] = n

        write_header = not self._header_written
        # CSV needs a stable header; if class columns change across runs,
        # easiest fix is to start a new log file rather than reconcile schemas.
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
