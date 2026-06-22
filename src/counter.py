"""Directional line-crossing counter for tracked objects.

Each tracked bee (identified by a persistent track ID from the tracker) gets
its centroid history recorded. When a track's centroid history shows it moved
from one side of the counting line to the other, we register a single
in/out crossing event for that track (and never count the same crossing
twice).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def _side(point, line_a, line_b):
    """Return >0, <0, or 0 depending on which side of the line A->B `point` is on."""
    ax, ay = line_a
    bx, by = line_b
    px, py = point
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


@dataclass
class TrackState:
    last_side: int | None = None
    class_id: int | None = None
    crossed: bool = False  # guards against double-counting the same track


@dataclass
class LineCounter:
    line_a: tuple[float, float]
    line_b: tuple[float, float]
    out_direction: str  # "a_to_b" or "b_to_a"
    class_names: dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        self._tracks: dict[int, TrackState] = defaultdict(TrackState)
        self.count_in = 0
        self.count_out = 0
        self.count_in_by_class: dict[str, int] = defaultdict(int)
        self.count_out_by_class: dict[str, int] = defaultdict(int)

    def update(self, track_id: int, centroid: tuple[float, float], class_id: int | None):
        state = self._tracks[track_id]
        side = _side(centroid, self.line_a, self.line_b)
        if side == 0:
            return  # exactly on the line, ambiguous, wait for next frame

        if state.last_side is None:
            state.last_side = side
            state.class_id = class_id
            return

        if side != state.last_side and not state.crossed:
            # crossing happened between previous frame and this one
            transition = "a_to_b" if state.last_side > 0 and side < 0 else "b_to_a"
            # NOTE: sign convention depends on point order; calibrate.py prints
            # a sample crossing so you can confirm a_to_b/b_to_a matches your
            # config's out_direction.
            direction_is_out = transition == self.out_direction
            class_name = self.class_names.get(class_id, str(class_id))

            if direction_is_out:
                self.count_out += 1
                self.count_out_by_class[class_name] += 1
            else:
                self.count_in += 1
                self.count_in_by_class[class_name] += 1

            state.crossed = True  # one crossing event per track lifetime

        state.last_side = side

    def reset(self):
        self.count_in = 0
        self.count_out = 0
        self.count_in_by_class.clear()
        self.count_out_by_class.clear()
        # NOTE: we intentionally do not clear self._tracks here, so tracks that
        # already crossed don't get double-counted if they briefly re-enter
        # frame within the same tracker session. Stale entries are harmless
        # (tracker assigns new IDs to new tracks); call drop_track() on the
        # tracker's "lost" callback if memory growth ever matters for very
        # long unattended runs.

    def drop_track(self, track_id: int):
        self._tracks.pop(track_id, None)
