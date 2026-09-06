"""Directional in/out counting across a virtual doorway line.

Given tracked centroids over time, decide when a track crosses an oriented
segment and in which direction. Net occupancy of the space = (entries - exits)
plus a known starting count. This is the piece that turns per-frame detection
into an operational "how many people are in the room right now" signal.
"""

from __future__ import annotations

import numpy as np


def _side(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Signed area sign of triangle (a, b, p): >0 left of a->b, <0 right."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class LineCounter:
    def __init__(self, line: tuple[tuple[float, float], tuple[float, float]], start_count: int = 0):
        self.a = np.asarray(line[0], dtype=float)
        self.b = np.asarray(line[1], dtype=float)
        self.entries = 0
        self.exits = 0
        self.start_count = start_count
        self._last_side: dict[int, float] = {}

    def update(self, tracked: dict[int, np.ndarray]) -> None:
        for oid, centroid in tracked.items():
            s = _side(np.asarray(centroid, dtype=float), self.a, self.b)
            prev = self._last_side.get(oid)
            self._last_side[oid] = s
            if prev is None or prev == 0 or s == 0:
                continue
            if prev < 0 < s:          # crossed left -> right == "in"
                self.entries += 1
            elif prev > 0 > s:        # right -> left == "out"
                self.exits += 1

    @property
    def occupancy(self) -> int:
        return max(0, self.start_count + self.entries - self.exits)
