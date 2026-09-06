"""Centroid tracker: stable IDs for detections across video frames.

Greedy nearest-centroid association with a "disappeared" grace period so a
person who is briefly missed (occlusion, a dropped detection) keeps their ID
instead of being counted as a new arrival.

This is deliberately dependency-light (numpy + scipy only). For production video
you would swap in ByteTrack / OC-SORT; the public API here matches closely
enough to make that drop-in.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_distance: float = 60.0, max_disappeared: int = 30):
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    # --- internal helpers ------------------------------------------------
    def _register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    @staticmethod
    def _centroids(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0, 2))
        boxes = np.asarray(boxes, dtype=float)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        return np.stack([cx, cy], axis=1)

    # --- public API ----------------------------------------------------
    def update(self, boxes: np.ndarray) -> OrderedDict[int, np.ndarray]:
        """boxes: (N, 4) xyxy. Returns {object_id: centroid} after this frame."""
        if len(boxes) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        input_centroids = self._centroids(boxes)

        if not self.objects:
            for c in input_centroids:
                self._register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        D = dist.cdist(object_centroids, input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols or D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        for col in set(range(D.shape[1])) - used_cols:
            self._register(input_centroids[col])

        return self.objects
