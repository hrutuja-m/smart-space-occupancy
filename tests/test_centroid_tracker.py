import numpy as np

from smart_space.tracking.centroid_tracker import CentroidTracker


def _box(cx, cy, s=10):
    return [cx - s, cy - s, cx + s, cy + s]


def test_stable_ids_across_small_motion():
    t = CentroidTracker(max_distance=30)
    ids1 = list(t.update(np.array([_box(50, 50), _box(200, 200)])))
    ids2 = list(t.update(np.array([_box(55, 52), _box(205, 198)])))
    assert ids1 == ids2 == [0, 1]


def test_new_object_registered():
    t = CentroidTracker(max_distance=30)
    t.update(np.array([_box(50, 50)]))
    objs = t.update(np.array([_box(52, 51), _box(400, 400)]))
    assert set(objs) == {0, 1}


def test_disappeared_grace_then_deregister():
    t = CentroidTracker(max_distance=30, max_disappeared=2)
    t.update(np.array([_box(50, 50)]))
    t.update(np.zeros((0, 4)))          # miss 1
    assert 0 in t.objects
    t.update(np.zeros((0, 4)))          # miss 2
    t.update(np.zeros((0, 4)))          # miss 3 -> exceeds grace
    assert 0 not in t.objects


def test_reid_after_brief_miss_keeps_id():
    t = CentroidTracker(max_distance=40, max_disappeared=5)
    t.update(np.array([_box(50, 50)]))
    t.update(np.zeros((0, 4)))
    objs = t.update(np.array([_box(58, 55)]))
    assert list(objs) == [0]
