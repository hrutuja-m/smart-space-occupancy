import numpy as np

from smart_space.data.density import density_map, head_points, read_yolo_label


def test_density_map_conserves_count(tmp_path):
    boxes = np.array([[0.5, 0.5, 0.1, 0.2], [0.2, 0.3, 0.1, 0.2], [0.8, 0.7, 0.1, 0.2]])
    dmap = density_map(boxes, (64, 64), sigma=2.0)
    assert abs(dmap.sum() - 3.0) < 1e-2
    assert (dmap >= 0).all()


def test_empty_label_zero_map():
    dmap = density_map(np.zeros((0, 4)), (32, 32), sigma=2.0)
    assert dmap.sum() == 0


def test_head_points_within_bounds():
    boxes = np.array([[0.99, 0.01, 0.4, 0.4]])
    pts = head_points(boxes, 100, 100)
    assert (pts[:, 0] < 100).all() and (pts[:, 1] >= 0).all()


def test_read_yolo_label_parsing(tmp_path):
    p = tmp_path / "000000.txt"
    p.write_text("0 0.5 0.5 0.1 0.2\n0 0.2 0.2 0.05 0.1\n")
    boxes = read_yolo_label(p)
    assert boxes.shape == (2, 4)
