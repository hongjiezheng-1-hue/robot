"""Checks for the LiDAR model, the occupancy grid and the grid planner (numpy only). Run: python tests/test_lidar_planning.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from control.planner import path_blocked, plan, plan_with_margins  # noqa: E402
from perception.lidar import LidarConfig, beam_angles, scan_boxes, sensor_origin  # noqa: E402
from perception.occupancy import OccupancyGrid  # noqa: E402

CFG = LidarConfig()


def test_scan_range_to_a_wall_box():
    pose = (0.0, 0.0, 0.0)
    ranges = scan_boxes(pose, [(1.5, 0.0, 0.2, 2.0)], CFG, noise=False)
    ox, _ = sensor_origin(pose, CFG)
    ahead = ranges[np.argmin(np.abs(beam_angles(CFG)))]
    assert np.isclose(ahead, 1.4 - ox, atol=1e-6), "beam straight ahead hits the near face of the box"
    behind = ranges[np.argmin(np.abs(np.abs(beam_angles(CFG)) - np.pi))]
    assert np.isnan(behind), "no return from an empty direction"


def test_scan_limits():
    far = scan_boxes((0.0, 0.0, 0.0), [(5.0, 0.0, 0.2, 1.0)], CFG, noise=False)
    assert np.all(np.isnan(far)), "nothing beyond the maximum range"
    near = scan_boxes((0.0, 0.0, 0.0), [(0.15, 0.0, 0.1, 0.1)], CFG, noise=False)
    assert np.all(np.isnan(near[np.isfinite(near) & (near < CFG.range_min)])), "no value below the minimum range"


def test_scan_heading_rotates_the_scan():
    box = [(0.0, 1.5, 2.0, 0.2)]
    a = scan_boxes((0.0, 0.0, np.pi / 2), box, CFG, noise=False)
    ahead = a[np.argmin(np.abs(beam_angles(CFG)))]
    assert np.isfinite(ahead) and 1.3 < ahead < 1.5, "facing +y the box is straight ahead"


def test_grid_learns_a_box_from_the_pose_estimate():
    grid = OccupancyGrid()
    box = [(1.0, 0.5, 0.3, 0.3)]
    rng = np.random.default_rng(0)
    for pose in [(0.0, 0.0, 0.0), (0.2, -0.2, 0.3), (0.0, 0.3, -0.2)]:
        for _ in range(3):
            grid.update(pose, scan_boxes(pose, box, CFG, rng), CFG)
    ix, iy = grid.to_index(1.0, 0.5)
    ox, oy = grid.to_index(0.0, 0.0)
    occ = grid.occupied()
    hits = occ[ix - 3:ix + 4, iy - 3:iy + 4]
    assert hits.sum() >= 4, "cells at the box are occupied"
    assert not occ[ox, oy] and grid.logodds[ox, oy] < 0, "the robot position is observed free"
    far = grid.occupied()[int(grid.shape[0] * 0.9):, :]
    assert not far.any(), "no false obstacles far away"


def test_planner_goes_around_a_box_and_keeps_clearance():
    grid = OccupancyGrid()
    box = [(0.0, 0.0, 0.3, 1.0)]
    rng = np.random.default_rng(1)
    # seen from the front and from the long sides: unobserved cells are planned as free, so the sides must be observed first
    for pose in [(-1.0, 0.0, 0.0), (-1.0, 0.4, 0.2), (-1.0, -0.4, -0.2), (0.5, -1.0, np.pi / 2), (0.5, 1.0, -np.pi / 2), (-0.5, -0.9, 0.9)]:
        grid.update(pose, scan_boxes(pose, box, CFG, rng), CFG)
    wp = plan(grid, (-1.0, 0.0), (1.0, 0.0), inflate_radius=0.25)
    assert wp is not None and np.allclose(wp[-1], (1.0, 0.0), atol=grid.res)
    pts = np.array([(-1.0, 0.0)] + wp)
    dense = np.concatenate([np.linspace(a, b, 40) for a, b in zip(pts[:-1], pts[1:])])
    gap = np.hypot(np.maximum(np.abs(dense[:, 0]) - 0.15, 0.0), np.maximum(np.abs(dense[:, 1]) - 0.5, 0.0))
    assert gap.min() > 0.20, "path stays at least 0.2 m from the box"
    assert abs(dense[:, 1]).max() > 0.5, "path detours around the end of the box"


def test_blocked_path_is_detected_and_unknown_cells_are_free():
    grid = OccupancyGrid()
    assert plan(grid, (-1.0, 0.0), (1.0, 0.0), 0.25) is not None
    assert len(plan(grid, (-1.0, 0.0), (1.0, 0.0), 0.25)) == 1, "straight line through unknown space"
    rng = np.random.default_rng(2)
    pose = (-1.0, 0.0, 0.0)
    for _ in range(3):
        grid.update(pose, scan_boxes(pose, [(0.0, 0.0, 0.3, 1.0)], CFG, rng), CFG)
    assert path_blocked(grid, [(-1.0, 0.0), (1.0, 0.0)], 0.25), "old straight path is now blocked"
    assert not path_blocked(grid, [(-1.0, 1.4), (1.0, 1.4)], 0.25)


def test_narrow_gap_is_planned_with_a_smaller_margin():
    grid = OccupancyGrid()
    wall = grid.to_index(0.0, np.linspace(-1.8, 1.8, 200))
    grid.logodds[wall] = 4.0
    gap = grid.to_index(0.0, np.linspace(-0.2, 0.2, 30))      # a 0.4 m opening: passable at margin 0.17, closed at 0.22
    grid.logodds[gap] = -4.0
    assert plan(grid, (-1.0, 0.0), (1.0, 0.0), 0.22) is None
    wp = plan_with_margins(grid, (-1.0, 0.0), (1.0, 0.0))
    assert wp is not None and np.allclose(wp[-1], (1.0, 0.0), atol=grid.res)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
