"""Checks for the reactive LiDAR avoidance and the PBVS docking law. Run: python tests/test_simple_control.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from control.pbvs import PBVSGains, pbvs_command, polar_error  # noqa: E402
from control.reactive import choose_heading, free_mask, front_clearance  # noqa: E402
from estimation.models import f_unicycle, wrap  # noqa: E402
from perception.lidar import LidarConfig, beam_angles, scan_boxes  # noqa: E402

CFG = LidarConfig()


def _points(pose, boxes):
    r = scan_boxes(pose, boxes, CFG, noise=False)
    ang = beam_angles(CFG)
    ok = np.isfinite(r)
    return np.stack([CFG.mount[0] + r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok])], axis=1)


def test_pbvs_gains_meet_the_stability_condition():
    assert PBVSGains().stable()


def test_pbvs_reaches_the_goal_pose_from_several_starts():
    gains = PBVSGains()
    goal = (1.0, 0.0, 0.0)
    for start in [(0.0, 0.0, 0.0), (0.0, 0.3, 0.2), (0.2, -0.3, -0.4), (0.0, 0.0, 1.2)]:
        x = np.array(start, dtype=float)
        for _ in range(1500):
            x = f_unicycle(x, pbvs_command(x, goal, gains), 0.05)
        rho, _, _ = polar_error(x, goal)
        assert rho < 0.02 and abs(wrap(x[2] - goal[2])) < np.radians(3.0), (start, x)


def test_goal_direction_kept_when_free_and_changed_when_blocked():
    pts = _points((0.0, 0.0, 0.0), [(5.0, 5.0, 0.1, 0.1)])        # nothing in the way
    assert choose_heading(pts, 0.0) == 0.0
    pts = _points((0.0, 0.0, 0.0), [(0.7, 0.0, 0.2, 0.6)])        # a board straight ahead
    h = choose_heading(pts, 0.0)
    assert h is not None and abs(h) > np.radians(20.0), "turns away from the board"
    assert free_mask(pts, np.array([h]), 0.22, 1.0)[0]


def test_front_clearance():
    pts = _points((0.0, 0.0, 0.0), [(0.7, 0.0, 0.2, 0.6)])
    assert abs(front_clearance(pts) - 0.6) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
