"""Checks for the pure helpers behind the demonstration video (no MuJoCo needed). Run: python tests/test_demo_video_helpers.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from perception.lidar import LidarConfig, scan_boxes  # noqa: E402
from sim.course import S_COURSE, START  # noqa: E402


def test_course_start_is_clear_of_the_boards_and_walls():
    x, y, _ = START
    assert x - 0.21 > -2.0, "the rear of the robot (about 0.2 m behind the origin) stays off the west wall with margin for the start offset"
    for cx, cy, sx, sy in S_COURSE:
        gap = np.hypot(max(abs(x - cx) - sx / 2, 0.0), max(abs(y - cy) - sy / 2, 0.0))
        assert gap > 0.3, "the start is not next to a board"


def test_start_scan_sees_the_first_board():
    ranges = scan_boxes(START, list(S_COURSE), LidarConfig(), noise=False)
    assert np.isfinite(ranges).sum() > 10, "the first board is within the LiDAR range from the start"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
