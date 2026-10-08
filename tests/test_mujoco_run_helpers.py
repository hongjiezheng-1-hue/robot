"""Checks for the pure-numpy parts of sim/mujoco_run.py. Run: python tests/test_mujoco_run_helpers.py

collect_run and calibrate_gains need MuJoCo and are validated in notebooks/05_ekf_on_mujoco.ipynb.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.mujoco_run import RunConfig, _box_distance, min_clearance  # noqa: E402
from sim.scene import SceneConfig  # noqa: E402


def test_box_distance():
    assert _box_distance(0.0, 0.0, 0.0, 0.0, 1.0, 1.0) == 0.0
    assert abs(_box_distance(2.0, 0.0, 0.0, 0.0, 1.0, 1.0) - 1.5) < 1e-12
    assert abs(_box_distance(1.0, 1.0, 0.0, 0.0, 1.0, 1.0) - np.hypot(0.5, 0.5)) < 1e-12


def test_default_waypoints_keep_clear_of_obstacles():
    scene, cfg = SceneConfig(), RunConfig()
    pts = [cfg.start[:2]] + list(cfg.waypoints)
    path = []
    for a, b in zip(pts, pts[1:] + pts[:1]):
        for t in np.linspace(0, 1, 60):
            path.append([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0.0])
    clearance = min_clearance(np.array(path), scene)
    assert clearance > 0.20, f"planned path clearance {clearance:.2f} m is too small for the robot footprint"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
