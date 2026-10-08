"""Checks for the pure-numpy parts of sim/mujoco_run.py. Run: python tests/test_mujoco_run_helpers.py

collect_run and calibrate_gains need MuJoCo and are validated in notebooks/05_ekf_on_mujoco.ipynb.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.mujoco_run import RunConfig, _box_distance, marker_possibly_visible, min_clearance, simulate_follower_path  # noqa: E402
from sim.scene import SceneConfig  # noqa: E402

ROBOT_REACH = 0.25   # [m] conservative distance from the axle centre to the robot's outline (rear casters reach about 0.19 m)


def test_box_distance():
    assert _box_distance(0.0, 0.0, 0.0, 0.0, 1.0, 1.0) == 0.0
    assert abs(_box_distance(2.0, 0.0, 0.0, 0.0, 1.0, 1.0) - 1.5) < 1e-12
    assert abs(_box_distance(1.0, 1.0, 0.0, 0.0, 1.0, 1.0) - np.hypot(0.5, 0.5)) < 1e-12


def test_followed_path_keeps_clear_of_obstacles():
    """The previous planned-polyline check missed corner cutting and slow turns, and the robot hit an obstacle."""
    scene, cfg = SceneConfig(), RunConfig()
    for s_v, s_w in ((0.86, 0.72), (1.0, 1.0)):
        path = simulate_follower_path(cfg, s_v, s_w, duration=600.0)
        clearance = min_clearance(path, scene)
        assert clearance > ROBOT_REACH, f"followed path comes within {clearance:.2f} m of an obstacle (gains {s_v}, {s_w})"


def test_validation_paths_keep_clear_and_see_the_marker():
    from sim.mujoco_run import VALIDATION_PATHS
    scene = SceneConfig()
    marker = np.array([scene.station_x, scene.station_y, scene.station_yaw])
    for name, spec in VALIDATION_PATHS.items():
        cfg = RunConfig(duration=300.0, **spec)
        assert len(cfg.waypoints) == len(cfg.speeds), name
        for s_v, s_w in ((0.86, 0.72), (1.0, 1.0)):
            path = simulate_follower_path(cfg, s_v, s_w)
            clearance = min_clearance(path, scene)
            assert clearance > ROBOT_REACH, f"{name}: path comes within {clearance:.2f} m of an obstacle (gains {s_v}, {s_w})"
        visible = np.mean([marker_possibly_visible(p, marker) for p in simulate_follower_path(cfg, 0.86, 0.72)])
        assert visible > 0.12, f"{name}: the marker could be in view for only {100 * visible:.0f}% of the path"


def test_marker_faces_the_lane():
    cfg = RunConfig()
    path = simulate_follower_path(cfg, duration=600.0)
    marker = np.array([1.85, 0.0, np.pi])
    visible = np.mean([marker_possibly_visible(p, marker) for p in path])
    assert visible > 0.15, f"marker could be in view for only {100 * visible:.0f}% of the path"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
