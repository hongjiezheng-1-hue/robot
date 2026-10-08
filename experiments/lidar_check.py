"""Compare the MuJoCo ray-cast LiDAR with the analytic box scan on the S course, noise off.

    python experiments/lidar_check.py --model-dir <robotis_tb3 folder>
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from drivers.mujoco_tb3 import MujocoTB3Driver  # noqa: E402
from lidar_navigation import S_COURSE  # noqa: E402
from perception.lidar import LidarConfig, scan_boxes, scan_mujoco  # noqa: E402
from sim.mujoco_run import pick_texture_type  # noqa: E402
from sim.scene import SceneConfig, write_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    cfg = pick_texture_type(SceneConfig(obstacles=S_COURSE), args.model_dir)
    drv = MujocoTB3Driver(write_scene(cfg, args.model_dir, scene_name="scene_lidar_check.xml"))
    lidar = LidarConfig(sigma_floor=0.0, sigma_rel=0.0)
    hx, hy = cfg.room_x / 2, cfg.room_y / 2
    walls = [(0, hy + 0.025, 2 * hx + 0.1, 0.05), (0, -hy - 0.025, 2 * hx + 0.1, 0.05), (hx + 0.025, 0, 0.05, 2 * hy), (-hx - 0.025, 0, 0.05, 2 * hy)]
    boxes = list(S_COURSE) + walls + [(cfg.station_x, 0.0, 0.16, 0.28)]
    rng = np.random.default_rng(0)
    base = drv.model.body("base").id
    for pose in [(-1.75, -0.6, 0.0), (-1.0, 0.0, 1.0), (0.0, 1.0, -2.0), (1.0, -0.3, 3.0)]:
        drv.reset(*pose)
        a = scan_mujoco(drv.model, drv.data, lidar, base, pose, rng)
        b = scan_boxes(pose, boxes, lidar, noise=False)
        both = np.isfinite(a) & np.isfinite(b)
        print(f"pose {pose}: mujoco hits {np.isfinite(a).sum()}, analytic hits {np.isfinite(b).sum()}, "
              f"hit mismatch {np.sum(np.isfinite(a) != np.isfinite(b))}, median |diff| {1000 * np.median(np.abs(a[both] - b[both])):.1f} mm, "
              f"max |diff| {1000 * np.max(np.abs(a[both] - b[both])):.1f} mm")


if __name__ == "__main__":
    main()
