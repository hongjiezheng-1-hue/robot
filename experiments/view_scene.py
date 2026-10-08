"""Interactive MuJoCo viewer of the Track A scene (room, obstacles, ArUco station, TurtleBot3).

Run from the repository root:
    python experiments/view_scene.py --model-dir <robotis_tb3 folder> [--pose 0.5 0.2 0.0]

Left mouse drag rotates, right drag pans, scroll zooms; double-click selects a body; the "Camera" panel switches to the
robot camera ("front"). The robot is placed at the given pose and the physics runs, so it can be pushed with Ctrl + right drag.
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
from sim.mujoco_run import pick_texture_type  # noqa: E402
from sim.scene import SceneConfig, write_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="folder robotis_tb3 of the ROBOTIS MuJoCo model")
    ap.add_argument("--pose", nargs=3, type=float, default=(-1.5, -0.08, 0.0), metavar=("X", "Y", "YAW"))
    args = ap.parse_args()
    cfg = pick_texture_type(SceneConfig(), args.model_dir)
    model = mujoco.MjModel.from_xml_path(write_scene(cfg, args.model_dir))
    data = mujoco.MjData(model)
    x, y, yaw = args.pose
    data.qpos[0], data.qpos[1] = x, y
    data.qpos[3], data.qpos[6] = np.cos(yaw / 2), np.sin(yaw / 2)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
