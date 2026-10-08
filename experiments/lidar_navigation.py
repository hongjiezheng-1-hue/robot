"""Navigate an unknown S-shaped obstacle course with LiDAR, then dock at the marker (MuJoCo, EKF and LQR in the loop).

    python experiments/lidar_navigation.py --model-dir <robotis_tb3 folder> --variant C --trials 3 [--out-dir navigation]

The three boards are not given to the robot. Each trial starts the true robot at a random offset (about 5 cm and 3 degrees, one
standard deviation) from the nominal start mark while the robot believes it is at the nominal mark. Provisional docking
tolerances (to be confirmed with the course staff): lateral below 3 cm, along-track below 5 cm, heading below 5 degrees.
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
from control.docking import DockSpec, dock_poses  # noqa: E402
from control.lqr_tracker import LQRTracker  # noqa: E402
from drivers.mujoco_tb3 import MujocoTB3Driver  # noqa: E402
from sim.mujoco_run import RunConfig, calibrate_gain_table, calibrate_gains, pick_texture_type  # noqa: E402
from sim.navigation import navigation_metrics, run_navigation  # noqa: E402
from sim.scene import SceneConfig, write_scene  # noqa: E402

# three thin boards, staggered so that the route weaves up, down and up again (x, y, size_x, size_y)
S_COURSE = ((-1.5, 0.65, 0.10, 1.70), (-0.5, -0.65, 0.10, 1.70), (0.5, 0.65, 0.10, 1.70))
START = (-1.65, -0.6, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--variant", choices=("table", "B", "C"), default="C")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--speed", type=float, default=0.10)
    ap.add_argument("--seed-base", type=int, default=200, help="start offsets use seeds seed_base + trial")
    ap.add_argument("--out-dir", default="navigation")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = f"s_{args.variant}_seed{args.seed_base}"

    print("MuJoCo", mujoco.__version__, "| variant", args.variant, flush=True)
    scene_cfg = pick_texture_type(SceneConfig(obstacles=S_COURSE), args.model_dir)
    drv = MujocoTB3Driver(write_scene(scene_cfg, args.model_dir, scene_name=f"scene_{label}.xml"))
    base = RunConfig()
    calib_cfg = SceneConfig(room_x=8.0, room_y=8.0, obstacles=(), station_x=3.5, marker_texture_type=scene_cfg.marker_texture_type)
    gain_fn, _ = calibrate_gain_table(MujocoTB3Driver(write_scene(calib_cfg, args.model_dir, scene_name=f"scene_calib_{label}.xml")), base)
    models = {"table": dict(gain_fn=gain_fn), "B": dict(gain_fn=None, estimate_scales=True),
              "C": dict(gain_fn=gain_fn, estimate_scales=True)}[args.variant]

    spec = DockSpec()
    standoff, dock = dock_poses((scene_cfg.station_x, scene_cfg.station_y, scene_cfg.station_yaw), spec)
    print(f"stand-off pose {np.round(standoff, 3)}, dock pose {np.round(dock, 3)}", flush=True)

    rows = []
    for trial in range(args.trials):
        rng = np.random.default_rng(args.seed_base + trial)
        nominal = np.array(START)
        start = nominal + rng.normal(0.0, [0.05, 0.05, np.radians(3.0)])
        log = run_navigation(drv, scene_cfg, standoff, dock, LQRTracker(0.1), start, nominal, speed=args.speed, seed=trial, **models)
        np.savez(out / f"navigation_{label}_trial{trial}.npz", **log)
        m = navigation_metrics(log, scene_cfg, spec, dock)
        rows.append(m)
        print(f"trial {trial}: {'FAILED to plan' if m['failed'] else ''} {m['seconds']:.0f} s, {m['replans']} plans, "
              f"{'COLLISION' if m['collided'] else 'no contact'} ({m['contact_steps']} steps), min clearance {1000 * m['clearance']:.0f} mm, "
              f"estimator {1000 * m['est_pos_rmse']:.0f} mm (max {1000 * m['est_pos_max']:.0f}), dock along {1000 * m['dock_along']:.0f} mm "
              f"lateral {1000 * m['dock_lateral']:.0f} mm heading {m['dock_heading_deg']:.1f} deg -> {'DOCKED' if m['docked'] else 'not docked'}",
              flush=True)

    print(f"\nvariant {args.variant}, {len(rows)} trials: docked {sum(r['docked'] for r in rows)} / {len(rows)}, "
          f"collisions {sum(r['collided'] for r in rows)}, planning failures {sum(r['failed'] for r in rows)}")


if __name__ == "__main__":
    main()
