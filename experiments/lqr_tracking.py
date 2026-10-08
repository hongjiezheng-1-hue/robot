"""LQR tracking in MuJoCo with the EKF in the loop, repeated over varied conditions. Run from the repository root:
    python experiments/lqr_tracking.py --model-dir <robotis_tb3 folder> --variant table --trials 4 [--out-dir tracking]

Each trial starts the true robot at a random offset (about 5 cm and 3 degrees, one standard deviation) from the nominal start mark,
while the robot believes it is at the nominal mark, and uses a different speed from --speeds in turn. The controller acts on the
estimated pose only. Provisional docking tolerances (to be confirmed with the course staff): lateral error below 3 cm,
along-track error below 5 cm, heading error below 5 degrees.
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import mujoco  # noqa: E402
from control.lqr_tracker import LQRTracker  # noqa: E402
from control.reference import build_reference  # noqa: E402
from drivers.mujoco_tb3 import MujocoTB3Driver  # noqa: E402
from sim.closed_loop import run_closed_loop, tracking_metrics  # noqa: E402
from sim.mujoco_run import RunConfig, calibrate_gain_table, calibrate_gains, min_clearance, pick_texture_type  # noqa: E402
from sim.scene import SceneConfig, write_scene  # noqa: E402

# a loop around the room that ends with a head-on approach to the charging station
START = (-1.4, 1.15)
WAYPOINTS = [(1.3, 0.6), (1.3, -1.15), (-1.3, -1.15), (-1.45, -0.08), (1.30, -0.08)]
DOCK_POSE = (1.30, -0.08, 0.0)
DOCK_TOLERANCE = dict(lateral=0.03, along=0.05, heading_deg=5.0)


def within_tolerance(m):
    return (abs(m["dock_lateral"]) < DOCK_TOLERANCE["lateral"] and abs(m["dock_along"]) < DOCK_TOLERANCE["along"]
            and abs(m["dock_heading_deg"]) < DOCK_TOLERANCE["heading_deg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.05, 0.08, 0.12])
    ap.add_argument("--fillet", type=float, default=0.3)
    ap.add_argument("--variant", choices=("table", "single", "B", "C"), default="table",
                    help="odometry model in the filter and for the command compensation")
    ap.add_argument("--no-compensation", action="store_true", help="do not invert the odometry gains in the command")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out-dir", default="tracking")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = args.variant + ("_nocomp" if args.no_compensation else "")

    print("MuJoCo", mujoco.__version__, "| variant", label)
    scene_cfg = pick_texture_type(SceneConfig(), args.model_dir)
    drv = MujocoTB3Driver(write_scene(scene_cfg, args.model_dir, scene_name=f"scene_{label}.xml"))
    base = RunConfig()
    single = calibrate_gains(drv, base)
    calib_cfg = SceneConfig(room_x=8.0, room_y=8.0, obstacles=(), station_x=3.5, marker_texture_type=scene_cfg.marker_texture_type)
    gain_fn, _ = calibrate_gain_table(MujocoTB3Driver(write_scene(calib_cfg, args.model_dir, scene_name=f"scene_calib_{label}.xml")), base)
    models = {"table": dict(gain_fn=gain_fn), "single": dict(gain_fn=lambda v, w: single),
              "B": dict(gain_fn=None, estimate_scales=True), "C": dict(gain_fn=gain_fn, estimate_scales=True)}[args.variant]

    rows = []
    for trial in range(args.trials):
        speed = args.speeds[trial % len(args.speeds)]
        ref = build_reference(START, WAYPOINTS, speed, args.fillet, 0.1)
        clearance = min_clearance(np.stack([ref["x"], ref["y"], ref["theta"]], axis=1), scene_cfg)
        if clearance < 0.30:
            raise SystemExit("the reference passes too close to an obstacle")
        rng = np.random.default_rng(100 + trial)
        nominal = np.array([ref["x"][0], ref["y"][0], ref["theta"][0]])
        start = nominal + rng.normal(0.0, [0.05, 0.05, np.radians(3.0)])
        log = run_closed_loop(drv, scene_cfg, ref, LQRTracker(0.1), start_pose=start, belief0=nominal,
                              compensate=not args.no_compensation, seed=trial, **models)
        np.savez(out / f"closed_loop_{label}_trial{trial}.npz", **log)
        m = tracking_metrics(log, DOCK_POSE)
        m.update(speed=speed, ok=within_tolerance(m), marker=float(np.mean(np.all(np.isfinite(log["marker"]), axis=1))))
        rows.append(m)
        print(f"trial {trial} speed {speed:.2f}: tracking RMSE {1000 * m['time_rmse']:.0f} mm (max {1000 * m['time_max']:.0f}), "
              f"path RMSE {1000 * m['path_rmse']:.0f} mm, estimator {1000 * m['est_pos_rmse']:.0f} mm / {m['est_head_rmse']:.1f} deg, "
              f"dock along {1000 * m['dock_along']:.0f} mm lateral {1000 * m['dock_lateral']:.0f} mm heading {m['dock_heading_deg']:.1f} deg "
              f"-> {'OK' if m['ok'] else 'outside'}", flush=True)

    a = lambda key: np.array([r[key] for r in rows])
    print(f"\nvariant {label}, {len(rows)} trials: docking within tolerance {int(a('ok').sum())} / {len(rows)}")
    print(f"tracking RMSE {1000 * a('time_rmse').mean():.0f} mm (worst {1000 * a('time_rmse').max():.0f}); "
          f"path RMSE {1000 * a('path_rmse').mean():.0f} mm; estimator position RMSE {1000 * a('est_pos_rmse').mean():.0f} mm; "
          f"|dock lateral| {1000 * np.abs(a('dock_lateral')).mean():.0f} mm, |dock along| {1000 * np.abs(a('dock_along')).mean():.0f} mm, "
          f"|dock heading| {np.abs(a('dock_heading_deg')).mean():.1f} deg")


if __name__ == "__main__":
    main()
