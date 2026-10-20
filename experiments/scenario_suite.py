"""Run scenarios and disturbances from sim/scenarios.py through the full navigation and docking pipeline.

    python experiments/scenario_suite.py --model-dir <robotis_tb3 folder> --variant C --scenarios s_mirrored gate \
        --disturbances none --trials 6 --seed-base 500 [--out results/scenarios/x.jsonl]

One line per run is printed and appended to --out as JSON. Outcome per run: docked in tolerance (and no contact), collision,
planning failure (no path), or not docked. The planner and estimator are not changed between runs.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
from control.docking import DockSpec, dock_poses  # noqa: E402
from control.lqr_tracker import LQRTracker  # noqa: E402
from drivers.mujoco_tb3 import MujocoTB3Driver  # noqa: E402
from perception.lidar import LidarConfig  # noqa: E402
from sim.mujoco_run import RunConfig, calibrate_gain_table, pick_texture_type  # noqa: E402
from sim.navigation import navigation_metrics, run_navigation, slam_prior_from_error  # noqa: E402
from sim.scenarios import DISTURBANCES, SCENARIOS  # noqa: E402
from sim.scene import SceneConfig, write_scene  # noqa: E402


def outcome(m):
    if m["failed"]:
        return "no_path"
    if m["collided"]:
        return "collision"
    return "docked" if m["docked"] else "not_docked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--variant", choices=("table", "B", "C"), default="C")
    ap.add_argument("--scenarios", nargs="+", default=["s_course"], choices=list(SCENARIOS))
    ap.add_argument("--disturbances", nargs="+", default=["none"], choices=list(DISTURBANCES))
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=500)
    ap.add_argument("--speed", type=float, default=0.10)
    ap.add_argument("--marker-unknown", type=float, default=None, help="marker guess error [m]; estimate the marker (EKF-SLAM)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    print("MuJoCo", mujoco.__version__, "| variant", args.variant, flush=True)

    tag = f"{args.variant}_{abs(hash(tuple(args.scenarios) + tuple(args.disturbances))) % 10000}"
    first = SCENARIOS[args.scenarios[0]]
    base_cfg = pick_texture_type(SceneConfig(obstacles=first.obstacles), args.model_dir)
    if args.variant == "B":
        models = dict(gain_fn=None, estimate_scales=True)
    else:
        calib_cfg = SceneConfig(room_x=8.0, room_y=8.0, obstacles=(), station_x=3.5, marker_texture_type=base_cfg.marker_texture_type)
        gain_fn, _ = calibrate_gain_table(MujocoTB3Driver(write_scene(calib_cfg, args.model_dir, scene_name=f"scene_suite_calib_{tag}.xml")), RunConfig())
        models = dict(gain_fn=gain_fn, estimate_scales=args.variant == "C")
    out = pathlib.Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)

    spec, rows = DockSpec(), []
    for sname in args.scenarios:
        scen = SCENARIOS[sname]
        sx, sy, syaw = scen.station
        scene_cfg = SceneConfig(obstacles=scen.obstacles, station_x=sx, station_y=sy, station_yaw=syaw,
                                marker_texture_type=base_cfg.marker_texture_type)
        drv = MujocoTB3Driver(write_scene(scene_cfg, args.model_dir, scene_name=f"scene_suite_{sname}_{tag}.xml"))
        standoff, dock = dock_poses(scen.station, spec)
        for dname in args.disturbances:
            d = DISTURBANCES[dname]
            for trial in range(args.trials):
                rng = np.random.default_rng(args.seed_base + trial)
                nominal = np.array(scen.start)
                start = nominal + d.start_offset_scale * rng.normal(0.0, [0.05, 0.05, np.radians(3.0)])
                lidar = LidarConfig(sigma_floor=0.01 * d.lidar_noise_scale, sigma_rel=0.01 * d.lidar_noise_scale)
                log = run_navigation(drv, scene_cfg, standoff, dock, LQRTracker(0.1), start, nominal, speed=args.speed, seed=trial,
                                     lidar_cfg=lidar, wheel_radius=0.033 * (1 + d.wheel_radius_error), marker_dropout=d.marker_dropout,
                                     push=d.push or None, dock_spec=spec,
                                     slam_prior=None if args.marker_unknown is None else slam_prior_from_error(scen.station, args.marker_unknown),
                                     **models)
                m = navigation_metrics(log, scene_cfg, spec, dock)
                row = dict(scenario=sname, disturbance=dname, variant=args.variant, marker_unknown=args.marker_unknown, seed=args.seed_base + trial, outcome=outcome(m),
                           seconds=m["seconds"], replans=m["replans"], recoveries=m["recoveries"], first_contact_s=m["first_contact_s"], collided=m["collided"], clearance=None if not np.isfinite(m["clearance"]) else m["clearance"],
                           est_rmse=None if not np.isfinite(m["est_pos_rmse"]) else m["est_pos_rmse"],
                           along=None if not np.isfinite(m["dock_along"]) else m["dock_along"],
                           lateral=None if not np.isfinite(m["dock_lateral"]) else m["dock_lateral"],
                           heading=None if not np.isfinite(m["dock_heading_deg"]) else m["dock_heading_deg"])
                rows.append(row)
                if out:
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                mm = lambda v: "-" if v is None else f"{1000 * v:.0f} mm"
                dock_txt = "-" if row["along"] is None else f"{1000 * row['along']:.0f}/{1000 * row['lateral']:.0f} mm {row['heading']:.1f} deg"
                print(f"{sname:14s} {dname:10s} seed {row['seed']}: {row['outcome']:10s} {row['seconds']:.0f} s, {row['replans']} plans, {row['recoveries']} recoveries, "
                      f"clearance {mm(row['clearance'])}, est RMSE {mm(row['est_rmse'])}, dock {dock_txt}")
    n = len(rows)
    print(f"\ntotal {n} runs: docked {sum(r['outcome'] == 'docked' for r in rows)}, collision {sum(r['outcome'] == 'collision' for r in rows)}, "
          f"no_path {sum(r['outcome'] == 'no_path' for r in rows)}, not_docked {sum(r['outcome'] == 'not_docked' for r in rows)}")


if __name__ == "__main__":
    main()
