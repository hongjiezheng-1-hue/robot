"""Single-command demonstration: navigate an unknown S-shaped board course with LiDAR and dock at the ArUco station (MuJoCo).

In short: one command runs everything. The robot finds unknown obstacles with LiDAR and docks at the marker.
The controller only uses the estimate; the simulator truth is only used for checking.

    python main.py --model-dir <robotis_mujoco_menagerie>/robotis_tb3

What happens (everything above the sensors uses only the estimated pose, never the simulator truth):
  1. the scene is built (4 x 3 m room, three boards the robot does not know about, charging station with an ArUco marker);
  2. the odometry gain table is measured on short arcs (variants table and C only);
  3. the robot starts slightly away from the mark it believes it is on, builds an occupancy grid from LiDAR scans and the EKF pose
     (wheel encoders + camera marker pose), plans and re-plans with A*, follows the path with an LQR tracker and docks along the
     marker normal;
  4. the result is checked against the provisional dock tolerance (lateral < 3 cm, along-track < 5 cm, heading < 5 deg) and a
     video and an overview image are written to --out-dir.
The exit code is 0 if the robot docked within tolerance without touching anything, 1 otherwise.
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def find_model_dir(given):
    candidates = [given, os.environ.get("TB3_MODEL_DIR"), ROOT / "robotis_mujoco_menagerie" / "robotis_tb3",
                  ROOT.parent / "robotis_mujoco_menagerie" / "robotis_tb3",
                  pathlib.Path.home() / "Documents" / "robotis_mujoco_menagerie" / "robotis_tb3"]
    for c in candidates:
        if c and (pathlib.Path(c) / "turtlebot3_waffle_pi.xml").exists():
            return str(c)
    raise SystemExit("Cannot find the robotis_tb3 model folder. Pass --model-dir <robotis_mujoco_menagerie>/robotis_tb3 "
                     "or set the environment variable TB3_MODEL_DIR.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="folder robotis_tb3 of the MuJoCo Menagerie (default: search common places)")
    ap.add_argument("--pipeline", choices=("full", "simple"), default="full",
                    help="full (default): LiDAR occupancy map, A* re-planning and LQR path tracking; "
                         "simple: EKF + reactive LiDAR avoidance + visual-servo docking, no map and no planner")
    ap.add_argument("--variant", choices=("C", "table", "B"), default="C",
                    help="odometry model: C = gain table x online correction (default), table = gain table, B = gains estimated online")
    ap.add_argument("--scenario", default="s_course", help="layout from sim/scenarios.py: s_course, s_mirrored, gate, scattered, u_trap, north_station, empty, chicane, facing_away")
    ap.add_argument("--disturbance", default="none", help="none, lidar_x3, start_x3, dropout50, radius+5, push (see sim/scenarios.py)")
    ap.add_argument("--marker-unknown", type=float, default=None, metavar="ERROR_M",
                    help="do not give the robot the marker position: start from a rough guess this many metres off and estimate "
                         "the marker with EKF-SLAM (for example 0.3)")
    ap.add_argument("--seed", type=int, default=401, help="seed of the random start offset and sensor noise")
    ap.add_argument("--speed", type=float, default=0.10, help="path-following speed [m/s]")
    ap.add_argument("--out-dir", default="demo_output")
    ap.add_argument("--no-video", action="store_true", help="skip the video (faster)")
    ap.add_argument("--video-step", type=int, default=2, help="draw every n-th control step (0.1 s each)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--live", action="store_true", help="show the run live in the MuJoCo viewer (LiDAR beams, estimated pose, current plan)")
    ap.add_argument("--show-truth", action="store_true", help="with --live: also draw the room, the true obstacles and the true path "
                                                                 "on the map (evaluation only; the robot does not know them)")
    ap.add_argument("--no-dashboard", action="store_true", help="with --live: no 2D map and camera window, only the MuJoCo viewer")
    ap.add_argument("--speedup", type=float, default=1.0, help="with --live: simulated seconds per wall-clock second (1 = real time)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    import mujoco

    from control.docking import DockSpec, dock_poses
    from control.lqr_tracker import LQRTracker
    from drivers.mujoco_tb3 import MujocoTB3Driver
    from perception.lidar import LidarConfig
    from sim.scenarios import DISTURBANCES, SCENARIOS
    from sim.mujoco_run import RunConfig, calibrate_gain_table, pick_texture_type
    from sim.navigation import navigation_metrics, run_navigation, slam_prior_from_error
    from sim.scene import SceneConfig, write_scene

    t0 = time.time()
    model_dir = find_model_dir(args.model_dir)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"MuJoCo {mujoco.__version__} | model folder {model_dir} | pipeline {args.pipeline} | variant {args.variant} | seed {args.seed} | scenario {args.scenario} | disturbance {args.disturbance}")

    if args.scenario not in SCENARIOS or args.disturbance not in DISTURBANCES:
        raise SystemExit(f"scenario must be one of {list(SCENARIOS)}, disturbance one of {list(DISTURBANCES)}")
    scen, dist = SCENARIOS[args.scenario], DISTURBANCES[args.disturbance]
    sx, sy, syaw = scen.station
    probe = pick_texture_type(SceneConfig(), model_dir)           # which marker texture this MuJoCo build renders (probe on the default scene)
    scene_cfg = SceneConfig(obstacles=scen.obstacles, station_x=sx, station_y=sy, station_yaw=syaw, marker_texture_type=probe.marker_texture_type)
    drv = MujocoTB3Driver(write_scene(scene_cfg, model_dir, scene_name=f"scene_demo_{os.getpid()}.xml"))
    if args.variant == "B":
        models = dict(gain_fn=None, estimate_scales=True)
    else:
        print("[1/3] measuring the odometry gain table on short arcs in an empty room ...", flush=True)
        calib_cfg = SceneConfig(room_x=8.0, room_y=8.0, obstacles=(), station_x=3.5, marker_texture_type=scene_cfg.marker_texture_type)
        gain_fn, _ = calibrate_gain_table(MujocoTB3Driver(write_scene(calib_cfg, model_dir, scene_name=f"scene_demo_calib_{os.getpid()}.xml")), RunConfig())
        models = dict(gain_fn=gain_fn, estimate_scales=args.variant == "C")

    spec = DockSpec()
    standoff, dock = dock_poses((scene_cfg.station_x, scene_cfg.station_y, scene_cfg.station_yaw), spec)
    rng = np.random.default_rng(args.seed)
    nominal = np.array(scen.start)
    start = nominal + dist.start_offset_scale * rng.normal(0.0, [0.05, 0.05, np.radians(3.0)])
    print(f"[2/3] navigating: the robot believes it starts at {np.round(nominal, 2)}, it really starts at {np.round(start, 2)}; "
          f"stand-off {np.round(standoff[:2], 2)}, dock {np.round(dock[:2], 2)}", flush=True)
    slam_prior = None
    if args.marker_unknown is not None:
        slam_prior = slam_prior_from_error(scen.station, args.marker_unknown)
        print(f"marker position NOT given: the robot starts from a guess {np.round(slam_prior[0], 2)} "
              f"(true {np.round(scen.station, 2)}) and estimates it (EKF-SLAM)", flush=True)
    t1 = time.time()
    on_step, viewer = None, None
    if args.live:
        from sim.live_view import LiveView

        live = LiveView(drv, speedup=args.speedup, scene_cfg=scene_cfg, standoff=standoff, dock=dock, dashboard=not args.no_dashboard,
                        start_pose=nominal, show_truth=args.show_truth, spec=spec)
        on_step, viewer = live.on_step, live
    common = dict(seed=args.seed, on_step=on_step,
                  lidar_cfg=LidarConfig(sigma_floor=0.01 * dist.lidar_noise_scale, sigma_rel=0.01 * dist.lidar_noise_scale),
                  wheel_radius=0.033 * (1 + dist.wheel_radius_error), marker_dropout=dist.marker_dropout, push=dist.push or None,
                  slam_prior=slam_prior, dock_spec=spec, **models)
    if args.pipeline == "simple":
        from sim.simple_navigation import run_simple

        log = run_simple(drv, scene_cfg, standoff, dock, start, nominal, go_speed=args.speed, **common)
    else:
        log = run_navigation(drv, scene_cfg, standoff, dock, LQRTracker(0.1), start, nominal, speed=args.speed, **common)
    if viewer is not None and not viewer.running():
        print("viewer closed before the run finished; stopping")
        return 2
    m = navigation_metrics(log, scene_cfg, spec, dock)
    ok = bool(m["docked"] and not m["collided"])

    extra = "" if args.pipeline == "simple" else f"; {m['replans']} plans, {len(log['map_reset_steps'])} map resets"
    print(f"\nsimulated {m['seconds']:.0f} s in {time.time() - t1:.0f} s of computing{extra}")
    print(f"contacts with boards, walls or station: {m['contact_steps']} steps ({'COLLISION' if m['collided'] else 'none'}); "
          f"closest approach to a board {1000 * m['clearance']:.0f} mm (centre distance)")
    if slam_prior is not None and np.all(np.isfinite(log["marker_est"])):
        me = log["marker_est"] - np.asarray(scen.station)
        print(f"marker estimate at the end: {1000 * np.hypot(me[0], me[1]):.0f} mm and {np.degrees(abs((me[2] + np.pi) % (2 * np.pi) - np.pi)):.1f} deg "
              f"from the true marker (the guess was {1000 * args.marker_unknown:.0f} mm off)")
    if m["failed"]:
        print("RESULT: no path to the stand-off pose was found")
    else:
        print(f"estimator position error: RMSE {1000 * m['est_pos_rmse']:.0f} mm, max {1000 * m['est_pos_max']:.0f} mm")
        print(f"final dock error: along-track {1000 * m['dock_along']:.0f} mm, lateral {1000 * m['dock_lateral']:.0f} mm, "
              f"heading {m['dock_heading_deg']:.1f} deg  (tolerance 50 mm, 30 mm, 5 deg; provisional)")
        print("RESULT: DOCKED within tolerance" if ok else "RESULT: outside tolerance or contact")

    np.savez(out / "demo_log.npz", **log)
    summary = dict(pipeline=args.pipeline, variant=args.variant, scenario=args.scenario, disturbance=args.disturbance, seed=args.seed, speed=args.speed, success=ok, **m)
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {out / 'demo_log.npz'} and {out / 'demo_summary.json'}")

    if not args.no_video and not args.live and not m["failed"]:
        from sim.demo_video import render_video

        print("[3/3] rendering the video ...", flush=True)
        verdict = "DOCKED" if ok else "NOT DOCKED"
        text = (f"{verdict}  along {1000 * m['dock_along']:.0f} mm, lateral {1000 * m['dock_lateral']:.0f} mm, "
                f"heading {m['dock_heading_deg']:.1f} deg")
        render_video(log, scene_cfg, drv, out / "demo.mp4", step=args.video_step, fps=args.fps, result_text=text,
                     label=f"{args.scenario} / {args.disturbance} / seed {args.seed}", overview_path=out / "demo_overview.png")
        print(f"wrote {out / 'demo.mp4'} and {out / 'demo_overview.png'}")
    print(f"total {time.time() - t0:.0f} s")
    if viewer is not None:
        verdict = "DOCKED" if ok else ("NO PATH" if m["failed"] else "NOT DOCKED")
        viewer.finish(verdict if m["failed"] else f"{verdict}  along {1000 * m['dock_along']:.0f} mm, lateral {1000 * m['dock_lateral']:.0f} mm, "
                      f"heading {m['dock_heading_deg']:.1f} deg", save_path=out / "live_dashboard.png")
        print(f"wrote {out / 'live_dashboard.png'}")
        print("The viewer stays open; close its window to exit.")
        viewer.wait_until_closed()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
