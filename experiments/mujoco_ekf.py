"""Collect a MuJoCo run and compare the EKF variants. Script version of notebooks 05 and 06 that runs locally.

Run from the repository root:
    python experiments/mujoco_ekf.py --model-dir <path to robotis_mujoco_menagerie/robotis_tb3> [--duration 240]
Rendering uses the default OpenGL backend; on Linux without a display set MUJOCO_GL=egl first.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import mujoco  # noqa: E402
from drivers.mujoco_tb3 import MujocoTB3Driver  # noqa: E402
from estimation.ekf import marker_measurement_cov_measured  # noqa: E402
from replay import (CHI2_3_999, inflated, load_run, marker_residual_by_distance, metrics,  # noqa: E402
                    odometry_ratio_by_speed, print_table, replay)
from sim.mujoco_run import (RunConfig, calibrate_gain_table, calibrate_gains, collect_run,  # noqa: E402
                            min_clearance, pick_texture_type)
from sim.scene import SceneConfig, write_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="folder robotis_tb3 of the ROBOTIS MuJoCo model")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--run-file", default=None, help="save the collected run here, or load it if it exists")
    args = ap.parse_args()

    print("MuJoCo", mujoco.__version__)
    scene_cfg = pick_texture_type(SceneConfig(), args.model_dir)
    print("marker texture type used:", scene_cfg.marker_texture_type)
    run_cfg = RunConfig(duration=args.duration)
    drv = MujocoTB3Driver(write_scene(scene_cfg, args.model_dir))

    s_v_cal, s_w_cal = calibrate_gains(drv, run_cfg)
    print(f"single-point calibration: s_v = {s_v_cal:.3f}, s_w = {s_w_cal:.3f}")
    calib_cfg = SceneConfig(room_x=8.0, room_y=8.0, obstacles=(), station_x=3.5, marker_texture_type=scene_cfg.marker_texture_type)
    calib_drv = MujocoTB3Driver(write_scene(calib_cfg, args.model_dir, scene_name="scene_calib.xml"))
    gain_fn, table = calibrate_gain_table(calib_drv, run_cfg)
    print("gain table s_v (rows: speed", list(table["speeds"]), "columns: yaw rate", list(table["yaw_rates"]), ")")
    print(np.round(table["s_v"], 3))
    print("gain table s_w")
    print(np.round(table["s_w"], 3))

    run_path = pathlib.Path(args.run_file) if args.run_file else None
    if run_path and run_path.exists():
        run = load_run(run_path)
        print("loaded", run_path)
    else:
        t0 = time.time()
        run = collect_run(drv, scene_cfg, run_cfg)
        print(f"collected {len(run['truth'])} samples in {time.time() - t0:.0f} s")
        if run_path:
            np.savez(run_path, **run)
    print(f"marker pose available in {100 * np.mean(np.all(np.isfinite(run['marker']), axis=1)):.0f}% of samples; "
          f"smallest clearance {min_clearance(run['truth'], scene_cfg):.2f} m")

    ratios = odometry_ratio_by_speed(run)
    print("true / nominal odometry, linear speed:", [(f"{lo:.2f}-{hi:.2f}", n, round(r, 3)) for lo, hi, n, r in ratios["v"]])
    print("true / nominal odometry, yaw rate:   ", [(f"{lo:.2f}-{hi:.2f}", n, round(r, 3)) for lo, hi, n, r in ratios["w"]])
    print("marker residuals: distance band | n | x std mm | y std mm | heading std deg")
    for lo, hi, n, sx, sy, sa, mp, ma in marker_residual_by_distance(run):
        print(f"   {lo:.1f}-{hi:.1f} | {n} | {1000 * sx:.1f} | {1000 * sy:.1f} | {sa:.2f}")

    aniso, cal = marker_measurement_cov_measured, (s_v_cal, s_w_cal)
    variants = [
        ("A0 nominal gains", dict(gains=(1.0, 1.0))),
        ("A single-point", dict(gains=cal)),
        ("A single-point + gate", dict(gains=cal, gate=CHI2_3_999)),
        ("A gain table", dict(gain_fn=gain_fn)),
        ("A table + aniso R", dict(gain_fn=gain_fn, meas_cov=aniso)),
        ("A table + aniso R, Q x2", dict(gain_fn=gain_fn, meas_cov=aniso, noise=inflated(2))),
        ("A table + aniso R, Q x4", dict(gain_fn=gain_fn, meas_cov=aniso, noise=inflated(4))),
        ("B", dict(estimate=True)),
        ("B + aniso R", dict(estimate=True, meas_cov=aniso)),
        ("B + aniso R, Q x2", dict(estimate=True, meas_cov=aniso, noise=inflated(2))),
        ("B + aniso R, Q x4", dict(estimate=True, meas_cov=aniso, noise=inflated(4))),
    ]
    rows = []
    for name, kw in variants:
        errs, nees, hist, rejected = replay(run, **kw)
        rows.append((name, metrics(errs, nees), rejected))
        if name == "B + aniso R, Q x2":
            print(f"B + aniso R, Q x2 final gains: s_v = {hist[-1, 0]:.3f}, s_w = {hist[-1, 1]:.3f}")
    print_table(rows)


if __name__ == "__main__":
    main()
