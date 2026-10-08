"""Independent validation of the filter variants on paths the coefficients were not read off. Run from the repository root:
    python experiments/validation_runs.py --model-dir <robotis_tb3 folder> [--duration 240] [--out-dir runs]

Nothing is refitted here: the noise coefficients (read off the default run), the hybrid switch distance (1.5 m), the gain
table and the single-point gains (calibrated in their own short runs) are used as they are. Sub-pixel corner refinement is on.
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
from estimation.models import h_marker_pose, wrap  # noqa: E402
from noise_model_study import hybrid, lateral_scaled, rms_cov  # noqa: E402
from replay import inflated, load_run, marker_bias_by_distance, metrics, replay  # noqa: E402
from sim.mujoco_run import (VALIDATION_PATHS, RunConfig, calibrate_gain_table, calibrate_gains, collect_run,  # noqa: E402
                            min_clearance, pick_texture_type)
from sim.scene import SceneConfig, write_scene  # noqa: E402


def first30(errs):
    return 1000 * np.linalg.norm(errs[:300, :2], axis=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--out-dir", default="validation_runs")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("MuJoCo", mujoco.__version__)
    scene_cfg = pick_texture_type(SceneConfig(), args.model_dir)
    drv = MujocoTB3Driver(write_scene(scene_cfg, args.model_dir))
    base_cfg = RunConfig()
    cal = calibrate_gains(drv, base_cfg)
    calib_cfg = SceneConfig(room_x=8.0, room_y=8.0, obstacles=(), station_x=3.5, marker_texture_type=scene_cfg.marker_texture_type)
    gain_fn, _ = calibrate_gain_table(MujocoTB3Driver(write_scene(calib_cfg, args.model_dir, scene_name="scene_calib.xml")), base_cfg)
    print(f"single-point gains {cal[0]:.3f}, {cal[1]:.3f}; gain table measured")

    variants = [
        ("A single-point", dict(gains=cal)),
        ("A gain table", dict(gain_fn=gain_fn)),
        ("A table + hybrid R", dict(gain_fn=gain_fn, meas_cov=hybrid)),
        ("B isotropic", dict(estimate=True)),
        ("B anisotropic, std-fitted", dict(estimate=True, meas_cov=marker_measurement_cov_measured)),
        ("B anisotropic, RMS-fitted", dict(estimate=True, meas_cov=rms_cov)),
        ("B anisotropic RMS, lateral x5", dict(estimate=True, meas_cov=lateral_scaled(5))),
        ("B hybrid", dict(estimate=True, meas_cov=hybrid)),
        ("B hybrid, Q x2", dict(estimate=True, meas_cov=hybrid, noise=inflated(2))),
        # C: the gain table as the base, with two multiplicative corrections estimated online (added after the first two paths)
        ("C table + online correction", dict(estimate=True, gain_fn=gain_fn, scale_sigma0=0.1)),
        ("C, Q x2", dict(estimate=True, gain_fn=gain_fn, scale_sigma0=0.1, noise=inflated(2))),
    ]

    summary = {name: {} for name, _ in variants}
    for path_name, spec in VALIDATION_PATHS.items():
        cfg = RunConfig(duration=args.duration, subpixel=True, **spec)
        file = out / f"validation_{path_name}.npz"
        if file.exists():
            run = load_run(file)
            print(f"\n=== path '{path_name}': loaded {file}")
        else:
            t0 = time.time()
            run = collect_run(drv, scene_cfg, cfg)
            np.savez(file, **run)
            print(f"\n=== path '{path_name}': collected {len(run['truth'])} samples in {time.time() - t0:.0f} s")
        seen = np.all(np.isfinite(run["marker"]), axis=1)
        obliq = np.array([np.degrees(abs(wrap(h_marker_pose(run["truth"][k], run["marker_world"])[2] - np.pi))) for k in np.where(seen)[0]])
        print(f"marker pose available in {100 * seen.mean():.0f}% of samples; obliquity when seen: median {np.median(obliq):.1f} deg, "
              f"{100 * np.mean(obliq > 15):.0f}% above 15 deg, {100 * np.mean(obliq > 30):.0f}% above 30 deg; "
              f"smallest clearance {min_clearance(run['truth'], scene_cfg):.2f} m")
        print("marker error mean / RMS by distance: band | n | x mm | y mm | heading deg")
        for lo, hi, n, mx, my, ma, rx, ry, ra in marker_bias_by_distance(run):
            print(f"   {lo:.1f}-{hi:.1f} | {n:3d} | {1000 * mx:5.1f}/{1000 * rx:5.1f} | {1000 * my:5.1f}/{1000 * ry:5.1f} | {ma:5.2f}/{ra:5.2f}")
        print("variant                          | first 30 s [mm] | pos RMSE 2nd half [mm] | max [mm] | heading RMSE [deg] | ANEES | inside 95%")
        for name, kw in variants:
            errs, nees, _, _ = replay(run, **kw)
            m = metrics(errs, nees)
            row = [first30(errs), 1000 * m["pos_rmse_half"], 1000 * m["pos_max"], m["head_rmse_deg"], m["anees"], m["inside"]]
            summary[name][path_name] = row
            print(f"{name:32s} | {row[0]:15.0f} | {row[1]:22.1f} | {row[2]:8.0f} | {row[3]:18.2f} | {row[4]:5.1f} | {row[5]:.2f}")

    header = "variant                          | first 30 s [mm] | pos RMSE 2nd half [mm] | max [mm] | heading RMSE [deg] | ANEES | inside 95%"
    groups = (("development paths (diagonal, reverse): the variants were designed with these in view",
               [p for p in ("diagonal", "reverse") if p in VALIDATION_PATHS]),
              ("final test path (zigzag): added after the variants were fixed", [p for p in ("zigzag",) if p in VALIDATION_PATHS]))
    for title, paths in groups:
        if not paths:
            continue
        print(f"\n=== mean over {title}")
        print(header)
        for name, _ in variants:
            a = np.mean([summary[name][p] for p in paths], axis=0)
            print(f"{name:32s} | {a[0]:15.0f} | {a[1]:22.1f} | {a[2]:8.0f} | {a[3]:18.2f} | {a[4]:5.1f} | {a[5]:.2f}")


if __name__ == "__main__":
    main()
