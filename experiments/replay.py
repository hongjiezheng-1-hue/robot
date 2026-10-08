"""Replay logged runs through the EKF variants and evaluate them. Pure numpy; works for synthetic and MuJoCo runs.

A run is a dict with:
  dt          sample period [s]
  pose0       true pose before the first interval (3,)
  wheel_omega measured wheel speeds [left, right] per interval, (N, 2) [rad/s]
  truth       true pose at the end of each interval, (N, 3)
  marker      measured marker pose [x, y, heading] in the robot frame, NaN when not detected, (N, 3)
  marker_world known marker pose in the world (3,)
  wheel_radius, track_nominal
"""

import numpy as np

from estimation.ekf import MarkerEKF, OdometryNoise
from estimation.models import h_marker_pose, wrap

CHI2_3_95 = 7.815
CHI2_3_999 = 16.27
P0 = np.diag([0.3**2, 0.3**2, np.deg2rad(10.0) ** 2])


def run_from_synthetic(cfg, data):
    truth, wheels, markers = data
    return dict(dt=cfg.dt, pose0=np.array(cfg.start, dtype=float), wheel_omega=wheels, truth=truth, marker=markers,
                marker_world=np.array(cfg.marker), wheel_radius=cfg.wheel_radius, track_nominal=cfg.track_nominal)


def inflated(k):
    """Default odometry noise scaled by k."""
    d = OdometryNoise()
    return OdometryNoise(d.v_floor * k, d.v_rel * k, d.w_floor * k, d.w_rel * k)


def load_run(path):
    """Load a run saved with np.savez, turning the scalar entries back into floats."""
    data = np.load(path)
    run = {k: data[k] for k in data.files}
    for k in ("dt", "wheel_radius", "track_nominal"):
        run[k] = float(run[k])
    return run


def replay(run, estimate=False, gains=(1.0, 1.0), gate=None, noise=None, seed=0, meas_cov=None, gain_fn=None):
    """Run one EKF variant over the log. Returns pose errors, NEES, gain history and the number of rejected updates."""
    rng = np.random.default_rng(10_000 + seed)
    x0 = run["pose0"] + rng.multivariate_normal(np.zeros(3), P0)
    ekf = MarkerEKF(x0, P0, run["marker_world"], run["wheel_radius"], run["track_nominal"], scales=gains,
                    estimate_scales=estimate, gate=gate, noise=noise, meas_cov=meas_cov, gain_fn=gain_fn)
    errs, nees, hist, rejected = [], [], [], 0
    for k in range(len(run["truth"])):
        ekf.predict(run["wheel_omega"][k, 0], run["wheel_omega"][k, 1], run["dt"])
        if np.all(np.isfinite(run["marker"][k])):
            rejected += 0 if ekf.update(run["marker"][k]) else 1
        e = run["truth"][k] - ekf.pose
        e[2] = wrap(e[2])
        errs.append(e)
        nees.append(float(e @ np.linalg.solve(ekf.pose_cov, e)))
        hist.append(np.array(ekf.scales))
    return np.array(errs), np.array(nees), np.array(hist), rejected


def metrics(errs, nees):
    half = len(errs) // 2
    pos = np.linalg.norm(errs[:, :2], axis=1)
    return dict(pos_rmse=float(np.sqrt(np.mean(pos**2))), pos_rmse_half=float(np.sqrt(np.mean(pos[half:] ** 2))),
                pos_max=float(pos.max()), head_rmse_deg=float(np.degrees(np.sqrt(np.mean(errs[:, 2] ** 2)))),
                anees=float(np.mean(nees)), inside=float(np.mean(nees <= CHI2_3_95)))


def print_table(rows):
    print("variant              | pos RMSE [mm] | pos RMSE 2nd half | max [mm] | heading RMSE [deg] | ANEES (ideal 3) | inside 95% | rejected")
    for name, m, rej in rows:
        print(f"{name:20s} | {1000 * m['pos_rmse']:13.1f} | {1000 * m['pos_rmse_half']:17.1f} | {1000 * m['pos_max']:8.0f} | "
              f"{m['head_rmse_deg']:18.2f} | {m['anees']:15.2f} | {m['inside']:10.3f} | {rej}")


def odometry_ratio_by_speed(run, v_bins=((0.02, 0.07), (0.07, 0.12), (0.12, 0.25)), w_bins=((0.1, 0.25), (0.25, 0.6))):
    """True motion divided by the nominal-encoder motion, per interval, grouped by nominal speed."""
    r, b, dt = run["wheel_radius"], run["track_nominal"], run["dt"]
    wl, wr = run["wheel_omega"][:, 0], run["wheel_omega"][:, 1]
    v_nom, w_nom = r * (wr + wl) / 2, r * (wr - wl) / b
    poses = np.vstack([run["pose0"], run["truth"]])
    d = np.diff(poses[:, :2], axis=0)
    heading = np.unwrap(poses[:, 2])
    mid = (heading[:-1] + heading[1:]) / 2
    v_true = (d[:, 0] * np.cos(mid) + d[:, 1] * np.sin(mid)) / dt
    w_true = np.diff(heading) / dt
    out = {"v": [], "w": []}
    for lo, hi in v_bins:
        sel = (v_nom >= lo) & (v_nom < hi) & (np.abs(w_nom) < 0.05)       # near-straight intervals only
        out["v"].append((lo, hi, int(sel.sum()), float(np.median(v_true[sel] / v_nom[sel])) if sel.any() else np.nan))
    for lo, hi in w_bins:
        sel = (np.abs(w_nom) >= lo) & (np.abs(w_nom) < hi)
        out["w"].append((lo, hi, int(sel.sum()), float(np.median(w_true[sel] / w_nom[sel])) if sel.any() else np.nan))
    return out


def marker_bias_by_distance(run, bins=((0.2, 0.7), (0.7, 1.1), (1.1, 1.5), (1.5, 2.0), (2.0, 3.5))):
    """Mean (systematic) and RMS (mean and spread together) of the marker pose error by distance."""
    rows = []
    for k in range(len(run["truth"])):
        z = run["marker"][k]
        if np.all(np.isfinite(z)):
            e = z - h_marker_pose(run["truth"][k], run["marker_world"])
            e[2] = wrap(e[2])
            rows.append((np.hypot(z[0], z[1]), e))
    out = []
    for lo, hi in bins:
        sel = np.array([e for d, e in rows if lo <= d < hi])
        if len(sel):
            out.append((lo, hi, len(sel), sel[:, 0].mean(), sel[:, 1].mean(), np.degrees(sel[:, 2].mean()),
                        np.sqrt(np.mean(sel[:, 0] ** 2)), np.sqrt(np.mean(sel[:, 1] ** 2)), np.degrees(np.sqrt(np.mean(sel[:, 2] ** 2)))))
    return out


def marker_residual_by_distance(run, bins=((0.2, 0.7), (0.7, 1.1), (1.1, 1.5), (1.5, 2.0), (2.0, 3.5))):
    """Measured minus true marker pose, grouped by distance; compares the real noise with the filter's model."""
    from estimation.ekf import marker_measurement_cov
    rows = []
    for k in range(len(run["truth"])):
        z = run["marker"][k]
        if np.all(np.isfinite(z)):
            truth = h_marker_pose(run["truth"][k], run["marker_world"])
            e = z - truth
            e[2] = wrap(e[2])
            rows.append((np.hypot(z[0], z[1]), e))
    out = []
    for lo, hi in bins:
        sel = [e for d, e in rows if lo <= d < hi]
        if sel:
            sel = np.array(sel)
            mid = (lo + hi) / 2
            model = np.sqrt(np.diag(marker_measurement_cov(mid)))
            out.append((lo, hi, len(sel), np.std(sel[:, 0]), np.std(sel[:, 1]), np.degrees(np.std(sel[:, 2])),
                        model[0], np.degrees(model[2])))
    return out
