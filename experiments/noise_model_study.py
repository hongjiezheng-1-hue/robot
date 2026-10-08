"""Compare marker measurement noise models on a saved run. Run from the repository root:
    python experiments/noise_model_study.py <run.npz>

Variants (all use the B filter, gains estimated online): the original isotropic model, the anisotropic model fitted to the
residual standard deviation, the anisotropic model fitted to the RMS (bias included), the effect of using only every 5th
marker update (the errors of successive frames are strongly correlated), a larger lateral sigma, and a hybrid model.
"""

import functools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from estimation.ekf import marker_measurement_cov, marker_measurement_cov_measured  # noqa: E402
from replay import load_run, metrics, replay  # noqa: E402

rms_cov = functools.partial(marker_measurement_cov_measured, depth_coeff=0.0062, lateral_coeff=0.0008, ang_coeff=0.06)


def thinned(run, n):
    out = dict(run)
    marker = run["marker"].copy()
    idx = np.where(np.all(np.isfinite(marker), axis=1))[0]
    marker[idx[np.arange(len(idx)) % n != 0]] = np.nan
    out["marker"] = marker
    return out


def lateral_scaled(k):
    def f(z):
        R = rms_cov(z)
        b = np.arctan2(z[1], z[0])
        c, s = np.cos(b), np.sin(b)
        rot = np.array([[c, -s], [s, c]])
        local = rot.T @ R[:2, :2] @ rot
        R[:2, :2] = rot @ np.diag([local[0, 0], local[1, 1] * k**2]) @ rot.T
        return R
    return f


def hybrid(z):
    d = np.hypot(z[0], z[1])
    return rms_cov(z) if d < 1.5 else marker_measurement_cov(d)


def first30(errs):
    return 1000 * np.linalg.norm(errs[:300, :2], axis=1).mean()


if __name__ == "__main__":
    run = load_run(sys.argv[1])
    cases = [
        ("isotropic (original)", run, None),
        ("anisotropic, std-fitted", run, marker_measurement_cov_measured),
        ("anisotropic, RMS-fitted", run, rms_cov),
        ("isotropic, every 5th update", thinned(run, 5), None),
        ("anisotropic RMS, every 5th update", thinned(run, 5), rms_cov),
        ("anisotropic RMS, every 10th update", thinned(run, 10), rms_cov),
        ("anisotropic RMS, lateral sigma x5", run, lateral_scaled(5)),
        ("hybrid (isotropic beyond 1.5 m)", run, hybrid),
    ]
    print("B filter | first 30 s [mm] | pos RMSE 2nd half [mm] | max [mm] | heading RMSE [deg] | ANEES | inside 95%")
    for name, r, cov in cases:
        errs, nees, _, _ = replay(r, estimate=True, meas_cov=cov)
        m = metrics(errs, nees)
        print(f"{name:36s} | {first30(errs):6.0f} | {1000 * m['pos_rmse_half']:6.1f} | {1000 * m['pos_max']:5.0f} | "
              f"{m['head_rmse_deg']:5.2f} | {m['anees']:5.1f} | {m['inside']:.2f}")
