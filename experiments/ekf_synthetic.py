"""Compare EKF variants on the synthetic plant. Run from the repository root: python experiments/ekf_synthetic.py [runs]

Variants (same data for every variant):
  A0   fixed nominal odometry gains (1, 1), no calibration
  A2   fixed gains with a calibration error of about 5 percent, default odometry noise
  A2t  as A2 with the odometry noise inflated by a factor chosen on separate tuning runs
  A1   fixed gains equal to the true gains (ideal calibration)
  B    gains estimated online, started from the nominal values
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from estimation.ekf import MarkerEKF, OdometryNoise  # noqa: E402
from estimation.models import wrap  # noqa: E402
from sim.synthetic import SyntheticConfig, simulate  # noqa: E402

CHI2_3_95 = 7.815
P0 = np.diag([0.3**2, 0.3**2, np.deg2rad(10.0) ** 2])


def inflated(k):
    d = OdometryNoise()
    return OdometryNoise(d.v_floor * k, d.v_rel * k, d.w_floor * k, d.w_rel * k)


def run_variant(cfg, data, name, seed, gate=None, noise=None, miscal=(1.05, 0.95)):
    truth, wheels, markers = data
    rng = np.random.default_rng(10_000 + seed)
    x0 = truth[0] + rng.multivariate_normal(np.zeros(3), P0)
    s_true = np.array([cfg.s_v_true, cfg.s_w_true])
    off = tuple(s_true * np.array(miscal))
    kinds = {"A0": dict(scales=(1.0, 1.0)), "A2": dict(scales=off), "A2t": dict(scales=off),
             "A1": dict(scales=tuple(s_true)), "B": dict(scales=(1.0, 1.0), estimate_scales=True)}
    ekf = MarkerEKF(x0, P0, cfg.marker, cfg.wheel_radius, cfg.track_nominal, gate=gate, noise=noise, **kinds[name])
    errs, nees, scale_hist = [], [], []
    for k in range(len(truth)):
        ekf.predict(wheels[k, 0], wheels[k, 1], cfg.dt)
        if np.all(np.isfinite(markers[k])):
            ekf.update(markers[k])
        e = truth[k] - ekf.pose
        e[2] = wrap(e[2])
        errs.append(e)
        nees.append(float(e @ np.linalg.solve(ekf.pose_cov, e)))
        scale_hist.append(np.array(ekf.scales))
    return np.array(errs), np.array(nees), np.array(scale_hist)


def tune_inflation(cfg, tune_seeds, factors=(1, 2, 3, 4, 6, 8)):
    """Pick the odometry-noise inflation for A2 whose mean ANEES is closest to 3, on separate runs."""
    best = None
    for k in factors:
        anees = np.mean([np.mean(run_variant(cfg, simulate(cfg, s), "A2t", s, noise=inflated(k))[1]) for s in tune_seeds])
        print(f"  tuning A2t: inflation {k}: ANEES {anees:.2f}")
        if best is None or abs(anees - 3) < abs(best[1] - 3):
            best = (k, anees)
    return best[0]


def summarise(cfg, seeds, names, k_tuned, gate=None, label=""):
    stats = {n: [] for n in names}
    scales_b = []
    for seed in seeds:
        data = simulate(cfg, seed)
        for n in names:
            noise = inflated(k_tuned) if n == "A2t" else None
            errs, nees, hist = run_variant(cfg, data, n, seed, gate, noise)
            half = len(errs) // 2
            stats[n].append([np.sqrt(np.mean(np.sum(errs[half:, :2] ** 2, axis=1))),
                             np.degrees(np.sqrt(np.mean(errs[half:, 2] ** 2))),
                             float(np.mean(nees)), float(np.mean(nees <= CHI2_3_95))])
            if n == "B":
                scales_b.append(hist[-1])
    if label:
        print(label)
    print("variant | position RMSE [mm] (second half) | heading RMSE [deg] | ANEES (ideal 3) | NEES inside 95% bound")
    for n in names:
        a = np.array(stats[n])
        print(f"{n:7s} | {1000 * a[:, 0].mean():8.1f} +/- {1000 * a[:, 0].std():6.1f}"
              f"            | {a[:, 1].mean():6.2f} +/- {a[:, 1].std():5.2f}"
              f"  | {a[:, 2].mean():9.2f} +/- {a[:, 2].std():7.2f} | {a[:, 3].mean():.3f}")
    if "B" in names:
        sb = np.array(scales_b)
        print(f"B final scale estimates: s_v {sb[:, 0].mean():.3f} +/- {sb[:, 0].std():.3f} (true {cfg.s_v_true}), "
              f"s_w {sb[:, 1].mean():.3f} +/- {sb[:, 1].std():.3f} (true {cfg.s_w_true:.3f})")


if __name__ == "__main__":
    cfg = SyntheticConfig()
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    truth, wheels, markers = simulate(cfg, 0)
    print(f"{n_seeds} runs of {cfg.duration:.0f} s; marker visible in {100 * np.mean(np.all(np.isfinite(markers), axis=1)):.0f}% of samples (seed 0)")
    k = tune_inflation(cfg, tune_seeds=range(100, 104))
    print(f"chosen inflation for A2t: {k}")
    summarise(cfg, range(n_seeds), ("A0", "A2", "A2t", "A1", "B"), k)
