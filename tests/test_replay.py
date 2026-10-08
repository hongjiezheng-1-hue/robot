"""Check the replay and evaluation helpers on a synthetic run. Run: python tests/test_replay.py"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from replay import marker_residual_by_distance, metrics, odometry_ratio_by_speed, replay, run_from_synthetic  # noqa: E402
from sim.synthetic import SyntheticConfig, simulate  # noqa: E402

CFG = SyntheticConfig(duration=300.0)
RUN = run_from_synthetic(CFG, simulate(CFG, 3))


def test_replay_matched_gains():
    errs, nees, hist, rejected = replay(RUN, gains=(CFG.s_v_true, CFG.s_w_true))
    m = metrics(errs, nees)
    assert m["anees"] < 6 and m["pos_rmse_half"] < 0.2 and rejected == 0


def test_replay_estimates_gains():
    _, _, hist, _ = replay(RUN, estimate=True)
    assert abs(hist[-1, 0] - CFG.s_v_true) < 0.05 and abs(hist[-1, 1] - CFG.s_w_true) < 0.05


def test_gating_rejects_outliers():
    cfg = SyntheticConfig(duration=300.0, outlier_rate=0.5)
    run = run_from_synthetic(cfg, simulate(cfg, 4))
    _, _, _, rejected = replay(run, gains=(cfg.s_v_true, cfg.s_w_true), gate=16.27)
    assert rejected > 0


def test_odometry_ratio_recovers_true_gains():
    out = odometry_ratio_by_speed(RUN)
    ratios_v = [r[3] for r in out["v"] if r[2] > 20]
    assert ratios_v and all(abs(r - CFG.s_v_true) < 0.06 for r in ratios_v), out
    ratios_w = [r[3] for r in out["w"] if r[2] > 20]
    assert ratios_w and all(abs(r - CFG.s_w_true) < 0.08 for r in ratios_w), out


def test_marker_residual_matches_model():
    rows = marker_residual_by_distance(RUN)
    assert rows
    for lo, hi, n, sx, sy, sang, mx, mang in rows:
        if n > 30:
            assert 0.5 < sx / mx < 1.6, (lo, hi, sx, mx)


def test_anisotropic_covariance():
    from estimation.ekf import marker_measurement_cov_measured
    R = marker_measurement_cov_measured(np.array([1.5, 0.0, 3.0]))
    assert np.all(np.linalg.eigvalsh(R) > 0) and np.allclose(R, R.T)
    assert R[0, 0] > 10 * R[1, 1], "depth (x here, on the line of sight) must dominate the lateral error"
    R2 = marker_measurement_cov_measured(np.array([0.0, 1.5, 3.0]))   # marker to the left: the line of sight is the y axis
    assert R2[1, 1] > 10 * R2[0, 0]


def test_gain_table_and_noise_variants_run():
    from sim.mujoco_run import make_gain_fn
    from estimation.ekf import marker_measurement_cov_measured
    from replay import inflated
    speeds, rates = np.array([0.03, 0.15]), np.array([0.0, 0.5])
    sv = np.full((2, 2), CFG.s_v_true)
    sw = np.full((2, 2), CFG.s_w_true)
    fn = make_gain_fn(speeds, rates, sv, sw)
    assert fn(0.1, -0.2) == (CFG.s_v_true, CFG.s_w_true) or np.allclose(fn(0.1, -0.2), (CFG.s_v_true, CFG.s_w_true))
    assert np.allclose(fn(5.0, 5.0), (CFG.s_v_true, CFG.s_w_true))   # clamped outside the grid
    errs, nees, _, _ = replay(RUN, gain_fn=fn, meas_cov=marker_measurement_cov_measured, noise=inflated(2.0))
    assert metrics(errs, nees)["pos_rmse_half"] < 0.4


def test_load_run_roundtrip(tmp_path=None):
    import tempfile
    from replay import load_run
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "run.npz"
        np.savez(path, **RUN)
        loaded = load_run(path)
    assert isinstance(loaded["dt"], float) and loaded["truth"].shape == RUN["truth"].shape
    errs, nees, _, _ = replay(loaded, gains=(CFG.s_v_true, CFG.s_w_true))
    assert metrics(errs, nees)["anees"] < 6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
