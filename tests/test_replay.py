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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
