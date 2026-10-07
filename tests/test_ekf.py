"""Behavioural checks for the EKF on the synthetic plant. Run: python tests/test_ekf.py

These test the filter implementation (consistency with a matched model, scale estimation), not the MuJoCo robot.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from ekf_synthetic import run_variant  # noqa: E402
from sim.synthetic import SyntheticConfig, simulate  # noqa: E402

CFG = SyntheticConfig(duration=400.0)
SEEDS = (0, 1, 2)


def _runs(name):
    return [run_variant(CFG, simulate(CFG, s), name, s) for s in SEEDS]


def test_matched_model_is_consistent():
    anees = np.mean([np.mean(nees) for _, nees, _ in _runs("A1")])
    assert 1.8 < anees < 4.5, anees   # ideal value for a 3-dimensional pose is 3


def test_scales_are_estimated():
    for _, _, hist in _runs("B"):
        assert abs(hist[-1, 0] - CFG.s_v_true) / CFG.s_v_true < 0.03, hist[-1]
        assert abs(hist[-1, 1] - CFG.s_w_true) / CFG.s_w_true < 0.03, hist[-1]


def test_missing_calibration_hurts():
    def rmse(name):
        return np.mean([np.sqrt(np.mean(np.sum(e[len(e) // 2:, :2] ** 2, axis=1))) for e, _, _ in _runs(name)])
    assert rmse("A0") > 3 * rmse("A1")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
