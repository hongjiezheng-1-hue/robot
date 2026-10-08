"""Checks for the reference trajectory and the LQR tracker. Run: python tests/test_control.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from control.lqr_tracker import LQRTracker, discretise, to_nominal, wheel_speeds  # noqa: E402
from control.reference import build_reference  # noqa: E402
from estimation.models import f_unicycle, tracking_error_model, wrap  # noqa: E402

DT = 0.1
WAYPOINTS = [(1.5, 0.0), (1.5, 1.0), (-0.5, 1.0)]


def test_reference_is_smooth_and_consistent():
    ref = build_reference((0.0, 0.0), WAYPOINTS, 0.1, 0.3, DT)
    p = np.stack([ref["x"], ref["y"]], axis=1)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    assert np.all(step < 0.1 * DT * 1.001) and np.all(step > 0.1 * DT * 0.98), "sample spacing must be about speed * dt"
    heading_from_motion = np.arctan2(np.diff(p[:, 1]), np.diff(p[:, 0]))
    assert np.max(np.abs(wrap(heading_from_motion - ref["theta"][:-1]))) < 0.2, "heading must follow the path tangent"
    on_arc = np.abs(ref["w"]) > 0
    assert on_arc.any() and np.allclose(np.abs(ref["w"][on_arc]), 0.1 / 0.3, atol=1e-6), "arc rate is v / radius"
    assert np.all(ref["v"] > 0), "the reference speed never reaches zero"
    assert ref["w"][on_arc][0] > 0 and abs(ref["theta"][-1]) > 3.0, "left turns give positive yaw rates and end heading is pi"


def test_lqr_gains_are_stable_over_the_operating_range():
    tracker = LQRTracker(DT)
    for v_r in (0.03, 0.06, 0.10, 0.15):
        for w_r in (-0.5, -0.2, 0.0, 0.2, 0.5):
            A, B = tracking_error_model(v_r, w_r)
            Ad, Bd = discretise(A, B, DT)
            eig = np.abs(np.linalg.eigvals(Ad - Bd @ tracker.gain(v_r, w_r)))
            assert np.all(eig < 1.0), (v_r, w_r, eig)


def _simulate(tracker, ref, start_pose, gain_true, gain_model=None):
    """Closed loop on an ideal unicycle whose true gains differ from the nominal ones; the controller sees the exact pose."""
    pose = np.array(start_pose, dtype=float)
    errs = []
    for k in range(len(ref["x"])):
        r = {key: ref[key][k] for key in ("x", "y", "theta", "v", "w")}
        v, w, e = tracker.command(pose, r)
        vn, wn = to_nominal(v, w, gain_model)
        pose = f_unicycle(pose, (gain_true[0] * vn, gain_true[1] * wn), DT)
        errs.append(e)
    return np.array(errs)


def test_tracking_converges_from_an_initial_offset():
    ref = build_reference((0.0, 0.0), WAYPOINTS, 0.1, 0.3, DT)
    errs = _simulate(LQRTracker(DT), ref, (-0.15, 0.2, 0.4), (1.0, 1.0))
    assert np.linalg.norm(errs[-100:, :2], axis=1).max() < 0.01, "position error must vanish with exact gains"
    assert abs(errs[-100:, 2]).max() < 0.02


def test_odometry_compensation_helps_with_wrong_gains():
    ref = build_reference((0.0, 0.0), WAYPOINTS, 0.1, 0.3, DT)
    true_gains = (0.86, 0.72)
    model = lambda v, w: true_gains
    unaware = _simulate(LQRTracker(DT), ref, (0.0, 0.0, 0.0), true_gains)
    aware = _simulate(LQRTracker(DT), ref, (0.0, 0.0, 0.0), true_gains, gain_model=model)
    rms = lambda e: float(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, axis=1))))
    assert rms(unaware) < 0.15, "the feedback alone must keep the error bounded"
    assert rms(aware) < rms(unaware) / 3, (rms(aware), rms(unaware))


def test_commands_respect_the_limits():
    tracker = LQRTracker(DT, v_max=0.15, w_max=0.6)
    ref = {"x": 3.0, "y": 3.0, "theta": 3.0, "v": 0.1, "w": 0.0}      # a large error: the commands must saturate
    v, w, _ = tracker.command(np.array([0.0, 0.0, 0.0]), ref)
    assert 0.0 <= v <= 0.15 and abs(w) <= 0.6


def test_wheel_speeds_and_nominal_inversion():
    vl, vr = wheel_speeds(0.1, 0.0, 0.033, 0.288)
    assert np.isclose(vl, vr) and np.isclose(vl, 0.1 / 0.033)
    vl, vr = wheel_speeds(0.0, 0.5, 0.033, 0.288)
    assert vl < 0 < vr
    vn, wn = to_nominal(0.086, 0.36, lambda v, w: (0.86, 0.72))
    assert np.isclose(vn, 0.1) and np.isclose(wn, 0.5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
