"""Plain-assert checks for estimation/models.py. Run: python tests/test_models.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from estimation.models import (F_unicycle, H_marker_pose, H_range_bearing, controllability_rank,  # noqa: E402
                               encoders_to_vw, f_unicycle, h_marker_pose, h_range_bearing,
                               observability_matrix, smallest_singular_value, tracking_error_model, wrap)

DT = 0.1


def num_jac(fun, x, eps=1e-6, angle_rows=()):
    cols = []
    for i in range(len(x)):
        d = np.zeros(len(x))
        d[i] = eps
        diff = np.asarray(fun(x + d)) - np.asarray(fun(x - d))
        for r in angle_rows:
            diff[r] = wrap(diff[r])
        cols.append(diff / (2 * eps))
    return np.array(cols).T


def test_jacobians():
    rng = np.random.default_rng(0)
    marker = np.array([2.0, 1.0, 0.5])
    for _ in range(20):
        x = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)])
        u = (rng.uniform(0.05, 0.2), rng.uniform(-0.5, 0.5))
        assert np.allclose(F_unicycle(x, u, DT), num_jac(lambda z: f_unicycle(z, u, DT), x, angle_rows=(2,)), atol=1e-5)
        assert np.allclose(H_marker_pose(x, marker), num_jac(lambda z: h_marker_pose(z, marker), x, angle_rows=(2,)), atol=1e-5)
        assert np.allclose(H_range_bearing(x, marker), num_jac(lambda z: h_range_bearing(z, marker), x, angle_rows=(1,)), atol=1e-5)
    assert np.allclose(F_unicycle(x, (0.1, 0.0), DT), num_jac(lambda z: f_unicycle(z, (0.1, 0.0), DT), x, angle_rows=(2,)), atol=1e-5)


def test_encoders():
    v, w = encoders_to_vw(1.0, 1.0, 0.033, 0.288)
    assert np.isclose(v, 0.033) and np.isclose(w, 0.0)
    v, w = encoders_to_vw(-1.0, 1.0, 0.033, 0.288)
    assert np.isclose(v, 0.0) and w > 0


def test_observability():
    rng = np.random.default_rng(1)
    marker = np.array([2.0, 1.0, 0.5])
    second = np.array([-1.0, 3.0, 0.0])
    sv_one, sv_two, sv_pose = [], [], []
    for _ in range(100):
        x = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)])
        if min(np.linalg.norm(x[:2] - marker[:2]), np.linalg.norm(x[:2] - second[:2])) < 0.5:
            continue
        us = [(rng.uniform(0.05, 0.2), rng.uniform(-0.4, 0.4))] * 6
        sv_one.append(smallest_singular_value(observability_matrix(x, us, DT, lambda z: H_range_bearing(z, marker))))
        sv_two.append(smallest_singular_value(observability_matrix(
            x, us, DT, lambda z: np.vstack([H_range_bearing(z, marker), H_range_bearing(z, second)]))))
        sv_pose.append(smallest_singular_value(observability_matrix(x, us, DT, lambda z: H_marker_pose(z, marker))))
    assert max(sv_one) < 1e-9, "one marker range+bearing should always be rank deficient"
    assert min(sv_two) > 1e-2 and min(sv_pose) > 1e-2


def test_controllability():
    for (vr, wr), rank in [((0.15, 0.0), 3), ((0.15, 0.3), 3), ((0.0, 0.3), 3), ((0.0, 0.0), 2)]:
        A, B = tracking_error_model(vr, wr)
        assert controllability_rank(A, B) == rank, (vr, wr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
