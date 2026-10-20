"""Checks for the one-landmark EKF-SLAM (marker position not known in advance). Run: python tests/test_slam.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from estimation.ekf import MarkerEKF, MarkerSLAM  # noqa: E402
from estimation.models import h_marker_pose, wrap  # noqa: E402

TRUE_MARKER = np.array([1.5, 0.0, np.pi])


def test_marker_jacobian_matches_finite_differences():
    pose, marker = np.array([0.2, -0.1, 0.3]), np.array([1.5, 0.2, 3.0])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    H = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    eps = 1e-6
    num = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps
        diff = h_marker_pose(pose, marker + d) - h_marker_pose(pose, marker - d)
        diff[2] = wrap(diff[2])
        num[:, j] = diff / (2 * eps)
    assert np.allclose(H, num, atol=1e-6)


def test_wrong_marker_prior_is_corrected_by_measurements():
    rng = np.random.default_rng(0)
    pose = np.array([0.0, 0.0, 0.0])
    prior = TRUE_MARKER + np.array([0.2, 0.3, np.radians(10.0)])
    slam = MarkerSLAM(pose, np.diag([1e-4, 1e-4, 1e-4]), prior, np.diag([0.5**2, 0.5**2, np.radians(20.0) ** 2]))
    for _ in range(50):
        slam.predict(0.0, 0.0, 0.1)
        z = h_marker_pose(pose, TRUE_MARKER) + rng.normal(0.0, [0.003, 0.003, 0.01])
        slam.update(z)
    err = slam.marker - TRUE_MARKER
    assert np.hypot(err[0], err[1]) < 0.01 and abs(wrap(err[2])) < np.radians(1.0), "marker found within 1 cm and 1 degree"
    assert np.sqrt(np.trace(slam.marker_cov[:2, :2])) < 0.05, "and the filter knows it"


def test_same_as_marker_ekf_when_the_marker_is_known_exactly():
    pose0, P0 = np.array([0.0, 0.0, 0.1]), np.diag([0.01, 0.01, 0.01])
    a = MarkerEKF(pose0, P0, TRUE_MARKER)
    b = MarkerSLAM(pose0, P0, TRUE_MARKER, np.diag([1e-12, 1e-12, 1e-12]))
    z = np.array([1.4, 0.15, 3.0])
    for f in (a, b):
        f.predict(3.0, 3.2, 0.1)
        f.update(z)
    assert np.allclose(a.pose, b.pose, atol=1e-6) and np.allclose(a.pose_cov, b.pose_cov, atol=1e-8)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
