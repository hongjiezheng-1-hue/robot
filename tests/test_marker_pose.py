"""Check the marker-pose frame conversions with synthetic projections. Run: python tests/test_marker_pose.py

A marker with known world pose is projected into a camera mounted on a robot at a known pose; the pose
recovered from the projected corners must equal estimation.models.h_marker_pose.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from estimation.models import h_marker_pose, wrap  # noqa: E402
from perception.marker_pose import R_CV_TO_ROBOT, camera_intrinsics, estimate_marker_pose  # noqa: E402

CAM_POS = np.array([0.083, 0.0, 0.107])
W, H, FOVY = 640, 480, 48.8
K = camera_intrinsics(FOVY, W, H)


def project_marker(robot, marker, size):
    """Pixel corners (TL, TR, BR, BL as seen by a viewer in front of the marker) of the marker."""
    mx, my, psi = marker
    n = np.array([np.cos(psi), np.sin(psi), 0.0])                 # outward normal
    right = np.cross(-n, np.array([0.0, 0.0, 1.0]))               # viewer looks along -n, so right = d x up
    up = np.array([0.0, 0.0, 1.0])
    centre = np.array([mx, my, 0.15])
    s = size / 2.0
    corners_w = [centre - right * s + up * s, centre + right * s + up * s,
                 centre + right * s - up * s, centre - right * s - up * s]
    px, py, th = robot
    c, s_ = np.cos(th), np.sin(th)
    pixels = []
    for p in corners_w:
        dx, dy = p[0] - px, p[1] - py
        p_robot = np.array([c * dx + s_ * dy, -s_ * dx + c * dy, p[2]])
        p_cv = R_CV_TO_ROBOT.T @ (p_robot - CAM_POS)
        assert p_cv[2] > 0, "marker behind the camera"
        pixels.append([K[0, 0] * p_cv[0] / p_cv[2] + K[0, 2], K[1, 1] * p_cv[1] / p_cv[2] + K[1, 2]])
    return np.array(pixels)


def test_recovers_h_marker_pose():
    marker = (1.85, 0.0, np.pi)      # marker on the +x wall facing -x
    size = 0.12
    for robot in [(1.0, 0.0, 0.0), (0.4, 0.3, -0.2), (1.2, -0.4, 0.3), (0.8, 0.5, -0.5)]:
        corners = project_marker(robot, marker, size)
        est = estimate_marker_pose(corners, K, size, CAM_POS)
        truth = h_marker_pose(np.array(robot), marker)
        err = est - truth
        err[2] = wrap(err[2])
        assert np.allclose(err, 0.0, atol=1e-6), (robot, est, truth)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
