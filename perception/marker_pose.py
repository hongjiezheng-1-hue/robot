"""Marker pose from detected ArUco corners, expressed in the robot frame.

Frames: robot frame is x forward, y left, z up, origin at the wheel-axle centre (the unicycle reference point).
OpenCV camera frame is x right, y down, z forward. The camera is assumed to look along the robot x axis; its
position in the robot frame is a configuration value (extrinsics, to be replaced by the real camera mounting).

The result matches estimation.models.h_marker_pose: [marker x in robot frame, marker y in robot frame,
direction of the marker outward normal relative to the robot heading].
"""

import cv2
import numpy as np

R_CV_TO_ROBOT = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])   # robot = R_CV_TO_ROBOT @ camera_cv


def camera_intrinsics(fovy_deg, width, height):
    """Pinhole intrinsics of a camera with square pixels and a centred principal point."""
    fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def estimate_marker_pose(corners, K, marker_size, cam_pos_robot, dist_coeffs=None):
    """corners: (4, 2) pixel corners ordered top-left, top-right, bottom-right, bottom-left as seen in the image."""
    s = marker_size / 2.0
    obj = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    # A planar target has two mirror-image pose solutions. Refine both from IPPE and keep the lower reprojection error;
    # with noisy corners at long range the wrong one can still win (see docs/findings.md).
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    for rvec0, tvec0 in zip(rvecs if count else [], tvecs if count else []):
        candidates.append(cv2.solvePnPRefineLM(obj, img, K, dist_coeffs, rvec0.copy(), tvec0.copy()))
    ok, rvec_it, tvec_it = cv2.solvePnP(obj, img, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        candidates.append((rvec_it, tvec_it))
    if not candidates:
        return None

    def rmse(rvec, tvec):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist_coeffs)
        return float(np.sqrt(np.mean((projected.reshape(4, 2) - img) ** 2)))

    rvec, tvec = min(candidates, key=lambda c: rmse(*c))
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]
    if normal[2] > 0:           # the outward normal faces the camera, i.e. points against the viewing direction (+z)
        normal = -normal
    position = R_CV_TO_ROBOT @ tvec.reshape(3) + np.asarray(cam_pos_robot, dtype=np.float64)
    normal_robot = R_CV_TO_ROBOT @ normal
    return np.array([position[0], position[1], np.arctan2(normal_robot[1], normal_robot[0])])
