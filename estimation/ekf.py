"""EKF for the Track A robot: wheel-encoder odometry for prediction, ArUco marker pose for update.

Two variants share this class:
  A: pose state [px, py, theta]; odometry scale factors are fixed, taken from a calibration.
  B: the two odometry scale factors are added to the state and estimated online.

Odometry: v_nom, w_nom come from the encoders with the nominal wheel radius and track width; the filter uses
v = s_v * v_nom and w = s_w * w_nom. The scale factors absorb radius, slip and effective-track errors.
Marker measurement z = [marker x, marker y in the robot frame, marker normal direction relative to the heading],
the quantities returned by perception.marker_pose.estimate_marker_pose.
"""

from dataclasses import dataclass

import numpy as np

from estimation.models import (F_unicycle, G_unicycle, H_marker_pose, encoders_to_vw, f_unicycle,
                               h_marker_pose, wrap)


@dataclass
class OdometryNoise:
    """Input noise std = floor + relative * |input|: slip and calibration residual grow with speed."""
    v_floor: float = 0.002     # [m/s]
    v_rel: float = 0.05
    w_floor: float = 0.01      # [rad/s]
    w_rel: float = 0.08


def marker_measurement_cov(distance, focal_px=530.5, marker_size=0.12, corner_sigma_px=0.5,
                           pos_floor=0.001, ang_floor=0.005, ang_gain=0.05):
    """Measurement covariance that grows with marker distance.

    Position: depth error from the apparent marker size, sigma = d^2 * corner_sigma / (f * L), plus a floor.
    Heading: floor + gain * d^2 [rad]. Constants are set from the render study in docs/findings.md.
    """
    sigma_pos = np.sqrt(pos_floor**2 + (distance**2 * corner_sigma_px / (focal_px * marker_size)) ** 2)
    sigma_ang = ang_floor + ang_gain * distance**2
    return np.diag([sigma_pos**2, sigma_pos**2, sigma_ang**2])


class MarkerEKF:
    def __init__(self, x0, P0, marker, wheel_radius=0.033, track_nominal=0.288, scales=(1.0, 1.0),
                 estimate_scales=False, scale_sigma0=0.2, scale_step_sigma=1e-3, noise=None, gate=None):
        self.marker = np.asarray(marker, dtype=float)
        self.r, self.b = wheel_radius, track_nominal
        self.estimate_scales = estimate_scales
        self.noise = noise or OdometryNoise()
        self.gate = gate                      # chi-square threshold on the innovation, None disables gating
        self.scale_step_sigma = scale_step_sigma
        self.fixed_scales = np.asarray(scales, dtype=float)
        if estimate_scales:
            self.x = np.concatenate([np.asarray(x0, dtype=float), self.fixed_scales])
            self.P = np.zeros((5, 5))
            self.P[:3, :3] = P0
            self.P[3:, 3:] = np.eye(2) * scale_sigma0**2
        else:
            self.x = np.asarray(x0, dtype=float).copy()
            self.P = np.array(P0, dtype=float)

    @property
    def scales(self):
        return self.x[3:5] if self.estimate_scales else self.fixed_scales

    @property
    def pose(self):
        return self.x[:3]

    @property
    def pose_cov(self):
        return self.P[:3, :3]

    def predict(self, w_left, w_right, dt):
        v_nom, w_nom = encoders_to_vw(w_left, w_right, self.r, self.b)
        s_v, s_w = self.scales
        u = (s_v * v_nom, s_w * w_nom)
        pose = self.x[:3]
        F3 = F_unicycle(pose, u, dt)
        G = G_unicycle(pose, u, dt)
        n = self.noise
        Qu = np.diag([(n.v_floor + n.v_rel * abs(u[0])) ** 2, (n.w_floor + n.w_rel * abs(u[1])) ** 2])
        Q3 = G @ Qu @ G.T
        if self.estimate_scales:
            F = np.eye(5)
            F[:3, :3] = F3
            F[:3, 3] = G[:, 0] * v_nom
            F[:3, 4] = G[:, 1] * w_nom
            Q = np.zeros((5, 5))
            Q[:3, :3] = Q3
            Q[3:, 3:] = np.eye(2) * self.scale_step_sigma**2
        else:
            F, Q = F3, Q3
        self.x[:3] = f_unicycle(pose, u, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z, R=None):
        """Marker pose update. Returns True if the measurement was used."""
        pose = self.x[:3]
        zhat = h_marker_pose(pose, self.marker)
        H3 = H_marker_pose(pose, self.marker)
        n = len(self.x)
        H = np.zeros((3, n))
        H[:, :3] = H3
        y = np.asarray(z, dtype=float) - zhat
        y[2] = wrap(y[2])
        if R is None:
            R = marker_measurement_cov(np.hypot(z[0], z[1]))
        S = H @ self.P @ H.T + R
        if self.gate is not None and float(y @ np.linalg.solve(S, y)) > self.gate:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        IKH = np.eye(n) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T
        return True
