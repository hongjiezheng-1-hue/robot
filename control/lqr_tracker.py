"""LQR tracking of a unicycle reference using the estimated pose only.

In short: LQR uses only the estimated pose and solves the Riccati equation at every step.
The estimator, not the controller, limits the accuracy.

Error in the robot frame e = [e_x, e_y, e_theta] = [R(theta_hat)^T (p_ref - p_hat), theta_ref - theta_hat]. About the
reference speeds (v_r, w_r) the error dynamics are e_dot = A e + B du with du = u - u_ref (estimation.models.tracking_error_model).
The gain is recomputed from the discrete Riccati equation at every step for the current (v_r, w_r) (gain scheduling).
"""

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from estimation.models import tracking_error_model, wrap


def discretise(A, B, dt):
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def to_nominal(v, w, gain_fn, iterations=4):
    """Nominal (encoder-model) command that produces the desired true speeds, inverting the odometry gains."""
    if gain_fn is None:
        return v, w
    vn, wn = v, w
    for _ in range(iterations):
        s_v, s_w = gain_fn(vn, wn)
        vn, wn = v / s_v, w / s_w
    return vn, wn


def wheel_speeds(v_nom, w_nom, wheel_radius, track_nominal):
    return ((v_nom - w_nom * track_nominal / 2) / wheel_radius, (v_nom + w_nom * track_nominal / 2) / wheel_radius)


class LQRTracker:
    def __init__(self, dt, q_std=(0.10, 0.05, np.radians(10.0)), r_std=(0.05, 0.3), v_max=0.15, w_max=0.6):
        """q_std and r_std are the largest acceptable errors and command corrections (Bryson's rule)."""
        self.dt = dt
        self.Q = np.diag(1.0 / np.asarray(q_std) ** 2)
        self.R = np.diag(1.0 / np.asarray(r_std) ** 2)
        self.v_max, self.w_max = v_max, w_max

    def gain(self, v_ref, w_ref):
        A, B = tracking_error_model(v_ref, w_ref)
        Ad, Bd = discretise(A, B, self.dt)
        P = solve_discrete_are(Ad, Bd, self.Q, self.R)
        return np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

    @staticmethod
    def error(pose_est, ref):
        dx, dy = ref["x"] - pose_est[0], ref["y"] - pose_est[1]
        c, s = np.cos(pose_est[2]), np.sin(pose_est[2])
        return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(ref["theta"] - pose_est[2])])

    def command(self, pose_est, ref):
        """Desired true speeds (v, w) and the error used. ref is a dict with x, y, theta, v, w for this step."""
        e = self.error(pose_est, ref)
        K = self.gain(ref["v"], ref["w"])
        dv, dw = -K @ e
        v = float(np.clip(ref["v"] + dv, 0.0, self.v_max))
        w = float(np.clip(ref["w"] + dw, -self.w_max, self.w_max))
        return v, w, e
