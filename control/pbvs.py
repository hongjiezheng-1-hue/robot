"""Position-based visual servoing (PBVS) for the unicycle: drive the estimated pose to a goal pose defined by the marker.

In short: the camera (through the EKF) tells the robot where the dock pose is relative to itself. The error is written in
polar form (distance rho, angle to the goal alpha, final heading error beta) and fed back with constant gains.

Control law (Siegwart, Nourbakhsh and Scaramuzza, "Introduction to Autonomous Mobile Robots", section 3.6):
    v = k_rho * rho,    w = k_alpha * alpha + k_beta * beta
The closed loop is locally exponentially stable when k_rho > 0, k_beta < 0 and k_alpha - k_rho > 0. It is a state-feedback
controller on the ESTIMATED state; the true pose is never used.
"""

from dataclasses import dataclass

import numpy as np

from estimation.models import wrap


@dataclass
class PBVSGains:
    k_rho: float = 0.4
    k_alpha: float = 1.2
    k_beta: float = -0.3
    v_max: float = 0.10
    w_max: float = 0.6

    def stable(self):
        return self.k_rho > 0 and self.k_beta < 0 and self.k_alpha - self.k_rho > 0


def polar_error(pose, goal):
    """rho, alpha, beta of the goal pose seen from the robot pose."""
    dx, dy = goal[0] - pose[0], goal[1] - pose[1]
    rho = float(np.hypot(dx, dy))
    alpha = float(wrap(np.arctan2(dy, dx) - pose[2]))
    beta = float(wrap(goal[2] - pose[2] - alpha))
    return rho, alpha, beta


def pbvs_command(pose_est, goal, gains, v_min=0.0):
    """(v, w) that moves the estimated pose to the goal pose. Forward motion only; turns in place if the goal is behind.
    v_min keeps a small forward speed so the last centimetres are not approached ever more slowly."""
    rho, alpha, beta = polar_error(pose_est, goal)
    if abs(alpha) > np.pi / 2:                     # goal behind: turn towards it first
        return 0.0, float(np.clip(gains.k_alpha * alpha, -gains.w_max, gains.w_max))
    v = float(np.clip(max(gains.k_rho * rho, v_min), 0.0, gains.v_max))
    w = float(np.clip(gains.k_alpha * alpha + gains.k_beta * beta, -gains.w_max, gains.w_max))
    return v, w
