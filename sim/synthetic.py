"""Synthetic plant used to validate the EKF without the simulator.

A unicycle follows waypoints around the room. The true odometry gains differ from the nominal ones, the
encoders add noise, and the marker pose is measured with distance-dependent noise only while the marker is
in view. The truth is used here to generate data and to evaluate; the filter never sees it.
"""

from dataclasses import dataclass

import numpy as np

from estimation.ekf import marker_measurement_cov
from estimation.models import f_unicycle, h_marker_pose, wrap


@dataclass
class SyntheticConfig:
    dt: float = 0.1
    duration: float = 600.0
    wheel_radius: float = 0.033
    track_nominal: float = 0.288
    s_v_true: float = 0.86                    # true odometry gains relative to the nominal model
    s_w_true: float = 0.288 / 0.40            # effective track 0.40 m instead of 0.288 m
    wheel_noise: float = 0.03                 # [rad/s] std of each measured wheel speed
    v_floor: float = 0.002                    # plant slip noise, same model as the filter's default
    v_rel: float = 0.05
    w_floor: float = 0.01
    w_rel: float = 0.08
    marker: tuple = (1.85, 0.0, np.pi)        # x, y, direction of the outward normal
    max_range: float = 3.0
    min_range: float = 0.2
    half_fov: float = np.radians(28.0)        # horizontal half field of view with a small margin
    waypoints: tuple = ((-1.5, -1.0), (1.0, -0.3), (1.0, 1.0), (-1.5, 1.0))
    start: tuple = (-1.5, -1.0, 0.3)
    outlier_rate: float = 0.0                 # probability of a heading-flipped marker measurement at range > 1.2 m


def marker_visible(pose, cfg):
    mx, my, psi = cfg.marker
    z = h_marker_pose(pose, cfg.marker)
    d = np.hypot(z[0], z[1])
    bearing = np.arctan2(z[1], z[0])
    in_front = np.cos(psi) * (pose[0] - mx) + np.sin(psi) * (pose[1] - my) > 0
    return in_front and cfg.min_range < d < cfg.max_range and abs(bearing) < cfg.half_fov


def simulate(cfg, seed):
    rng = np.random.default_rng(seed)
    n_steps = int(round(cfg.duration / cfg.dt))
    pose = np.array(cfg.start, dtype=float)
    wp = 0
    truth = np.zeros((n_steps, 3))
    wheels = np.zeros((n_steps, 2))
    markers = np.full((n_steps, 3), np.nan)
    for k in range(n_steps):
        dx, dy = cfg.waypoints[wp][0] - pose[0], cfg.waypoints[wp][1] - pose[1]
        if np.hypot(dx, dy) < 0.15:
            wp = (wp + 1) % len(cfg.waypoints)
            dx, dy = cfg.waypoints[wp][0] - pose[0], cfg.waypoints[wp][1] - pose[1]
        ang_err = wrap(np.arctan2(dy, dx) - pose[2])
        v_target = 0.1 if abs(ang_err) < 0.5 else 0.03
        w_target = float(np.clip(1.5 * ang_err, -0.6, 0.6))
        v_nom, w_nom = v_target / cfg.s_v_true, w_target / cfg.s_w_true
        wr = (v_nom + w_nom * cfg.track_nominal / 2) / cfg.wheel_radius
        wl = (v_nom - w_nom * cfg.track_nominal / 2) / cfg.wheel_radius
        v = v_target + rng.normal(0, cfg.v_floor + cfg.v_rel * abs(v_target))
        w = w_target + rng.normal(0, cfg.w_floor + cfg.w_rel * abs(w_target))
        pose = f_unicycle(pose, (v, w), cfg.dt)
        truth[k] = pose
        wheels[k] = (wl + rng.normal(0, cfg.wheel_noise), wr + rng.normal(0, cfg.wheel_noise))
        if marker_visible(pose, cfg):
            z = h_marker_pose(pose, cfg.marker)
            d = np.hypot(z[0], z[1])
            z = z + rng.multivariate_normal(np.zeros(3), marker_measurement_cov(d))
            if d > 1.2 and rng.random() < cfg.outlier_rate:
                z[2] = wrap(z[2] + rng.choice([-1, 1]) * rng.uniform(0.3, 0.8))
            markers[k] = z
    return truth, wheels, markers
