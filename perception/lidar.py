"""2D LiDAR model: 360 beams, ranges with noise, in MuJoCo (ray casting) and in plain numpy (for tests).

The parameters follow my recollection of the TurtleBot3 LDS-01 (range 0.12 to 3.5 m, 1 degree resolution, 5 Hz, accuracy
about 1 cm below 0.5 m and 3 percent above); they have not been checked against the documentation or the real sensor. The
noise standard deviation is set to 1 percent of the range, which reads the 3 percent accuracy as three standard deviations.
The mount position is the position of the LDS mesh in the Waffle Pi XML; the height is an assumption.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LidarConfig:
    n_beams: int = 360
    range_min: float = 0.12
    range_max: float = 3.5
    mount: tuple = (-0.049, 0.0, 0.145)      # [m] in the robot base frame (x forward, y left, z up)
    sigma_floor: float = 0.01                # [m]
    sigma_rel: float = 0.01                  # fraction of the range


def beam_angles(cfg):
    """Beam directions relative to the robot heading."""
    return np.linspace(-np.pi, np.pi, cfg.n_beams, endpoint=False)


def sensor_origin(pose, cfg):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return x + c * cfg.mount[0] - s * cfg.mount[1], y + s * cfg.mount[0] + c * cfg.mount[1]


def _add_noise(ranges, cfg, rng):
    valid = np.isfinite(ranges)
    sigma = np.maximum(cfg.sigma_floor, cfg.sigma_rel * ranges[valid])
    out = ranges.copy()
    out[valid] = ranges[valid] + rng.normal(0.0, sigma)
    return out


def scan_mujoco(model, data, cfg, base_body_id, true_pose, rng):
    """Simulated scan from the true robot pose by casting rays in the MuJoCo scene. NaN means no return within the range."""
    import mujoco

    ox, oy = sensor_origin(true_pose, cfg)
    origin = np.array([ox, oy, cfg.mount[2]], dtype=np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    ranges = np.full(cfg.n_beams, np.nan)
    for i, a in enumerate(beam_angles(cfg)):
        direction = np.array([np.cos(true_pose[2] + a), np.sin(true_pose[2] + a), 0.0])
        d = mujoco.mj_ray(model, data, origin, direction, None, 1, base_body_id, geomid)
        if cfg.range_min <= d <= cfg.range_max:
            ranges[i] = d
    return _add_noise(ranges, cfg, rng)


def scan_boxes(true_pose, boxes, cfg, rng=None, noise=True):
    """The same scan against axis-aligned boxes (cx, cy, sx, sy) in plain numpy; used by the tests."""
    ox, oy = sensor_origin(true_pose, cfg)
    ang = true_pose[2] + beam_angles(cfg)
    dx, dy = np.cos(ang), np.sin(ang)
    best = np.full(cfg.n_beams, np.inf)
    for cx, cy, sx, sy in boxes:
        lo = np.array([cx - sx / 2, cy - sy / 2]) - np.array([ox, oy])
        hi = np.array([cx + sx / 2, cy + sy / 2]) - np.array([ox, oy])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1x, t2x = lo[0] / dx, hi[0] / dx
            t1y, t2y = lo[1] / dy, hi[1] / dy
        t_near = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
        t_far = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
        t_near = np.where(np.isnan(t_near), -np.inf, t_near)
        hit = (t_far >= np.maximum(t_near, 0.0)) & np.isfinite(t_far)
        t = np.where(t_near > 0, t_near, t_far)       # inside a box: the exit point
        best = np.where(hit & (t > 0), np.minimum(best, t), best)
    ranges = np.where((best >= cfg.range_min) & (best <= cfg.range_max), best, np.nan)
    return _add_noise(ranges, cfg, rng) if (noise and rng is not None) else ranges
