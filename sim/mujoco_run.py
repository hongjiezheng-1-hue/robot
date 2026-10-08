"""Data collection on the MuJoCo robot for EKF validation, plus a gain calibration run.

The waypoint follower below reads the TRUE pose. It exists only to generate a motion for the data log and is not
the project's controller; the estimator replays the encoder and camera data without ever seeing the truth.
"""

from dataclasses import dataclass

import numpy as np

from estimation.models import f_unicycle, h_marker_pose, wrap


@dataclass
class RunConfig:
    dt: float = 0.1
    duration: float = 240.0
    wheel_radius: float = 0.033
    track_nominal: float = 0.288
    start: tuple = (-1.5, -0.08, 0.0)
    # Lane along y = -0.08 faces the marker (+x) and runs between the two obstacles; then a loop around the room.
    waypoints: tuple = ((1.3, -0.08), (1.3, 1.1), (-1.5, 1.1), (-1.5, -0.08),
                        (1.3, -0.08), (1.3, 1.1), (-1.5, 1.1), (-1.5, -0.08))
    speeds: tuple = (0.05, 0.10, 0.15, 0.10, 0.15, 0.05, 0.10, 0.15)   # nominal speed on the segment that ends at each waypoint
    arrive_radius: float = 0.15


def follower_step(pose, wp, cfg):
    """Waypoint follower: returns the (possibly advanced) waypoint index and the nominal v, w command."""
    x, y, yaw = pose
    dx, dy = cfg.waypoints[wp][0] - x, cfg.waypoints[wp][1] - y
    if np.hypot(dx, dy) < cfg.arrive_radius:
        wp = (wp + 1) % len(cfg.waypoints)
        dx, dy = cfg.waypoints[wp][0] - x, cfg.waypoints[wp][1] - y
    ang_err = wrap(np.arctan2(dy, dx) - yaw)
    v = cfg.speeds[wp] if abs(ang_err) < 0.5 else 0.02
    w = float(np.clip(1.5 * ang_err, -0.5, 0.5))
    return wp, v, w


def simulate_follower_path(cfg, s_v=0.86, s_w=0.72, duration=None):
    """Kinematic prediction of the followed path with the measured odometry gains (turns are slower than commanded)."""
    pose = np.array(cfg.start, dtype=float)
    wp, path = 0, []
    for _ in range(int(round((duration or cfg.duration) / cfg.dt))):
        wp, v, w = follower_step(pose, wp, cfg)
        pose = f_unicycle(pose, (s_v * v, s_w * w), cfg.dt)
        path.append(pose.copy())
    return np.array(path)


def _box_distance(px, py, cx, cy, sx, sy):
    dx = max(abs(px - cx) - sx / 2, 0.0)
    dy = max(abs(py - cy) - sy / 2, 0.0)
    return float(np.hypot(dx, dy))


def min_clearance(truth, scene_cfg):
    """Smallest distance from the robot centre to an obstacle, the station base or a wall along the path."""
    best = np.inf
    for px, py in truth[:, :2]:
        for (cx, cy, sx, sy) in scene_cfg.obstacles:
            best = min(best, _box_distance(px, py, cx, cy, sx, sy))
        best = min(best, _box_distance(px, py, scene_cfg.station_x, scene_cfg.station_y, 0.16, 0.28))
        best = min(best, scene_cfg.room_x / 2 - abs(px), scene_cfg.room_y / 2 - abs(py))
    return float(best)


def marker_possibly_visible(pose, marker_world, max_range=4.0, half_fov=np.radians(45.0)):
    """Cheap geometric pre-check used only to skip rendering frames in which the marker cannot be in view."""
    z = h_marker_pose(np.asarray(pose, dtype=float), marker_world)
    return z[0] > 0 and np.hypot(z[0], z[1]) < max_range and abs(np.arctan2(z[1], z[0])) < half_fov


def collect_run(drv, scene_cfg, cfg):
    import mujoco

    from drivers.mujoco_tb3 import TICKS_PER_REV
    from perception.aruco import detect_markers
    from perception.marker_pose import camera_intrinsics, estimate_marker_pose

    renderer = mujoco.Renderer(drv.model, height=scene_cfg.camera_height, width=scene_cfg.camera_width)
    K = camera_intrinsics(scene_cfg.camera_fovy, scene_cfg.camera_width, scene_cfg.camera_height)
    marker_world = np.array([scene_cfg.station_x, scene_cfg.station_y, scene_cfg.station_yaw])
    sub = int(round(cfg.dt / drv.control_dt))
    n = int(round(cfg.duration / cfg.dt))
    drv.reset(*cfg.start)
    for _ in range(int(0.5 / drv.control_dt)):
        drv.step(0.0, 0.0)
    pose0 = np.array(drv.true_pose())
    ticks = [drv.ticks()]
    truth = np.zeros((n, 3))
    marker = np.full((n, 3), np.nan)
    wp = 0
    v_log = np.zeros(n)
    for k in range(n):
        wp, v, w = follower_step(drv.true_pose(), wp, cfg)
        v_log[k] = v
        wr = (v + w * cfg.track_nominal / 2) / cfg.wheel_radius
        wl = (v - w * cfg.track_nominal / 2) / cfg.wheel_radius
        for _ in range(sub):
            drv.step(wl, wr)
        ticks.append(drv.ticks())
        truth[k] = drv.true_pose()
        if k >= 150:
            moved = np.sum(np.linalg.norm(np.diff(truth[k - 100:k + 1, :2], axis=0), axis=1))
            if moved < 0.03 and v_log[k - 100:k + 1].mean() > 0.03:
                raise RuntimeError(f"robot appears stuck at t = {k * cfg.dt:.0f} s, pose {truth[k]}: possible collision")
        if marker_possibly_visible(truth[k], marker_world):
            renderer.update_scene(drv.data, camera=scene_cfg.camera_name)
            ids, corners = detect_markers(renderer.render(), scene_cfg.marker_dictionary)
            if ids is not None and scene_cfg.marker_id in ids.flatten():
                c = corners[list(ids.flatten()).index(scene_cfg.marker_id)].reshape(4, 2)
                z = estimate_marker_pose(c, K, scene_cfg.marker_size, scene_cfg.camera_pos)
                if z is not None:
                    marker[k] = z
    wheel_omega = np.diff(np.array(ticks), axis=0) * (2 * np.pi / TICKS_PER_REV) / cfg.dt
    return dict(dt=cfg.dt, pose0=pose0, wheel_omega=wheel_omega, truth=truth, marker=marker, marker_world=marker_world,
                wheel_radius=cfg.wheel_radius, track_nominal=cfg.track_nominal)


def calibrate_gains(drv, cfg, straight_wheel=3.0, spin_wheel=1.0, seconds=8.0):
    """Estimate the odometry gains from one straight run and one spin at a single operating point."""
    def run(wl, wr):
        drv.reset(0.0, 0.0, 0.0)
        hist = []
        for _ in range(int(seconds / drv.control_dt)):
            enc = drv.step(wl, wr)
            x, y, yaw = drv.true_pose()
            hist.append([enc.time, enc.omega[0], enc.omega[1], x, y, yaw])
        s = np.array(hist[-int(1.0 / drv.control_dt):])
        dt = s[-1, 0] - s[0, 0]
        heading = s[:, 5].mean()
        v_true = ((s[-1, 3] - s[0, 3]) * np.cos(heading) + (s[-1, 4] - s[0, 4]) * np.sin(heading)) / dt
        w_true = (np.unwrap(s[:, 5])[-1] - np.unwrap(s[:, 5])[0]) / dt
        wl_m, wr_m = s[:, 1].mean(), s[:, 2].mean()
        return v_true, w_true, cfg.wheel_radius * (wr_m + wl_m) / 2, cfg.wheel_radius * (wr_m - wl_m) / cfg.track_nominal

    v_true, _, v_nom, _ = run(straight_wheel, straight_wheel)
    _, w_true, _, w_nom = run(-spin_wheel, spin_wheel)
    return v_true / v_nom, w_true / w_nom
