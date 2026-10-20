"""Simple pipeline (the default): EKF + reactive LiDAR avoidance + position-based visual servoing (PBVS) to dock.

In short: the EKF estimates the pose from the wheel encoders and the marker seen by the camera. The robot drives towards
the stand-off pose in front of the marker; if the LiDAR shows something in the way, it steers to the nearest free direction.
At the stand-off it docks with the PBVS law. No map is built and no path is planned. Controller input: the estimate only.

Phases: go (to the stand-off, avoiding obstacles) -> dock (PBVS to the dock pose) -> hold.
The truth is read from the simulator only to place the simulated LiDAR beams and to log it for evaluation.
"""

import numpy as np

from control.docking import DockSpec, dock_poses
from control.lqr_tracker import to_nominal, wheel_speeds
from control.pbvs import PBVSGains, pbvs_command, polar_error
from control.reactive import BugNavigator, front_clearance, scan_points
from estimation.ekf import MarkerEKF, MarkerSLAM
from estimation.models import wrap
from perception.lidar import LidarConfig, beam_angles, scan_mujoco
from sim.closed_loop import P0_START
from sim.navigation import contact_ids

MODES = ("go", "dock", "hold", "failed", "turn")


def run_simple(drv, scene_cfg, standoff, dock, start_pose, belief0, gain_fn=None, estimate_scales=False, compensate=True,
               lidar_cfg=None, go_speed=0.10, dock_gains=None, max_seconds=240.0, hold_seconds=10.0, half_width=0.24,
               meas_cov=None, noise=None, subpixel=True, seed=0, wheel_radius=0.033, track_nominal=0.288, on_step=None,
               marker_dropout=0.0, push=None, slam_prior=None, dock_spec=None):
    import mujoco

    from drivers.mujoco_tb3 import TICKS_PER_REV
    from perception.aruco import detect_markers
    from perception.marker_pose import camera_intrinsics, estimate_marker_pose

    lidar_cfg = lidar_cfg or LidarConfig()
    dock_gains = dock_gains or PBVSGains(v_max=0.05)
    approach_gains = PBVSGains(v_max=go_speed)
    angles = beam_angles(lidar_cfg)
    dt, lidar_every = 0.1, 2
    sub = int(round(dt / drv.control_dt))
    renderer = mujoco.Renderer(drv.model, height=scene_cfg.camera_height, width=scene_cfg.camera_width)
    K = camera_intrinsics(scene_cfg.camera_fovy, scene_cfg.camera_width, scene_cfg.camera_height)
    marker_world = np.array([scene_cfg.station_x, scene_cfg.station_y, scene_cfg.station_yaw])
    base_id = drv.model.body("base").id
    robot_geoms, env_geoms = contact_ids(drv.model, base_id)

    drv.reset(*start_pose)
    for _ in range(int(0.5 / drv.control_dt)):
        drv.step(0.0, 0.0)
    pose0 = np.array(drv.true_pose())
    if slam_prior is None:
        ekf = MarkerEKF(np.asarray(belief0, dtype=float), P0_START, marker_world, wheel_radius, track_nominal,
                        estimate_scales=estimate_scales, gain_fn=gain_fn, meas_cov=meas_cov, noise=noise)
    else:
        ekf = MarkerSLAM(np.asarray(belief0, dtype=float), P0_START, slam_prior[0], slam_prior[1], wheel_radius=wheel_radius,
                         track_nominal=track_nominal, estimate_scales=estimate_scales, gain_fn=gain_fn, meas_cov=meas_cov, noise=noise)
    spec = dock_spec or DockSpec()

    def goal():
        return (standoff, dock) if slam_prior is None else dock_poses(ekf.marker, spec)

    rng = np.random.default_rng(seed)
    drop_rng = np.random.default_rng(seed + 12345)
    n_max = int(round(max_seconds / dt))
    keys = (("truth", 3), ("est_pose", 3), ("est_cov", 9), ("cmd", 2), ("marker", 3), ("wheel_omega", 2), ("ref", 3))
    log = {k: np.full((n_max, d), np.nan) for k, d in keys}
    log["mode"] = np.zeros(n_max, dtype=int)
    log["contact"] = np.zeros(n_max, dtype=bool)
    scans, points, heading_choice, last_ranges = [], np.zeros((0, 2)), None, None
    navigator = BugNavigator(half_width=half_width)
    mode, hold_left = "go", int(hold_seconds / dt)
    ticks_prev = drv.ticks()
    for step in range(n_max):
        if step % lidar_every == 0:
            ranges = scan_mujoco(drv.model, drv.data, lidar_cfg, base_id, drv.true_pose(), rng)
            points, last_ranges = scan_points(ranges, angles, lidar_cfg.mount[0]), ranges
            scans.append((step, ekf.pose.copy(), ranges))

        so, dk = goal()
        v = w = 0.0
        target = None
        if mode == "go":
            rho, alpha, _ = polar_error(ekf.pose, so)
            target = so
            if rho < 0.6 and navigator._goal_free(points, alpha, rho):
                v, w = pbvs_command(ekf.pose, so, approach_gains)   # close to the stand-off with a free way: arrive facing the marker
            else:
                heading_choice = navigator.heading(points, alpha, rho)
                if heading_choice is None:
                    w = float(np.clip(1.2 * alpha, -0.5, 0.5))      # nothing free: turn on the spot towards the goal
                else:
                    w = float(np.clip(1.2 * heading_choice, -0.6, 0.6))
                    slow = min(1.0, max(0.0, (front_clearance(points) - 0.25) / 0.4))
                    v = go_speed * max(0.0, np.cos(heading_choice)) * slow
            if rho < 0.15:
                mode = "dock"                                       # near the stand-off: the docking law also fixes the heading
        elif mode == "dock":
            target = dk
            rho, _, _ = polar_error(ekf.pose, dk)
            if rho < 0.005:
                mode = "turn"                                       # position reached: finish the heading on the spot
            else:
                v, w = pbvs_command(ekf.pose, dk, dock_gains, v_min=0.02)  # visual servoing on the estimated pose
        elif mode == "turn":
            err = float(wrap(dk[2] - ekf.pose[2]))
            w = float(np.clip(1.5 * err, -0.3, 0.3))
            if abs(err) < np.radians(1.0):
                mode, w = "hold", 0.0
        elif mode == "hold":
            hold_left -= 1

        vn, wn = to_nominal(v, w, gain_fn if compensate else None)
        wl, wr = wheel_speeds(vn, wn, wheel_radius, track_nominal)
        hit = False
        pushing = bool(push) and push["time"] <= step * dt < push["time"] + push["duration"]
        if pushing:
            c_, s_ = np.cos(drv.true_pose()[2]), np.sin(drv.true_pose()[2])
            drv.data.xfrc_applied[base_id, :2] = push["force"] * np.array([-s_, c_])
        for _ in range(sub):
            drv.step(wl, wr)
            for c in drv.data.contact[:drv.data.ncon]:
                if (c.geom1 in robot_geoms and c.geom2 in env_geoms) or (c.geom2 in robot_geoms and c.geom1 in env_geoms):
                    hit = True
        if pushing:
            drv.data.xfrc_applied[base_id, :] = 0.0
        ticks = drv.ticks()
        omega = (ticks - ticks_prev) * (2 * np.pi / TICKS_PER_REV) / dt
        ticks_prev = ticks
        ekf.predict(omega[0], omega[1], dt)
        renderer.update_scene(drv.data, camera=scene_cfg.camera_name)
        image = renderer.render()
        ids, corners = detect_markers(image, scene_cfg.marker_dictionary, subpixel=subpixel)
        marker_corners = None
        if ids is not None and scene_cfg.marker_id in ids.flatten() and drop_rng.random() >= marker_dropout:
            marker_corners = corners[list(ids.flatten()).index(scene_cfg.marker_id)].reshape(4, 2)
            z = estimate_marker_pose(marker_corners, K, scene_cfg.marker_size, scene_cfg.camera_pos)
            if z is not None:
                ekf.update(z)
                log["marker"][step] = z
        log["truth"][step] = drv.true_pose()                       # logged for evaluation only
        log["est_pose"][step], log["est_cov"][step], log["wheel_omega"][step] = ekf.pose, ekf.pose_cov.ravel(), omega
        log["cmd"][step], log["mode"][step], log["contact"][step] = (v, w), MODES.index(mode), hit
        if target is not None:
            log["ref"][step] = target
        if on_step is not None and on_step(dict(step=step, dt=dt, mode=mode, est=ekf.pose.copy(), ranges=last_ranges,
                                                plan=[tuple(target[:2])] if target is not None else None, image=image,
                                                marker_corners=marker_corners, grid=None, marker_est=ekf.marker.copy())):
            n_max = step + 1
            break
        if mode == "hold" and hold_left <= 0:
            n_max = step + 1
            break

    out = {key: val[:n_max] for key, val in log.items()}
    out["est_cov"] = out["est_cov"].reshape(n_max, 3, 3)
    so, dk = goal()
    out.update(dt=dt, pose0=pose0, marker_world=marker_world, standoff=np.array(so), dock=np.array(dk),
               marker_est=np.array(ekf.marker), slam=slam_prior is not None, pipeline="simple",
               scan_steps=np.array([s[0] for s in scans]), scan_poses=np.array([s[1] for s in scans]),
               scan_ranges=np.array([s[2] for s in scans]), replan_steps=np.zeros(0, dtype=int), replan_len=np.zeros(0, dtype=int),
               replan_xy=np.zeros((0, 2)), map_reset_steps=np.zeros(0, dtype=int), logodds=np.zeros((1, 1)),
               failed=False, n_replans=0, n_recoveries=0, n_steps=n_max)
    return out
