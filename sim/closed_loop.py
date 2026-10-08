"""Closed-loop run on the MuJoCo robot: encoders and camera feed the EKF, the LQR tracker acts on the estimate only.

The controller and the estimator never see the true pose. The truth is read from the simulator only after each step, to log it
for evaluation. The initial belief is the true start pose plus a small random error (the robot is placed on a known mark).
"""

import numpy as np

from control.lqr_tracker import to_nominal, wheel_speeds
from estimation.ekf import MarkerEKF
from estimation.models import wrap

P0_START = np.diag([0.05**2, 0.05**2, np.deg2rad(3.0) ** 2])    # belief about the start pose: placed on a known mark


def run_closed_loop(drv, scene_cfg, ref, tracker, gain_fn=None, estimate_scales=False, compensate=True, meas_cov=None, noise=None,
                    start_pose=None, belief0=None, hold_seconds=15.0, subpixel=True, seed=0, wheel_radius=0.033, track_nominal=0.288):
    """gain_fn: odometry gains by speed and yaw rate, used by the filter and, if compensate, to invert them in the command.

    start_pose: true start pose of the robot; belief0: the pose the robot believes it starts at (for example the nominal mark).
    Without belief0 the belief is the true start pose plus a small random error.
    """
    import mujoco

    from drivers.mujoco_tb3 import TICKS_PER_REV
    from perception.aruco import detect_markers
    from perception.marker_pose import camera_intrinsics, estimate_marker_pose

    dt = float(ref["t"][1] - ref["t"][0])
    sub = int(round(dt / drv.control_dt))
    renderer = mujoco.Renderer(drv.model, height=scene_cfg.camera_height, width=scene_cfg.camera_width)
    K = camera_intrinsics(scene_cfg.camera_fovy, scene_cfg.camera_width, scene_cfg.camera_height)
    marker_world = np.array([scene_cfg.station_x, scene_cfg.station_y, scene_cfg.station_yaw])

    drv.reset(*(start_pose if start_pose is not None else (ref["x"][0], ref["y"][0], ref["theta"][0])))
    for _ in range(int(0.5 / drv.control_dt)):
        drv.step(0.0, 0.0)
    pose0 = np.array(drv.true_pose())
    rng = np.random.default_rng(seed)
    x0 = np.asarray(belief0, dtype=float) if belief0 is not None else pose0 + rng.multivariate_normal(np.zeros(3), P0_START)
    ekf = MarkerEKF(x0, P0_START, marker_world, wheel_radius, track_nominal,
                    estimate_scales=estimate_scales, gain_fn=gain_fn, meas_cov=meas_cov, noise=noise)

    n_ref = len(ref["x"])
    n = n_ref + int(round(hold_seconds / dt))
    log = {k: np.full((n, d), np.nan) for k, d in (("truth", 3), ("est_pose", 3), ("est_cov", 9), ("ref", 3), ("ref_vw", 2), ("err", 3),
                                                    ("cmd", 2), ("marker", 3), ("wheel_omega", 2))}
    ticks_prev = drv.ticks()
    for k in range(n):
        if k < n_ref:
            r = {key: ref[key][k] for key in ("x", "y", "theta", "v", "w")}
            v, w, e = tracker.command(ekf.pose, r)                  # estimated pose only
            vn, wn = to_nominal(v, w, gain_fn if compensate else None)
            wl, wr = wheel_speeds(vn, wn, wheel_radius, track_nominal)
            log["ref"][k], log["ref_vw"][k], log["err"][k], log["cmd"][k] = (r["x"], r["y"], r["theta"]), (r["v"], r["w"]), e, (v, w)
        else:
            wl = wr = 0.0                                           # reference finished: stop
            log["cmd"][k] = (0.0, 0.0)
        for _ in range(sub):
            drv.step(wl, wr)
        ticks = drv.ticks()
        omega = (ticks - ticks_prev) * (2 * np.pi / TICKS_PER_REV) / dt
        ticks_prev = ticks
        ekf.predict(omega[0], omega[1], dt)
        renderer.update_scene(drv.data, camera=scene_cfg.camera_name)
        ids, corners = detect_markers(renderer.render(), scene_cfg.marker_dictionary, subpixel=subpixel)
        if ids is not None and scene_cfg.marker_id in ids.flatten():
            c = corners[list(ids.flatten()).index(scene_cfg.marker_id)].reshape(4, 2)
            z = estimate_marker_pose(c, K, scene_cfg.marker_size, scene_cfg.camera_pos)
            if z is not None:
                ekf.update(z)
                log["marker"][k] = z
        log["truth"][k] = drv.true_pose()                          # logged for evaluation only
        log["est_pose"][k], log["est_cov"][k], log["wheel_omega"][k] = ekf.pose, ekf.pose_cov.ravel(), omega
    log["est_cov"] = log["est_cov"].reshape(n, 3, 3)
    log.update(dt=dt, pose0=pose0, marker_world=marker_world, wheel_radius=wheel_radius, track_nominal=track_nominal,
               ref_end=n_ref, ref_xy=np.stack([ref["x"], ref["y"]], axis=1))
    return log


def tracking_metrics(log, dock_pose=None):
    """Path and time-indexed tracking error of the true pose, estimator error, command effort and the final dock error."""
    n_ref = int(log["ref_end"])
    truth, ref_xy = log["truth"][:n_ref], log["ref_xy"]
    time_err = np.linalg.norm(truth[:, :2] - log["ref"][:n_ref, :2], axis=1)
    path_err = np.array([np.min(np.linalg.norm(ref_xy - p, axis=1)) for p in truth[:, :2]])     # distance to the sampled reference
    est_err = np.linalg.norm(log["truth"][:, :2] - log["est_pose"][:, :2], axis=1)
    head_err = np.degrees(np.abs(wrap(log["truth"][:, 2] - log["est_pose"][:, 2])))
    m = dict(time_rmse=float(np.sqrt(np.mean(time_err**2))), time_max=float(time_err.max()),
             path_rmse=float(np.sqrt(np.mean(path_err**2))), path_max=float(path_err.max()),
             est_pos_rmse=float(np.sqrt(np.mean(est_err**2))), est_head_rmse=float(np.sqrt(np.mean(head_err**2))),
             dv_rms=float(np.sqrt(np.mean((log["cmd"][:n_ref, 0] - log["ref_vw"][:n_ref, 0]) ** 2))),
             dw_rms=float(np.sqrt(np.mean((log["cmd"][:n_ref, 1] - log["ref_vw"][:n_ref, 1]) ** 2))))
    if dock_pose is not None:
        final = log["truth"][-1]
        c, s = np.cos(dock_pose[2]), np.sin(dock_pose[2])
        d = final[:2] - np.asarray(dock_pose[:2])
        m.update(dock_along=float(c * d[0] + s * d[1]), dock_lateral=float(-s * d[0] + c * d[1]),
                 dock_heading_deg=float(np.degrees(wrap(final[2] - dock_pose[2]))))
    return m
