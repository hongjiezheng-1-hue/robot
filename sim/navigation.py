"""Closed-loop navigation through unknown obstacles, then docking, on the MuJoCo robot.

In short: the map is built from the estimated pose, and the robot plans again when a scan shows the path is blocked.
Result (findings stage 11): 126 of 126 runs docked on 7 layouts; 26 of 30 with disturbances.

Sensing: wheel encoders and the camera feed the EKF; the 2D LiDAR builds an occupancy grid from the ESTIMATED pose. The planner
(A* on the inflated grid), the LQR tracker and the docking logic only use the estimate and the grid. The obstacles are not given:
unknown cells are planned as free, and the path is re-planned whenever a scan shows that it is blocked. The truth is read from the
simulator only to place the simulated LiDAR beams (the world generating the measurement) and to log it for evaluation.

Phases: plan -> rotate -> follow -> (re-plan ...) -> align -> dock -> hold. The stand-off and dock poses come from the marker pose.
"""

import numpy as np

from control.docking import DockSpec, dock_error, dock_poses
from control.lqr_tracker import to_nominal, wheel_speeds
from control.planner import path_blocked, plan_with_margins
from control.reference import build_reference
from estimation.ekf import MarkerEKF, MarkerSLAM
from estimation.models import wrap
from perception.lidar import LidarConfig, scan_mujoco
from perception.occupancy import OccupancyGrid
from sim.closed_loop import P0_START

MODES = ("plan", "rotate", "follow", "align", "dock", "hold", "failed", "search", "look")


def _valid_reference(grid, ref, radius, skip=0.35):
    blocked = grid.inflated(radius)
    ix, iy = grid.to_index(ref["x"], ref["y"])
    ok = grid.inside(ix, iy)
    far = np.hypot(ref["x"] - ref["x"][0], ref["y"] - ref["y"][0]) >= skip       # the stretch under the robot cannot be fixed
    return bool(ok.all() and not blocked[ix, iy][far].any())


def make_reference(grid, start_xy, waypoints, speed, dt, hard, fillets=(0.3, 0.2, 0.12, 0.05)):
    """Reference with the largest corner radius whose rounded path keeps the hard clearance to the observed obstacles."""
    ref = None
    for r in fillets:
        ref = build_reference(start_xy, waypoints, speed, r, dt)
        if _valid_reference(grid, ref, hard):
            break
    return ref


def contact_ids(model, base_body):
    import mujoco

    robot = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == model.body_rootid[base_body]}
    names = {g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)}
    env = {g for g, n in names.items() if n.startswith(("obstacle", "wall", "station"))}
    return robot, env


def run_navigation(drv, scene_cfg, standoff, dock, tracker, start_pose, belief0, gain_fn=None, estimate_scales=False, compensate=True,
                   lidar_cfg=None, speed=0.10, dock_speed=0.05, inflate=0.22, hard=0.17, max_seconds=240.0, hold_seconds=10.0, max_replans=40, map_reset_jump=0.10,
                   meas_cov=None, noise=None, subpixel=True, seed=0, wheel_radius=0.033, track_nominal=0.288, grid_kwargs=None, on_step=None, marker_dropout=0.0, push=None, max_recoveries=3, slam_prior=None, dock_spec=None):
    """slam_prior: None when the marker pose is known; otherwise (rough marker pose, its covariance). Then the marker pose is
    estimated (EKF-SLAM) and the stand-off and dock poses follow the estimate instead of the given standoff and dock."""
    import mujoco

    from drivers.mujoco_tb3 import TICKS_PER_REV
    from perception.aruco import detect_markers
    from perception.marker_pose import camera_intrinsics, estimate_marker_pose

    lidar_cfg = lidar_cfg or LidarConfig()
    dt, lidar_every = 0.1, 2                                       # LiDAR at 5 Hz
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
        """(stand-off, dock): fixed when the marker is known, otherwise from the current marker estimate."""
        return (standoff, dock) if slam_prior is None else dock_poses(ekf.marker, spec)
    rng = np.random.default_rng(seed)
    drop_rng = np.random.default_rng(seed + 12345)
    grid = OccupancyGrid(**(grid_kwargs or {}))

    n_max = int(round(max_seconds / dt))
    keys = (("truth", 3), ("est_pose", 3), ("est_cov", 9), ("cmd", 2), ("marker", 3), ("wheel_omega", 2), ("ref", 3))
    log = {k: np.full((n_max, d), np.nan) for k, d in keys}
    log["mode"] = np.zeros(n_max, dtype=int)
    log["contact"] = np.zeros(n_max, dtype=bool)
    map_resets, force_scan, recoveries = [], False, 0
    search_start, last_update_step = 0, -10**9
    look_start, looked, dock_length = 0, False, 0.0
    scans, replans = [], []                                        # (step, estimated pose, ranges), (step, waypoints)
    standoff_xy = np.array(goal()[0][:2])

    mode, ref, k, waypoints, hold_left, n_done = "plan", None, 0, None, int(hold_seconds / dt), 0
    last_ranges = None
    ticks_prev = drv.ticks()
    for step in range(n_max):
        if step % lidar_every == 0 or force_scan:
            force_scan = False
            ranges = scan_mujoco(drv.model, drv.data, lidar_cfg, base_id, drv.true_pose(), rng)
            if mode not in ("dock", "hold", "search"):
                grid.update(ekf.pose, ranges, lidar_cfg)           # mapped at the estimated pose
            scans.append((step, ekf.pose.copy(), ranges))
            last_ranges = ranges
            if mode == "follow" and k < len(ref["x"]) - 1:
                rest = np.stack([ref["x"][k::3], ref["y"][k::3]], axis=1)
                if len(rest) >= 2 and path_blocked(grid, [tuple(p) for p in rest], hard):
                    mode = "plan"
                elif np.linalg.norm(np.array(goal()[0][:2]) - standoff_xy) > 0.15:
                    mode = "plan"                                  # the marker estimate moved the goal: plan again

        v = w = 0.0
        if mode == "plan":
            if step >= 2 * lidar_every:                            # a couple of scans first
                standoff_xy = np.array(goal()[0][:2])
                waypoints = plan_with_margins(grid, tuple(ekf.pose[:2]), tuple(standoff_xy), (inflate, 0.19, hard), relax_radius=hard)
                since = map_resets[-1] if map_resets else -1
                thrash = sum(1 for r in replans if r[0] > since and step - r[0] < 100) >= 12  # many plans within 10 s: map probably corrupted
                if (waypoints is None or thrash) and recoveries < max_recoveries:
                    # recovery: the pose estimate (and so the map) may be wrong, for example after a large start offset or a push the
                    # encoders cannot see. Turn towards where the marker should be, wait for a marker update to correct the pose,
                    # then forget the map and plan again.
                    recoveries += 1
                    search_start, mode, waypoints = step, "search", None
                elif waypoints is None or len(replans) >= max_replans:
                    mode = "failed"
                elif len(waypoints) == 0:                          # already at the stand-off cell
                    target_heading, mode, n_done = goal()[1][2], "align", 0
                else:
                    replans.append((step, [tuple(ekf.pose[:2])] + waypoints))
                    first = np.arctan2(waypoints[0][1] - ekf.pose[1], waypoints[0][0] - ekf.pose[0])
                    target_heading, mode, n_done = first, "rotate", 0
        elif mode == "search":
            bearing = np.arctan2(ekf.marker[1] - ekf.pose[1], ekf.marker[0] - ekf.pose[0])   # where the marker should be
            err = wrap(bearing - ekf.pose[2])
            w = float(np.clip(1.5 * err, -0.5, 0.5)) if abs(err) > np.radians(3.0) else 0.0
            corrected = last_update_step >= search_start and step - last_update_step >= 10     # marker seen, estimate settled
            if corrected or step - search_start > 100:
                grid = OccupancyGrid(**(grid_kwargs or {}))
                map_resets.append(step)
                force_scan, mode, w = True, "plan", 0.0
        elif mode == "look":
            # marker position not well known yet: sweep the camera around where the marker should be until it is seen
            bearing = np.arctan2(ekf.marker[1] - ekf.pose[1], ekf.marker[0] - ekf.pose[0])
            sweep = 0.8 * np.sin(2 * np.pi * (step - look_start) * dt / 12.0)
            err = wrap(bearing + sweep - ekf.pose[2])
            w = float(np.clip(1.5 * err, -0.5, 0.5))
            if last_update_step >= look_start and step - last_update_step >= 10:
                mode, w = "plan", 0.0                              # marker found: plan to the corrected stand-off
            elif step - look_start > 250:
                target_heading, mode, n_done, w = goal()[1][2], "align", 0, 0.0     # not found in 25 s: dock on the best guess
        elif mode in ("rotate", "align"):
            err = wrap(target_heading - ekf.pose[2])
            w = float(np.clip(1.5 * err, -0.5, 0.5))
            n_done = n_done + 1 if abs(err) < np.radians(3.0) else 0
            if n_done >= 3:
                w = 0.0
                if mode == "rotate":
                    ref = make_reference(grid, tuple(ekf.pose[:2]), waypoints, speed, dt, hard)
                    mode, k = "follow", 0
                else:
                    # straight line along the dock heading, starting at the projection of the estimate onto the dock axis:
                    # a line from the estimate itself would be slanted and end with a heading error
                    dock_now = goal()[1]
                    axis = np.array([np.cos(dock_now[2]), np.sin(dock_now[2])])
                    foot = np.asarray(dock_now[:2]) + axis * float(axis @ (ekf.pose[:2] - np.asarray(dock_now[:2])))
                    ref = build_reference(tuple(foot), [tuple(dock_now[:2])], dock_speed, 0.3, dt)
                    dock_length = float(np.linalg.norm(np.asarray(dock_now[:2]) - foot))
                    mode, k = "dock", 0
        elif mode in ("follow", "dock"):
            if k < len(ref["x"]):
                r = {key: ref[key][k] for key in ("x", "y", "theta", "v", "w")}
                if mode == "dock" and slam_prior is not None:
                    # the dock pose follows the marker estimate, which keeps improving on the way in
                    d_now = goal()[1]
                    a_now = np.array([np.cos(d_now[2]), np.sin(d_now[2])])
                    p_now = np.asarray(d_now[:2]) - a_now * (dock_length - min(k * dock_speed * dt, dock_length))
                    r.update(x=p_now[0], y=p_now[1], theta=d_now[2])
                v, w, _ = tracker.command(ekf.pose, r)             # estimated pose only
                log["ref"][step] = (r["x"], r["y"], r["theta"])
                k += 1
            elif mode == "follow":
                if np.linalg.norm(ekf.pose[:2] - np.array(goal()[0][:2])) < 0.15:
                    if slam_prior is not None and np.sqrt(np.trace(ekf.marker_cov[:2, :2])) > 0.05 and not looked:
                        mode, look_start, looked = "look", step, True
                    else:
                        target_heading, mode, n_done = goal()[1][2], "align", 0
                else:
                    mode = "plan"
            else:
                mode = "hold"
        elif mode == "hold":
            hold_left -= 1
        if mode == "failed":
            log["mode"][step:] = MODES.index("failed")
            n_max = step + 1
            break

        vn, wn = to_nominal(v, w, gain_fn if compensate else None)
        wl, wr = wheel_speeds(vn, wn, wheel_radius, track_nominal)
        hit = False
        pushing = bool(push) and push["time"] <= step * dt < push["time"] + push["duration"]
        if pushing:                                                # disturbance: a lateral force the encoders cannot see
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
            c = corners[list(ids.flatten()).index(scene_cfg.marker_id)].reshape(4, 2)
            marker_corners = c
            z = estimate_marker_pose(c, K, scene_cfg.marker_size, scene_cfg.camera_pos)
            if z is not None:
                before = ekf.pose[:2].copy()
                ekf.update(z)
                last_update_step = step
                log["marker"][step] = z
                if np.linalg.norm(ekf.pose[:2] - before) > map_reset_jump and mode in ("plan", "rotate", "follow"):
                    # the map was built with a pose that has just been corrected: it is inconsistent with the new estimate
                    grid = OccupancyGrid(**(grid_kwargs or {}))
                    map_resets.append(step)
                    force_scan = True
                    mode = "plan"
        log["truth"][step] = drv.true_pose()                       # logged for evaluation only
        log["est_pose"][step], log["est_cov"][step], log["wheel_omega"][step] = ekf.pose, ekf.pose_cov.ravel(), omega
        log["cmd"][step], log["mode"][step], log["contact"][step] = (v, w), MODES.index(mode), hit
        if on_step is not None and on_step(dict(step=step, dt=dt, mode=mode, est=ekf.pose.copy(), ranges=last_ranges, plan=waypoints,
                                                    image=image, marker_corners=marker_corners, grid=grid,
                                                    marker_est=ekf.marker.copy())):
            n_max = step + 1                                       # the caller asked to stop (for example the viewer was closed)
            break
        if mode == "hold" and hold_left <= 0:
            n_max = step + 1
            break

    out = {key: val[:n_max] for key, val in log.items()}
    out["est_cov"] = out["est_cov"].reshape(n_max, 3, 3)
    out.update(dt=dt, pose0=pose0, marker_world=marker_world, standoff=np.array(goal()[0]), dock=np.array(goal()[1]), marker_est=np.array(ekf.marker), slam=slam_prior is not None,
               logodds=grid.logodds, grid_origin=np.array([grid.x0, grid.y0, grid.res]),
               scan_steps=np.array([s[0] for s in scans]), scan_poses=np.array([s[1] for s in scans]),
               scan_ranges=np.array([s[2] for s in scans]), replan_steps=np.array([r[0] for r in replans]), replan_len=np.array([len(r[1]) for r in replans], dtype=int),
               replan_xy=np.array([pt for r in replans for pt in r[1]]).reshape(-1, 2),
               failed=(MODES[out["mode"][-1]] == "failed"), n_replans=len(replans), n_recoveries=recoveries, n_steps=n_max, map_reset_steps=np.array(map_resets, dtype=int))
    return out


def slam_prior_from_error(marker_true, error):
    """Rough marker guess (as from a floor plan) that is `error` metres off, partly across and partly along the marker normal,
    and 10 degrees off per 0.3 m, with a deliberately large uncertainty (0.5 m, 20 degrees)."""
    mx, my, psi = marker_true
    n = np.array([np.cos(psi), np.sin(psi)])
    t = np.array([-n[1], n[0]])
    guess = np.array([mx, my, psi]) + np.concatenate([error * (0.5 * n + 0.87 * t), [np.radians(10.0) * error / 0.3]])
    return guess, np.diag([0.5**2, 0.5**2, np.radians(20.0) ** 2])


def box_clearance(truth_xy, boxes, wall_half):
    """Smallest distance from the true robot centre to any obstacle box or room wall over the run (evaluation only)."""
    best = np.inf
    for cx, cy, sx, sy in boxes:
        dx = np.maximum(np.abs(truth_xy[:, 0] - cx) - sx / 2, 0.0)
        dy = np.maximum(np.abs(truth_xy[:, 1] - cy) - sy / 2, 0.0)
        best = min(best, float(np.hypot(dx, dy).min()))
    wx, wy = wall_half
    best = min(best, float((wx - np.abs(truth_xy[:, 0])).min()), float((wy - np.abs(truth_xy[:, 1])).min()))
    return best


def navigation_metrics(log, scene_cfg, spec, dock_pose):
    from control.docking import docked

    truth = log["truth"]
    along, lateral, heading = dock_error(truth[-1], dock_pose)
    est_err = np.linalg.norm(truth[:, :2] - log["est_pose"][:, :2], axis=1)
    moving = np.arange(len(truth))[np.abs(log["cmd"]).sum(axis=1) > 0]
    return dict(failed=bool(log["failed"]), replans=int(log["n_replans"]), seconds=float(len(truth) * log["dt"]),
                collided=bool(log["contact"].any()), contact_steps=int(log["contact"].sum()),
                clearance=box_clearance(truth[:, :2], scene_cfg.obstacles, (scene_cfg.room_x / 2, scene_cfg.room_y / 2)),
                est_pos_rmse=float(np.sqrt(np.mean(est_err**2))), est_pos_max=float(est_err.max()),
                dock_along=along, dock_lateral=lateral, dock_heading_deg=heading,
                docked=bool((not log["failed"]) and docked(truth[-1], dock_pose, spec)),
                moving_steps=int(len(moving)), recoveries=int(log.get("n_recoveries", 0)),
                first_contact_s=float(np.argmax(log["contact"]) * log["dt"]) if log["contact"].any() else None)
