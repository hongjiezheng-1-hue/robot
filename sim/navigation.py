"""Closed-loop navigation through unknown obstacles, then docking, on the MuJoCo robot.

Sensing: wheel encoders and the camera feed the EKF; the 2D LiDAR builds an occupancy grid from the ESTIMATED pose. The planner
(A* on the inflated grid), the LQR tracker and the docking logic only use the estimate and the grid. The obstacles are not given:
unknown cells are planned as free, and the path is re-planned whenever a scan shows that it is blocked. The truth is read from the
simulator only to place the simulated LiDAR beams (the world generating the measurement) and to log it for evaluation.

Phases: plan -> rotate -> follow -> (re-plan ...) -> align -> dock -> hold. The stand-off and dock poses come from the marker pose.
"""

import numpy as np

from control.docking import dock_error
from control.lqr_tracker import to_nominal, wheel_speeds
from control.planner import path_blocked, plan_with_margins
from control.reference import build_reference
from estimation.ekf import MarkerEKF
from estimation.models import wrap
from perception.lidar import LidarConfig, scan_mujoco
from perception.occupancy import OccupancyGrid
from sim.closed_loop import P0_START

MODES = ("plan", "rotate", "follow", "align", "dock", "hold", "failed")


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
                   meas_cov=None, noise=None, subpixel=True, seed=0, wheel_radius=0.033, track_nominal=0.288, grid_kwargs=None):
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
    ekf = MarkerEKF(np.asarray(belief0, dtype=float), P0_START, marker_world, wheel_radius, track_nominal,
                    estimate_scales=estimate_scales, gain_fn=gain_fn, meas_cov=meas_cov, noise=noise)
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid(**(grid_kwargs or {}))

    n_max = int(round(max_seconds / dt))
    keys = (("truth", 3), ("est_pose", 3), ("est_cov", 9), ("cmd", 2), ("marker", 3), ("wheel_omega", 2), ("ref", 3))
    log = {k: np.full((n_max, d), np.nan) for k, d in keys}
    log["mode"] = np.zeros(n_max, dtype=int)
    log["contact"] = np.zeros(n_max, dtype=bool)
    map_resets, force_scan = [], False
    scans, replans = [], []                                        # (step, estimated pose, ranges), (step, waypoints)
    standoff_xy = np.array(standoff[:2])

    mode, ref, k, waypoints, hold_left, n_done = "plan", None, 0, None, int(hold_seconds / dt), 0
    ticks_prev = drv.ticks()
    for step in range(n_max):
        if step % lidar_every == 0 or force_scan:
            force_scan = False
            ranges = scan_mujoco(drv.model, drv.data, lidar_cfg, base_id, drv.true_pose(), rng)
            if mode not in ("dock", "hold"):
                grid.update(ekf.pose, ranges, lidar_cfg)           # mapped at the estimated pose
            scans.append((step, ekf.pose.copy(), ranges))
            if mode == "follow" and k < len(ref["x"]) - 1:
                rest = np.stack([ref["x"][k::3], ref["y"][k::3]], axis=1)
                if len(rest) >= 2 and path_blocked(grid, [tuple(p) for p in rest], hard):
                    mode = "plan"

        v = w = 0.0
        if mode == "plan":
            if step >= 2 * lidar_every:                            # a couple of scans first
                waypoints = plan_with_margins(grid, tuple(ekf.pose[:2]), tuple(standoff_xy), (inflate, 0.19, hard), relax_radius=hard)
                if waypoints is None or len(replans) >= max_replans:
                    mode = "failed"
                else:
                    replans.append((step, [tuple(ekf.pose[:2])] + waypoints))
                    first = np.arctan2(waypoints[0][1] - ekf.pose[1], waypoints[0][0] - ekf.pose[0])
                    target_heading, mode, n_done = first, "rotate", 0
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
                    axis = np.array([np.cos(dock[2]), np.sin(dock[2])])
                    foot = np.asarray(dock[:2]) + axis * float(axis @ (ekf.pose[:2] - np.asarray(dock[:2])))
                    ref = build_reference(tuple(foot), [tuple(dock[:2])], dock_speed, 0.3, dt)
                    mode, k = "dock", 0
        elif mode in ("follow", "dock"):
            if k < len(ref["x"]):
                r = {key: ref[key][k] for key in ("x", "y", "theta", "v", "w")}
                v, w, _ = tracker.command(ekf.pose, r)             # estimated pose only
                log["ref"][step] = (r["x"], r["y"], r["theta"])
                k += 1
            elif mode == "follow":
                if np.linalg.norm(ekf.pose[:2] - standoff_xy) < 0.15:
                    target_heading, mode, n_done = dock[2], "align", 0
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
        for _ in range(sub):
            drv.step(wl, wr)
            for c in drv.data.contact[:drv.data.ncon]:
                if (c.geom1 in robot_geoms and c.geom2 in env_geoms) or (c.geom2 in robot_geoms and c.geom1 in env_geoms):
                    hit = True
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
                before = ekf.pose[:2].copy()
                ekf.update(z)
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
        if mode == "hold" and hold_left <= 0:
            n_max = step + 1
            break

    out = {key: val[:n_max] for key, val in log.items()}
    out["est_cov"] = out["est_cov"].reshape(n_max, 3, 3)
    out.update(dt=dt, pose0=pose0, marker_world=marker_world, standoff=np.array(standoff), dock=np.array(dock),
               logodds=grid.logodds, grid_origin=np.array([grid.x0, grid.y0, grid.res]),
               scan_steps=np.array([s[0] for s in scans]), scan_poses=np.array([s[1] for s in scans]),
               scan_ranges=np.array([s[2] for s in scans]), replan_steps=np.array([r[0] for r in replans]),
               failed=(MODES[out["mode"][-1]] == "failed"), n_replans=len(replans), n_steps=n_max, map_reset_steps=np.array(map_resets, dtype=int))
    return out


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
                moving_steps=int(len(moving)))
