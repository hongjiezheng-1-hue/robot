"""Render a navigation log as a video: map built by the robot, LiDAR beams, plans, true and estimated path, robot camera, MuJoCo view.

The truth is used here only to draw the robot and to measure the error; the robot never sees it.
"""

import pathlib

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402

from estimation.models import wrap  # noqa: E402
from perception.aruco import detect_markers  # noqa: E402
from perception.lidar import LidarConfig, beam_angles, sensor_origin  # noqa: E402
from perception.occupancy import OccupancyGrid  # noqa: E402
from sim.navigation import MODES  # noqa: E402

MODE_TEXT = {"plan": "planning", "rotate": "turning towards the path", "follow": "following the path (LQR on the EKF estimate)",
             "align": "aligning with the marker normal", "dock": "docking along the marker normal", "hold": "docked",
             "failed": "no path found"}


def _triangle(pose, size=0.14):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    pts = np.array([[size, 0.0], [-0.6 * size, 0.55 * size], [-0.6 * size, -0.55 * size]])
    return pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def _ellipse(cov_xy, k=2.0):
    w, v = np.linalg.eigh(cov_xy)
    w = np.maximum(w, 0.0)
    return 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), np.degrees(np.arctan2(v[1, 1], v[0, 1]))


class _GridReplay:
    """Rebuilds the occupancy grid over time from the logged scans, estimated poses and map resets."""

    def __init__(self, log, lidar_cfg):
        self.log, self.cfg = log, lidar_cfg
        self.events = sorted([(int(s), 0, i) for i, s in enumerate(log["scan_steps"])] + [(int(s), 1, -1) for s in log["map_reset_steps"]])
        self.pos, self.grid = 0, OccupancyGrid()

    def at(self, step):
        while self.pos < len(self.events) and self.events[self.pos][0] <= step:
            s, kind, i = self.events[self.pos]
            if kind == 1:
                self.grid = OccupancyGrid()
            elif MODES[int(self.log["mode"][min(s, len(self.log["mode"]) - 1)])] not in ("dock", "hold"):
                self.grid.update(self.log["scan_poses"][i], self.log["scan_ranges"][i], self.cfg)
            self.pos += 1
        return self.grid


def render_video(log, scene_cfg, drv, out_path, step=2, fps=15, hold_seconds=3.0, result_text="", lidar_cfg=None, overview_path=None):
    lidar_cfg = lidar_cfg or LidarConfig()
    truth, est, cov, dt = log["truth"], log["est_pose"], log["est_cov"], float(log["dt"])
    n = len(truth)
    t = np.arange(n) * dt
    err_mm = 1000 * np.linalg.norm(truth[:, :2] - est[:, :2], axis=1)
    seen = np.isfinite(log["marker"][:, 0])
    scan_steps = log["scan_steps"]
    offsets = np.concatenate([[0], np.cumsum(log["replan_len"])])

    model, data = drv.model, drv.data
    cam_r = mujoco.Renderer(model, height=scene_cfg.camera_height, width=scene_cfg.camera_width)
    ov_r = mujoco.Renderer(model, height=360, width=640)
    free = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, free)
    free.lookat[:] = [0.0, -0.1, 0.0]
    free.distance, free.azimuth, free.elevation = 5.2, 90.0, -48.0

    def place(pose):
        mujoco.mj_resetData(model, data)
        data.qpos[0], data.qpos[1] = pose[0], pose[1]
        data.qpos[3], data.qpos[6] = np.cos(pose[2] / 2), np.sin(pose[2] / 2)
        mujoco.mj_forward(model, data)

    fig = plt.figure(figsize=(14.4, 8.1), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0, 0.7], hspace=0.25, wspace=0.08,
                          left=0.04, right=0.99, top=0.95, bottom=0.07)
    ax_map, ax_cam, ax_3d, ax_err = fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])

    hx, hy = scene_cfg.room_x / 2, scene_cfg.room_y / 2
    g0 = OccupancyGrid()
    extent = (g0.x0, g0.x0 + g0.shape[0] * g0.res, g0.y0, g0.y0 + g0.shape[1] * g0.res)
    im_grid = ax_map.imshow(np.full(g0.shape[::-1], 0.5), origin="lower", cmap="Greys", vmin=0, vmax=1.6, extent=extent, alpha=0.6, zorder=0)
    ax_map.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, ec="0.3", lw=1.5, zorder=1))
    for i, (cx, cy, sx, sy) in enumerate(scene_cfg.obstacles):
        ax_map.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False, ec="#c9803a", lw=1.6, ls="--", zorder=1,
                                   label="true board (unknown to the robot)" if i == 0 else None))
    ax_map.add_patch(Rectangle((scene_cfg.station_x - 0.08, scene_cfg.station_y - 0.14), 0.16, 0.28, fc="#2f9a3c", ec="0.2", zorder=1))
    ax_map.plot(*log["dock"][:2], "g*", ms=11, zorder=3, label="dock pose")
    ax_map.plot(*log["standoff"][:2], "m*", ms=9, zorder=3, label="stand-off pose")
    beams = LineCollection([], colors="tab:cyan", lw=0.4, alpha=0.45, zorder=2)
    ax_map.add_collection(beams)
    (plan_line,) = ax_map.plot([], [], "-", c="tab:purple", lw=1.4, zorder=3, label="current plan")
    (tr_true,) = ax_map.plot([], [], "k-", lw=1.4, zorder=4, label="true path")
    (tr_est,) = ax_map.plot([], [], "-", c="tab:red", lw=1.0, zorder=4, label="EKF estimate")
    body = Circle((0, 0), 0.15, fill=False, ec="0.4", lw=0.8, zorder=4)
    tri_true = Polygon(_triangle((0, 0, 0)), closed=True, fc="k", zorder=5)
    tri_est = Polygon(_triangle((0, 0, 0)), closed=True, fc="none", ec="tab:red", lw=1.5, zorder=5)
    ell = Ellipse((0, 0), 0.1, 0.1, fill=False, ec="tab:red", lw=1.0, ls=":", zorder=5)
    for p in (body, tri_true, tri_est, ell):
        ax_map.add_patch(p)
    ax_map.set_xlim(-hx - 0.1, hx + 0.1)
    ax_map.set_ylim(-hy - 0.1, hy + 0.1)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    ax_map.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
    title = ax_map.set_title("", fontsize=10)
    banner = ax_map.text(0.5, 0.9, "", transform=ax_map.transAxes, ha="center", va="center", fontsize=15, weight="bold", color="white",
                         bbox=dict(boxstyle="round", fc="#2f6f3c", ec="none", alpha=0.92), zorder=10, visible=False)

    im_cam = ax_cam.imshow(np.zeros((scene_cfg.camera_height, scene_cfg.camera_width, 3), dtype=np.uint8))
    ax_cam.axis("off")
    ax_cam.set_title("robot camera (green box: detected ArUco marker)", fontsize=10)
    im_3d = ax_3d.imshow(np.zeros((360, 640, 3), dtype=np.uint8))
    ax_3d.axis("off")
    ax_3d.set_title("MuJoCo scene", fontsize=10)

    ax_err.plot(t, err_mm, c="tab:red", lw=1.1, label="estimator position error")
    ax_err.fill_between(t, 0, 1, where=seen, transform=ax_err.get_xaxis_transform(), color="tab:green", alpha=0.15, label="marker in view")
    for s in log["replan_steps"]:
        ax_err.axvline(s * dt, color="tab:purple", lw=0.5, alpha=0.5)
    for s in log["map_reset_steps"]:
        ax_err.axvline(s * dt, color="k", lw=1.0, ls=":")
    ax_err.set_xlim(0, t[-1])
    ax_err.set_ylim(0, max(100.0, 1.1 * np.nanmax(err_mm)))
    ax_err.set_xlabel("time [s]   (purple lines: re-plans, dotted black: map cleared after a large pose correction)")
    ax_err.set_ylabel("error [mm]")
    ax_err.grid(alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=8, ncol=2)
    cursor = ax_err.axvline(0, c="k", lw=1)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1440, 810))
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    replay = _GridReplay(log, lidar_cfg)
    angles = beam_angles(lidar_cfg)
    last = None
    n_hold = int(hold_seconds * fps)
    for count, k in enumerate(indices + [n - 1] * n_hold):
        final = count >= len(indices)
        grid = replay.at(k)
        lo = grid.logodds.T
        im_grid.set_data(np.where(lo > 0.5, 1.0, np.where(lo < -0.3, 0.0, 0.5)))
        place(truth[k])
        cam_r.update_scene(data, camera=scene_cfg.camera_name)
        img = cam_r.render().copy()
        ids, corners = detect_markers(img, scene_cfg.marker_dictionary, subpixel=True)
        if ids is not None and scene_cfg.marker_id in ids.flatten():
            c = corners[list(ids.flatten()).index(scene_cfg.marker_id)].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [c], True, (0, 255, 0), 3)
        ov_r.update_scene(data, camera=free)
        im_cam.set_data(img)
        im_3d.set_data(ov_r.render())
        j = int(np.searchsorted(scan_steps, k, side="right")) - 1
        if j >= 0:
            r = log["scan_ranges"][j]
            ok = np.isfinite(r)
            ox, oy = sensor_origin(truth[k], lidar_cfg)
            a = truth[k, 2] + angles[ok]
            ends = np.stack([ox + r[ok] * np.cos(a), oy + r[ok] * np.sin(a)], axis=1)
            beams.set_segments([[(ox, oy), tuple(e)] for e in ends])
        m = int(np.searchsorted(log["replan_steps"], k, side="right")) - 1
        if m >= 0:
            pts = log["replan_xy"][offsets[m]:offsets[m + 1]]
            plan_line.set_data(pts[:, 0], pts[:, 1])
        tr_true.set_data(truth[:k + 1, 0], truth[:k + 1, 1])
        tr_est.set_data(est[:k + 1, 0], est[:k + 1, 1])
        body.center = (truth[k, 0], truth[k, 1])
        tri_true.set_xy(_triangle(truth[k]))
        tri_est.set_xy(_triangle(est[k]))
        w_, h_, ang = _ellipse(cov[k][:2, :2])
        ell.set_center((est[k, 0], est[k, 1]))
        ell.set_width(max(w_, 1e-3))
        ell.set_height(max(h_, 1e-3))
        ell.set_angle(ang)
        mode = MODES[int(log["mode"][k])]
        title.set_text(f"t = {t[k]:5.1f} s   {MODE_TEXT[mode]}   |   EKF error {err_mm[k]:4.0f} mm, "
                       f"heading {np.degrees(abs(wrap(truth[k, 2] - est[k, 2]))):.1f} deg")
        cursor.set_xdata([t[k], t[k]])
        if final and result_text:
            banner.set_text(result_text)
            banner.set_visible(True)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        last = frame
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if count % 100 == 0:
            print(f"  video frame {count + 1} / {len(indices) + n_hold}", flush=True)
    writer.release()
    if overview_path is not None and last is not None:
        cv2.imwrite(str(overview_path), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    plt.close(fig)
