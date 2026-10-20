"""Live 2D dashboard shown while main.py --live runs.

In short: by default it shows only what the robot itself knows: the LiDAR points, the map it built from them, its estimated
path, its plan and its estimate of the marker, in coordinates measured from the start point (x = the direction the robot
faced at the start). The room walls, the true obstacles and the true path are not drawn, because the robot does not know
them. With show_truth=True they are added for evaluation. The error curve always needs the simulator truth and is labelled so.
"""

import numpy as np
from matplotlib.transforms import Affine2D

MODE_TEXT = {"plan": "planning", "rotate": "turning towards the path", "follow": "following the path (LQR on the EKF estimate)",
             "align": "aligning with the marker", "dock": "docking along the marker normal", "hold": "docked",
             "failed": "no path found", "search": "looking for the marker", "look": "searching for the marker before docking",
             "go": "driving to the stand-off (reactive LiDAR avoidance)",
             "turn": "turning to the final heading"}


class LiveDashboard:
    def __init__(self, scene_cfg, standoff, dock, every=3, start_pose=(0.0, 0.0, 0.0), show_truth=False, spec=None):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle

        from control.docking import DockSpec
        from perception.occupancy import OccupancyGrid

        self.plt, self.every, self.show_truth = plt, every, show_truth
        self.spec = spec or DockSpec()
        self.t, self.err = [], []
        self.est_xy, self.true_xy = [], []
        plt.ion()
        self.fig = plt.figure("Robot view: map, plan, camera", figsize=(13, 7.2))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.8], left=0.05, right=0.99, top=0.93,
                                   bottom=0.08, wspace=0.12, hspace=0.3)
        ax, self.ax_cam, self.ax_err = self.fig.add_subplot(gs[:, 0]), self.fig.add_subplot(gs[0, 1]), self.fig.add_subplot(gs[1, 1])
        self.ax = ax
        # the robot's map frame -> coordinates from the start point, x along the heading at the start
        sx, sy, sth = start_pose
        self.to_start = Affine2D().translate(-sx, -sy).rotate(-sth)
        T = self.to_start + ax.transData

        g = OccupancyGrid()
        extent = (g.x0, g.x0 + g.shape[0] * g.res, g.y0, g.y0 + g.shape[1] * g.res)
        self.im_grid = ax.imshow(np.full(g.shape[::-1], 0.5), origin="lower", cmap="Greys", vmin=0, vmax=1.6, extent=extent, alpha=0.6)
        self.im_grid.set_transform(T)
        if show_truth:
            hx, hy = scene_cfg.room_x / 2, scene_cfg.room_y / 2
            ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, ec="0.3", lw=1.5, transform=T))
            for i, (cx, cy, bx, by) in enumerate(scene_cfg.obstacles):
                ax.add_patch(Rectangle((cx - bx / 2, cy - by / 2), bx, by, fill=False, ec="#c9803a", lw=1.5, ls="--", transform=T,
                                       label="true obstacle (evaluation only)" if i == 0 else None))
            c, s = np.cos(scene_cfg.station_yaw), np.sin(scene_cfg.station_yaw)
            box = np.array([[-0.08, -0.14], [0.08, -0.14], [0.08, 0.14], [-0.08, 0.14]]) @ np.array([[c, s], [-s, c]])
            ax.add_patch(Polygon(box + [scene_cfg.station_x, scene_cfg.station_y], closed=True, fc="#2f9a3c", ec="0.2", transform=T))
            (self.true_line,) = ax.plot([], [], "k-", lw=1.4, transform=T, label="true path (evaluation only)")
        (self.hits,) = ax.plot([], [], ".", c="tab:cyan", ms=2, transform=T, label="LiDAR hits")
        (self.plan_line,) = ax.plot([], [], "-", c="tab:purple", lw=1.6, transform=T, label="current plan")
        (self.est_line,) = ax.plot([], [], "-", c="tab:red", lw=1.3, transform=T, label="estimated path")
        (self.est_dot,) = ax.plot([], [], "o", c="tab:red", ms=7, transform=T)
        (self.est_head,) = ax.plot([], [], "-", c="tab:red", lw=2.5, transform=T)
        (self.marker_line,) = ax.plot([], [], "-", c="tab:green", lw=4, transform=T, label="marker (estimate)")
        (self.standoff_pt,) = ax.plot([], [], "m*", ms=10, transform=T, label="stand-off")
        (self.dock_pt,) = ax.plot([], [], "g*", ms=12, transform=T, label="dock")
        ax.plot([0], [0], "ks", ms=6, label="start point")
        ax.set_aspect("equal")
        ax.set_xlabel("metres from the start point (x: direction faced at the start)")
        ax.set_ylabel("metres from the start point")
        ax.grid(alpha=0.25)
        corners = self.to_start.transform(np.array([[extent[0], extent[2]], [extent[1], extent[2]], [extent[0], extent[3]], [extent[1], extent[3]]]))
        ax.set_xlim(corners[:, 0].min(), corners[:, 0].max())
        ax.set_ylim(corners[:, 1].min(), corners[:, 1].max())
        ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
        self.title = self.fig.suptitle("", fontsize=11)
        self.banner = ax.text(0.5, 0.92, "", transform=ax.transAxes, ha="center", fontsize=13, weight="bold", color="white",
                              bbox=dict(boxstyle="round", fc="#2f6f3c", ec="none"), visible=False)
        self.im_cam = self.ax_cam.imshow(np.zeros((scene_cfg.camera_height, scene_cfg.camera_width, 3), dtype=np.uint8))
        self.ax_cam.axis("off")
        self.cam_title = self.ax_cam.set_title("robot camera", fontsize=10)
        (self.err_line,) = self.ax_err.plot([], [], c="tab:red", lw=1.1)
        self.ax_err.set_xlabel("time [s]")
        self.ax_err.set_ylabel("error [mm]")
        self.ax_err.set_title("estimate error, for evaluation only (uses the simulator truth)", fontsize=9)
        self.ax_err.grid(alpha=0.3)
        self.plt.show(block=False)

    def update(self, info, true_pose, lidar_cfg, angles):
        from control.docking import dock_poses
        from perception.lidar import sensor_origin

        k, dt = info["step"], info["dt"]
        est = info["est"]
        self.t.append(k * dt)
        self.err.append(1000 * float(np.hypot(true_pose[0] - est[0], true_pose[1] - est[1])))
        self.est_xy.append(est[:2].copy())
        self.true_xy.append(np.asarray(true_pose[:2], dtype=float))
        if k % self.every:
            return
        grid = info.get("grid")
        if grid is not None:
            lo = grid.logodds.T
            self.im_grid.set_data(np.where(lo > 0.5, 1.0, np.where(lo < -0.3, 0.0, 0.5)))
        r = info.get("ranges")
        if r is not None:
            ok = np.isfinite(r)
            ox, oy = sensor_origin(est, lidar_cfg)                # where the robot believes the beams start
            a = est[2] + angles[ok]
            self.hits.set_data(ox + r[ok] * np.cos(a), oy + r[ok] * np.sin(a))
        plan = info.get("plan")
        if plan:
            pts = np.array([est[:2]] + [p for p in plan])
            self.plan_line.set_data(pts[:, 0], pts[:, 1])
        es = np.array(self.est_xy)
        self.est_line.set_data(es[:, 0], es[:, 1])
        self.est_dot.set_data([est[0]], [est[1]])
        self.est_head.set_data([est[0], est[0] + 0.2 * np.cos(est[2])], [est[1], est[1] + 0.2 * np.sin(est[2])])
        if self.show_truth:
            tr = np.array(self.true_xy)
            self.true_line.set_data(tr[:, 0], tr[:, 1])
        m = info.get("marker_est")
        if m is not None:
            t = np.array([-np.sin(m[2]), np.cos(m[2])]) * 0.09
            self.marker_line.set_data([m[0] - t[0], m[0] + t[0]], [m[1] - t[1], m[1] + t[1]])
            so, dk = dock_poses(m, self.spec)
            self.standoff_pt.set_data([so[0]], [so[1]])
            self.dock_pt.set_data([dk[0]], [dk[1]])
        img = info.get("image")
        if img is not None:
            img = img.copy()
            corners = info.get("marker_corners")
            if corners is not None:
                import cv2

                cv2.polylines(img, [np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
            self.im_cam.set_data(img)
            self.cam_title.set_text("robot camera: " + ("marker detected (green box)" if corners is not None else "no marker in view"))
        self.err_line.set_data(self.t, self.err)
        self.ax_err.set_xlim(0, max(10.0, self.t[-1]))
        self.ax_err.set_ylim(0, max(100.0, 1.1 * max(self.err)))
        self.title.set_text(f"t = {self.t[-1]:5.1f} s   {MODE_TEXT.get(info['mode'], info['mode'])}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def finish(self, text, save_path=None):
        self.banner.set_text(text)
        self.banner.set_visible(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if save_path is not None:
            self.fig.savefig(save_path, dpi=110)

    def pump(self):
        self.fig.canvas.flush_events()


class SensorDashboard:
    """What the robot senses, nothing else: the LiDAR scan around the robot and the camera image.

    In short: the LiDAR panel is drawn in the robot's own frame (robot in the centre, facing up, rings every metre), so no
    room coordinates are needed. On top of the scan: the next point the robot is heading for and where it believes the marker
    is (both from the EKF estimate). The camera panel shows the image the ArUco detector gets, with the detected marker in green.
    """

    def __init__(self, scene_cfg, max_range=3.5, every=2):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon

        self.plt, self.every = plt, every
        plt.ion()
        self.fig = plt.figure("Robot sensors: LiDAR and camera", figsize=(12.5, 6.2))
        ax = self.fig.add_axes([0.04, 0.08, 0.42, 0.82])
        self.ax_cam = self.fig.add_axes([0.5, 0.15, 0.48, 0.7])
        self.ax = ax
        for r in np.arange(1.0, max_range + 0.01, 1.0):
            ax.add_patch(Circle((0, 0), r, fill=False, ec="0.75", lw=0.8, ls="--"))
            ax.text(0.05, r + 0.03, f"{r:.0f} m", color="0.5", fontsize=8)
        ax.add_patch(Polygon([[0, 0.17], [-0.12, -0.12], [0.12, -0.12]], closed=True, fc="tab:red", ec="k"))
        (self.hits,) = ax.plot([], [], ".", c="tab:blue", ms=3, label="LiDAR returns")
        (self.target,) = ax.plot([], [], "-", c="tab:purple", lw=1.5, label="heading for")
        (self.marker,) = ax.plot([], [], "s", c="tab:green", ms=9, label="marker (estimate)")
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("LiDAR around the robot (robot in the centre, facing up)", fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
        self.im_cam = self.ax_cam.imshow(np.zeros((scene_cfg.camera_height, scene_cfg.camera_width, 3), dtype=np.uint8))
        self.ax_cam.axis("off")
        self.cam_title = self.ax_cam.set_title("robot camera", fontsize=10)
        self.title = self.fig.suptitle("", fontsize=11)
        self.banner = self.fig.text(0.5, 0.03, "", ha="center", fontsize=13, weight="bold", color="white",
                                    bbox=dict(boxstyle="round", fc="#2f6f3c", ec="none"), visible=False)
        self.plt.show(block=False)

    @staticmethod
    def _to_view(est, xy):
        """World point -> robot frame -> plot (right = robot's right, up = robot's forward)."""
        dx, dy = xy[0] - est[0], xy[1] - est[1]
        c, s = np.cos(est[2]), np.sin(est[2])
        fwd, left = c * dx + s * dy, -s * dx + c * dy
        return -left, fwd

    def update(self, info, true_pose, lidar_cfg, angles):
        k, est = info["step"], info["est"]
        if k % self.every:
            return
        r = info.get("ranges")
        if r is not None:
            ok = np.isfinite(r)
            fwd = lidar_cfg.mount[0] + r[ok] * np.cos(angles[ok])
            left = r[ok] * np.sin(angles[ok])
            self.hits.set_data(-left, fwd)
        plan, m = info.get("plan"), info.get("marker_est")
        if info["mode"] in ("align", "dock", "turn", "hold", "look") and m is not None:
            from control.docking import DockSpec, dock_poses

            plan = [dock_poses(m, DockSpec())[1][:2]]              # docking: heading for the dock pose in front of the marker
        if plan:
            px, py = self._to_view(est, plan[0])
            self.target.set_data([0, px], [0, py])
        if m is not None:
            mx, my = self._to_view(est, m[:2])
            self.marker.set_data([mx], [my])
        img = info.get("image")
        if img is not None:
            img = img.copy()
            corners = info.get("marker_corners")
            if corners is not None:
                import cv2

                cv2.polylines(img, [np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 3)
            self.im_cam.set_data(img)
            self.cam_title.set_text("robot camera: " + ("marker detected (green box)" if corners is not None else "no marker in view"))
        self.title.set_text(f"t = {k * info['dt']:5.1f} s   {MODE_TEXT.get(info['mode'], info['mode'])}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def finish(self, text, save_path=None):
        self.banner.set_text(text)
        self.banner.set_visible(True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if save_path is not None:
            self.fig.savefig(save_path, dpi=110)

    def pump(self):
        self.fig.canvas.flush_events()
