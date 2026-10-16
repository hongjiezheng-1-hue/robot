"""Live 2D dashboard shown while main.py --live runs: the map the robot builds, its plan, its estimated and true path,
the robot camera image and the estimator error.

In short: this is the same picture as the demo video, but drawn while the code runs.
Everything except the black true path and the error curve comes from the robot's side (estimate, map, plan, camera);
the truth is drawn only so the viewer can judge the estimate.
"""

import numpy as np

MODE_TEXT = {"plan": "planning", "rotate": "turning towards the path", "follow": "following the path (LQR on the EKF estimate)",
             "align": "aligning with the marker", "dock": "docking along the marker normal", "hold": "docked",
             "failed": "no path found", "search": "looking for the marker"}


class LiveDashboard:
    def __init__(self, scene_cfg, standoff, dock, every=3):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle

        from perception.occupancy import OccupancyGrid

        self.plt, self.every = plt, every
        self.t, self.err, self.seen = [], [], []
        self.est_xy, self.true_xy = [], []
        plt.ion()
        self.fig = plt.figure("Robot view: map, plan, camera", figsize=(13, 7.2))
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.8], left=0.05, right=0.99, top=0.93,
                                   bottom=0.08, wspace=0.12, hspace=0.3)
        ax, self.ax_cam, self.ax_err = self.fig.add_subplot(gs[:, 0]), self.fig.add_subplot(gs[0, 1]), self.fig.add_subplot(gs[1, 1])
        self.ax = ax
        g = OccupancyGrid()
        extent = (g.x0, g.x0 + g.shape[0] * g.res, g.y0, g.y0 + g.shape[1] * g.res)
        self.im_grid = ax.imshow(np.full(g.shape[::-1], 0.5), origin="lower", cmap="Greys", vmin=0, vmax=1.6, extent=extent, alpha=0.6)
        hx, hy = scene_cfg.room_x / 2, scene_cfg.room_y / 2
        ax.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, ec="0.3", lw=1.5))
        for i, (cx, cy, sx, sy) in enumerate(scene_cfg.obstacles):
            ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False, ec="#c9803a", lw=1.5, ls="--",
                                   label="true obstacle (unknown to the robot)" if i == 0 else None))
        c, s = np.cos(scene_cfg.station_yaw), np.sin(scene_cfg.station_yaw)
        box = np.array([[-0.08, -0.14], [0.08, -0.14], [0.08, 0.14], [-0.08, 0.14]]) @ np.array([[c, s], [-s, c]])
        ax.add_patch(Polygon(box + [scene_cfg.station_x, scene_cfg.station_y], closed=True, fc="#2f9a3c", ec="0.2"))
        ax.plot(*standoff[:2], "m*", ms=10, label="stand-off")
        ax.plot(*dock[:2], "g*", ms=12, label="dock")
        (self.hits,) = ax.plot([], [], ".", c="tab:cyan", ms=2, label="LiDAR hits")
        (self.plan_line,) = ax.plot([], [], "-", c="tab:purple", lw=1.6, label="current plan")
        (self.true_line,) = ax.plot([], [], "k-", lw=1.4, label="true path")
        (self.est_line,) = ax.plot([], [], "-", c="tab:red", lw=1.1, label="EKF estimate")
        (self.est_dot,) = ax.plot([], [], "o", c="tab:red", ms=7)
        ax.set_xlim(-hx - 0.1, hx + 0.1)
        ax.set_ylim(-hy - 0.1, hy + 0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.9)
        self.title = self.fig.suptitle("", fontsize=11)
        self.banner = ax.text(0.5, 0.92, "", transform=ax.transAxes, ha="center", fontsize=13, weight="bold", color="white",
                              bbox=dict(boxstyle="round", fc="#2f6f3c", ec="none"), visible=False)
        self.im_cam = self.ax_cam.imshow(np.zeros((scene_cfg.camera_height, scene_cfg.camera_width, 3), dtype=np.uint8))
        self.ax_cam.axis("off")
        self.cam_title = self.ax_cam.set_title("robot camera", fontsize=10)
        (self.err_line,) = self.ax_err.plot([], [], c="tab:red", lw=1.1)
        self.ax_err.set_xlabel("time [s]")
        self.ax_err.set_ylabel("estimator error [mm]")
        self.ax_err.set_title("position error of the estimate (truth used only to draw this)", fontsize=9)
        self.ax_err.grid(alpha=0.3)
        self.plt.show(block=False)

    def update(self, info, true_pose, lidar_cfg, angles):
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
            ox, oy = sensor_origin(true_pose, lidar_cfg)
            a = true_pose[2] + angles[ok]
            self.hits.set_data(ox + r[ok] * np.cos(a), oy + r[ok] * np.sin(a))
        plan = info.get("plan")
        if plan:
            pts = np.array([est[:2]] + [p for p in plan])
            self.plan_line.set_data(pts[:, 0], pts[:, 1])
        tr, es = np.array(self.true_xy), np.array(self.est_xy)
        self.true_line.set_data(tr[:, 0], tr[:, 1])
        self.est_line.set_data(es[:, 0], es[:, 1])
        self.est_dot.set_data([est[0]], [est[1]])
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
        self.title.set_text(f"t = {self.t[-1]:5.1f} s   {MODE_TEXT.get(info['mode'], info['mode'])}   |   estimator error {self.err[-1]:4.0f} mm")
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
