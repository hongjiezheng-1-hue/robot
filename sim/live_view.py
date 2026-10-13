"""Live MuJoCo viewer for the navigation demo: the scene, the LiDAR beams, the estimated pose and the current plan.

The robot, boards and station are drawn by MuJoCo from the simulator state (the "world"). Everything added on top comes from the
robot's own side: red = estimated pose, cyan lines = LiDAR beams that returned a range, purple = the path being followed.
The beams start at the true sensor position because that is where the simulated sensor is; this is for display only.
"""

import time

import numpy as np

from perception.lidar import LidarConfig, beam_angles, sensor_origin


class LiveView:
    def __init__(self, drv, speedup=1.0, lidar_cfg=None):
        import mujoco.viewer

        self.drv, self.speedup, self.cfg = drv, speedup, lidar_cfg or LidarConfig()
        self.viewer = mujoco.viewer.launch_passive(drv.model, drv.data)
        cam = self.viewer.cam
        cam.lookat[:] = [0.0, -0.1, 0.0]
        cam.distance, cam.azimuth, cam.elevation = 5.0, 90.0, -55.0
        self.t_start = time.time()
        self.angles = beam_angles(self.cfg)

    def running(self):
        return self.viewer.is_running()

    def _line(self, scn, p0, p1, width, rgba):
        import mujoco

        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
        scn.ngeom += 1

    def _sphere(self, scn, pos, radius, rgba):
        import mujoco

        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0.0, 0.0]), np.asarray(pos, dtype=np.float64),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        scn.ngeom += 1

    def on_step(self, info):
        """Called by run_navigation after every control step; returns True when the window was closed."""
        if not self.viewer.is_running():
            return True
        true_pose = self.drv.true_pose()                              # only to place the drawn beams
        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            r = info["ranges"]
            if r is not None:
                ox, oy = sensor_origin(true_pose, self.cfg)
                ok = np.isfinite(r)
                a = true_pose[2] + self.angles[ok]
                for rr, aa in zip(r[ok][::2], a[::2]):
                    self._line(scn, (ox, oy, 0.145), (ox + rr * np.cos(aa), oy + rr * np.sin(aa), 0.145), 0.002, (0.1, 0.8, 0.9, 0.5))
            est = info["est"]
            self._sphere(scn, (est[0], est[1], 0.30), 0.03, (0.9, 0.1, 0.1, 1.0))
            self._line(scn, (est[0], est[1], 0.30), (est[0] + 0.15 * np.cos(est[2]), est[1] + 0.15 * np.sin(est[2]), 0.30), 0.01, (0.9, 0.1, 0.1, 1.0))
            plan = info["plan"]
            if plan:
                pts = [(est[0], est[1])] + [tuple(p) for p in plan]
                for p0, p1 in zip(pts[:-1], pts[1:]):
                    self._line(scn, (p0[0], p0[1], 0.02), (p1[0], p1[1], 0.02), 0.012, (0.6, 0.2, 0.8, 0.9))
        self.viewer.sync()
        wait = info["step"] * info["dt"] / self.speedup - (time.time() - self.t_start)
        if wait > 0:
            time.sleep(wait)
        return False

    def wait_until_closed(self):
        while self.viewer.is_running():
            self.viewer.sync()
            time.sleep(0.05)
