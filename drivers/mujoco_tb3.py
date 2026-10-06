"""MuJoCo driver for the ROBOTIS TurtleBot3 model.

The rest of the pipeline only sees this interface: command desired wheel speeds, read encoders.
Ground truth is exposed through true_pose(), which is for evaluation only and must never be used
by the estimator or the controller.

Not yet validated against MuJoCo: run notebooks/02_driver_validation.ipynb first.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from drivers.wheel_speed_controller import WheelSpeedController

TICKS_PER_REV = 4096   # encoder resolution assumed for the simulated encoders; to be checked against the ROBOTIS documentation


@dataclass
class EncoderReading:
    ticks: np.ndarray    # integer tick counts [left, right]
    omega: np.ndarray    # wheel speeds [rad/s] from tick differences over one control period
    time: float          # simulation time [s]


class MujocoTB3Driver:
    def __init__(self, scene_xml, control_dt=0.02, **loop_kwargs):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.control_dt = control_dt
        self.substeps = int(round(control_dt / self.model.opt.timestep))
        self._act = (self.model.actuator("wheel_left").id, self.model.actuator("wheel_right").id)
        limit = float(self.model.actuator_ctrlrange[self._act[0], 1])
        self._loops = [WheelSpeedController(control_dt, cmd_limit=limit, **loop_kwargs) for _ in range(2)]
        self.reset()

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[1] = x, y
        self.data.qpos[3], self.data.qpos[6] = np.cos(yaw / 2), np.sin(yaw / 2)   # free-joint quaternion (w, x, y, z)
        mujoco.mj_forward(self.model, self.data)
        for loop in self._loops:
            loop.reset()
        self._last_ticks = self._read_ticks()

    def _read_ticks(self):
        angles = np.array([self.data.joint("wheel_left").qpos[0], self.data.joint("wheel_right").qpos[0]])
        return np.round(angles * TICKS_PER_REV / (2 * np.pi)).astype(int)

    def step(self, omega_left_des, omega_right_des):
        """Advance one control period with the desired wheel speeds [rad/s]; return the encoder reading."""
        ticks = self._read_ticks()
        omega = (ticks - self._last_ticks) * (2 * np.pi / TICKS_PER_REV) / self.control_dt
        self._last_ticks = ticks
        desired = (omega_left_des, omega_right_des)
        for i in range(2):
            self.data.ctrl[self._act[i]] = self._loops[i].step(desired[i], omega[i])
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return EncoderReading(ticks=ticks, omega=omega, time=float(self.data.time))

    def true_pose(self):
        """Ground-truth base pose (x, y, yaw). Evaluation only."""
        w, qx, qy, qz = self.data.qpos[3:7]
        yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(self.data.qpos[0]), float(self.data.qpos[1]), float(yaw)
