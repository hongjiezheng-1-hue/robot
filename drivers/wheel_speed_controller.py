"""Wheel-speed loop for one wheel: feedforward offset plus PI on the measured wheel speed.

The upstream TurtleBot3 MuJoCo model has a dead zone and a constant offset between the
velocity command and the resulting wheel speed (measured: roughly command minus 2.04 rad/s,
no motion below that; see docs/findings.md). This loop hides that behind a wheel-speed
interface: the rest of the pipeline commands a desired wheel speed and reads encoders.
"""

import numpy as np


class WheelSpeedController:
    def __init__(self, dt, offset=2.04, kp=0.5, ki=2.0, cmd_limit=7.88, sign_eps=0.02, alpha=0.3):
        self.dt = dt
        self.offset = offset          # [rad/s] measured command offset of the actuator
        self.kp = kp
        self.ki = ki
        self.cmd_limit = cmd_limit    # [rad/s] actuator command range
        self.sign_eps = sign_eps      # [rad/s] smoothing width of the sign() in the feedforward
        self.alpha = alpha            # speed measurement low-pass coefficient (1 = no filtering)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.meas_filtered = 0.0

    def step(self, omega_des, omega_meas):
        self.meas_filtered += self.alpha * (omega_meas - self.meas_filtered)
        feedforward = omega_des + self.offset * np.tanh(omega_des / self.sign_eps)
        err = omega_des - self.meas_filtered
        unsat = feedforward + self.kp * err + self.integral
        cmd = float(np.clip(unsat, -self.cmd_limit, self.cmd_limit))
        saturated_against_error = (unsat > self.cmd_limit and err > 0) or (unsat < -self.cmd_limit and err < 0)
        if not saturated_against_error:
            self.integral += self.ki * err * self.dt
        return cmd
