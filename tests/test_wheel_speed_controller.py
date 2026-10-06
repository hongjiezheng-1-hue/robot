"""Logic checks for the wheel-speed loop against a stand-in plant. Run: python tests/test_wheel_speed_controller.py

The stand-in plant only imitates the behaviour measured in docs/findings.md (command offset, dead zone,
first-order lag). Passing here does not validate the loop on the MuJoCo model; that is done in
notebooks/02_driver_validation.ipynb.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from drivers.wheel_speed_controller import WheelSpeedController  # noqa: E402

DT = 0.02


class StandInPlant:
    def __init__(self, offset=2.04, tau=0.2, noise=0.05, seed=0):
        self.offset, self.tau, self.noise = offset, tau, noise
        self.rng = np.random.default_rng(seed)
        self.w = 0.0

    def step(self, cmd):
        target = np.sign(cmd) * max(abs(cmd) - self.offset, 0.0)
        self.w += (target - self.w) * DT / self.tau
        return self.w + self.rng.normal(0.0, self.noise)


def run(controller, omega_des, seconds=8.0):
    plant, meas, log_w, log_cmd = StandInPlant(), 0.0, [], []
    controller.reset()
    for _ in range(int(seconds / DT)):
        cmd = controller.step(omega_des, meas)
        meas = plant.step(cmd)
        log_w.append(plant.w)
        log_cmd.append(cmd)
    return np.array(log_w), np.array(log_cmd)


def test_tracks_desired_speed():
    for des in (0.5, 1.5, 3.0, -2.0):
        w, _ = run(WheelSpeedController(DT), des)
        assert abs(w[-50:].mean() - des) < 0.1, (des, w[-50:].mean())


def test_zero_desired_stops():
    w, _ = run(WheelSpeedController(DT), 0.0)
    assert abs(w[-50:].mean()) < 0.05


def test_compensation_matters():
    w, _ = run(WheelSpeedController(DT, offset=0.0, kp=0.5, ki=0.0), 3.0)
    assert abs(w[-50:].mean() - 3.0) > 1.0, "without offset and integral action the stand-in plant should be far off"


def test_saturation_is_bounded():
    c = WheelSpeedController(DT)
    w, cmd = run(c, 20.0)
    assert np.all(np.abs(cmd) <= c.cmd_limit + 1e-9)
    assert abs(c.integral) < 50.0, "integrator wound up while saturated"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
