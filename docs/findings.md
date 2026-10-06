# Findings log: stage 0 (simulation setup) and stage 1 (model analysis)

All numbers below come from runs of the notebooks in this repository. They are single runs with the default MuJoCo settings unless stated; they are evidence for design decisions, not calibrated parameters yet.

## Platform and simulator

- Simulator: MuJoCo. Model: ROBOTIS TurtleBot3 from `ROBOTIS-GIT/robotis_mujoco_menagerie` (`robotis_tb3`, Apache-2.0), provided by the unit coordinator. The upstream XML is used unmodified and fetched at run time by `notebooks/00_mujoco_tb3_setup_check.ipynb`.
- Default robot: Waffle Pi (it carries the camera; the XML marks the camera position). Burger is selectable with one variable. Which robot the hardware lab holds is still to be confirmed.
- The upstream model has differential-drive wheels and velocity actuators only. It has no LiDAR, camera or encoder sensors, no obstacles and no ArUco marker. These are added in this project.
- Offscreen rendering works in Colab with `MUJOCO_GL=egl`, so camera rendering is feasible.

## Stage 0: drive characterisation (Waffle Pi, upstream XML)

Method: constant wheel velocity commands for 4 s, statistics over the last 1 s (`notebooks/00_mujoco_tb3_setup_check.ipynb`). Wheel radius 0.033 m, XML wheel separation 0.288 m.

Straight driving (both wheels commanded equally):

| Command [rad/s] | Measured wheel speed L / R [rad/s] | Signed forward speed [m/s] | r x mean wheel speed [m/s] |
|---|---|---|---|
| 1.0 | 0.01 / 0.01 | 0.000 | 0.000 |
| 2.0 | 0.01 / 0.01 | 0.000 | 0.000 |
| 3.0 | 0.95 / 0.95 | 0.031 | 0.031 |
| 5.0 | 2.97 / 2.98 | 0.084 | 0.098 |
| 7.88 | 5.83 / 5.84 | 0.166 | 0.193 |

In-place spin (left and right commanded with opposite signs):

| Command [rad/s] | Measured wheel speed L / R [rad/s] | Yaw rate [rad/s] | Yaw rate predicted from measured wheels and XML separation [rad/s] |
|---|---|---|---|
| 1.0 | -0.01 / 0.01 | 0.001 | 0.001 |
| 3.0 | -0.60 / 0.68 | 0.112 | 0.147 |
| 7.88 | -5.44 / 5.55 | 0.925 | 1.259 |

Observations:

1. **Dead zone and offset.** Above about 2 rad/s the wheel speed is roughly the command minus 2.04 rad/s; below it the wheels do not move. This is consistent with a constant resisting torque of about `kv x 2.04` = 0.2 N m. A first estimate from the XML (`frictionloss / kv` = 1 rad/s) was half of the measured offset, so the XML friction term alone does not explain it; the remaining resistance has not been identified.
2. **Usable speed range.** At the maximum command (7.88 rad/s) the robot reaches about 0.17 m/s.
3. **Odometry scale.** Forward speed divided by `r x mean wheel speed` is about 1.0 at command 3.0 and about 0.86 at commands 5.0 and 7.88: encoder-based odometry overestimates speed by about 14 percent at higher speeds.
4. **Effective wheel separation.** The in-place yaw rate is about 74 to 76 percent of the kinematic prediction, which corresponds to an effective separation of about 0.38 to 0.39 m instead of 0.288 m. The cause (wheel scrub or caster resistance) has not been identified.
5. Left and right wheels are symmetric on straights (yaw rate within 0.002 rad/s).

Decision: keep the upstream XML unchanged and compensate in a driver layer (command offset plus an encoder-based wheel-speed loop), so the rest of the pipeline sees a wheel-speed interface. This also matches the course guidance that moving to hardware should only change the sensor and actuator driver layer. The odometry scale and effective separation are to be calibrated, or treated as process-noise terms in the EKF.

## Stage 1: models, observability and controllability

Code: `estimation/models.py`; checks: `tests/test_models.py`; analysis: `notebooks/01_model_analysis.ipynb`.

Models:

- State `x = [px, py, theta]`, inputs from the encoders `v = r (wR + wL) / 2`, `w = r (wR - wL) / b`, exact-arc discrete prediction `f_unicycle`.
- Marker measurement: full marker pose seen from the robot (marker position in the robot frame and marker orientation relative to the heading), or range and bearing to the marker. Analytical Jacobians are checked against finite differences in the tests.
- Tracking-error model about a reference `(v_r, w_r)`, `e = [e_x, e_y, e_theta]` in the robot frame: `A = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]]`, `B = [[-1, 0], [0, 0], [0, -1]]`.

Observability (smallest singular value of the observability matrix over 289 random poses and motions; near zero means unobservable):

| Measurement | Min | Median |
|---|---|---|
| One marker, range and bearing | 6e-17 | 2e-16 |
| Two markers, range and bearing | 0.66 | 1.5 |
| One marker, full pose | 0.39 | 0.74 |

- Range and bearing to a single marker are rank deficient at every sampled pose and motion, including after sweeping the robot heading. The unobservable direction is consistent with a rotation of the whole trajectory about the marker, which changes neither the body-frame odometry nor the range and bearing. This is an interpretation of the numerical result, not a proof.
- Full marker pose (marker orientation included) or a second marker restores full rank. Decision: the EKF uses the full ArUco pose as its camera measurement.
- With the marker out of view only odometry remains, nothing is observable, and the covariance grows. Quantifying this is part of the robustness evaluation.

Controllability of the tracking-error model:

| Reference (v_r, w_r) | Rank (3 is controllable) |
|---|---|
| (0.15, 0.0) | 3 |
| (0.15, 0.3) | 3 |
| (0.0, 0.3) | 3 |
| (0.0, 0.0) | 2 |

The model loses controllability only when the reference speed is zero in both components, so the final docking approach must not let the reference speed drop exactly to zero under the linear tracker.

## Open items

- Confirm the robot (Burger or Waffle Pi) held by the hardware lab.
- Camera extrinsics (mounting offset) are ignored in the stage-1 measurement models and belong to the perception stage.
- Calibrate or model the odometry scale and effective wheel separation, and repeat the drive characterisation over a finer command sweep.
- Implement the driver layer, the sensors (LiDAR, camera, encoders), the room with obstacles and the ArUco marker.
