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

## Stage 2: driver validation (Waffle Pi, wheel-speed loop on)

Code: `drivers/wheel_speed_controller.py`, `drivers/mujoco_tb3.py`; run: `notebooks/02_driver_validation.ipynb`. Control period 0.02 s, constant desired wheel speeds for 8 s, statistics over the last 1 s. The loop logic is also tested against a stand-in plant in `tests/test_wheel_speed_controller.py`; the MuJoCo results below are the real validation.

Straight driving:

| Desired [rad/s] | Measured wheel speed L / R [rad/s] | True v [m/s] | Odometry v [m/s] | True v / odometry v |
|---|---|---|---|---|
| 0.5 | 0.50 / 0.50 | 0.016 | 0.017 | 0.99 |
| 1.0 | 1.00 / 1.00 | 0.032 | 0.033 | 0.98 |
| 2.0 | 1.99 / 2.01 | 0.058 | 0.066 | 0.88 |
| 3.0 | 3.00 / 3.01 | 0.085 | 0.099 | 0.85 |
| 5.0 | 5.00 / 4.99 | 0.141 | 0.165 | 0.86 |

In-place spin:

| Desired [rad/s] | Measured wheel speed L / R [rad/s] | True yaw rate [rad/s] | Odometry yaw rate, nominal track 0.288 m [rad/s] | Effective track [m] |
|---|---|---|---|---|
| 0.5 | -0.50 / 0.50 | 0.093 | 0.115 | 0.353 |
| 1.0 | -1.00 / 1.01 | 0.167 | 0.231 | 0.398 |
| 2.0 | -2.00 / 2.01 | 0.318 | 0.458 | 0.416 |

Observations:

1. The wheel-speed loop tracks the desired speed to within 0.01 rad/s, including 0.5 and 1.0 rad/s, which the raw actuator could not reach because of its dead zone.
2. The odometry speed scale is not constant: about 1.0 at wheel speeds of 1 rad/s or less and about 0.86 at 2 rad/s or more, with the change between 1 and 2 rad/s. The effective track width during spins grows with spin rate (0.35 to 0.42 m, nominal 0.288 m). The cause is not identified.
3. Not covered: arc motions (simultaneous forward and turning motion), repeated runs. A constant calibration from these tables is therefore not yet justified.

## Stage 3: scene, ArUco marker and marker pose

Code: `sim/scene.py`, `perception/aruco.py`, `perception/marker_pose.py`; checks: `tests/test_scene.py`, `tests/test_marker_pose.py`, `notebooks/03_scene_check.ipynb`.

Scene check in MuJoCo (OpenCV 5.0.0, `notebooks/03_scene_check.ipynb`): the generated room, two obstacles and the charging station load and render. The marker texture is detected in 3 of 8 viewing azimuths (0, 45 and 315 degrees, the views in front of the marker) with the default orientation, and in none with the mirrored texture, so the default texture orientation is correct. Views from the side or from behind are not expected to detect it. The floor reflects a faint mirrored copy of the marker (floor reflectance 0.2, copied from the upstream scene); no false detection was seen.

Marker pose in the robot frame (`estimate_marker_pose`): position of the marker and the direction of its outward normal relative to the robot heading, the same quantities as `h_marker_pose`. The frame conversion was verified on synthetic projections against `h_marker_pose` to 1e-6.

Solver finding: with the locally installed OpenCV 4.3.0, the IPPE planar solver gave reprojection errors of 0.4 to 5 px on exact, noise-free synthetic corners, while the iterative solver recovered the true pose exactly, so the synthetic geometry is right and that solver version is inaccurate. The estimator therefore refines the two IPPE candidates and also tries the iterative solution, and keeps the candidate with the smallest reprojection error. Behaviour with OpenCV 5.0.0 on MuJoCo renders is checked in `notebooks/04_camera_pose_check.ipynb`.

Corner-noise sensitivity of the marker pose (synthetic projections with Gaussian pixel noise, 300 poses per row, marker 0.12 m, camera 48.8 deg vertical field of view at 640x480; these are not MuJoCo renders):

| Corner noise | Distance [m] | Marker width [px] | Position error median / 95% [mm] | Heading error median / 95% [deg] | Heading error above 10 deg |
|---|---|---|---|---|---|
| 0.3 px | 0.5 | 127 | 0.5 / 1.6 | 0.3 / 1.0 | 0% |
| 0.3 px | 0.8 | 79 | 1.6 / 4.3 | 0.8 / 2.9 | 0% |
| 0.3 px | 1.2 | 53 | 4.1 / 11.6 | 2.0 / 9.7 | 5% |
| 0.3 px | 1.6 | 40 | 7.8 / 21.9 | 3.4 / 16.8 | 16% |
| 1.0 px | 0.5 | 127 | 1.8 / 5.4 | 1.1 / 3.3 | 0% |
| 1.0 px | 0.8 | 79 | 6.1 / 15.9 | 3.1 / 12.0 | 8% |
| 1.0 px | 1.2 | 53 | 14.3 / 40.4 | 6.8 / 24.9 | 36% |
| 1.0 px | 1.6 | 40 | 28.2 / 71.8 | 9.7 / 25.2 | 48% |

Marker pose from MuJoCo renders (`notebooks/04_camera_pose_check.ipynb`, OpenCV 5.0.0, MuJoCo 3.15.0, camera placeholders as above): the model contains one camera and the marker was detected at all 36 tested poses (distance about 0.4 to 1.9 m, robot roughly facing the marker within 0.2 rad). Errors against ground truth, 9 poses per band (percentiles are rough):

| Distance [m] | Position error median / 95% [mm] | Heading error median / 95% [deg] |
|---|---|---|
| 0.2 to 0.7 | 1.2 / 1.8 | 0.2 / 0.5 |
| 0.7 to 1.1 | 2.6 / 4.2 | 1.5 / 1.7 |
| 1.1 to 1.5 | 12.6 / 14.3 | 0.9 / 5.1 |
| 1.5 to 1.9 | 19.9 / 41.9 | 2.0 / 18.0 |

The trend matches the synthetic study (error grows with distance, heading worst at long range), but the render errors are not equivalent to a single Gaussian corner noise: position error corresponds to about 1 px of synthetic noise while heading error is better than the 1 px synthetic case. A possible cause of the position error is a depth error along the line of sight from a small bias in the apparent marker size; this has not been checked (it needs the error split into along-sight and lateral components). Not covered: distances beyond 1.9 m (the room allows about 3.7 m, where the marker is about 20 px wide), large viewing angles, repeated noisy renders. The range over which the marker is detected is therefore still unknown, and it determines when the EKF receives updates.

Consequence for the EKF (a design hypothesis to test, not yet a result): the heading component of the marker pose becomes unreliable at long range and with noisy corners, while the position component stays accurate to a few centimetres. The measurement noise for the heading component should grow with distance, or the heading component should be dropped when the marker appears narrower than roughly 80 px.

## Open items

- Confirm the robot (Burger or Waffle Pi) held by the hardware lab.
- Camera extrinsics (mounting offset) are ignored in the stage-1 measurement models and belong to the perception stage.
- Characterise odometry over arc motions inside the planned operating envelope, then either calibrate by speed range or absorb the residual in the EKF process noise; measure dead-reckoning error with and without calibration.
- Add the sensors (LiDAR, camera), the room with obstacles and the ArUco marker. Encoders are already provided by the driver.
