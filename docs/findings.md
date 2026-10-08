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

## Stage 4: EKF with fixed and estimated odometry gains (synthetic validation only)

Code: `estimation/ekf.py`, `sim/synthetic.py`; experiments: `experiments/ekf_synthetic.py`, `experiments/ekf_sensitivity.py` (outputs in `results/`); tests: `tests/test_ekf.py`, `tests/test_models.py`.

Design:

- State `[px, py, theta]`. Prediction from encoder odometry `v = s_v v_nom`, `w = s_w w_nom`, where `v_nom, w_nom` use the nominal wheel radius and track width and the gains `s_v, s_w` absorb radius, slip and effective-track errors. Process noise comes from input noise (`floor + relative x |input|`) mapped through the input Jacobian, so it grows with speed.
- Update with the full marker pose; measurement noise grows with the measured marker distance (position sigma from `d^2 x corner_sigma / (f x L)` plus a floor, heading sigma `0.005 + 0.05 d^2` rad), set from the render study above.
- Variant A: gains fixed from a calibration. Variant B: the two gains are added to the state (random walk, start at the nominal value 1 with sigma 0.2) and estimated online.

Synthetic plant: unicycle following waypoints, true gains `s_v = 0.86`, `s_w = 0.72` (from the stage-2 MuJoCo measurements, constant here), encoder noise, slip noise with the same model as the filter's, marker pose measured with the filter's noise model while the marker is in view (distance 0.2 to 3 m, within 28 degrees of the heading, robot in front of the marker). Marker visible in about 25% of samples. No obstacles occlude the marker and no heading-flip outliers are included in these runs.

Main comparison (20 runs of 600 s, position and heading RMSE over the second half, mean +/- std over runs):

| Variant | Position RMSE [mm] | Heading RMSE [deg] | ANEES (ideal 3) | NEES inside 95% bound |
|---|---|---|---|---|
| A0: nominal gains, no calibration | 3079 +/- 83 | 105 +/- 3.5 | 74959 | 0.01 |
| A2: gains about 5% off | 294 +/- 27 | 8.3 +/- 0.8 | 35.0 | 0.12 |
| A2t: as A2, odometry noise inflated 4x | 293 +/- 26 | 8.2 +/- 0.7 | 3.34 | 0.98 |
| A1: exact gains | 75.5 +/- 23.7 | 2.2 +/- 0.7 | 3.36 | 0.94 |
| B: gains estimated online | 97.4 +/- 32.3 | 3.1 +/- 1.0 | 2.27 | 0.97 |

The inflation factor of A2t was chosen on separate runs (seeds 100 to 103) as the one whose ANEES is closest to 3. B's final gain estimates are `s_v = 0.858 +/- 0.012` (true 0.86) and `s_w = 0.719 +/- 0.008` (true 0.720).

Findings:

1. The filter implementation is consistent when the model matches (A1: ANEES 3.36, 94% inside the bound).
2. Inflating the process noise makes a miscalibrated filter honest (ANEES 35 to 3.3) but does not improve its accuracy (294 mm to 293 mm).
3. B recovers most of the accuracy of an exactly calibrated filter and stays consistent.
4. Without calibration (A0) the filter is lost: the odometry bias is too large to be corrected by a marker that is visible 25% of the time.

Sensitivity (8 runs per row, second-half RMSE):

| Calibration error of A | A position [mm] | A ANEES | A inside bound |
|---|---|---|---|
| 0% | 83.5 | 2.97 | 0.96 |
| 1% | 90.2 | 4.09 | 0.87 |
| 2% | 126.5 | 7.74 | 0.60 |
| 5% | 287.8 | 35.1 | 0.12 |
| 10% | 588.2 | 144.0 | 0.03 |
| B (independent of the error) | 111.1 | 2.11 | 0.98 |

| Max detection range | Marker visible | A (5% off) position [mm] | B position [mm] | B ANEES |
|---|---|---|---|---|
| 1.5 m | 6% | 409 | 108 | 4.07 |
| 2.0 m | 12% | 365 | 106 | 2.21 |
| 3.0 m | 25% | 288 | 111 | 2.11 |
| 4.0 m | 30% | 284 | 95 | 2.14 |

Mission length, whole-run RMSE including the convergence phase of B (12 runs per row):

| Length | A (5% off) [mm] | A (exact) [mm] | B [mm] |
|---|---|---|---|
| 60 s | 185 | 133 | 158 |
| 120 s | 249 | 104 | 162 |
| 300 s | 267 | 91 | 131 |
| 600 s | 279 | 88 | 122 |

Reading the sensitivity results: with exact calibration A is better than B (84 vs 111 mm), B is better in consistency from about 1% calibration error and in position error from about 2%. B's accuracy barely depends on marker visibility down to 6% of samples. B beats a 5%-miscalibrated A at every mission length, with an advantage that grows with length (about 15% at 60 s, 35% at 120 s, 50% at 300 s); it stays behind an exactly calibrated A at every length.

Limits of these results (not yet tested): the true gains are constant in the synthetic plant, while the stage-2 MuJoCo measurements show gains that depend on speed (speed scale about 1.0 at low wheel speed and about 0.86 at 2 rad/s or more; effective track 0.35 to 0.42 m), so neither variant models the real plant exactly; the marker pose noise is Gaussian and consistent with the filter's model, whereas the render study shows heavy-tailed heading errors at long range (gating is implemented but untested); marker occlusion by obstacles is not modelled; no data from the MuJoCo robot has been run through the filter yet.

## Stage 5: EKF on MuJoCo data (first attempt invalid, calibration usable)

Code: `sim/mujoco_run.py`, `experiments/replay.py`; run: `notebooks/05_ekf_on_mujoco.ipynb`.

First collection attempt (240 to 300 s of driving, 3000 samples) is **invalid**: the robot hit the corner of an obstacle at about 45 s and stayed stuck, so the odometry ratios (0.188, -0.078, -0.008) and the filter results (position error 340 to 420 mm, ANEES above 365, marker visible in 2% of samples) are artefacts and are not reported as findings. Cause: the path check only tested the planned polyline, but the follower turns when it comes within 0.15 m of a waypoint (corner cutting) and the real robot turns slower than commanded (`s_w` about 0.72). Fixes: the clearance is now checked on a kinematic simulation of the follower with the measured gains (and with unit gains), required above 0.25 m (0.28 m reached after moving the lane to y = -0.08); the collection raises an error if the robot barely moves for 10 s; the lane along y = -0.08 faces the marker so it is in view for a large part of the run. Rendering is also skipped when the marker cannot be in view, because the first run took 984 s (about 0.33 s per frame).

Usable results from that attempt:

- Gain calibration in MuJoCo (one straight run at wheel speed 3 rad/s, one spin at 1 rad/s): `s_v = 0.855`, `s_w = 0.724`, equivalent effective track 0.398 m. This agrees with the stage-2 measurements (0.86 and about 0.72).
- Marker pose noise, 64 samples taken before the collision (preliminary, only two distance bins): position std x 8.2 mm and y 3.2 mm at 1.1 to 1.5 m (n = 51), x 10.0 mm and y 2.6 mm at 1.5 to 2.0 m (n = 13), heading std 0.87 and 0.58 degrees. The filter's model gives 13.3 and 24.1 mm and 5.1 and 9.1 degrees at the bin centres, so the model looks conservative on these samples; this is not a conclusion because the sample is small and does not cover short or long range.

### Stage 5: second attempt (valid run)

Run: 2400 samples (240 s), marker pose available in 47% of samples, smallest clearance 0.25 m, no collision; the robot drives a lane along y = -0.08 facing the marker, then a loop around the room, at nominal speeds 0.05, 0.10 and 0.15 m/s. Single-point calibration in the same session: `s_v = 0.855`, `s_w = 0.724`.

Odometry ratio (true motion over nominal-encoder motion) by speed, from the run itself:

| Quantity | Range | Median ratio | n |
|---|---|---|---|
| Linear speed, near-straight | 0.02 to 0.07 m/s | 0.940 | 1056 |
| Linear speed, near-straight | 0.07 to 0.12 m/s | 0.853 | 447 |
| Linear speed, near-straight | 0.12 to 0.25 m/s | 0.854 | 381 |
| Yaw rate while driving | 0.10 to 0.25 rad/s | 0.611 | 97 |
| Yaw rate while driving | 0.25 to 0.60 rad/s | 0.689 | 360 |

The gains depend on speed and yaw rate, and the yaw gain while driving (0.61 to 0.69) is lower than the in-place spin calibration (0.724), so a single-point calibration does not transfer to arcs. The synthetic study assumed constant gains.

Marker pose residuals (measured minus true) with the robot facing the marker, so the robot x axis is the line of sight:

| Distance [m] | n | x (depth) std [mm] | y (lateral) std [mm] | Heading std [deg] | Filter model position sigma [mm] | Filter model heading sigma [deg] |
|---|---|---|---|---|---|---|
| 0.2 to 0.7 | 26 | 1.1 | 0.6 | 1.21 | 1.9 | 0.87 |
| 0.7 to 1.1 | 116 | 3.1 | 0.4 | 2.69 | 6.4 | 2.61 |
| 1.1 to 1.5 | 114 | 4.9 | 0.7 | 5.66 | 13.3 | 5.13 |
| 1.5 to 2.0 | 189 | 10.9 | 0.8 | 7.11 | 24.1 | 9.06 |
| 2.0 to 3.5 | 685 | 34.4 | 3.9 | 8.02 | 59.4 | 21.95 |

The position error is mostly along the line of sight (depth), about 8 to 10 times the lateral error, which supports the earlier hypothesis of a depth error from the apparent marker size. The isotropic model over-estimates depth about 1.7 times and lateral error 5 to 30 times; the heading model is slightly low at short range and high at long range. These are standard deviations and do not include a systematic bias, which has not been examined; they cover only head-on views.

EKF variants replayed on this run (the filters never see the truth):

| Variant | Position RMSE [mm] | Position RMSE, second half [mm] | Max [mm] | Heading RMSE [deg] | ANEES (ideal 3) | Inside 95% bound | Rejected updates |
|---|---|---|---|---|---|---|---|
| A0 nominal gains | 2963 | 3730 | 6299 | 105.8 | 159155 | 0.09 | 0 |
| A single-point calibration | 377 | 387 | 817 | 12.4 | 133.6 | 0.005 | 0 |
| A calibrated + gate | 739 | 972 | 1636 | 40.0 | 209.2 | 0.005 | 553 |
| B estimated gains | 312 | 138 | 908 | 10.7 | 58.8 | 0.26 | 0 |
| B + gate | 313 | 139 | 908 | 10.7 | 58.9 | 0.26 | 1 |

B's final gains are `s_v = 0.959` and `s_w = 0.673`.

Findings:

1. B is better than the single-point-calibrated A in the second half (138 against 387 mm), as in the synthetic study, but every variant is overconfident (ANEES 59 to 134 against an ideal 3) and errors reach 0.8 to 0.9 m.
2. Position error drops to a few millimetres while the marker is in view in the lane and grows to about 1 m within roughly 45 s of dead reckoning around the room.
3. Gating at the 99.9% level hurts here: with A the filter drifts away, genuine marker measurements look like outliers and 553 of them are rejected (972 mm against 387 mm). Gating is only safe when the filter is already close.
4. B's `s_w` (0.673) agrees with the measured yaw ratio while driving (0.61 to 0.69), but its `s_v` (0.959) is higher than any measured linear ratio, so it is probably absorbing other errors instead of estimating the linear gain.

Planned next (not yet done): an anisotropic marker noise model with coefficients taken from the residual table, a speed- and yaw-rate-dependent odometry gain table calibrated with arc motions in an obstacle-free room (a fairer fixed-gain baseline), and a sensitivity study of the odometry process-noise scale.

## Stage 6: the same experiment run locally (MuJoCo 3.3.7) with the improved filter variants

Run with `experiments/mujoco_ekf.py` (output in `results/mujoco_ekf_local_mujoco3.3.7.txt`; environment in `docs/dev_environment.md`): 2400 samples, marker pose available in 48% of samples, smallest clearance 0.24 m. **These numbers come from MuJoCo 3.3.7 and differ from the Colab run (MuJoCo 3.15.0) above, so they must not be mixed with it.**

Dependence on the MuJoCo version:

| Quantity | Colab, 3.15.0 | Local, 3.3.7 |
|---|---|---|
| Single-point calibration `s_v`, `s_w` | 0.855, 0.724 | 0.869, 0.725 |
| Linear ratio, 0.02 to 0.07 / 0.07 to 0.12 / 0.12 to 0.25 m/s | 0.940 / 0.853 / 0.854 | 0.931 / 0.869 / 0.865 |
| Yaw ratio while driving, 0.10 to 0.25 / 0.25 to 0.60 rad/s | 0.611 / 0.689 | 0.842 / 0.727 |

The linear behaviour is similar, the yaw behaviour while driving is not (0.61 against 0.84 at low yaw rates). The odometry error of the simulated robot therefore depends on the simulator version, a reproducibility risk for any quantity that is calibrated.

Gain table measured on arcs in an obstacle-free room (rows: nominal speed 0.03, 0.06, 0.10, 0.15 m/s; columns: yaw rate 0, 0.15, 0.3, 0.5 rad/s):

| s_v | 0 | 0.15 | 0.3 | 0.5 |
|---|---|---|---|---|
| 0.03 | 0.968 | 0.857 | 0.793 | 0.535 |
| 0.06 | 0.902 | 0.895 | 0.865 | 0.765 |
| 0.10 | 0.869 | 0.867 | 0.880 | 0.855 |
| 0.15 | 0.867 | 0.857 | 0.856 | 0.866 |

| s_w | 0.15 | 0.3 | 0.5 |
|---|---|---|---|
| 0.03 | 0.839 | 0.809 | 0.709 |
| 0.06 | 0.877 | 0.879 | 0.802 |
| 0.10 | 0.854 | 0.859 | 0.825 |
| 0.15 | 0.646 | 0.765 | 0.831 |

Variants replayed on this run:

| Variant | Position RMSE [mm] | Second half [mm] | Max [mm] | Heading RMSE [deg] | ANEES (ideal 3) | Inside 95% | Rejected |
|---|---|---|---|---|---|---|---|
| A0 nominal gains | 1199 | 1348 | 2916 | 40.2 | 2169 | 0.18 | 0 |
| A single-point | 185 | 183 | 413 | 5.6 | 67.6 | 0.02 | 0 |
| A single-point + gate | 185 | 183 | 413 | 5.6 | 67.6 | 0.02 | 0 |
| A gain table | 102 | 96 | 250 | 3.0 | 28.3 | 0.26 | 0 |
| A table + anisotropic R | 158 | 93 | 746 | 3.9 | 71.7 | 0.17 | 0 |
| A table + anisotropic R, Q x2 | 147 | 111 | 712 | 3.6 | 35.7 | 0.39 | 0 |
| A table + anisotropic R, Q x4 | 134 | 126 | 627 | 3.4 | 18.4 | 0.50 | 0 |
| B | 107 | 80 | 260 | 2.1 | 31.8 | 0.36 | 0 |
| B + anisotropic R | 164 | 92 | 746 | 3.2 | 57.3 | 0.13 | 0 |
| B + anisotropic R, Q x2 | 149 | 107 | 713 | 2.7 | 31.6 | 0.27 | 0 |
| B + anisotropic R, Q x4 | 144 | 117 | 629 | 4.0 | 17.7 | 0.48 | 0 |

Findings (single run per variant, MuJoCo 3.3.7 only, so differences between neighbouring rows are not established):

1. The gain table measured on arcs is a much better fixed-gain baseline than the single-point calibration: 96 against 183 mm in the second half and ANEES 28 against 68.
2. B (80 mm, heading 2.1 degrees, ANEES 32) is close to the gain-table filter (96 mm, 3.0 degrees, ANEES 28): slightly better in position and heading and about equal in consistency. The advantage of B over the single-point baseline reproduces (80 against 183 mm), but against the better fixed-gain baseline it is small and not conclusive.
3. The anisotropic marker noise model made both families worse in the maximum error (250 to 746 mm for the table filter) and in ANEES (28 to 72 for the table filter, 32 to 57 for B). The reason is not established. Hypotheses, none tested: it trusts the lateral and heading measurements too much at long range because the position and heading errors of one marker measurement are correlated and the model treats them as independent; it was fitted to head-on views only while the replay includes other views.
4. Scaling the odometry process noise improves consistency (ANEES 72, 36, 18 for factors 1, 2, 4 with the table filter) but never reaches 3 and does not improve accuracy.
5. No update was rejected by the gate in this run, since the filters stay close enough to the truth; the earlier Colab run showed the opposite.
6. All variants remain overconfident (ANEES at least 17).

## Open items

- Confirm the robot (Burger or Waffle Pi) held by the hardware lab.
- Camera extrinsics (mounting offset) are ignored in the stage-1 measurement models and belong to the perception stage.
- Characterise odometry over arc motions inside the planned operating envelope, then either calibrate by speed range or absorb the residual in the EKF process noise; measure dead-reckoning error with and without calibration.
- Add the sensors (LiDAR, camera), the room with obstacles and the ArUco marker. Encoders are already provided by the driver.
