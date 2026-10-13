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

Code: `estimation/ekf.py`, `sim/synthetic.py`; experiments: `experiments/ekf_synthetic.py`, `experiments/ekf_sensitivity.py` (outputs in `results/synthetic/`); tests: `tests/test_ekf.py`, `tests/test_models.py`.

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

Code: `sim/mujoco_run.py`, `experiments/replay.py`; run: `notebooks/05_ekf_on_mujoco.ipynb` (the Colab runs below used the first version of this notebook; it was later merged with the improved-filter notebook and now contains all the variants), or `experiments/mujoco_ekf.py` locally.

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

Run with `experiments/mujoco_ekf.py` (output in `results/mujoco_3.3.7/mujoco_ekf.txt`; environment in `docs/dev_environment.md`): 2400 samples, marker pose available in 48% of samples, smallest clearance 0.24 m. **These numbers come from MuJoCo 3.3.7 and differ from the Colab run (MuJoCo 3.15.0) above, so they must not be mixed with it.**

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

## Stage 7: why the anisotropic marker noise model was worse, and sub-pixel corner refinement (MuJoCo 3.3.7)

Scripts: `experiments/noise_model_study.py`, `experiments/mujoco_ekf.py --subpixel`; outputs in `results/mujoco_3.3.7/`. Two local runs of 2400 samples: corner refinement off (the stage 6 run) and on. Single runs, so small differences are not established.

Diagnosis on the run without refinement:

1. The marker pose error has a systematic part, not only scatter. Depth (x) mean / std [mm] by distance 0.2 to 0.7, 0.7 to 1.1, 1.1 to 1.5, 1.5 to 2.0, 2.0 to 3.5 m: 1.5 / 1.0, 4.6 / 3.1, 9.1 / 5.7, 12.1 / 11.6, 32.3 / 33.9. Lateral (y): 1.0 / 0.4, 1.4 / 0.4, 2.0 / 0.7, 2.4 / 0.9, 4.4 / 4.2. The anisotropic model of stage 5 was fitted to the standard deviation only, so it under-stated the real error, which the filter does not model as a bias.
2. The errors are correlated: corr(y, heading) about -0.3 to -0.5 and corr(x, y) about +0.4 to +0.6 below 2 m; the model treats them as independent.
3. Lateral error grows with obliquity (mean 3.4 mm head-on, 11.8 and 17.7 mm at 5 to 15 and 15 to 30 degrees, few samples), while 859 of the 901 samples beyond 1.5 m were head-on, which is what the model was fitted to.
4. The largest filter error (746 mm) occurs at t = 0.2 s, during the initial convergence, and the first 30 s account for the difference: mean position error 270 mm with the anisotropic model against 56 mm with the isotropic one. Over all marker updates the isotropic model is over-conservative (mean normalised innovation 0.7, ideal 3) and the anisotropic one is closer (2.1), yet its estimates are worse.

Noise model variants with the B filter (first 30 s mean position error / second-half position RMSE [mm]; ANEES in brackets):

| Model | No corner refinement | Sub-pixel refinement |
|---|---|---|
| Isotropic (original) | 56 / 80.0 (31.8) | 75 / 66.7 (33.7) |
| Anisotropic, fitted to std | 270 / 92.3 (57.3) | 282 / 57.4 (50.1) |
| Anisotropic, fitted to RMS (bias included) | 143 / 97.3 (42.2) | 162 / 58.1 (37.9) |
| Anisotropic RMS, lateral sigma x5 | 53 / 99.1 (40.7) | 46 / 62.1 (35.6) |
| Hybrid (anisotropic RMS below 1.5 m, isotropic beyond) | 56 / 73.2 (33.1) | 75 / 60.1 (34.4) |
| Isotropic, every 5th update only | 101 / 70.6 (19.5) | 139 / 77.7 (20.0) |
| Anisotropic RMS, every 5th update only | 229 / 73.4 (24.7) | 262 / 68.8 (25.0) |

Findings consistent over both runs: the poor start of the anisotropic model comes from a lateral sigma that is too small at long range (a larger lateral sigma restores the start); the hybrid model keeps the isotropic start and is about 10% better than the isotropic model in the second half (73.2 against 80.0 and 60.1 against 66.7). Not reproduced: using only every 5th update helps consistency (ANEES) but not accuracy; the earlier idea that correlated frame-to-frame errors explain the problem is therefore not supported as an explanation of accuracy.

Sub-pixel corner refinement (OpenCV `CORNER_REFINE_SUBPIX`), marker error mean / RMS [mm], heading RMS [deg], by distance:

| Distance [m] | Depth x, off | Depth x, on | Heading, off | Heading, on |
|---|---|---|---|---|
| 0.7 to 1.1 | 4.6 / 5.5 | 0.3 / 1.5 | 2.9 | 0.66 |
| 1.1 to 1.5 | 9.1 / 10.7 | 5.5 / 5.9 | 5.3 | 0.64 |
| 1.5 to 2.0 | 12.1 / 16.8 | 10.7 / 10.8 | 7.2 | 1.05 |
| 2.0 to 3.5 | 32.3 / 46.8 | 22.3 / 29.8 | 8.2 | 7.5 |

Refinement reduces the scatter several times over (heading error 5 to 8 times at 1.1 to 2 m) but not the systematic depth bias (about 0.003 d squared) or the lateral bias of 1 to 4 mm. It is therefore not caused by corner localisation. A possible cause is the texture filtering of the simulated render shrinking the black square by about 0.2 px and a half-pixel principal-point convention; neither was tested, and a real camera may behave differently. At 2 to 3.5 m the marker is about 20 px wide and the heading error stays near 7.5 degrees.

With refinement the second-half position error of the B filter improves from 80.0 to 66.7 mm (isotropic model) and the anisotropic variants become better than the isotropic one (57 against 67 mm); the maximum error stays at 640 to 720 mm for the anisotropic model because of the initial convergence problem above.

Not yet done: the noise coefficients were read off the same runs they are evaluated on, so a held-out run with a different path and different speeds is needed before the hybrid model is adopted; the origin of the depth bias.

## Stage 8: independent validation on new paths (MuJoCo 3.3.7, sub-pixel corner refinement on)

Script: `experiments/validation_runs.py`; output: `results/mujoco_3.3.7/validation_runs.txt`; paths in `sim/mujoco_run.py` (`VALIDATION_PATHS`). Nothing was refitted: the noise coefficients (read off the default run), the hybrid switch distance (1.5 m), the gain table and the single-point gains (calibrated in their own short runs) are used as they are. Each path is 240 s, 2400 samples, one run.

| Path | Marker in view | Viewing angle when seen | Smallest clearance |
|---|---|---|---|
| diagonal (development) | 35% | median 24 degrees, 98% above 15 degrees | 0.26 m |
| reverse (development) | 35% | median 0.1 degrees (head-on) | 0.28 m |
| zigzag (final test) | 50% | median 9.6 degrees | 0.25 m |

The marker pose error reproduces on all three paths: depth error about +10 mm at 1.5 to 2.0 m and +28 to +34 mm at 2.0 to 3.5 m, lateral error 1 to 3 mm, heading error 0.4 to 0.8 degrees up to 2 m and 11 to 14 degrees RMS at 2 to 3.5 m.

The paths diagonal and reverse were used to see how the variants behave and variant C (below) was designed after seeing them, so they are development data. The path zigzag was added after all variants were fixed and is the final test; its geometry was only adjusted for obstacle clearance before any filter was run on it.

Second-half position RMSE [mm] (ANEES in brackets; ideal ANEES is 3):

| Variant | diagonal | reverse | zigzag (final test) | Mean of the three |
|---|---|---|---|---|
| A single-point calibration | 258.8 (41.9) | 106.8 (36.7) | 133.1 (36.9) | 166.3 |
| A gain table (measured on arcs) | 131.4 (18.6) | 66.0 (19.5) | **41.6** (17.7) | 79.7 |
| A gain table + hybrid noise | 130.1 (18.8) | 66.7 (20.6) | 40.2 (17.9) | 79.0 |
| B gains estimated online, isotropic noise | 89.1 (18.6) | 72.5 (19.3) | 113.9 (20.1) | 91.8 |
| B + hybrid noise | 92.1 (17.7) | 63.6 (21.5) | 111.5 (20.4) | 89.1 |
| B + anisotropic noise, fitted to std | 129.1 (85.1) | 114.6 (49.2) | 126.3 (32.8) | 123.3 |
| B + anisotropic noise, fitted to RMS | 134.0 (52.2) | 129.0 (33.6) | 132.1 (29.0) | 131.7 |
| B + anisotropic RMS, lateral sigma x5 | 122.7 (47.1) | 132.7 (30.0) | 132.9 (31.6) | 129.4 |
| C gain table with online correction | 61.2 (11.9) | 102.2 (12.0) | 57.1 (11.3) | 73.5 |
| C with odometry noise x2 | 92.4 (7.2) | 113.1 (7.3) | 74.8 (7.5) | 93.4 |

Maximum position error on the final test path: A gain table 299 mm, B 300 mm, C 299 mm, anisotropic variants 521 to 766 mm.

Findings:

1. The gain table measured on arcs is the strongest fixed-gain baseline and the result holds on all three paths (41.6 against 133.1 mm for the single-point calibration on the final test path).
2. Estimating constant gains online (B) is **not** better than that table: 113.9 against 41.6 mm on the final test path. B looked better on the development paths (80.8 against 98.7 mm), and on the diagonal path it produced a 1.4 m excursion. The conclusion of stage 6 that B is better than a fixed-gain filter held only against the single-point calibration.
3. Cause of the excursion (diagonal path): the marker was last seen during a slow oblique approach, where B estimated `s_w = 0.66`; the filter then dead-reckoned for 60 s without the marker (position error 0.4 to 1.4 m, heading error up to 41 degrees). The true gains depend on speed and yaw rate, which a constant gain cannot represent, and B's estimate depends on the regime in which the marker happened to be in view. At 3.1 m the marker heading also shows outliers of +25 to +37 degrees between measurements within a few degrees, the planar pose ambiguity.
4. The anisotropic marker noise models are worse than the isotropic one on all three paths (123 to 132 against 92 mm on average, ANEES 29 to 85). They should not be used.
5. The hybrid noise model has no reliable effect (on top of the gain table: 40.2 against 41.6, 130.1 against 131.4, 66.7 against 66.0 mm). The earlier statement that it is about 10% better is withdrawn; it was a single-run coincidence.
6. Variant C is better than B on the final test path (57.1 against 113.9 mm) but not better than the table alone (41.6 mm), and wins on one path and loses on two. Its benefit is consistency (ANEES 11.3 against 17.7, 61% inside the bound against 42%), not reliable accuracy.
7. Every variant remains overconfident; the best ANEES is 7.2 to 7.5 (C with doubled odometry noise), against an ideal 3.

Limits: the gain table was calibrated in the same simulator under ideal conditions, which a real robot will not allow; with a noisier or coarser calibration the online estimation (B or C) may matter more, so these results cannot simply be transferred to hardware. Only MuJoCo 3.3.7 and three paths were used, and the simulator is deterministic, so "independent" means independent paths. The gain table needs 16 short runs in an obstacle-free room.

## Stage 9: LQR trajectory tracking with the EKF in the loop (MuJoCo 3.3.7)

Setup: `experiments/lqr_tracking.py`, `sim/closed_loop.py`, `control/`. The reference is a 10.5 m loop of the room with rounded corners that ends in a head-on approach to the charging station (131 s at 0.08 m/s). The LQR tracker (error model in the robot frame, discrete gains recomputed at every step, Bryson weights 0.10 m, 0.05 m, 10 degrees and 0.05 m/s, 0.3 rad/s) acts on the **estimated** pose only; the truth is logged after each step. The true start is offset from the nominal mark (5 cm, 3 degrees, one standard deviation) while the robot believes it is at the mark. Three trials per variant, with speeds 0.05, 0.08 and 0.12 m/s in turn (speed and start offset are therefore confounded). Dock tolerance (provisional, to be confirmed with staff): lateral below 3 cm, along-track below 5 cm, heading below 5 degrees.

| Variant | Docked in tolerance | Tracking RMSE [mm] (per trial) | Estimator position RMSE [mm] |
|---|---|---|---|
| table, command compensated | 2 / 3 | 907, 67, 56 | 1063, 61, 50 |
| table, no compensation | 2 / 3 | 900, 92, 53 | 1038, 83, 57 |
| single-point gains | 0 / 3 | 631, 634, 612 | 839, 720, 624 |
| B (gains estimated online) | 3 / 3 | 108, 105, 64 | 101, 94, 62 |
| C (table x online correction) | 3 / 3 | 103, 101, 52 | 99, 95, 48 |

Findings:

1. When the filter is good the loop works: the final dock error was 1 to 29 mm along-track, 0 to 6 mm lateral and below 1 degree in every trial that did not diverge, with tracking RMSE 52 to 108 mm. The tracking error is of the same size as the estimator error (for example 103 against 99 mm), so the estimator, not the LQR, limits the tracking accuracy.
2. The single failure of the table variants (trial 0, 0.05 m/s, with and without command compensation) is estimator drift without the marker. On the long leg facing away from the marker the filter has no correction for about 90 s; the heading error grew to about 7 degrees, the robot (which trusts the estimate) left its reference line and the estimate then diverged by metres. The robot was about 0.16 m from the south wall at that moment, so it probably touched the wall (contacts were not logged in this experiment). Variant C had the same drift (heading error up to about 9 degrees) but the marker came back into view in time; this is not evidence that C is better, only that it was less unlucky in one trial.
3. The single-point odometry calibration fails closed-loop in all three trials (estimator RMSE 624 to 839 mm), as stage 8 suggested for dead-reckoning stretches.
4. I cannot rank table, B and C from three trials with confounded speed and start offset. Compensating the command with the inverse gain table helped in one of two comparable trials (67 against 92 mm) and made no difference in the other (56 against 53 mm); this is not enough to claim a benefit.
5. Two of the five runs of the first batch exited silently (variants table and B printed only the header). They were rerun with stderr captured and finished normally; the cause was not identified.

Consequence for the design: a route on which the marker is out of view for more than about a minute is not safe with this estimator alone. The LiDAR map (stage 10) can detect a growing mismatch only if scans are matched against it, which is not implemented.

## Stage 10: LiDAR navigation through unknown obstacles and docking (MuJoCo 3.3.7)

Setup: `experiments/lidar_navigation.py`, `sim/navigation.py`, `perception/lidar.py`, `perception/occupancy.py`, `control/planner.py`, `control/docking.py`. Three thin boards (0.10 x 1.70 m) are placed so that the route weaves up, down and up again (an S); the robot is not told where they are. LiDAR: 360 beams, 0.12 to 3.5 m, 5 Hz, noise 1% of the range (floor 1 cm); these numbers are from my memory of the LDS-01, not checked against the documentation, and the mount height is an assumption. The MuJoCo ray-cast scan matches an analytic box scan exactly on the boards and walls (`experiments/lidar_check.py`; the station is only approximated analytically). The robot builds a log-odds occupancy grid **from the estimated pose**, plans with A* on the inflated grid (unknown cells count as free, so it re-plans when a scan shows the path is blocked), rotates in place to the first segment, follows the rounded path with the LQR tracker, stops at a stand-off pose 0.95 m in front of the marker, aligns, and drives straight to the dock pose 0.45 m from the marker plane at 0.05 m/s. Provisional dock tolerance (to be confirmed with staff): lateral below 3 cm, along-track below 5 cm, heading below 5 degrees. Each trial starts the true robot 5 cm and 3 degrees (one standard deviation) away from the nominal start the robot believes. Collisions are counted from MuJoCo contacts between the robot and the boards, walls or station (evaluation only).

Results (docked within tolerance / collisions / planning failures; 3 odometry variants):

| Seeds | Status of the seeds | table | B | C |
|---|---|---|---|---|
| 200 to 204 | development (planner tuned on failures seen here) | 5/5, 0, 0 | 5/5, 0, 0 | 5/5, 0, 0 |
| 300 to 305 | before the map reset (below) | 4/6, **4**, 0 | not run | 4/6, **4**, 0 |
| 300 to 305 | after the map reset (tuned on these) | 6/6, 0, 0 | not run | 6/6, 0, 0 |
| 400 to 407 | **fresh test**, planner as after the map reset | 7/8, 0, 1 | 8/8, 0, 0 | 7/8, 0, 1 |
| 400 to 407 | after the margin fallback (no longer independent) | 8/8, 0, 0 | not run | 8/8, 0, 0 |

Final dock errors in the successful runs: along-track at most 9 mm, lateral at most 20 mm, heading at most 3.9 degrees. The heading error was up to 6 degrees in earlier versions, where the stand-off was 0.80 m and the dock line started at the estimate instead of its projection on the dock axis. Run time 68 to 95 s, 6 to 20 plans per run. The smallest distance from the true robot centre to a board was 174 mm over all runs of the final planner (164 mm in an earlier version), which is only about 1 cm more than the half-width of the robot (about 153 mm, from the 306 mm width I recall for the Waffle Pi; not measured), so the margins are thin.

What was learned and fixed (each fix was made after seeing the failure, which is why the seed sets are labelled):

1. Re-plan loop: the planner relaxes the inflation near the start so a robot close to an obstacle can leave, but the blocked-path test did not, and the robot re-planned every 0.4 s without moving. Now both ignore the first 0.35 m and use the same hard radius (0.17 m); the relaxation shrinks further if the robot's own cell would otherwise be blocked.
2. Stale map after a pose correction (the important one): with the marker out of view the position error grew to 0.4 to 0.85 m; when the marker came back the estimate jumped, but the boards were already in the grid at the wrong place, and the robot hit the end of a board in 2 of 6 trials (the other two contacts on seeds 300 to 305 were the robot touching the wall at its start, see 4). Now the grid is cleared when a marker update moves the position estimate by more than 0.10 m, and the robot re-plans. This removed all collisions on the 300 series and none occurred on the fresh 400 series.
3. Fresh test failure: in 1 of 8 trials (both table and C, same start) no path existed at the 0.22 m margin in the narrow channel between two boards, because the map was shifted by about 0.15 m. Planning now retries with margins 0.22, 0.19 and 0.17 m. That trial then docked, but the seed set is no longer independent after this change.
4. The nominal start (-1.75, -0.6) was too close to the west wall (the rear of the robot touches the wall at x of about -1.8); it was moved to (-1.65, -0.6). Contacts at the very start were a scenario artefact, not navigation errors.

Observations and limits:

- The estimator error is large between marker sightings (position RMSE 12 to 430 mm per run, maximum 754 mm in the final series, 848 mm before the map reset), yet the robot avoided the boards in all runs after fix 2, because the LiDAR sees obstacles relative to the robot. Docking depends on the marker being reacquired; the stand-off pose lies in front of the marker, where it is visible.
- B (gains estimated online) docked in 8 of 8 fresh trials, table and C in 7 of 8. With one course and 8 trials this is not a difference I can claim; the failure of table and C was a planning-margin problem, not an estimator problem, and was fixed afterwards.
- One course, one marker placement, the board positions known to me when tuning, a deterministic simulator and the ideal gain table calibrated in the same simulator. A different board layout is the next independent check; the 400 series is independent only with respect to the start offset and noise.
- The robot always rotates in place before following a re-planned path; there is no obstacle avoidance while docking (the station is excluded from the map by design); no scan matching, so the map is only as good as the pose estimate.
- The marker world pose is assumed known (it is in the scene). Docking is done with the EKF estimate, which fuses the marker measurement, rather than with a separate image-based controller on the measured marker pose.
- LiDAR parameters, dock distance and dock tolerance are unverified assumptions; real LiDAR returns (reflectivity, the thin board edges, the robot's own mast) will behave differently.

## Stage 11: more layouts and disturbances (MuJoCo 3.3.7)

Setup: `sim/scenarios.py`, `experiments/scenario_suite.py`, `experiments/summarise_scenarios.py`, raw results in `results/scenarios/`. Seven layouts (the S course, its mirror image, a gate with a 0.7 m gap, five scattered boxes, a U-shaped pocket on the straight line to the goal, a station on the north wall, an empty room) with 6 start offsets and noise seeds each, and five disturbances on the S course (LiDAR noise x3, start offset x3 = 15 cm and 9 degrees, half of the camera frames lost, believed wheel radius 5 percent too large, a lateral push of 30 N for 0.15 s at t = 20 s that the encoders cannot see), 6 seeds each. Variants C and B. The layouts were written down before the first run and only checked for solvability with a known map (`experiments/scenario_feasibility.py`); none was changed afterwards. Outcome per run: docked in tolerance without contact, collision, no path, or not docked.

| Round | What changed before it | Seeds | C layouts | C disturbances | B layouts | B disturbances |
|---|---|---|---|---|---|---|
| R1 | nothing (planner frozen) | 500 to 505, 600 to 605 | 42/42 | 25/30 | 36/42 | 25/30 |
| R2 | map reset and re-plan when no path is found or plans thrash | 700 to 705, 800 to 805 | 42/42 | 22/30 | 37/42 | 24/30 |
| R3 | marker search, footprint clearing, goal relaxation (below) | 900 to 905, 1000 to 1005 | 42/42 | 26/30 | 38/42 | 25/30 |

Rounds R2 and R3 used new seeds, but the scenarios themselves were already seen in R1, and the changes were made after looking at R1 and R2 failures, so only R1 is a clean first-run result and R2 and R3 are development results on new seeds. Seeds 800 to 805 were also used to test the changes before R3 (push 6/6, start offset 5/6), so the R3 seeds are the only ones not used for any decision.

Per scenario (docked / 6, C and B, R1 / R2 / R3): all layouts except u_trap and one B run are 6/6 for C in every round (126 of 126 layouts runs). B on u_trap: 0, 2, 3 of 6. Disturbances, C: LiDAR x3 5, 4, 5; start offset x3 4, 3, 5; half frames lost 6, 6, 4; wheel radius +5 percent 6, 6, 6; push 4, 3, 6. B: 6, 6, 5; 4, 4, 4; 6, 6, 6; 6, 5, 5; 3, 3, 5.

Docked runs in R3: along-track at most 12 mm, lateral at most 25 mm, heading at most 4.9 degrees; closest approach to a board 154 mm (the robot half-width is about 153 mm, not measured), so contact is close in the worst case.

Findings and the changes made:

1. **In-place rotation with variant B.** B starts with both gains at 1 and only learns them while moving, but in place the real yaw gain is about 0.72; a large first turn then smears the first scans into the map and no path is found (u_trap, a 53 degree first turn: 0 of 6). Variants with the gain table (table, C) do not have this problem in the layouts used. B is therefore not suitable as the default.
2. **A wrong pose makes the goal unreachable (the main weakness).** After a large start offset or a push, the estimated pose and so the whole map are shifted; the stand-off pose, defined in the world frame, then falls inside a mapped board and no path exists. Changes: (a) the planner now starts from inside a conflicting map (the cells under the robot are cleared), (b) a blocked goal is replaced by the nearest free cell within 0.8 m (the robot gets closer, sees the marker and the pose is corrected), (c) a recovery that turns towards the marker for up to 10 s before the map is cleared. On development seeds this took the push case from 3/6 to 6/6 and the start offset case from 3/6 to 5/6; on R3 seeds 6/6 and 5/6.
3. **Remaining failures (R3, C): four collisions.** LiDAR noise x3 (1 of 6), start offset x3 (1 of 6, in contact at 0 s because the 15 cm offset places the robot against the wall, then lost with 0.66 m estimator error) and half of the camera frames lost (2 of 6, estimator error 0.2 and 1.0 m). With the marker unavailable for long stretches the pose estimate drifts and the map is built on it; there is no scan matching against the known room walls, which is the missing piece.
4. The marker-jump map reset covers position jumps only; a heading correction of 20 degrees without a position jump is not detected.
5. Two parallel runs crashed once because they rewrote the marker texture at the same time; scene files are now written atomically and only when they change.

Limits: simulation only, one robot model, MuJoCo 3.3.7, 6 seeds per cell (a cell of 6/6 does not show a rate above about 60 percent with 95 percent confidence). The layouts are mine; nothing was tested on a layout drawn by someone else.

## Open items

- Confirm the robot (Burger or Waffle Pi) held by the hardware lab.
- Camera extrinsics (mounting offset) are ignored in the stage-1 measurement models and belong to the perception stage.
- Characterise odometry over arc motions inside the planned operating envelope, then either calibrate by speed range or absorb the residual in the EKF process noise; measure dead-reckoning error with and without calibration.
- Test on a different board layout and with several marker placements; add scan matching so that the map does not depend on the marker; check LiDAR spec and dock tolerance with staff; hardware measurements per the Thursday checklist.
