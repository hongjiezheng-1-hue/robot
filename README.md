# 41014/42043 Integrated Project - Track A: Visual Waypoint Navigation

TurtleBot3 with 2D LiDAR and camera in MuJoCo. The robot navigates a room with obstacles and visually docks to an ArUco-marked charging station. The controller uses the estimated state only; simulator ground truth is used for evaluation only.

Status: work in progress (simulation first, hardware optional afterwards). Running notes with measured results: [docs/findings.md](docs/findings.md).

## 1. Overview
<!-- Task, platform, what the pipeline does end to end. -->

## 2. Model and frames
<!-- State-space model, assumptions, frame diagram (world / base / camera / marker), controllability and observability analysis. -->

## 3. Sensing
<!-- LiDAR processing, camera intrinsics, ArUco detection, noise characterisation, frame transforms. -->

## 4. Estimation
<!-- EKF design: state, motion model, measurement models, Q and R with justification, NEES/consistency results. -->

## 5. Control
<!-- LQR tracker on the estimated state: weights and why, saturation handling, docking logic. -->

## 6. Results
<!-- Quantitative results over repeated trials with varied conditions; failure analysis; trade-offs. -->

## 7. Install and run
```bash
pip install -r requirements.txt
```
<!-- Exact commands to reproduce the results from a fresh environment. -->

## 8. Demo video
<!-- Link (5 minutes): task completion, a varied/disturbed trial, estimate and uncertainty alongside the true state. -->

## 9. Contribution statement
<!-- Individual project (42043): one member. -->

## 10. Generative AI declaration
<!-- State the tools used and what they were used for. -->

## 11. Acknowledgements
<!-- Weekly tutorial code, MuJoCo examples and Robotics Toolbox material adapted in this project. -->
