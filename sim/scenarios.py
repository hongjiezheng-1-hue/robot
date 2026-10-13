"""Test scenarios (obstacle layouts and station placements) and disturbances for the navigation experiments.

The scenarios were written down before they were run, with the planner and estimator frozen at the commit that precedes this
file. Only whether a path exists at all was checked offline (experiments/scenario_feasibility.py); no scenario was made easier
after seeing a result. Room 4 x 3 m (walls at x = +-2, y = +-1.5). Boards are (x, y, size_x, size_y).
"""

import math
from dataclasses import dataclass, field

STATION_EAST = (1.85, 0.0, math.pi)             # on the east wall, marker faces -x (the default)
STATION_NORTH = (0.5, 1.42, -math.pi / 2)       # on the north wall, marker faces -y


@dataclass
class Scenario:
    name: str
    description: str
    obstacles: tuple
    start: tuple                                 # nominal start pose (x, y, heading) that the robot believes
    station: tuple = STATION_EAST                # (x, y, yaw of the marker normal)


SCENARIOS = {s.name: s for s in [
    Scenario("s_course", "three staggered boards (the course used so far)",
             ((-1.5, 0.65, 0.10, 1.70), (-0.5, -0.65, 0.10, 1.70), (0.5, 0.65, 0.10, 1.70)), (-1.65, -0.6, 0.0)),
    Scenario("s_mirrored", "the S course mirrored in y",
             ((-1.5, -0.65, 0.10, 1.70), (-0.5, 0.65, 0.10, 1.70), (0.5, -0.65, 0.10, 1.70)), (-1.65, 0.6, 0.0)),
    Scenario("gate", "one wall with a 0.7 m gap in the middle, start off-axis",
             ((-0.4, 0.95, 0.10, 1.10), (-0.4, -0.95, 0.10, 1.10)), (-1.65, 0.9, 0.0)),
    Scenario("scattered", "five 0.3 m boxes in the middle of the room",
             ((-1.1, 0.5, 0.30, 0.30), (-0.5, -0.6, 0.30, 0.30), (0.0, 0.4, 0.30, 0.30), (0.35, -0.7, 0.30, 0.30), (-0.2, -1.15, 0.30, 0.30)),
             (-1.65, -0.2, 0.0)),
    Scenario("u_trap", "a U-shaped pocket open to the west on the straight line to the goal",
             ((-0.3, 0.55, 1.00, 0.10), (-0.3, -0.55, 1.00, 0.10), (0.15, 0.0, 0.10, 1.10)), (-1.65, 0.0, 0.0)),
    Scenario("north_station", "station on the north wall (marker faces south), one board",
             ((-0.4, 0.35, 0.10, 1.20),), (-1.65, -0.6, 0.0), STATION_NORTH),
    Scenario("empty", "no obstacles", (), (-1.65, -0.6, 0.0)),
]}


@dataclass
class Disturbance:
    name: str
    description: str
    lidar_noise_scale: float = 1.0               # multiplies the LiDAR noise
    start_offset_scale: float = 1.0              # multiplies the start offset (5 cm, 3 deg, one standard deviation)
    marker_dropout: float = 0.0                  # probability that a camera frame is lost
    wheel_radius_error: float = 0.0              # the robot believes a wheel radius that is this fraction too large
    push: dict = field(default_factory=dict)     # lateral force pulse on the base: {"time": s, "force": N, "duration": s}


DISTURBANCES = {d.name: d for d in [
    Disturbance("none", "nominal"),
    Disturbance("lidar_x3", "LiDAR noise three times larger", lidar_noise_scale=3.0),
    Disturbance("start_x3", "start offset three times larger (15 cm, 9 deg)", start_offset_scale=3.0),
    Disturbance("dropout50", "half of the camera frames lost", marker_dropout=0.5),
    Disturbance("radius+5", "believed wheel radius 5 percent too large", wheel_radius_error=0.05),
    Disturbance("push", "lateral push on the base while driving", push={"time": 20.0, "force": 30.0, "duration": 0.15}),
]}
