"""Docking geometry relative to the marker, and the provisional acceptance test.

The charging station is defined by its marker: the dock pose is a point on the marker normal, `dock_distance` in front of the marker
plane, heading towards the marker. The stand-off pose lies further out on the same normal. Both distances are provisional values
(the real station dimensions and the dock tolerance have to be confirmed with the course staff).
"""

from dataclasses import dataclass

import numpy as np

from estimation.models import wrap


@dataclass
class DockSpec:
    standoff: float = 0.95          # [m] base centre to marker plane at the stand-off pose
    distance: float = 0.45          # [m] base centre to marker plane when docked
    lateral_tol: float = 0.03       # [m]
    along_tol: float = 0.05         # [m]
    heading_tol_deg: float = 5.0    # [deg]


def dock_poses(marker_world, spec):
    """(stand-off pose, dock pose) as (x, y, heading). The marker normal points out of the marker; the robot faces against it."""
    mx, my, normal = marker_world
    n = np.array([np.cos(normal), np.sin(normal)])
    heading = float(wrap(normal + np.pi))
    standoff = (mx + spec.standoff * n[0], my + spec.standoff * n[1], heading)
    dock = (mx + spec.distance * n[0], my + spec.distance * n[1], heading)
    return standoff, dock


def dock_error(pose, dock):
    """Along-track, lateral and heading error (deg) of a pose with respect to the dock pose, in the dock frame."""
    c, s = np.cos(dock[2]), np.sin(dock[2])
    d = np.asarray(pose[:2]) - np.asarray(dock[:2])
    return float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1]), float(np.degrees(wrap(pose[2] - dock[2])))


def docked(pose, dock, spec):
    along, lateral, heading = dock_error(pose, dock)
    return abs(along) < spec.along_tol and abs(lateral) < spec.lateral_tol and abs(heading) < spec.heading_tol_deg
