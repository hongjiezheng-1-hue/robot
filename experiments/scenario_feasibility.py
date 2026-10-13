"""Offline check that a path exists in each scenario when the map is known (it says nothing about whether the robot finds it).

    python experiments/scenario_feasibility.py

The true boards and walls are rasterised into an occupancy grid and planned on with the same margins as the robot. Only
scenarios that fail here (no path even with full knowledge) would be unsolvable; the check does not make scenarios easier.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.docking import DockSpec, dock_poses  # noqa: E402
from control.planner import plan, plan_with_margins  # noqa: E402
from perception.occupancy import OccupancyGrid  # noqa: E402
from sim.scenarios import SCENARIOS  # noqa: E402


def true_grid(scen, room=(4.0, 3.0)):
    g = OccupancyGrid()
    ix, iy = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]), indexing="ij")
    x, y = g.to_world(ix, iy)
    hx, hy = room[0] / 2, room[1] / 2
    boxes = list(scen.obstacles) + [(0, hy + 0.025, 2 * hx + 0.1, 0.05), (0, -hy - 0.025, 2 * hx + 0.1, 0.05),
                                    (hx + 0.025, 0, 0.05, 2 * hy), (-hx - 0.025, 0, 0.05, 2 * hy)]
    sx, sy, syaw = scen.station
    c, s = np.cos(syaw), np.sin(syaw)
    u, v = (x - sx) * c + (y - sy) * s, -(x - sx) * s + (y - sy) * c           # station frame: depth along the normal
    occ = (np.abs(u) < 0.08) & (np.abs(v) < 0.14)
    for cx, cy, bx, by in boxes:
        occ |= (np.abs(x - cx) <= bx / 2) & (np.abs(y - cy) <= by / 2)
    g.logodds[occ] = 4.0
    return g


def main():
    spec = DockSpec()
    for name, scen in SCENARIOS.items():
        standoff, dock = dock_poses(scen.station, spec)
        g = true_grid(scen)
        widest = next((m for m in (0.22, 0.19, 0.17) if plan(g, scen.start[:2], standoff[:2], m, relax_radius=min(0.17, m)) is not None), None)
        wp = plan_with_margins(g, scen.start[:2], standoff[:2])
        length = None if wp is None else sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip([scen.start[:2]] + wp[:-1], wp))
        print(f"{name:14s} stand-off {np.round(standoff[:2], 2)}  path with margin {widest}  length {'-' if length is None else f'{length:.1f} m'}")


if __name__ == "__main__":
    main()
