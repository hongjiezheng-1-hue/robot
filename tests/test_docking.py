"""Checks for the docking geometry. Run: python tests/test_docking.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from control.docking import DockSpec, dock_error, dock_poses, docked  # noqa: E402


def test_poses_lie_on_the_marker_normal_facing_the_marker():
    spec = DockSpec()
    standoff, dock = dock_poses((1.85, 0.0, np.pi), spec)
    assert np.allclose(standoff, (1.85 - spec.standoff, 0.0, 0.0), atol=1e-9) and np.allclose(dock, (1.40, 0.0, 0.0), atol=1e-9)
    standoff, dock = dock_poses((0.0, 1.0, -np.pi / 2), spec)
    assert np.allclose(dock[:2], (0.0, 1.0 - spec.distance)) and np.isclose(dock[2], np.pi / 2), "robot faces the marker"


def test_error_is_expressed_in_the_dock_frame():
    dock = (1.4, 0.0, 0.0)
    along, lateral, heading = dock_error((1.43, -0.02, np.radians(4.0)), dock)
    assert np.allclose((along, lateral, heading), (0.03, -0.02, 4.0))
    dock = (0.0, 0.55, np.pi / 2)       # rotated dock: forward is +y
    along, lateral, _ = dock_error((0.02, 0.60, np.pi / 2), dock)
    assert np.isclose(along, 0.05) and np.isclose(lateral, -0.02)


def test_tolerance_is_checked_per_component():
    spec = DockSpec()
    dock = (1.4, 0.0, 0.0)
    assert docked((1.42, 0.02, np.radians(3)), dock, spec)
    assert not docked((1.42, 0.04, 0.0), dock, spec), "lateral 4 cm"
    assert not docked((1.47, 0.0, 0.0), dock, spec), "along 7 cm"
    assert not docked((1.40, 0.0, np.radians(6)), dock, spec), "heading 6 deg"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
