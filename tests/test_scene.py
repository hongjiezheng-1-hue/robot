"""Well-formedness checks for the generated scene XML (no MuJoCo needed). Run: python tests/test_scene.py"""

import pathlib
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sim.scene import SceneConfig, build_scene_xml, inject_camera  # noqa: E402


def test_scene_structure():
    cfg = SceneConfig()
    root = ET.fromstring(build_scene_xml(cfg))
    geoms = {g.get("name"): g for g in root.iter("geom")}
    for name in ("wall_n", "wall_s", "wall_e", "wall_w", "station_base", "station_marker"):
        assert name in geoms, name
    assert sum(1 for n in geoms if n and n.startswith("obstacle_")) == len(cfg.obstacles)
    assert root.find("include").get("file") == cfg.robot_xml
    assert geoms["station_marker"].get("contype") == "0", "the marker plate must not collide"
    assert root.find("worldbody/body[@name='station']") is not None


def test_marker_plate_size():
    cfg = SceneConfig(marker_size=0.12, marker_side_px=400, marker_border_px=100)
    root = ET.fromstring(build_scene_xml(cfg))
    size = [float(s) for s in [g for g in root.iter("geom") if g.get("name") == "station_marker"][0].get("size").split()]
    assert abs(size[1] - 0.09) < 1e-9 and abs(size[2] - 0.09) < 1e-9   # plate = marker plus quiet zone, half-size 0.09


def test_marker_texture_types():
    for texture_type, expected in (("2d", "2d"), ("cube", "cube")):
        root = ET.fromstring(build_scene_xml(SceneConfig(marker_texture_type=texture_type)))
        tex = [t for t in root.iter("texture") if t.get("name") == "marker_tex"][0]
        assert tex.get("type") == expected
        assert (tex.get("gridsize") == "1 1") == (expected == "cube")
    try:
        build_scene_xml(SceneConfig(marker_texture_type="bogus"))
        raise AssertionError("expected a ValueError for an unknown texture type")
    except ValueError:
        pass


def test_camera_injection():
    robot = ('<mujoco><worldbody><body name="base" pos="0 0 0">\n      <joint type="free" name="base_joint"/>\n'
             '      <geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
    cfg = SceneConfig()
    root = ET.fromstring(inject_camera(robot, cfg))
    cam = root.find("worldbody/body[@name='base']/camera")
    assert cam is not None and cam.get("name") == cfg.camera_name
    assert [float(v) for v in cam.get("pos").split()] == list(cfg.camera_pos)
    assert float(cam.get("fovy")) == cfg.camera_fovy
    try:
        inject_camera("<mujoco/>", cfg)
        raise AssertionError("expected a ValueError when the base joint is missing")
    except ValueError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
