"""Scene generation for Track A: a rectangular room, box obstacles and an ArUco-marked charging station.

The generated MJCF includes a derived copy of the upstream TurtleBot3 model (the upstream file is left
untouched) with a camera element added to the base body. Everything is written into the folder of the
upstream model because the model references its mesh files relatively. All dimensions are configuration
values; the defaults are placeholders to be replaced by the lab measurements.
"""

import dataclasses
import math
import os
from dataclasses import dataclass


@dataclass
class SceneConfig:
    room_x: float = 4.0                 # [m] interior length along x, centred on the origin
    room_y: float = 3.0                 # [m] interior width along y
    wall_height: float = 0.30           # [m]
    wall_thickness: float = 0.05        # [m]
    obstacles: tuple = ((-0.5, 0.4, 0.30, 0.30), (0.7, -0.6, 0.25, 0.40))   # (x, y, size_x, size_y) boxes [m]
    obstacle_height: float = 0.30       # [m], taller than the LiDAR so it is seen
    station_x: float = 1.85             # [m] charging station position
    station_y: float = 0.0
    station_yaw: float = math.pi        # [rad] marker faces the +x direction rotated by this yaw (pi: faces -x, into the room)
    marker_id: int = 0
    marker_dictionary: str = "DICT_4X4_50"
    marker_size: float = 0.12           # [m] side of the black marker square
    marker_height: float = 0.15         # [m] height of the marker centre above the floor
    marker_side_px: int = 400
    marker_border_px: int = 100         # white quiet zone around the marker in the texture
    robot_xml: str = "turtlebot3_waffle_pi.xml"
    marker_texture: str = "aruco_marker.png"
    camera_name: str = "front"
    camera_pos: tuple = (0.083, 0.0, 0.107)   # [m] in the robot base frame (placeholder: position of the camera block in the upstream XML)
    camera_fovy: float = 48.8           # [deg] vertical field of view (placeholder: Raspberry Pi camera v2)
    camera_width: int = 640
    camera_height: int = 480


_BASE_ANCHOR = '<joint type="free" name="base_joint"/>'


def inject_camera(robot_xml_text, cfg):
    """Return the robot XML with a forward-looking camera added to the base body (x forward, y left, z up)."""
    if _BASE_ANCHOR not in robot_xml_text:
        raise ValueError("base joint not found in the robot XML; cannot attach the camera")
    x, y, z = cfg.camera_pos
    # xyaxes: image right = robot -y, image up = robot +z, so the camera looks along robot +x
    camera = (f'{_BASE_ANCHOR}\n      <camera name="{cfg.camera_name}" pos="{x} {y} {z}" '
              f'xyaxes="0 -1 0 0 0 1" fovy="{cfg.camera_fovy}"/>')
    return robot_xml_text.replace(_BASE_ANCHOR, camera, 1)


def build_scene_xml(cfg):
    hx, hy = cfg.room_x / 2, cfg.room_y / 2
    t, h = cfg.wall_thickness, cfg.wall_height
    walls = [
        ("wall_n", 0.0, hy + t / 2, hx + t, t / 2),
        ("wall_s", 0.0, -(hy + t / 2), hx + t, t / 2),
        ("wall_e", hx + t / 2, 0.0, t / 2, hy),
        ("wall_w", -(hx + t / 2), 0.0, t / 2, hy),
    ]
    wall_xml = "\n".join(
        f'    <geom name="{n}" type="box" pos="{x} {y} {h / 2}" size="{sx} {sy} {h / 2}" rgba="0.8 0.8 0.8 1"/>'
        for n, x, y, sx, sy in walls)
    oh = cfg.obstacle_height
    obstacle_xml = "\n".join(
        f'    <geom name="obstacle_{i}" type="box" pos="{x} {y} {oh / 2}" size="{sx / 2} {sy / 2} {oh / 2}" rgba="0.75 0.45 0.2 1"/>'
        for i, (x, y, sx, sy) in enumerate(cfg.obstacles))
    plate_half = cfg.marker_size * (cfg.marker_side_px + 2 * cfg.marker_border_px) / cfg.marker_side_px / 2
    qw, qz = math.cos(cfg.station_yaw / 2), math.sin(cfg.station_yaw / 2)
    return f"""<mujoco model="track_a_scene">
  <include file="{cfg.robot_xml}"/>

  <statistic center="0 0 0.3" extent="{max(cfg.room_x, cfg.room_y)}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <texture type="2d" name="marker_tex" file="{cfg.marker_texture}"/>
    <material name="marker" texture="marker_tex" specular="0" shininess="0"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
{wall_xml}
{obstacle_xml}
    <body name="station" pos="{cfg.station_x} {cfg.station_y} 0" quat="{qw} 0 0 {qz}">
      <geom name="station_base" type="box" pos="0 0 0.03" size="0.08 0.14 0.03" rgba="0.2 0.6 0.2 1"/>
      <geom name="station_post" type="box" pos="-0.01 0 {cfg.marker_height / 2}" size="0.008 0.008 {cfg.marker_height / 2}" rgba="0.3 0.3 0.3 1"/>
      <geom name="station_marker" type="box" pos="0 0 {cfg.marker_height}" size="0.002 {plate_half} {plate_half}"
            material="marker" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def write_scene(cfg, robot_dir, flip_marker=False, scene_name="scene_track_a.xml", with_camera=True):
    """Write the marker texture, the derived robot XML (with camera) and the scene into robot_dir; return the scene path."""
    import cv2
    from perception.aruco import make_marker_image

    img = make_marker_image(cfg.marker_id, cfg.marker_dictionary, cfg.marker_side_px, cfg.marker_border_px, flip_marker)
    cv2.imwrite(os.path.join(robot_dir, cfg.marker_texture), img)
    if with_camera:
        with open(os.path.join(robot_dir, cfg.robot_xml)) as f:
            derived_name = os.path.splitext(cfg.robot_xml)[0] + "_cam.xml"
            derived = inject_camera(f.read(), cfg)
        with open(os.path.join(robot_dir, derived_name), "w") as f:
            f.write(derived)
        cfg = dataclasses.replace(cfg, robot_xml=derived_name)
    path = os.path.join(robot_dir, scene_name)
    with open(path, "w") as f:
        f.write(build_scene_xml(cfg))
    return path
