"""Visualise a logged MuJoCo run: video (and a final-frame overview image) of the true robot, the EKF estimate with its
2-sigma position ellipse, the odometry-only trajectory, what the robot camera sees, and the position error.

Run from the repository root:
    python experiments/visualize_run.py --model-dir <robotis_tb3 folder> --run-file run.npz --out-dir viz [--step 4 --fps 15]

The truth is used here only to draw and to measure the error; the filter never sees it.
"""

import argparse
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle  # noqa: E402

from drivers.mujoco_tb3 import MujocoTB3Driver  # noqa: E402
from estimation.models import f_unicycle  # noqa: E402
from perception.aruco import detect_markers  # noqa: E402
from replay import load_run, replay_states  # noqa: E402
from sim.mujoco_run import RunConfig, calibrate_gains, pick_texture_type  # noqa: E402
from sim.scene import SceneConfig, write_scene  # noqa: E402


def dead_reckoning(run, gains):
    """Integrate the encoder odometry from the true start pose, without any marker update."""
    from estimation.models import encoders_to_vw
    pose, out = np.array(run["pose0"], dtype=float), []
    for wl, wr in run["wheel_omega"]:
        v, w = encoders_to_vw(wl, wr, run["wheel_radius"], run["track_nominal"])
        pose = f_unicycle(pose, (gains[0] * v, gains[1] * w), run["dt"])
        out.append(pose.copy())
    return np.array(out)


def triangle(pose, size=0.14):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    pts = np.array([[size, 0.0], [-0.6 * size, 0.55 * size], [-0.6 * size, -0.55 * size]])
    return pts @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def ellipse_params(cov_xy, k=2.0):
    w, v = np.linalg.eigh(cov_xy)
    w = np.maximum(w, 0.0)
    return 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), np.degrees(np.arctan2(v[1, 1], v[0, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--run-file", required=True)
    ap.add_argument("--out-dir", default="viz")
    ap.add_argument("--step", type=int, default=4, help="draw every n-th sample")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--filter", choices=("B", "A"), default="B", help="B: gains estimated online, A: single-point calibration")
    ap.add_argument("--subpixel", action="store_true", help="draw detections with sub-pixel corner refinement")
    ap.add_argument("--no-video", action="store_true", help="only write the final-frame overview image")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = load_run(args.run_file)
    scene_cfg = pick_texture_type(SceneConfig(), args.model_dir)
    drv = MujocoTB3Driver(write_scene(scene_cfg, args.model_dir))
    gains = calibrate_gains(drv, RunConfig())
    kw = dict(estimate=True) if args.filter == "B" else dict(gains=gains)
    errs, nees, hist, rejected, poses, covs = replay_states(run, **kw)
    dr = dead_reckoning(run, gains)
    truth, dt, n = run["truth"], run["dt"], len(run["truth"])
    seen = np.all(np.isfinite(run["marker"]), axis=1)
    err_mm = 1000 * np.linalg.norm(errs[:, :2], axis=1)
    dr_mm = 1000 * np.linalg.norm(dr[:, :2] - truth[:, :2], axis=1)
    t = (np.arange(n) + 1) * dt
    print(f"filter {args.filter}: position RMSE {np.sqrt(np.mean(err_mm ** 2)):.0f} mm, odometry only {np.sqrt(np.mean(dr_mm ** 2)):.0f} mm, "
          f"marker pose used in {100 * seen.mean():.0f}% of samples")

    model, data = drv.model, drv.data
    cam_r = mujoco.Renderer(model, height=scene_cfg.camera_height, width=scene_cfg.camera_width)
    ov_r = mujoco.Renderer(model, height=360, width=640)   # the default offscreen framebuffer is 640 pixels wide
    free = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, free)
    free.lookat[:] = [0.0, -0.2, 0.0]
    free.distance, free.azimuth, free.elevation = 5.2, 90.0, -48.0

    def place(pose):
        mujoco.mj_resetData(model, data)
        data.qpos[0], data.qpos[1] = pose[0], pose[1]
        data.qpos[3], data.qpos[6] = np.cos(pose[2] / 2), np.sin(pose[2] / 2)
        mujoco.mj_forward(model, data)

    fig = plt.figure(figsize=(14.4, 8.1), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0, 0.8], hspace=0.22, wspace=0.08,
                          left=0.04, right=0.99, top=0.95, bottom=0.07)
    ax_map, ax_cam, ax_3d, ax_err = fig.add_subplot(gs[:2, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])

    hx, hy = scene_cfg.room_x / 2, scene_cfg.room_y / 2
    ax_map.add_patch(Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False, ec="0.3", lw=2))
    for cx, cy, sx, sy in scene_cfg.obstacles:
        ax_map.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc="#c9803a", ec="0.3"))
    ax_map.add_patch(Rectangle((scene_cfg.station_x - 0.08, scene_cfg.station_y - 0.14), 0.16, 0.28, fc="#2f9a3c", ec="0.2"))
    ax_map.plot([scene_cfg.station_x - 0.05] * 2, [scene_cfg.station_y - 0.09, scene_cfg.station_y + 0.09], "k-", lw=3)
    ax_map.text(scene_cfg.station_x, scene_cfg.station_y + 0.2, "ArUco\nstation", ha="center", fontsize=8)
    ax_map.set_xlim(-hx - 0.1, hx + 0.1)
    ax_map.set_ylim(-hy - 0.1, hy + 0.1)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")
    (tr_true,) = ax_map.plot([], [], "k-", lw=1.4, label="true path")
    (tr_est,) = ax_map.plot([], [], "-", c="tab:red", lw=1.1, label="EKF estimate")
    (tr_dr,) = ax_map.plot([], [], "--", c="tab:blue", lw=1.0, label="odometry only")
    (ray,) = ax_map.plot([], [], "-", c="tab:green", lw=1.5, label="marker in view")
    body = Circle((0, 0), 0.15, fill=False, ec="0.5", lw=0.8)
    ax_map.add_patch(body)
    tri_true = Polygon(triangle((0, 0, 0)), closed=True, fc="k")
    tri_est = Polygon(triangle((0, 0, 0)), closed=True, fc="none", ec="tab:red", lw=1.5)
    ell = Ellipse((0, 0), 0.1, 0.1, fill=False, ec="tab:red", lw=1.0, ls=":")
    for p in (tri_true, tri_est, ell):
        ax_map.add_patch(p)
    ax_map.legend(loc="upper left", fontsize=8, ncol=2)
    title = ax_map.set_title("")

    im_cam = ax_cam.imshow(np.zeros((scene_cfg.camera_height, scene_cfg.camera_width, 3), dtype=np.uint8))
    ax_cam.axis("off")
    ax_cam.set_title("robot camera (green box: detected marker)", fontsize=10)
    im_3d = ax_3d.imshow(np.zeros((360, 640, 3), dtype=np.uint8))
    ax_3d.axis("off")
    ax_3d.set_title("MuJoCo scene", fontsize=10)

    ax_err.semilogy(t, np.maximum(dr_mm, 1.0), c="tab:blue", lw=0.9, label="odometry only")
    ax_err.semilogy(t, np.maximum(err_mm, 1.0), c="tab:red", lw=1.1, label="EKF")
    ax_err.fill_between(t, 0.5, 2000, where=seen, color="tab:green", alpha=0.12, label="marker in view")
    ax_err.set_xlim(0, t[-1])
    ax_err.set_ylim(1, 2000)
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("position error [mm]")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(loc="upper right", fontsize=8, ncol=3)
    cursor = ax_err.axvline(0, c="k", lw=1)

    writer = None if args.no_video else cv2.VideoWriter(str(out / "run.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1440, 810))
    indices = list(range(0, n, args.step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    last = None
    for count, k in enumerate(indices):
        place(truth[k])
        cam_r.update_scene(data, camera=scene_cfg.camera_name)
        img = cam_r.render().copy()
        ids, corners = detect_markers(img, scene_cfg.marker_dictionary, subpixel=args.subpixel)
        if ids is not None and scene_cfg.marker_id in ids.flatten():
            c = corners[list(ids.flatten()).index(scene_cfg.marker_id)].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(img, [c], True, (0, 255, 0), 3)
        ov_r.update_scene(data, camera=free)
        im_cam.set_data(img)
        im_3d.set_data(ov_r.render())
        tr_true.set_data(truth[:k + 1, 0], truth[:k + 1, 1])
        tr_est.set_data(poses[:k + 1, 0], poses[:k + 1, 1])
        tr_dr.set_data(dr[:k + 1, 0], dr[:k + 1, 1])
        body.center = (truth[k, 0], truth[k, 1])
        tri_true.set_xy(triangle(truth[k]))
        tri_est.set_xy(triangle(poses[k]))
        w_, h_, ang = ellipse_params(covs[k][:2, :2])
        ell.set_center((poses[k, 0], poses[k, 1]))
        ell.set_width(max(w_, 1e-3))
        ell.set_height(max(h_, 1e-3))
        ell.set_angle(ang)
        if seen[k]:
            ray.set_data([truth[k, 0], scene_cfg.station_x], [truth[k, 1], scene_cfg.station_y])
        else:
            ray.set_data([], [])
        title.set_text(f"t = {t[k]:5.1f} s    EKF error {err_mm[k]:6.0f} mm    odometry-only error {dr_mm[k]:6.0f} mm    "
                       f"2-sigma ellipse (red dotted)")
        cursor.set_xdata([t[k], t[k]])
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        last = frame
        if writer is not None:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if count % 100 == 0:
            print(f"frame {count + 1} / {len(indices)}")
    if writer is not None:
        writer.release()
        print("wrote", out / "run.mp4")
    cv2.imwrite(str(out / "overview.png"), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))
    print("wrote", out / "overview.png")


if __name__ == "__main__":
    main()
