"""Plot one LiDAR navigation log: map built from the estimated pose, plans, true and estimated path, and the estimator error.

    python experiments/visualize_navigation.py navigation_s_C_trial0.npz out.png
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from estimation.models import wrap  # noqa: E402

S_COURSE = ((-1.5, 0.65, 0.10, 1.70), (-0.5, -0.65, 0.10, 1.70), (0.5, 0.65, 0.10, 1.70))


def main(path, out, course=S_COURSE, room=(4.0, 3.0), station_x=1.85):
    d = np.load(path, allow_pickle=True)
    truth, est, dt = d["truth"], d["est_pose"], float(d["dt"])
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw=dict(height_ratios=[2.3, 1]))
    x0, y0, res = d["grid_origin"]
    lo = d["logodds"]
    ax.imshow(np.where(lo.T > 0.5, 1.0, np.where(lo.T < -0.3, 0.0, 0.5)), origin="lower", cmap="Greys", vmin=0, vmax=1.6,
              extent=(x0, x0 + lo.shape[0] * res, y0, y0 + lo.shape[1] * res), alpha=0.55)
    ax.add_patch(Rectangle((-room[0] / 2, -room[1] / 2), *room, fill=False, ec="k", lw=1.5))
    for cx, cy, sx, sy in course:
        ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc="none", ec="tab:orange", lw=2, ls="--", label="true board (hidden from the robot)" if cx == course[0][0] else None))
    ax.add_patch(Rectangle((station_x - 0.08, -0.14), 0.16, 0.28, fc="tab:green", alpha=0.6))
    steps = d["replan_steps"]
    ax.plot(truth[:, 0], truth[:, 1], "b-", lw=1.8, label="true path")
    ax.plot(est[:, 0], est[:, 1], "r--", lw=1.2, label="estimated path")
    ax.plot(*d["standoff"][:2], "m*", ms=12, label="stand-off")
    ax.plot(*d["dock"][:2], "g*", ms=12, label="dock")
    ax.plot(*truth[0, :2], "ko", ms=6)
    ax.set_aspect("equal"); ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.6, 1.6); ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{pathlib.Path(path).stem}: {len(steps)} plans, grey = occupancy grid from the estimated pose")
    t = np.arange(len(truth)) * dt
    err = np.hypot(*(truth[:, :2] - est[:, :2]).T)
    bx.plot(t, 1000 * err, "r-", label="estimator position error [mm]")
    bx.plot(t, np.degrees(np.abs(wrap(truth[:, 2] - est[:, 2]))), "b-", lw=0.8, label="heading error [deg]")
    seen = np.isfinite(d["marker"][:, 0])
    bx.fill_between(t, 0, 1, where=seen, transform=bx.get_xaxis_transform(), color="g", alpha=0.12, label="marker in view")
    for s in steps:
        bx.axvline(s * dt, color="k", lw=0.5, alpha=0.5)
    bx.set_xlabel("time [s]"); bx.legend(fontsize=8, ncol=2); bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
