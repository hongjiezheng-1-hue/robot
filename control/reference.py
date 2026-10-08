"""Reference trajectory for the unicycle: straight segments joined by circular arcs, followed at constant speed.

Corners are rounded so the reference angular rate is never a step and the linear speed never drops to zero, which keeps
the tracking-error model controllable (docs/findings.md, stage 1).
"""

import numpy as np

from estimation.models import wrap


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def build_reference(start_xy, waypoints, speed, fillet_radius, dt):
    """Sample the reference at the control period. Returns a dict of arrays t, x, y, theta, v, w and the path length."""
    pts = [np.asarray(start_xy, dtype=float)] + [np.asarray(w, dtype=float) for w in waypoints]
    d = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    length = [float(np.hypot(*v)) for v in d]
    psi = [float(np.arctan2(v[1], v[0])) for v in d]
    n_seg = len(d)

    # fillet at the interior vertex i (between segment i-1 and segment i)
    t_off = [0.0] * (n_seg + 1)     # tangent length at the start of segment i
    radius = [0.0] * (n_seg + 1)
    turn = [0.0] * (n_seg + 1)
    for i in range(1, n_seg):
        phi = float(wrap(psi[i] - psi[i - 1]))
        if abs(phi) < 1e-6:
            continue
        tan_half = np.tan(abs(phi) / 2)
        t = min(fillet_radius * tan_half, 0.45 * min(length[i - 1], length[i]))
        t_off[i], radius[i], turn[i] = t, t / tan_half, phi

    pieces = []     # (kind, length, params)
    for i in range(n_seg):
        start = pts[i] + (t_off[i] * d[i] / length[i] if i > 0 else 0.0)
        end_t = t_off[i + 1] if i + 1 < n_seg else 0.0
        line_len = length[i] - (t_off[i] if i > 0 else 0.0) - end_t
        pieces.append(("line", line_len, (start, psi[i])))
        if i + 1 < n_seg and turn[i + 1] != 0.0:
            a = pts[i + 1] - t_off[i + 1] * d[i] / length[i]
            s = np.sign(turn[i + 1])
            normal = np.array([-np.sin(psi[i]), np.cos(psi[i])])
            r = radius[i + 1]
            centre = a + r * s * normal
            pieces.append(("arc", r * abs(turn[i + 1]), (centre, a - centre, psi[i], s, r)))

    total = sum(p[1] for p in pieces)
    n = int(np.floor(total / (speed * dt))) + 1
    out = {k: np.zeros(n) for k in ("t", "x", "y", "theta", "v", "w")}
    cum = np.concatenate([[0.0], np.cumsum([p[1] for p in pieces])])
    for k in range(n):
        s_total = min(k * speed * dt, total)
        j = min(int(np.searchsorted(cum, s_total, side="right")) - 1, len(pieces) - 1)
        s_loc = s_total - cum[j]
        kind, _, params = pieces[j]
        if kind == "line":
            start, ang = params
            p = start + s_loc * np.array([np.cos(ang), np.sin(ang)])
            theta, w = ang, 0.0
        else:
            centre, rv0, ang, s, r = params
            alpha = s * s_loc / r
            p = centre + _rot(alpha) @ rv0
            theta, w = ang + alpha, s * speed / r
        out["t"][k], out["x"][k], out["y"][k], out["theta"][k], out["v"][k], out["w"][k] = k * dt, p[0], p[1], wrap(theta), speed, w
    out["length"] = total
    return out
