"""Platform and measurement models for the TurtleBot3 Track A pipeline.

Conventions: world-frame state x = [px, py, theta]; body-frame input u = [v, w]
(linear and angular velocity); angles in radians. A marker is described by its
known world pose (mx, my, psi).

Simplification: measurements are expressed at the robot base frame. The camera
extrinsics (mounting offset on the robot) are handled in the perception stage.
"""

import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def encoders_to_vw(w_left, w_right, wheel_radius, track_width):
    """Wheel angular speeds [rad/s] to body linear and angular velocity."""
    v = wheel_radius * (w_right + w_left) / 2.0
    w = wheel_radius * (w_right - w_left) / track_width
    return v, w


def f_unicycle(x, u, dt):
    """Exact-arc unicycle prediction over one sample."""
    v, w = u
    px, py, th = x
    if abs(w) < 1e-9:
        return np.array([px + v * dt * np.cos(th), py + v * dt * np.sin(th), th])
    thn = th + w * dt
    return np.array([px + v / w * (np.sin(thn) - np.sin(th)),
                     py + v / w * (np.cos(th) - np.cos(thn)),
                     wrap(thn)])


def F_unicycle(x, u, dt):
    """Jacobian of f_unicycle with respect to the state."""
    v, w = u
    _, _, th = x
    if abs(w) < 1e-9:
        return np.array([[1.0, 0.0, -v * dt * np.sin(th)],
                         [0.0, 1.0, v * dt * np.cos(th)],
                         [0.0, 0.0, 1.0]])
    thn = th + w * dt
    return np.array([[1.0, 0.0, v / w * (np.cos(thn) - np.cos(th))],
                     [0.0, 1.0, v / w * (np.sin(thn) - np.sin(th))],
                     [0.0, 0.0, 1.0]])


def G_unicycle(x, u, dt):
    """Jacobian of f_unicycle with respect to the input u = [v, w] (3x2)."""
    v, w = u
    th = x[2]
    if abs(w) < 1e-4:   # series expansion, the exact expressions lose precision as w goes to 0
        return np.array([[dt * np.cos(th), -v * dt**2 * np.sin(th) / 2],
                         [dt * np.sin(th), v * dt**2 * np.cos(th) / 2],
                         [0.0, dt]])
    thn = th + w * dt
    return np.array([[(np.sin(thn) - np.sin(th)) / w,
                      -v / w**2 * (np.sin(thn) - np.sin(th)) + v / w * dt * np.cos(thn)],
                     [(np.cos(th) - np.cos(thn)) / w,
                      -v / w**2 * (np.cos(th) - np.cos(thn)) + v / w * dt * np.sin(thn)],
                     [0.0, dt]])


def h_marker_pose(x, marker):
    """Marker pose seen from the robot: its position in the robot frame and its orientation relative to the robot heading."""
    mx, my, psi = marker
    c, s = np.cos(x[2]), np.sin(x[2])
    dx, dy = mx - x[0], my - x[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(psi - x[2])])


def H_marker_pose(x, marker):
    mx, my, _ = marker
    c, s = np.cos(x[2]), np.sin(x[2])
    dx, dy = mx - x[0], my - x[1]
    return np.array([[-c, -s, -s * dx + c * dy],
                     [s, -c, -c * dx - s * dy],
                     [0.0, 0.0, -1.0]])


def h_range_bearing(x, marker):
    dx, dy = marker[0] - x[0], marker[1] - x[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])


def H_range_bearing(x, marker):
    dx, dy = marker[0] - x[0], marker[1] - x[1]
    r2 = dx * dx + dy * dy
    r = np.sqrt(r2)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / r2, -dx / r2, -1.0]])


def tracking_error_model(v_ref, w_ref):
    """Linearised tracking-error dynamics e = [e_x, e_y, e_theta] in the robot frame.

    e_dot = A e + B [dv, dw] about the reference speeds (v_ref, w_ref).
    """
    A = np.array([[0.0, w_ref, 0.0],
                  [-w_ref, 0.0, v_ref],
                  [0.0, 0.0, 0.0]])
    B = np.array([[-1.0, 0.0],
                  [0.0, 0.0],
                  [0.0, -1.0]])
    return A, B


def controllability_rank(A, B):
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


def observability_matrix(x0, controls, dt, H_fn):
    """Stack H_k Phi_k along a motion starting at x0; H_fn(x) returns the measurement Jacobian."""
    rows, phi, x = [], np.eye(len(x0)), np.asarray(x0, dtype=float).copy()
    for u in controls:
        rows.append(H_fn(x) @ phi)
        phi = F_unicycle(x, u, dt) @ phi
        x = f_unicycle(x, u, dt)
    return np.vstack(rows)


def smallest_singular_value(M):
    return float(np.linalg.svd(M, compute_uv=False)[-1])
