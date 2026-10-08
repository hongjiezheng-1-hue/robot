"""Grid path planning: A* on the inflated occupancy grid, then line-of-sight shortcuts. Unknown cells are treated as free."""

import heapq

import numpy as np

_NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))]


def astar(blocked, start, goal):
    """Shortest 8-connected path of free cells between two cell indices, or None."""
    h = lambda c: np.hypot(c[0] - goal[0], c[1] - goal[1])
    open_heap = [(h(start), 0.0, start)]
    came, cost = {start: None}, {start: 0.0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        if g > cost[cur]:
            continue
        for dx, dy, step in _NEIGHBOURS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < blocked.shape[0] and 0 <= nxt[1] < blocked.shape[1]) or blocked[nxt]:
                continue
            if dx and dy and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue                                           # no cutting through a corner
            ng = g + step
            if ng < cost.get(nxt, np.inf):
                cost[nxt], came[nxt] = ng, cur
                heapq.heappush(open_heap, (ng + h(nxt), ng, nxt))
    return None


def line_free(blocked, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
    return not blocked[np.round(xs).astype(int), np.round(ys).astype(int)].any()


def shortcut(blocked, path):
    """Greedy line-of-sight smoothing: keep the farthest visible cell from each kept cell."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(blocked, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan(grid, start_xy, goal_xy, inflate_radius, relax_radius=None, relax_zone=0.35):
    """Waypoints (world coordinates, excluding the start) from start to goal, or None when no path is known.

    Obstacles are inflated by inflate_radius. Within relax_zone of the start the smaller relax_radius is used instead, so a robot
    that ended up closer to an obstacle than the planning margin can still leave, while the path keeps the hard clearance.
    """
    blocked = grid.inflated(inflate_radius)
    si = tuple(int(v) for v in grid.to_index(*start_xy))
    gi = tuple(int(v) for v in grid.to_index(*goal_xy))
    if relax_radius is not None:
        ix, iy = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing="ij")
        near = np.hypot((ix - si[0]) * grid.res, (iy - si[1]) * grid.res) <= relax_zone
        for r in (relax_radius, 0.12, 0.08, 0.04):         # shrink further until the robot's own cell is free (cells round the radius up)
            if r > relax_radius:
                continue
            relaxed = grid.inflated(r)
            if grid.inside(*si) and not relaxed[si]:
                break
        blocked[near] = relaxed[near]
    if not grid.inside(*gi) or not grid.inside(*si) or blocked[gi]:
        return None
    path = astar(blocked, si, gi)
    if path is None:
        return None
    pts = shortcut(blocked, path)
    return [tuple(float(v) for v in grid.to_world(i, j)) for i, j in pts[1:]]


def path_blocked(grid, waypoints_xy, inflate_radius, max_length=1.5, skip_length=0.35):
    """True when the stretch of the polyline between skip_length and max_length from its start passes through an inflated occupied cell.

    The first skip_length is ignored: a robot cannot do anything about cells directly under it, and checking them only causes
    endless re-planning.
    """
    blocked = grid.inflated(inflate_radius)
    length = 0.0
    for a, b in zip(waypoints_xy[:-1], waypoints_xy[1:]):
        seg = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        n = int(seg / (grid.res / 2)) + 1
        s = length + np.linspace(0.0, seg, n)
        xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
        ix, iy = grid.to_index(xs, ys)
        ok = grid.inside(ix, iy) & (s >= skip_length)
        if blocked[ix[ok], iy[ok]].any():
            return True
        length += seg
        if length > max_length:
            break
    return False


def plan_with_margins(grid, start_xy, goal_xy, margins=(0.22, 0.19, 0.17), relax_radius=0.17):
    """plan() with the widest margin that still leaves a path; None when even the smallest margin is blocked."""
    for margin in margins:
        waypoints = plan(grid, start_xy, goal_xy, margin, relax_radius=min(relax_radius, margin))
        if waypoints is not None:
            return waypoints
    return None
