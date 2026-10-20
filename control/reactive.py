"""Reactive LiDAR obstacle avoidance: no map and no path planning.

In short: from the current scan, measure how far a corridor as wide as the robot is free in each direction, and steer in the
direction that gets closest to the goal. Only the current scan is used, so nothing has to be remembered.

The scan is expressed in the robot frame (x forward, y left). A known limitation of reactive methods: a dead end (a U-shaped
pocket) can trap the robot, because it only looks at what is in front of it now.
"""

import numpy as np

from estimation.models import wrap


def scan_points(ranges, angles, mount_x=-0.049):
    """Valid LiDAR returns as points in the robot frame."""
    ok = np.isfinite(ranges)
    return np.stack([mount_x + ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])], axis=1)


def free_mask(points, directions, half_width, lookahead):
    """True for each direction (robot frame) whose straight corridor of the given half width is free up to `lookahead`."""
    c, s = np.cos(directions)[:, None], np.sin(directions)[:, None]
    along = c * points[None, :, 0] + s * points[None, :, 1]
    lateral = -s * points[None, :, 0] + c * points[None, :, 1]
    blocked = (along > 0.0) & (along < lookahead) & (np.abs(lateral) < half_width)
    return ~blocked.any(axis=1)


def free_length(points, directions, half_width, cap):
    """For each direction (robot frame): how far a corridor of the given half width is free (at most `cap`)."""
    c, s = np.cos(directions)[:, None], np.sin(directions)[:, None]
    along = c * points[None, :, 0] + s * points[None, :, 1]
    lateral = -s * points[None, :, 0] + c * points[None, :, 1]
    blocking = (along > 0.0) & (np.abs(lateral) < half_width)
    first = np.where(blocking, along, np.inf).min(axis=1) if len(points) else np.full(len(directions), np.inf)
    return np.minimum(first, cap)


def choose_heading(points, goal_angle, goal_distance=np.inf, half_width=0.22, lookahead=1.5, previous=None, n_candidates=72,
                   min_free=0.3):
    """Direction to drive in (robot frame), or None if no direction is free for at least `min_free`.

    If the corridor towards the goal is free all the way (up to the goal or `lookahead`), drive towards the goal. Otherwise
    pick the direction that promises the most progress towards the goal: free length times the cosine of the angle to the
    goal, with a small penalty for changing the previous choice (so the robot does not zigzag).
    """
    cap = min(lookahead, goal_distance)
    if free_length(points, np.array([goal_angle]), half_width, cap)[0] >= cap:
        return float(goal_angle)
    candidates = np.linspace(-np.pi, np.pi, n_candidates, endpoint=False)
    length = free_length(points, candidates, half_width, cap)
    score = np.cos(wrap(candidates - goal_angle)) * length
    if previous is not None:
        score = score - 0.1 * np.abs(wrap(candidates - previous))
    score[length < min_free] = -np.inf
    if not np.isfinite(score).any():
        return None
    return float(candidates[int(np.argmax(score))])


def front_clearance(points, half_width=0.2):
    """Distance to the nearest return straight ahead within the robot's width (inf if none)."""
    ahead = (points[:, 0] > 0.0) & (np.abs(points[:, 1]) < half_width)
    return float(points[ahead, 0].min()) if ahead.any() else np.inf


class BugNavigator:
    """Reactive navigation with a memory of one number: a simple version of the Bug algorithm.

    In short: drive straight at the goal while the way is free. When it is blocked, pick the side (left or right) with the
    smaller detour and follow the obstacle on that side, until the way to the goal is free again AND the robot is closer to
    the goal than where it got blocked. That escapes the dead ends that pure "steer to the freest direction" gets stuck in.
    """

    def __init__(self, half_width=0.24, clear_ahead=0.8, follow_length=0.5, step_deg=5.0, flip_after=400):
        self.half_width, self.clear_ahead, self.follow_length = half_width, clear_ahead, follow_length
        self.step = np.radians(step_deg)
        self.flip_after = flip_after                 # control steps of following before trying the other side
        self.following, self.side, self.hit_distance, self.steps_following = False, 1.0, np.inf, 0

    def _goal_free(self, points, goal_angle, goal_distance):
        cap = min(self.clear_ahead, goal_distance)
        return free_length(points, np.array([goal_angle]), self.half_width, cap)[0] >= cap

    def _first_free(self, points, goal_angle, side):
        """Turning from the goal direction towards `side` (+1 left, -1 right): angle offset of the first free direction."""
        offsets = np.arange(self.step, np.pi + 1e-9, self.step)
        dirs = goal_angle + side * offsets
        free = free_length(points, dirs, self.half_width, self.follow_length) >= self.follow_length
        return (float(offsets[np.argmax(free)]), float(wrap(dirs[np.argmax(free)]))) if free.any() else (np.inf, None)

    def _start_following(self, points, goal_angle, goal_distance):
        left, right = self._first_free(points, goal_angle, 1.0), self._first_free(points, goal_angle, -1.0)
        self.side = 1.0 if left[0] <= right[0] else -1.0
        self.following, self.hit_distance, self.steps_following = True, goal_distance, 0

    def heading(self, points, goal_angle, goal_distance):
        """Direction to drive in (robot frame), or None if no direction is free."""
        if self._goal_free(points, goal_angle, goal_distance):
            if not self.following or goal_distance < self.hit_distance - 0.1:
                self.following = False
                return float(goal_angle)
        elif not self.following or goal_distance < self.hit_distance - 0.1:
            self._start_following(points, goal_angle, goal_distance)     # newly blocked (or blocked again after progress)
        self.steps_following += 1
        if self.steps_following > self.flip_after:                       # following for too long: try the other side
            self.side, self.steps_following = -self.side, 0
        return self._first_free(points, goal_angle, self.side)[1]
