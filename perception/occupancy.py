"""Log-odds occupancy grid built from LiDAR scans and the ESTIMATED robot pose."""

import numpy as np
from scipy.ndimage import binary_dilation

from perception.lidar import beam_angles, sensor_origin


class OccupancyGrid:
    def __init__(self, x_range=(-2.4, 2.4), y_range=(-1.9, 1.9), resolution=0.05, l_free=-0.4, l_occ=0.85, l_clamp=4.0):
        self.res = resolution
        self.x0, self.y0 = x_range[0], y_range[0]
        self.shape = (int(round((x_range[1] - x_range[0]) / resolution)), int(round((y_range[1] - y_range[0]) / resolution)))
        self.logodds = np.zeros(self.shape)
        self.l_free, self.l_occ, self.l_clamp = l_free, l_occ, l_clamp

    def to_index(self, x, y):
        return np.floor((np.asarray(x) - self.x0) / self.res).astype(int), np.floor((np.asarray(y) - self.y0) / self.res).astype(int)

    def to_world(self, ix, iy):
        return self.x0 + (np.asarray(ix) + 0.5) * self.res, self.y0 + (np.asarray(iy) + 0.5) * self.res

    def inside(self, ix, iy):
        return (ix >= 0) & (ix < self.shape[0]) & (iy >= 0) & (iy < self.shape[1])

    def update(self, pose_est, ranges, cfg):
        """Integrate one scan taken from the estimated pose."""
        ox, oy = sensor_origin(pose_est, cfg)
        ang = pose_est[2] + beam_angles(cfg)
        r_hit = np.where(np.isfinite(ranges), ranges, np.nan)
        step = self.res / 2
        ds = np.arange(0.0, cfg.range_max, step)
        reach = np.where(np.isfinite(r_hit), r_hit - 0.5 * self.res, cfg.range_max)          # free up to the hit, or the whole range
        mask = ds[None, :] < reach[:, None]
        px = ox + np.cos(ang)[:, None] * ds[None, :]
        py = oy + np.sin(ang)[:, None] * ds[None, :]
        ix, iy = self.to_index(px[mask], py[mask])
        ok = self.inside(ix, iy)
        free = np.unique(ix[ok] * self.shape[1] + iy[ok])
        hit = np.isfinite(r_hit)
        hx, hy = self.to_index(ox + np.cos(ang[hit]) * r_hit[hit], oy + np.sin(ang[hit]) * r_hit[hit])
        hok = self.inside(hx, hy)
        occ = np.unique(hx[hok] * self.shape[1] + hy[hok])
        free = np.setdiff1d(free, occ)
        flat = self.logodds.reshape(-1)
        flat[free] = np.maximum(flat[free] + self.l_free, -self.l_clamp)
        flat[occ] = np.minimum(flat[occ] + self.l_occ, self.l_clamp)

    def occupied(self, threshold=0.5):
        return self.logodds > threshold

    def inflated(self, radius):
        """Occupied cells grown by the robot radius plus a margin."""
        n = int(np.ceil(radius / self.res))
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        return binary_dilation(self.occupied(), structure=(xx**2 + yy**2) <= n**2)
