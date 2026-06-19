"""Box (1): robot-centric local elevation map, accumulated in the odom frame.

Fuses multiple depth frames into one heightmap (running mean per cell). Single
depth frames are sparse and forward-only; accumulating over a few seconds of
walking fills a wider, denser patch -> sharper terrain match. Lives in the odom
frame (smooth over the short accumulation window, so the slow drift is negligible
there); the matcher pulls a robot-centric patch out of it.
"""
from __future__ import annotations
import numpy as np


class LocalElevationMap:
    def __init__(self, xr, yr, res, intrinsics, depth_max=40.0,
                 decay=1.0, min_cnt=0.5):
        self.x0, self.x1 = xr; self.y0, self.y1 = yr; self.res = res
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.depth_max = depth_max
        self.decay = decay        # <1 = sliding window (old frames fade)
        self.min_cnt = min_cnt    # cells below this weight are treated as empty
        self.nx = int(round((self.x1-self.x0)/res)) + 1
        self.ny = int(round((self.y1-self.y0)/res)) + 1
        self.sum = np.zeros((self.ny, self.nx))
        self.cnt = np.zeros((self.ny, self.nx))
        self.gx = self.x0 + np.arange(self.nx)*res
        self.gy = self.y0 + np.arange(self.ny)*res

    def add(self, depth, T_odom_cam):
        """Project a depth image (HxW metres, optical frame) into the odom grid."""
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        d = depth
        mask = np.isfinite(d) & (d > 0.3) & (d < self.depth_max)
        u, v, d = u[mask], v[mask], d[mask]
        if d.size == 0:
            return 0
        if self.decay < 1.0:               # fade old observations (sliding window)
            self.sum *= self.decay
            self.cnt *= self.decay
        xc = (u-self.cx)/self.fx*d; yc = (v-self.cy)/self.fy*d
        pts = np.stack([xc, yc, d, np.ones_like(d)])           # 4xN optical
        w = (T_odom_cam @ pts)[:3]                             # 3xN odom
        ix = np.round((w[0]-self.x0)/self.res).astype(int)
        iy = np.round((w[1]-self.y0)/self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        np.add.at(self.sum, (iy[ok], ix[ok]), w[2][ok])
        np.add.at(self.cnt, (iy[ok], ix[ok]), 1.0)
        return int(ok.sum())

    def add_cloud(self, points, T_odom_lidar):
        """Accumulate a 3D point cloud (Nx3, lidar frame) into the odom grid."""
        if points.size == 0:
            return 0
        if self.decay < 1.0:
            self.sum *= self.decay
            self.cnt *= self.decay
        pts = np.vstack([points.T, np.ones(points.shape[0])])   # 4xN
        w = (T_odom_lidar @ pts)[:3]
        ix = np.round((w[0]-self.x0)/self.res).astype(int)
        iy = np.round((w[1]-self.y0)/self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        np.add.at(self.sum, (iy[ok], ix[ok]), w[2][ok])
        np.add.at(self.cnt, (iy[ok], ix[ok]), 1.0)
        return int(ok.sum())

    def mean_grid(self):
        g = np.full_like(self.sum, np.nan)
        nz = self.cnt >= self.min_cnt
        g[nz] = self.sum[nz]/self.cnt[nz]
        return g

    def _sample(self, X, Y, g):
        fx = np.interp(X, self.gx, np.arange(self.nx), left=np.nan, right=np.nan)
        fy = np.interp(Y, self.gy, np.arange(self.ny), left=np.nan, right=np.nan)
        out = np.full(np.shape(X), np.nan)
        ok = np.isfinite(fx) & np.isfinite(fy)
        x0 = np.clip(np.floor(fx[ok]).astype(int), 0, self.nx-2)
        y0 = np.clip(np.floor(fy[ok]).astype(int), 0, self.ny-2)
        tx = fx[ok]-x0; ty = fy[ok]-y0
        out[ok] = ((1-tx)*(1-ty)*g[y0, x0] + tx*(1-ty)*g[y0, x0+1]
                   + (1-tx)*ty*g[y0+1, x0] + tx*ty*g[y0+1, x0+1])
        return out

    def patch(self, x, y, yaw, px, py):
        """Robot-centric observed patch (px forward, py left). NaN where empty."""
        g = self.mean_grid()
        c, s = np.cos(yaw), np.sin(yaw)
        X = x + c*px - s*py
        Y = y + s*px + c*py
        return self._sample(X, Y, g)

    def fill_fraction(self):
        return float((self.cnt > 0).mean())
