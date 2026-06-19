from __future__ import annotations
import os
import numpy as np


class DEM:
    def __init__(self, model_path, offset, xr, yr, res, cache_path=None):
        self.offset = np.asarray(offset, float)
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path)
            self.gx, self.gy, self.Z = z["gx"], z["gy"], z["Z"]
        else:
            self.gx, self.gy, self.Z = self._raycast(model_path, xr, yr, res)
            if cache_path:
                np.savez(cache_path, gx=self.gx, gy=self.gy, Z=self.Z)

    def _raycast(self, model_path, xr, yr, res):
        import mujoco
        m = mujoco.MjModel.from_xml_path(model_path)
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        top = float(m.hfield_size[0][2]) + 60.0
        gx = np.arange(xr[0], xr[1] + 1e-6, res)
        gy = np.arange(yr[0], yr[1] + 1e-6, res)
        Z = np.full((len(gy), len(gx)), np.nan)
        gid = np.zeros(1, dtype=np.int32)
        for j, Y in enumerate(gy):
            for i, X in enumerate(gx):
                mx, my = X - self.offset[0], Y - self.offset[1]
                dist = mujoco.mj_ray(m, d, np.array([mx, my, top]),
                                     np.array([0, 0, -1.0]), None, 1, -1, gid)
                if dist >= 0:
                    Z[j, i] = (top - dist) + self.offset[2]
        return gx, gy, Z

    def sample(self, X, Y):
        """Bilinear elevation at world (X,Y) arrays; NaN outside."""
        gx, gy, Z = self.gx, self.gy, self.Z
        fx = np.interp(X, gx, np.arange(len(gx)), left=np.nan, right=np.nan)
        fy = np.interp(Y, gy, np.arange(len(gy)), left=np.nan, right=np.nan)
        out = np.full(np.shape(X), np.nan)
        ok = np.isfinite(fx) & np.isfinite(fy)
        x0 = np.clip(np.floor(fx[ok]).astype(int), 0, len(gx)-2)
        y0 = np.clip(np.floor(fy[ok]).astype(int), 0, len(gy)-2)
        tx = fx[ok]-x0; ty = fy[ok]-y0
        out[ok] = ((1-tx)*(1-ty)*Z[y0, x0] + tx*(1-ty)*Z[y0, x0+1]
                   + (1-tx)*ty*Z[y0+1, x0] + tx*ty*Z[y0+1, x0+1])
        return out

    def patch(self, x, y, yaw, px, py):
        """Robot-centric patch: px forward, py left (matches probe convention)."""
        c, s = np.cos(yaw), np.sin(yaw)
        X = x + c*px - s*py
        Y = y + s*px + c*py
        return self.sample(X, Y)
