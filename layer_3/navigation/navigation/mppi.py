from __future__ import annotations
import numpy as np


class MPPI:
    def __init__(self, K=512, H=20, dt=0.1,
                 sigma=(0.25, 0.0, 0.4),
                 vx_lim=(-0.05, 0.4), vy_lim=0.0, wz_lim=0.8,
                 lam=1.0,
                 w_terrain=6.0, w_goal=2.0, w_goal_term=8.0, w_ctrl=0.05,
                 lethal_cost=12.0):
        self.K, self.H, self.dt = K, H, dt
        self.sigma = np.asarray(sigma, float)
        self.vx_lim = vx_lim
        self.vy_lim = vy_lim
        self.wz_lim = wz_lim
        self.lam = lam
        self.w_terrain, self.w_goal = w_terrain, w_goal
        self.w_goal_term, self.w_ctrl = w_goal_term, w_ctrl
        self.lethal_cost = lethal_cost
        self.U = np.zeros((H, 3))            # nominal command sequence

    def _clip(self, V):
        V[..., 0] = np.clip(V[..., 0], *self.vx_lim)
        V[..., 1] = np.clip(V[..., 1], -self.vy_lim, self.vy_lim)
        V[..., 2] = np.clip(V[..., 2], -self.wz_lim, self.wz_lim)
        return V

    def compute(self, x0, y0, yaw0, carrot, cost_query):
        K, H, dt = self.K, self.H, self.dt
        eps = np.random.randn(K, H, 3) * self.sigma
        V = self._clip(self.U[None] + eps)               # (K,H,3)

        # rollout: integrate holonomic kinematics in the map frame
        wz = V[:, :, 2]
        yaw = yaw0 + dt * np.cumsum(wz, axis=1)
        yaw = np.concatenate([np.full((K, 1), yaw0), yaw[:, :-1]], axis=1)  # yaw at step start
        c, s = np.cos(yaw), np.sin(yaw)
        wvx = V[:, :, 0] * c - V[:, :, 1] * s
        wvy = V[:, :, 0] * s + V[:, :, 1] * c
        xs = x0 + dt * np.cumsum(wvx, axis=1)
        ys = y0 + dt * np.cumsum(wvy, axis=1)

        # cost
        terr = cost_query(xs.ravel(), ys.ravel()).reshape(K, H)
        cx, cy = carrot
        dist = np.hypot(xs - cx, ys - cy)
        ctrl = np.sum(V**2, axis=(1, 2))
        cost = (self.w_terrain * terr.sum(axis=1)
                + self.w_goal * dist.sum(axis=1)
                + self.w_goal_term * dist[:, -1]
                + self.w_ctrl * ctrl)

        # softmax weighting -> update nominal sequence
        beta = cost.min()
        w = np.exp(-(cost - beta) / self.lam)
        w /= w.sum() + 1e-9
        self.U = np.einsum('k,khc->hc', w, V)
        self.U = self._clip(self.U[None])[0]

        cmd = self.U[0].copy()
        # warm start: shift the nominal sequence forward one step
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0

        # best trajectory for viz
        best = np.stack([xs[np.argmin(cost)], ys[np.argmin(cost)]], axis=1)
        return cmd, best

    def reset(self):
        self.U[:] = 0.0
