from __future__ import annotations
import numpy as np


def search(dem, obs, px, py, sx, sy, syaw, dxs, dys, dyaws, min_overlap=80):
    valid0 = np.isfinite(obs)
    o_flat = obs.ravel()
    best = None
    cost_xy = np.full((len(dys), len(dxs)), np.nan)
    mid = len(dyaws) // 2
    for ka, dyaw in enumerate(dyaws):
        for jj, dy in enumerate(dys):
            for ii, dx in enumerate(dxs):
                cand = dem.patch(sx+dx, sy+dy, syaw+dyaw, px, py)
                m2 = valid0 & np.isfinite(cand)
                if m2.sum() < min_overlap:
                    continue
                o = o_flat[m2.ravel()]; c = cand.ravel()[m2.ravel()]
                o = o - o.mean(); c = c - c.mean()
                cost = float(np.mean((o-c)**2))
                if ka == mid:
                    cost_xy[jj, ii] = cost
                if best is None or cost < best[0]:
                    best = (cost, dx, dy, dyaw, ii, jj)
    return best, cost_xy


def covariance(cost_xy, dxs, dys, best, sig_min=0.1, sig_max=1.0, scale=0.5):
    _, _, _, _, ii, jj = best
    bc = cost_xy[jj, ii]
    finite = cost_xy[np.isfinite(cost_xy)]
    thr = bc + 0.5 * (np.median(finite) - bc)
    rx, ry = dxs[1]-dxs[0], dys[1]-dys[0]

    def half_width(prof, idx, step):
        n = len(prof); lo = hi = idx
        while lo > 0 and np.isfinite(prof[lo-1]) and prof[lo-1] <= thr:
            lo -= 1
        while hi < n-1 and np.isfinite(prof[hi+1]) and prof[hi+1] <= thr:
            hi += 1
        return max(idx-lo, hi-idx) * step

    # basin half-width over-estimates the true match accuracy (~0.3 m) on the
    # gentle slope, so scale it down to give the prior real authority in the graph.
    sx = float(np.clip(scale * half_width(cost_xy[jj, :], ii, rx), sig_min, sig_max))
    sy = float(np.clip(scale * half_width(cost_xy[:, ii], jj, ry), sig_min, sig_max))
    return np.array([[sx*sx, 0.0], [0.0, sy*sy]])


def confident(best, cost_xy, max_cost=0.02, min_peak_ratio=1.8):
   
    if best is None:
        return False
    bc = best[0]
    finite = cost_xy[np.isfinite(cost_xy)]
    if bc > max_cost or finite.size < 10:
        return False
    return float(np.median(finite)) > min_peak_ratio * bc
