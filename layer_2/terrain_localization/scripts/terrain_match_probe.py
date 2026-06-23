#!/usr/bin/env python3
"""HiRISE orbital-prior FEASIBILITY PROBE (Stage A: DEM matchability).

Question: can a robot-centric local elevation patch localize against the global
HiRISE DEM on THIS terrain? If the patch only matches at the true pose (sharp,
unique cost minimum) terrain-relative localization is viable and will bound
drift. If the minimum is flat/ambiguous, the terrain lacks relief (the TRN
analogue of VIO's textureless-wall problem).

Method (no depth/extrinsics yet — idealized local map = DEM sampled at truth):
  1. Build world-frame DEM by ray-casting the terrain model.
  2. For sample poses along the GT trajectory: extract a robot-centric patch at
     the TRUE pose, then search (dx,dy,dyaw) for the best-matching patch in the
     DEM. Cost = SSD after optimal z-alignment (mean removal) — i.e. shape match,
     since the robot's absolute altitude is a free parameter it must also solve.
  3. Report recovered-pose error + per-axis cost sensitivity (which DOF are
     well-constrained vs degenerate).

Usage:
  MUJOCO_GL=egl python3 terrain_match_probe.py <bag> [--no-show]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

TERRAIN_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "mars_gazebo", "unitree_g1_mjcf", "mars_nav_200", "model.xml")
OFFSET = np.array([92.0, 92.0, -3.947])   # terrain body pos in the scene


def build_dem(xr, yr, res):
    """World-frame elevation grid via ray-cast. Returns (gx, gy, Z)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(TERRAIN_MODEL)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    top = float(m.hfield_size[0][2]) + 60.0
    gx = np.arange(xr[0], xr[1] + 1e-6, res)
    gy = np.arange(yr[0], yr[1] + 1e-6, res)
    Z = np.full((len(gy), len(gx)), np.nan)
    gid = np.zeros(1, dtype=np.int32)
    for j, Y in enumerate(gy):
        for i, X in enumerate(gx):
            mx, my = X - OFFSET[0], Y - OFFSET[1]          # world -> terrain-local
            pnt = np.array([mx, my, top]); vec = np.array([0, 0, -1.0])
            dist = mujoco.mj_ray(m, d, pnt, vec, None, 1, -1, gid)
            if dist >= 0:
                Z[j, i] = (top - dist) + OFFSET[2]          # -> world elevation
    return gx, gy, Z


def sample(gx, gy, Z, X, Y):
    """Bilinear DEM lookup at world (X,Y) arrays. NaN outside."""
    fx = np.interp(X, gx, np.arange(len(gx)), left=np.nan, right=np.nan)
    fy = np.interp(Y, gy, np.arange(len(gy)), left=np.nan, right=np.nan)
    out = np.full(np.shape(X), np.nan)
    ok = np.isfinite(fx) & np.isfinite(fy)
    x0 = np.floor(fx[ok]).astype(int); y0 = np.floor(fy[ok]).astype(int)
    x0 = np.clip(x0, 0, len(gx)-2); y0 = np.clip(y0, 0, len(gy)-2)
    tx = fx[ok]-x0; ty = fy[ok]-y0
    v = ((1-tx)*(1-ty)*Z[y0,x0] + tx*(1-ty)*Z[y0,x0+1]
         + (1-tx)*ty*Z[y0+1,x0] + tx*ty*Z[y0+1,x0+1])
    out[ok] = v
    return out


def patch_coords(half, step):
    a = np.arange(-half, half+1e-6, step)
    return np.meshgrid(a, a)   # px, py in robot frame


def extract(gx, gy, Z, x, y, yaw, px, py):
    c, s = np.cos(yaw), np.sin(yaw)
    X = x + c*px - s*py
    Y = y + s*px + c*py
    return sample(gx, gy, Z, X, Y)


def yaw_of(q):  # geometry_msgs quaternion -> yaw
    return np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    r = SequentialReader(); r.open(StorageOptions(uri=a.bag, storage_id=""), ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    gc = get_message(types["/ground_truth/odom"])
    poses = []
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/ground_truth/odom":
            m = deserialize_message(data, gc); p = m.pose.pose.position
            poses.append((p.x, p.y, yaw_of(m.pose.pose.orientation)))
    poses = np.array(poses)
    print(f"GT poses: {len(poses)}")

    print("building DEM (ray-cast)...")
    gx, gy, Z = build_dem((-8, 30), (-14, 14), 0.25)
    print(f"DEM grid {Z.shape}, valid {np.isfinite(Z).mean()*100:.0f}%, "
          f"relief {np.nanmax(Z)-np.nanmin(Z):.2f} m")

    px, py = patch_coords(half=5.0, step=0.25)     # idealized 10x10 m local map
    # search grid
    dxs = np.arange(-4, 4.01, 0.25); dys = np.arange(-4, 4.01, 0.25)
    dyaws = np.deg2rad(np.arange(-25, 25.01, 5))
    rng = np.random.default_rng(0)

    sample_idx = np.linspace(len(poses)*0.1, len(poses)*0.9, 8).astype(int)
    errs = []
    cost_demo = None
    for n, idx in enumerate(sample_idx):
        x, y, yaw = poses[idx]
        obs = extract(gx, gy, Z, x, y, yaw, px, py) + rng.normal(0, 0.02, px.shape)
        valid = np.isfinite(obs)
        if valid.mean() < 0.5:
            continue
        best = (1e18, 0, 0, 0); cost_xy = np.full((len(dys), len(dxs)), np.nan)
        for ka, dyaw in enumerate(dyaws):
            for jj, dy in enumerate(dys):
                for ii, dx in enumerate(dxs):
                    cand = extract(gx, gy, Z, x+dx, y+dy, yaw+dyaw, px, py)
                    m2 = valid & np.isfinite(cand)
                    if m2.mean() < 0.5:
                        continue
                    o = obs[m2]-obs[m2].mean(); c = cand[m2]-cand[m2].mean()
                    cost = np.mean((o-c)**2)
                    if ka == len(dyaws)//2:
                        cost_xy[jj, ii] = cost
                    if cost < best[0]:
                        best = (cost, dx, dy, dyaw)
        _, bdx, bdy, bdyaw = best
        e = np.hypot(bdx, bdy)
        errs.append((e, bdx, bdy, np.rad2deg(bdyaw)))
        print(f"  pose {n} (x={x:5.1f} y={y:5.1f}): recovered offset "
              f"dx={bdx:+.2f} dy={bdy:+.2f} dyaw={np.rad2deg(bdyaw):+.0f}deg  |err|={e:.2f} m")
        if cost_demo is None:
            cost_demo = (cost_xy, dxs, dys, x, y)

    errs = np.array([e[0] for e in errs])
    print(f"\nVERDICT: median |xy err| = {np.median(errs):.2f} m   "
          f"(<0.5 m = matchable; >2 m = ambiguous/degenerate)")

    import matplotlib
    if a.no_show: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = ax[0].imshow(Z, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap="terrain")
    ax[0].plot(poses[:, 0], poses[:, 1], "r-", lw=1.5, label="GT path")
    ax[0].set_title("World DEM + trajectory"); ax[0].legend(); fig.colorbar(im0, ax=ax[0], label="elev [m]")
    cost_xy, dxs, dys, x, y = cost_demo
    im1 = ax[1].imshow(cost_xy, origin="lower", extent=[dxs[0], dxs[-1], dys[0], dys[-1]], cmap="viridis")
    ax[1].plot(0, 0, "r*", ms=14, label="true pose")
    ax[1].set_title(f"Match cost surface (dx,dy) @ pose x={x:.0f}\nsharp dip at ★ = localizable")
    ax[1].set_xlabel("dx [m]"); ax[1].set_ylabel("dy [m]"); ax[1].legend()
    fig.colorbar(im1, ax=ax[1], label="match cost")
    fig.tight_layout(); fig.savefig("/tmp/terrain_probe.png", dpi=120)
    print("saved /tmp/terrain_probe.png")
    if not a.no_show: plt.show()


if __name__ == "__main__":
    main()
