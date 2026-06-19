#!/usr/bin/env python3
"""HiRISE orbital-prior FEASIBILITY PROBE — Stage B (real depth).

Stage A proved the DEM is matchable with an idealized local map. Stage B builds
the local elevation patch from ACTUAL /d435/depth (forward-only FOV, real noise,
sky pixels rejected), projected to world via the camera pose, then matches it
against the DEM. If it still localizes near ground truth, the real sensor path
works and we can build the node chain (matcher -> GTSAM prior factor).

Camera pose = GT base pose (odom->pelvis) composed with the static pelvis->cam
extrinsic (recovered from the bag TF + optical-frame rotation).

Usage:
  MUJOCO_GL=egl python3 terrain_match_stage_b.py <bag> [--no-show]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terrain_match_probe import build_dem, extract, patch_coords, yaw_of  # noqa

# d435 intrinsics (from /d435/camera_info)
FX = FY = 432.9714612651417
CX, CY = 320.0, 240.0
DEPTH_MAX = 40.0   # reject sky (pixels sit at ~8251 m)
R_LINK_OPT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # link->optical


def quat_R(x, y, z, w):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def read_bag(bag):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id=""), ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    gc = get_message(types["/ground_truth/odom"])
    dc = get_message(types["/d435/depth/image_raw"])
    tfc = {n: get_message(types[n]) for n in ["/tf", "/tf_static"] if n in types}
    gt = []          # t, x,y,z, qx,qy,qz,qw
    depth = []       # (t, HxW float32)
    edges = {}       # child -> (parent, 4x4)   (TF snapshot for the extrinsic)
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/ground_truth/odom":
            m = deserialize_message(data, gc); p = m.pose.pose.position; q = m.pose.pose.orientation
            gt.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        elif topic == "/d435/depth/image_raw":
            m = deserialize_message(data, dc)
            t = m.header.stamp.sec+m.header.stamp.nanosec*1e-9
            img = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
            depth.append((t, img))
        elif topic in tfc:
            for tr in deserialize_message(data, tfc[topic]).transforms:
                # keep the FIRST sighting (neutral stance) — the waist rotates
                # later in the bag, which corrupts a static extrinsic snapshot.
                if tr.child_frame_id in edges:
                    continue
                tt = tr.transform.translation; q = tr.transform.rotation
                M = np.eye(4); M[:3, :3] = quat_R(q.x, q.y, q.z, q.w); M[:3, 3] = [tt.x, tt.y, tt.z]
                edges[tr.child_frame_id] = (tr.header.frame_id, M)
    return np.array(gt), depth, edges


SCENE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "mars_gazebo", "unitree_g1_mjcf", "mars_nav_scene.xml")


def model_pelvis_cam():
    """Exact pelvis->d435 OPTICAL extrinsic from the MuJoCo model (how the image
    was actually rendered). MuJoCo camera frame is +x right, +y up, looks down -z;
    optical frame (x right, y down, +z forward) = cam_mat @ diag(1,-1,-1)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "d435")
    R_opt = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    Twc = np.eye(4); Twc[:3, :3] = R_opt; Twc[:3, 3] = d.cam_xpos[cid]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    Twp = np.eye(4); Twp[:3, :3] = d.xmat[bid].reshape(3, 3); Twp[:3, 3] = d.xpos[bid]
    return np.linalg.inv(Twp) @ Twc


def gt_pose_at(gt, t):
    """Interp GT base pose -> 4x4 T_world_pelvis at time t."""
    i = np.searchsorted(gt[:, 0], t)
    i = np.clip(i, 1, len(gt)-1)
    t0, t1 = gt[i-1, 0], gt[i, 0]
    a = 0.0 if t1 == t0 else (t-t0)/(t1-t0)
    pos = (1-a)*gt[i-1, 1:4] + a*gt[i, 1:4]
    q = gt[i, 4:8] if a > 0.5 else gt[i-1, 4:8]   # nearest quat (fine at 50Hz)
    M = np.eye(4); M[:3, :3] = quat_R(*q); M[:3, 3] = pos
    yaw = np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]*q[1]+q[2]*q[2]))
    return M, pos, yaw


def depth_patch(img, T_world_cam, rx, ry, ryaw, px, py):
    """Project depth -> world points -> robot-local elevation grid (NaN where empty)."""
    H, W = img.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = img
    mask = np.isfinite(d) & (d > 0.3) & (d < DEPTH_MAX)
    u, v, d = u[mask], v[mask], d[mask]
    xc = (u-CX)/FX*d; yc = (v-CY)/FY*d; zc = d
    pts_cam = np.stack([xc, yc, zc, np.ones_like(d)])      # 4xN
    pts_w = (T_world_cam @ pts_cam)[:3]                    # 3xN world
    # to robot-local (px forward, py left)
    c, s = np.cos(ryaw), np.sin(ryaw)
    dx, dy = pts_w[0]-rx, pts_w[1]-ry
    lpx = c*dx + s*dy; lpy = -s*dx + c*dy; lz = pts_w[2]
    # bin into the patch grid
    gx1 = px[0]; gy1 = py[:, 0]
    step = gx1[1]-gx1[0]
    grid = np.full(px.shape, np.nan); cnt = np.zeros(px.shape)
    acc = np.zeros(px.shape)
    ix = np.round((lpx-gx1[0])/step).astype(int)
    iy = np.round((lpy-gy1[0])/step).astype(int)
    ok = (ix >= 0)&(ix < px.shape[1])&(iy >= 0)&(iy < px.shape[0])
    np.add.at(acc, (iy[ok], ix[ok]), lz[ok])
    np.add.at(cnt, (iy[ok], ix[ok]), 1.0)
    nz = cnt > 0
    grid[nz] = acc[nz]/cnt[nz]
    return grid


def match(gx, gy, Z, obs, x, y, yaw, px, py, dxs, dys, dyaws):
    valid0 = np.isfinite(obs)
    best = (1e18, 0, 0, 0); cost_xy = np.full((len(dys), len(dxs)), np.nan)
    mid = len(dyaws)//2
    for ka, dyaw in enumerate(dyaws):
        for jj, dy in enumerate(dys):
            for ii, dx in enumerate(dxs):
                cand = extract(gx, gy, Z, x+dx, y+dy, yaw+dyaw, px, py)
                m2 = valid0 & np.isfinite(cand)
                if m2.sum() < 50:
                    continue
                o = obs[m2]-obs[m2].mean(); c = cand[m2]-cand[m2].mean()
                cost = np.mean((o-c)**2)
                if ka == mid:
                    cost_xy[jj, ii] = cost
                if cost < best[0]:
                    best = (cost, dx, dy, dyaw)
    return best, cost_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag"); ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()

    print("reading bag...")
    gt, depth, edges = read_bag(a.bag)
    print(f"GT {len(gt)}  depth frames {len(depth)}")
    Tpc = model_pelvis_cam()
    print("pelvis->cam translation:", np.round(Tpc[:3, 3], 3))
    # report camera pitch (angle of optical +z below horizontal) as a sanity check
    fwd = Tpc[:3, 2]
    print(f"camera pitch below horizontal: {np.rad2deg(np.arctan2(-fwd[2], np.hypot(fwd[0],fwd[1]))):.0f} deg")

    print("building DEM...")
    gx, gy, Z = build_dem((-8, 30), (-14, 14), 0.25)

    # forward-biased patch (camera looks ahead+down): px forward 0..9, py +-5
    pxv = np.arange(-1.0, 9.01, 0.25); pyv = np.arange(-5.0, 5.01, 0.25)
    px, py = np.meshgrid(pxv, pyv)
    dxs = np.arange(-4, 4.01, 0.25); dys = np.arange(-4, 4.01, 0.25)
    dyaws = np.deg2rad(np.arange(-25, 25.01, 5))

    idxs = np.linspace(len(depth)*0.15, len(depth)*0.9, 6).astype(int)
    errs = []; demo = None
    for n, di in enumerate(idxs):
        t, img = depth[di]
        Twp, pos, yaw = gt_pose_at(gt, t)
        Twc = Twp @ Tpc
        obs = depth_patch(img, Twc, pos[0], pos[1], yaw, px, py)
        cov = np.isfinite(obs).mean()
        if cov < 0.05:
            print(f"  frame {n}: too few depth cells ({cov*100:.0f}%), skip"); continue
        (cost, bdx, bdy, bdyaw), cost_xy = match(gx, gy, Z, obs, pos[0], pos[1], yaw, px, py, dxs, dys, dyaws)
        e = np.hypot(bdx, bdy); errs.append(e)
        print(f"  frame {n} (x={pos[0]:5.1f} y={pos[1]:5.1f}): patch fill {cov*100:3.0f}%  "
              f"recovered dx={bdx:+.2f} dy={bdy:+.2f} dyaw={np.rad2deg(bdyaw):+.0f}  |err|={e:.2f} m")
        if demo is None:
            demo = (obs, cost_xy, dxs, dys, pos[0])

    if errs:
        errs = np.array(errs)
        print(f"\nVERDICT (real depth): median |xy err| = {np.median(errs):.2f} m  "
              f"(<1 m great, <2 m usable, >3 m degenerate)")

    import matplotlib
    if a.no_show: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if demo:
        obs, cost_xy, dxs, dys, x0 = demo
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
        im0 = ax[0].imshow(obs, origin="lower",
                           extent=[pxv[0], pxv[-1], pyv[0], pyv[-1]], cmap="terrain")
        ax[0].set_title("Depth-derived local patch (robot frame)\nforward = +x")
        ax[0].set_xlabel("forward [m]"); ax[0].set_ylabel("left [m]")
        fig.colorbar(im0, ax=ax[0], label="elev [m]")
        im1 = ax[1].imshow(cost_xy, origin="lower",
                           extent=[dxs[0], dxs[-1], dys[0], dys[-1]], cmap="viridis")
        ax[1].plot(0, 0, "r*", ms=14, label="true pose")
        ax[1].set_title(f"Match cost (dx,dy) @ x={x0:.0f}"); ax[1].legend()
        ax[1].set_xlabel("dx [m]"); ax[1].set_ylabel("dy [m]")
        fig.colorbar(im1, ax=ax[1], label="cost")
        fig.tight_layout(); fig.savefig("/tmp/terrain_stage_b.png", dpi=120)
        print("saved /tmp/terrain_stage_b.png")
        if not a.no_show: plt.show()


if __name__ == "__main__":
    main()
