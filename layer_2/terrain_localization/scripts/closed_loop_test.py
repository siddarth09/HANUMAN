#!/usr/bin/env python3
"""Deterministic closed-loop test of the orbital-prior CORRECTION logic.

Isolates the matcher from the flaky live EKF: we inject a KNOWN growing drift as
the odometry seed (simulating EKF/leg-odom drift) and check that the matcher's
maintained map<-odom correction keeps the estimate LOCKED to ground truth. If the
corrected error stays small while the raw drift grows to many metres, the orbital
prior bounds drift — which is the whole point.

  MUJOCO_GL=egl python3 closed_loop_test.py <bag>
"""
import os
import sys
from glob import glob

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HANUMAN = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))   # .../src/HANUMAN
_WS = os.path.abspath(os.path.join(_HANUMAN, "..", ".."))           # colcon workspace root
_sp = glob(os.path.join(
    _WS, "install", "terrain_localization", "lib", "python*", "site-packages"))
if _sp:
    sys.path.insert(0, _sp[0])
from terrain_localization.dem import DEM
from terrain_localization.local_map import LocalElevationMap
from terrain_localization import matcher
from terrain_localization.calib import model_pelvis_cam, quat_R, yaw_of

_MJCF = os.path.join(_HANUMAN, "mars_gazebo", "unitree_g1_mjcf")
SCENE = os.path.join(_MJCF, "mars_nav_scene.xml")
TERRAIN = os.path.join(_MJCF, "mars_nav_200", "model.xml")


def compose(T, p):
    tx, ty, tyaw = T; ox, oy, oyaw = p
    c, s = np.cos(tyaw), np.sin(tyaw)
    return np.array([tx + c*ox - s*oy, ty + s*ox + c*oy, tyaw + oyaw])


def correction(mp, op):
    mx, my, myaw = mp; ox, oy, oyaw = op
    tyaw = myaw - oyaw; c, s = np.cos(tyaw), np.sin(tyaw)
    return np.array([mx - (c*ox - s*oy), my - (s*ox + c*oy), tyaw])


def grids(scale, base_xy=3.0, step=0.25, base_yaw=np.deg2rad(15), ystep=np.deg2rad(5), max_xy=16):
    w = min(base_xy*scale, max_xy); st = step*scale
    dxs = np.arange(-w, w+1e-6, st); dys = dxs.copy()
    yw = min(base_yaw*scale, np.pi)
    dyaws = np.arange(-yw, yw+1e-6, ystep*scale)
    return dxs, dys, dyaws


def main():
    bag = sys.argv[1]
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    r = SequentialReader(); r.open(StorageOptions(uri=bag, storage_id=""), ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    gc = get_message(types["/ground_truth/odom"]); dc = get_message(types["/d435/depth/image_raw"])
    gt = []; depth = []
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/ground_truth/odom":
            m = deserialize_message(data, gc); p = m.pose.pose.position; q = m.pose.pose.orientation
            gt.append([m.header.stamp.sec+m.header.stamp.nanosec*1e-9, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        elif topic == "/d435/depth/image_raw":
            m = deserialize_message(data, dc)
            depth.append((m.header.stamp.sec+m.header.stamp.nanosec*1e-9,
                          np.frombuffer(m.data, np.float32).reshape(m.height, m.width)))
    gt = np.array(gt)

    dem = DEM(TERRAIN, [92.0, 92.0, -3.947], (-8, 30), (-14, 14), 0.25, cache_path="/tmp/hanuman_dem.npz")
    Tbc = model_pelvis_cam(SCENE)
    lmap = LocalElevationMap((-8, 30), (-14, 14), 0.2, (432.97, 432.97, 320.0, 240.0), decay=0.99)
    a = np.arange(-5, 5.01, 0.4); px, py = np.meshgrid(a, a)

    def gt_at(t):
        i = np.clip(np.searchsorted(gt[:, 0], t), 1, len(gt)-1)
        q = gt[i, 4:8]
        T = np.eye(4); T[:3, :3] = quat_R(*q); T[:3, 3] = gt[i, 1:4]
        return T, gt[i, 1:4], yaw_of(*q)

    map_T_odom = np.zeros(3); fail = 0
    match_every = 60          # ~ depth 30Hz -> match ~0.5 Hz
    raw_errs, cor_errs = [], []
    print(f"depth {len(depth)}  (injecting growing drift into the odom seed)")
    for di, (t, img) in enumerate(depth):
        T, pos, yaw = gt_at(t)
        lmap.add(img, T @ Tbc)                       # clean local map (GT pose)
        if di % match_every or di < match_every:
            continue
        # INJECT drift: odom lags GT, growing ~ leg-odom under-travel + lateral wander
        prog = pos[0]                                 # forward distance ~ drift driver
        drift = np.array([0.18*prog, 0.10*prog, np.deg2rad(2.0)*prog/5])
        odom = np.array([pos[0], pos[1], yaw]) + drift
        obs = lmap.patch(pos[0], pos[1], yaw, px, py)  # true robot-centric terrain
        if np.isfinite(obs).mean() < 0.1:
            continue
        seed = compose(map_T_odom, odom)
        scale = 1 if fail < 2 else min(fail, 8)
        dxs, dys, dyaws = grids(scale)
        best, cxy = matcher.search(dem, obs, px, py, seed[0], seed[1], seed[2], dxs, dys, dyaws)
        raw_err = np.hypot(*(seed[:2] - np.array([pos[0], pos[1]])))  # error if we trusted the seed
        if not matcher.confident(best, cxy):
            fail += 1; continue
        cost, dx, dy, dyaw, _, _ = best
        mp = np.array([seed[0]+dx, seed[1]+dy, seed[2]+dyaw])
        map_T_odom = correction(mp, odom); fail = 0
        cor_err = np.hypot(*(mp[:2] - np.array([pos[0], pos[1]])))
        raw_errs.append(np.hypot(*(odom[:2]-np.array([pos[0],pos[1]]))))
        cor_errs.append(cor_err)
        print(f"  x={pos[0]:5.1f}: raw-drift={raw_errs[-1]:5.2f}m  seed-err={raw_err:4.2f}m  "
              f"CORRECTED-err={cor_err:.2f}m  scale=x{scale}")
    print(f"\nRAW drift (uncorrected): median {np.median(raw_errs):.2f} m, max {np.max(raw_errs):.2f} m")
    print(f"CORRECTED (orbital prior): median {np.median(cor_errs):.2f} m, max {np.max(cor_errs):.2f} m")
    ok = np.median(cor_errs) < 1.0 and np.max(cor_errs) < 2.0
    print(f"VERDICT: {'PASS — drift bounded ✓' if ok else 'needs work'}")


if __name__ == "__main__":
    main()
