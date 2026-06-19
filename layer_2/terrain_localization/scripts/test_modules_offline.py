#!/usr/bin/env python3
"""Offline end-to-end test of the terrain_localization modules on a bag.

Exercises dem.DEM + local_map.LocalElevationMap (with multi-frame ACCUMULATION) +
matcher.search through the real package code. Uses GT for the camera pose + search
seed (the live node uses the EKF). Reports match error vs GT at several checkpoints
— should improve on the ~1.9 m single-frame Stage-B result thanks to accumulation.

  MUJOCO_GL=egl python3 test_modules_offline.py <bag>
"""
import os, sys
from glob import glob

import numpy as np

# import the installed package modules
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
    print(f"GT {len(gt)}  depth {len(depth)}")

    dem = DEM(TERRAIN, [92.0, 92.0, -3.947], (-8, 30), (-14, 14), 0.25, cache_path="/tmp/hanuman_dem.npz")
    Tbc = model_pelvis_cam(SCENE)
    lmap = LocalElevationMap((-8, 30), (-14, 14), 0.2, (432.97, 432.97, 320.0, 240.0))

    def gt_at(t):
        i = np.clip(np.searchsorted(gt[:, 0], t), 1, len(gt)-1)
        q = gt[i, 4:8]
        T = np.eye(4); T[:3, :3] = quat_R(*q); T[:3, 3] = gt[i, 1:4]
        return T, gt[i, 1:4], yaw_of(*q)

    a = np.arange(-5, 5.01, 0.4); px, py = np.meshgrid(a, a)
    dxs = np.arange(-2.5, 2.51, 0.25); dys = dxs.copy(); dyaws = np.deg2rad(np.arange(-15, 15.1, 5))

    checkpoints = set(np.linspace(len(depth)*0.2, len(depth)*0.95, 6).astype(int))
    errs = []
    for di, (t, img) in enumerate(depth):
        T, pos, yaw = gt_at(t)
        lmap.add(img, T @ Tbc)                         # ACCUMULATE every frame
        if di in checkpoints:
            obs = lmap.patch(pos[0], pos[1], yaw, px, py)
            best, cost_xy = matcher.search(dem, obs, px, py, pos[0], pos[1], yaw, dxs, dys, dyaws)
            ok = matcher.confident(best, cost_xy)
            if best:
                cost, dx, dy, dyaw, _, _ = best; e = np.hypot(dx, dy)
                cov = matcher.covariance(cost_xy, dxs, dys, best)
                errs.append(e)
                print(f"  ckpt x={pos[0]:5.1f}: fill={lmap.fill_fraction()*100:3.0f}% "
                      f"err={e:.2f}m corr=({dx:+.2f},{dy:+.2f}) cost={cost:.4f} "
                      f"sig=({np.sqrt(cov[0,0]):.2f},{np.sqrt(cov[1,1]):.2f}) "
                      f"confident={ok}")
    if errs:
        print(f"\nACCUMULATED median |xy err| = {np.median(errs):.2f} m "
              f"(Stage-B single-frame was ~1.9 m)")


if __name__ == "__main__":
    main()
