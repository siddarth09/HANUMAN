#!/usr/bin/env python3
"""Overlay GT / EKF / GTSAM-SLAM trajectories from a rosbag2 (mcap/sqlite3).

Reads /ground_truth/odom, /odometry/filtered, /slam/odometry and produces:
  * top-down XY path (the money plot: drift looks smooth, a fall looks jagged)
  * Z vs time (altitude blow-up diagnosis)
  * position error vs GT, for EKF and SLAM

Usage (ROS env):
  python3 plot_slam_traj.py ~/projects25/rosbag2_2026_06_16-17_04_02 --no-show
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

GT = "/ground_truth/odom"
EKF = "/odometry/filtered"
SLAM = "/slam/odometry"


def read_bag(path: str) -> dict[str, np.ndarray]:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=path, storage_id=""),
                ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    want = (GT, EKF, SLAM)
    msg_cls = {n: get_message(types[n]) for n in want if n in types}
    rows: dict[str, list] = {GT: [], EKF: [], SLAM: []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in msg_cls:
            continue
        m = deserialize_message(data, msg_cls[topic])
        st = m.header.stamp
        p = m.pose.pose.position
        rows[topic].append([st.sec + st.nanosec * 1e-9, p.x, p.y, p.z])
    return {k: (np.array(v) if v else np.empty((0, 4))) for k, v in rows.items()}


def _interp_xyz(t_ref, src):
    if len(src) < 2:
        return None
    return np.column_stack([np.interp(t_ref, src[:, 0], src[:, c])
                            for c in (1, 2, 3)])


def _csv_xyz(path, cols=(0, 1, 2, 3)):
    if not os.path.exists(path):
        return np.empty((0, 4))
    with open(path) as f:
        r = csv.reader(f); next(r, None)
        rows = [[float(row[i]) for i in cols] for row in r if row]
    return np.array(rows) if rows else np.empty((0, 4))


def read_csv_dir(path):
    """validation_node output dir: gt.csv / ekf.csv / slam.csv (+ terrain.csv)."""
    d = {GT: _csv_xyz(os.path.join(path, "gt.csv")),
         EKF: _csv_xyz(os.path.join(path, "ekf.csv")),
         SLAM: _csv_xyz(os.path.join(path, "slam.csv"))}
    terr = _csv_xyz(os.path.join(path, "terrain.csv"), cols=(0, 1, 2, 2))  # t,x,y
    return d, terr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="rosbag2 dir OR validation_node CSV dir")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", default="slam_traj.png")
    args = ap.parse_args()

    terrain = None
    if os.path.isdir(args.bag) and os.path.exists(os.path.join(args.bag, "gt.csv")):
        d, terrain = read_csv_dir(args.bag)
    else:
        d = read_bag(args.bag)
    for name, key in (("GT", GT), ("EKF", EKF), ("SLAM", SLAM)):
        print(f"{name:4s} {key:22s} {len(d[key]):6d} msgs")
    if terrain is not None:
        print(f"terrain fixes: {len(terrain)}")
    if len(d[GT]) < 2:
        sys.exit("No ground-truth samples — nothing to compare against.")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = d[GT][0, 0]
    style = {GT: ("k", "ground truth"), EKF: ("tab:blue", "EKF (50 Hz)"),
             SLAM: ("tab:red", "GTSAM SLAM")}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    for key, (c, lbl) in style.items():
        a = d[key]
        if len(a):
            ax[0].plot(a[:, 1], a[:, 2], color=c, lw=1.6, label=lbl)
            ax[0].plot(a[0, 1], a[0, 2], color=c, marker="o", ms=6)
    if terrain is not None and len(terrain):
        ax[0].scatter(terrain[:, 1], terrain[:, 2], c="gold", marker="*",
                      s=70, edgecolors="k", zorder=5, label="terrain fixes")
    ax[0].set_title("Top-down trajectory (XY)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].axis("equal"); ax[0].grid(alpha=0.3); ax[0].legend()

    for key, (c, lbl) in style.items():
        a = d[key]
        if len(a):
            ax[1].plot(a[:, 0] - t0, a[:, 3], c, lw=1.4, label=lbl)
    ax[1].set_title("Altitude (z) vs time")
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("z [m]")
    ax[1].grid(alpha=0.3); ax[1].legend()

    t_ref = d[GT][:, 0]
    gt_xyz = d[GT][:, 1:4]
    for key in (EKF, SLAM):
        est = _interp_xyz(t_ref, d[key])
        if est is not None:
            err = np.linalg.norm(est - gt_xyz, axis=1)
            c, lbl = style[key]
            ax[2].plot(t_ref - t0, err, c, lw=1.4,
                       label=f"{lbl}  (final {err[-1]:.2f} m, "
                             f"max {err.max():.2f} m)")
    ax[2].set_title("Position error vs ground truth")
    ax[2].set_xlabel("t [s]"); ax[2].set_ylabel("|error| [m]")
    ax[2].grid(alpha=0.3); ax[2].legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
