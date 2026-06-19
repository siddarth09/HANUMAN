#!/usr/bin/env python3
"""HANUMAN navigation dashboard: HiRISE basemap + cost overlay + click-to-goal.

Tools: 2D Pose Estimate (/initialpose), 2D Nav Goal (/goal_pose), Clear.
Needs a display (TkAgg). Loads the same DEM the planner uses.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from matplotlib.patches import Circle
from matplotlib.image import imread

from navigation.costmap import build_costmap

BG = "#0d1117"
PANEL = "#161b22"
ACCENT = "#ff7b00"
TXT = "#e6edf3"


class Dashboard(Node):
    def __init__(self):
        super().__init__("nav_dashboard")
        gp = self.declare_parameter
        gp("dem_cache", "/tmp/hanuman_dem.npz")
        gp("albedo_path",
           "/home/sid/projects25/src/HANUMAN/mars_gazebo/unitree_g1_mjcf/"
           "mars_nav_200/mars_nav_200_albedo.png")
        gp("dem_offset", [92.0, 92.0])      # map = terrain + offset
        gp("hfield_half_m", 100.0)          # hfield radius (200 m tile)
        gp("map_frame", "map")
        gp("footprint_radius", 0.45)        # robot safety circle (m)
        gp("slope_max_deg", 25.0)
        gp("rough_radius", 2); gp("rough_max", 0.15)
        gp("w_slope", 0.6); gp("w_rough", 0.4)
        P = lambda n: self.get_parameter(n).value
        self.map_frame = P("map_frame")
        self.footprint_r = P("footprint_radius")

        z = np.load(P("dem_cache"))
        self.gx, self.gy, self.Z = z["gx"], z["gy"], z["Z"]
        self.res = float(self.gx[1] - self.gx[0])
        self.cost, self.lethal = build_costmap(
            self.Z, self.res, P("slope_max_deg"), int(P("rough_radius")),
            P("rough_max"), P("w_slope"), P("w_rough"))
        self.albedo = imread(P("albedo_path"))
        ox, oy = P("dem_offset"); R = P("hfield_half_m")
        self.img_extent = [ox - R, ox + R, oy - R, oy + R]   # albedo in map coords

        # live data
        self.gpath = self.mpath = None
        self.gtsam = self.ekf = self.gt = self.goal = None     # (x,y) or (x,y,yaw)
        self.cmd = (0.0, 0.0)                                   # (vx, wz)
        self.mode = "goal"                                     # 'goal' | 'pose'
        self._press = None

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 1)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        self.create_subscription(Path, "/nav/global_path", lambda m: setattr(self, "gpath", self._xy(m)), 1)
        self.create_subscription(Path, "/nav/mppi_path", lambda m: setattr(self, "mpath", self._xy(m)), 1)
        self.create_subscription(PoseWithCovarianceStamped, "/terrain_match/pose", self._gtsam_cb, 10)
        self.create_subscription(Odometry, "/odometry/filtered", self._ekf_cb, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self._gt_cb, 10)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)

        self._build_figure()
        self.get_logger().info("dashboard up — pick a tool on the right, then click-drag the map.")

    # ---- ROS in ----
    @staticmethod
    def _xy(m):
        return np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses]) if m.poses else None

    @staticmethod
    def _yaw(q):
        return float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))

    def _gtsam_cb(self, m): self.gtsam = (m.pose.pose.position.x, m.pose.pose.position.y, self._yaw(m.pose.pose.orientation))
    def _ekf_cb(self, m): self.ekf = (m.pose.pose.position.x, m.pose.pose.position.y, self._yaw(m.pose.pose.orientation))
    def _gt_cb(self, m): self.gt = (m.pose.pose.position.x, m.pose.pose.position.y)
    def _cmd_cb(self, m): self.cmd = (m.linear.x, m.angular.z)

    def _robot_xy(self):
        for v in (self.gtsam, self.ekf, self.gt):
            if v is not None:
                return v
        return None

    # ---- figure ----
    def _build_figure(self):
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(15, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title("HANUMAN — Mars Navigation")
        self.ax = self.fig.add_axes([0.04, 0.06, 0.66, 0.88])      # map
        self.ax.set_facecolor(BG)

        self.ax.imshow(self.albedo, extent=self.img_extent, origin="upper", zorder=0)
        dem_ext = [self.gx[0]-self.res/2, self.gx[-1]+self.res/2,
                   self.gy[0]-self.res/2, self.gy[-1]+self.res/2]
        self.ax.imshow(np.where(self.lethal, np.nan, self.cost), cmap="RdYlGn_r",
                       extent=dem_ext, origin="lower", alpha=0.28, vmin=0, vmax=1, zorder=1)
        self.ax.imshow(np.where(self.lethal, 1.0, np.nan), cmap="autumn_r",
                       extent=dem_ext, origin="lower", alpha=0.45, zorder=2)

        (self.l_gpath,) = self.ax.plot([], [], "-", color="#39d353", lw=2.5, label="global path", zorder=5)
        (self.l_mpath,) = self.ax.plot([], [], "-", color=ACCENT, lw=2.0, label="MPPI", zorder=6)
        (self.p_gt,) = self.ax.plot([], [], "o", color="#ffffff", ms=7, mec="k", label="ground truth", zorder=7)
        (self.p_ekf,) = self.ax.plot([], [], "o", color="#58a6ff", ms=7, label="EKF", zorder=7)
        (self.p_gtsam,) = self.ax.plot([], [], "*", color="#f2cc60", ms=15, mec="k", label="GTSAM", zorder=9)
        (self.p_goal,) = self.ax.plot([], [], "X", color="#ff5db1", ms=14, mec="k", label="goal", zorder=10)
        (self.l_head,) = self.ax.plot([], [], "-", color="#f2cc60", lw=2, zorder=9)   # heading
        self.footprint = Circle((0, 0), self.footprint_r, fill=False, ec="#58a6ff",
                                lw=1.5, ls="--", alpha=0.9, zorder=8)
        self.ax.add_patch(self.footprint); self.footprint.set_visible(False)

        self.ax.set_xlim(self.gx[0], self.gx[-1])
        self.ax.set_ylim(self.gy[0], self.gy[-1])
        self.ax.set_aspect("equal")
        self.ax.set_title("Jezero Crater — HiRISE basemap + traversability",
                          color=TXT, fontsize=13)
        self.ax.tick_params(colors="#7d8590")
        self.ax.legend(loc="upper right", framealpha=0.85, fontsize=8)

        # ---- right tool rail ----
        self.fig.text(0.78, 0.93, "HANUMAN", color=ACCENT, fontsize=20, weight="bold")
        self.fig.text(0.78, 0.90, "Mars Navigation Console", color="#7d8590", fontsize=9)
        self.b_pose = Button(self.fig.add_axes([0.74, 0.80, 0.22, 0.05]), "2D Pose Estimate",
                             color=PANEL, hovercolor="#21262d")
        self.b_goal = Button(self.fig.add_axes([0.74, 0.73, 0.22, 0.05]), "2D Nav Goal",
                             color=PANEL, hovercolor="#21262d")
        self.b_clear = Button(self.fig.add_axes([0.74, 0.66, 0.22, 0.05]), "Clear",
                              color=PANEL, hovercolor="#21262d")
        for b in (self.b_pose, self.b_goal, self.b_clear):
            b.label.set_color(TXT)
        self.b_pose.on_clicked(lambda e: self._set_mode("pose"))
        self.b_goal.on_clicked(lambda e: self._set_mode("goal"))
        self.b_clear.on_clicked(self._clear)
        self.status = self.fig.text(0.74, 0.10, "", color=TXT, fontsize=10,
                                    family="monospace", va="bottom")
        self._set_mode("goal")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.ani = FuncAnimation(self.fig, self._update, interval=100,
                                 blit=False, cache_frame_data=False)

    def _set_mode(self, mode):
        self.mode = mode
        self.b_pose.ax.set_facecolor(ACCENT if mode == "pose" else PANEL)
        self.b_goal.ax.set_facecolor(ACCENT if mode == "goal" else PANEL)
        self.fig.canvas.draw_idle()

    def _clear(self, _e):
        self.goal = None
        self.gpath = None
        self.l_gpath.set_data([], [])
        self.p_goal.set_data([], [])

    # ---- click-drag: position on press, heading from drag, publish on release ----
    def _on_press(self, e):
        if e.inaxes is self.ax and e.button == 1 and e.xdata is not None:
            self._press = (e.xdata, e.ydata)

    def _on_release(self, e):
        if self._press is None or e.inaxes is not self.ax or e.xdata is None:
            self._press = None
            return
        x0, y0 = self._press
        self._press = None
        yaw = np.arctan2(e.ydata - y0, e.xdata - x0) if np.hypot(e.xdata-x0, e.ydata-y0) > 0.3 else 0.0
        qz, qw = float(np.sin(yaw/2)), float(np.cos(yaw/2))
        if self.mode == "goal":
            m = PoseStamped()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.pose.position.x, m.pose.position.y = float(x0), float(y0)
            m.pose.orientation.z, m.pose.orientation.w = qz, qw
            self.goal_pub.publish(m)
            self.goal = (x0, y0)
            self.get_logger().info(f"GOAL ({x0:.1f},{y0:.1f})")
        else:
            m = PoseWithCovarianceStamped()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.pose.pose.position.x, m.pose.pose.position.y = float(x0), float(y0)
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = qz, qw
            self.pose_pub.publish(m)
            self.get_logger().info(f"POSE RESET ({x0:.1f},{y0:.1f},{np.rad2deg(yaw):.0f}deg)")

    def _update(self, _f):
        rclpy.spin_once(self, timeout_sec=0.0)
        if self.gpath is not None:
            self.l_gpath.set_data(self.gpath[:, 0], self.gpath[:, 1])
        if self.mpath is not None:
            self.l_mpath.set_data(self.mpath[:, 0], self.mpath[:, 1])
        if self.gt is not None:
            self.p_gt.set_data([self.gt[0]], [self.gt[1]])
        if self.ekf is not None:
            self.p_ekf.set_data([self.ekf[0]], [self.ekf[1]])
        if self.gtsam is not None:
            self.p_gtsam.set_data([self.gtsam[0]], [self.gtsam[1]])
        if self.goal is not None:
            self.p_goal.set_data([self.goal[0]], [self.goal[1]])

        r = self._robot_xy()
        if r is not None:
            self.footprint.center = (r[0], r[1]); self.footprint.set_visible(True)
            if len(r) == 3:
                hl = 0.8
                self.l_head.set_data([r[0], r[0]+hl*np.cos(r[2])], [r[1], r[1]+hl*np.sin(r[2])])

        d = (f"{np.hypot(self.goal[0]-r[0], self.goal[1]-r[1]):.1f} m"
             if (self.goal and r) else "—")
        pose = f"({r[0]:+.1f},{r[1]:+.1f})" if r else "—"
        self.status.set_text(
            f"TOOL : {self.mode.upper()}\n"
            f"pose : {pose}\n"
            f"goal : {('(%.1f,%.1f)' % self.goal) if self.goal else '—'}\n"
            f"dist : {d}\n"
            f"cmd  : vx={self.cmd[0]:+.2f}  wz={self.cmd[1]:+.2f}\n"
            f"safety r = {self.footprint_r:.2f} m")
        return []


def main(args=None):
    rclpy.init(args=args)
    node = Dashboard()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
