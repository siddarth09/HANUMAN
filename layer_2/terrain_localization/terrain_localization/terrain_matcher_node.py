#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header

from terrain_localization.dem import DEM
from terrain_localization.local_map import LocalElevationMap
from terrain_localization import matcher
from terrain_localization.calib import model_pelvis_cam, quat_R, yaw_of

_SCENE = ("/home/sid/projects25/src/HANUMAN/mars_gazebo/"
          "unitree_g1_mjcf/mars_nav_scene.xml")
_TERRAIN = ("/home/sid/projects25/src/HANUMAN/mars_gazebo/"
            "unitree_g1_mjcf/mars_nav_200/model.xml")


class TerrainMatcherNode(Node):
    def __init__(self):
        super().__init__("terrain_matcher_node")
        gp = self.declare_parameter
        gp("scene_path", _SCENE); gp("terrain_model", _TERRAIN)
        gp("dem_offset", [92.0, 92.0, -3.947])
        gp("dem_bounds", [-8.0, 30.0, -14.0, 14.0]); gp("dem_res", 0.25)
        gp("dem_cache", "/tmp/hanuman_dem.npz")
        gp("map_frame", "map")
        gp("depth_topic", "/d435/depth/image_raw")
        gp("camera_info_topic", "/d435/camera_info")
        gp("leg_odom_topic", "/leg_odometry")   # world-frame velocity -> own DR position
        gp("imu_topic", "/imu_broadcaster/imu") # absolute orientation -> own DR heading
        gp("match_period", 2.0)
        gp("patch_half", 5.0); gp("patch_res", 0.4)
        gp("search_xy", 3.0); gp("search_xy_step", 0.25)
        gp("max_search_xy", 16.0)   # escalated window cap when re-acquiring lock
        gp("search_yaw_deg", 15.0); gp("search_yaw_step_deg", 5.0)
        gp("min_fill", 0.10)
        gp("local_bounds", [-8.0, 30.0, -14.0, 14.0]); gp("local_res", 0.2)
        gp("depth_max", 40.0)
        gp("local_decay", 0.99)   # <1: sliding window so the map stays consistent under drift
        # terrain-relative z: pin base height to DEM ground + nominal stand height
        gp("base_height", 0.74)        # nominal pelvis height above terrain (m)
        gp("base_z_sigma", 0.15)       # z prior sigma (m); ~gait CoM bob
        # motion gate + re-localization hysteresis
        gp("gate_base", 3.0)           # motion-gate tolerance at last fix (m)
        gp("gate_growth", 0.3)         # gate growth per second since last fix (m/s)
        gp("gate_max", 4.0)            # hard cap on the gate (m)
        gp("reloc_min_fixes", 3)       # consecutive agreeing far-fixes to commit a relocalize
        gp("reloc_cluster", 1.5)       # how tightly those fixes must agree (m)
        gp("reloc_sigma", 1.0)         # inflated x/y/z sigma for an escalated/relocalized fix (m)

        P = lambda n: self.get_parameter(n).value
        self.map_frame = P("map_frame")
        self.get_logger().info("loading DEM (ray-cast, cached)...")
        b = P("dem_bounds")
        self.dem = DEM(P("terrain_model"), P("dem_offset"), (b[0], b[1]), (b[2], b[3]),
                       P("dem_res"), cache_path=P("dem_cache"))
        self.T_base_cam = model_pelvis_cam(P("scene_path"))
        self.get_logger().info(f"extrinsic base->cam t={np.round(self.T_base_cam[:3,3],3)}")

        # patch + search grids
        ph, pr = P("patch_half"), P("patch_res")
        a = np.arange(-ph, ph+1e-6, pr)
        self.px, self.py = np.meshgrid(a, a)
        # base search window + escalation (recovery) settings
        self.base_xy = P("search_xy"); self.base_step = P("search_xy_step")
        self.base_height = P("base_height"); self.base_z_sigma = P("base_z_sigma")
        self.gate_base = P("gate_base"); self.gate_growth = P("gate_growth")
        self.gate_max = P("gate_max"); self.reloc_min = P("reloc_min_fixes")
        self.reloc_cluster = P("reloc_cluster"); self.reloc_sigma = P("reloc_sigma")
        self._reloc_xy = None; self._reloc_n = 0   # pending re-localization cluster
        self.base_yaw = np.deg2rad(P("search_yaw_deg"))
        self.base_yaw_step = np.deg2rad(P("search_yaw_step_deg"))
        self.max_xy = P("max_search_xy")
        self.min_fill = P("min_fill")

        # map<-odom correction maintained by the matcher (seeds the search)
        self.map_T_odom = np.zeros(3)     # (tx, ty, tyaw)
        self.fail_count = 0
        # motion-consistency gate state
        self.last_map_xy = None
        self.last_dr_xy = None
        self.last_fix_t = None

        self.K = None
        self.lmap = None
        self.local_cfg = (P("local_bounds"), P("local_res"), P("depth_max"), P("local_decay"))

        # smooth dead-reckoning: position from leg-odom world velocity, orientation from IMU
        self.dr_pos = np.zeros(3)
        self.dr_R = np.eye(3)
        self.dr_yaw = 0.0
        self.last_leg_t = None
        self.have_dr = False

        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, P("camera_info_topic"), self._info_cb, be)
        self.create_subscription(Image, P("depth_topic"), self._depth_cb, be)
        self.create_subscription(Odometry, P("leg_odom_topic"), self._leg_cb, be)
        self.create_subscription(Imu, P("imu_topic"), self._imu_cb, be)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "/terrain_match/pose", 10)
        # manual localization reset: snap map<-odom to the clicked pose
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                 self._initialpose_cb, 10)

        # --- RViz debug clouds: DEM (latched, map frame) + live local map (odom) ---
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.dem_pub = self.create_publisher(PointCloud2, "/terrain/dem_cloud", latched)
        self.lmap_pub = self.create_publisher(PointCloud2, "/terrain/local_cloud", 1)
        self._publish_dem_cloud()

        self.create_timer(P("match_period"), self._match)
        self.get_logger().info("terrain matcher up — accumulating depth, matching @"
                               f"{1.0/P('match_period'):.2f} Hz")

    # ---- callbacks ----
    def _info_cb(self, m: CameraInfo):
        if self.K is None:
            fx, fy, cx, cy = m.k[0], m.k[4], m.k[2], m.k[5]
            lb, lr, dmax, dec = self.local_cfg
            self.lmap = LocalElevationMap((lb[0], lb[1]), (lb[2], lb[3]), lr,
                                          (fx, fy, cx, cy), depth_max=dmax, decay=dec)
            self.K = (fx, fy, cx, cy)
            self.get_logger().info(f"camera_info: fx={fx:.1f} cx={cx:.1f}")

    def _imu_cb(self, m: Imu):
        q = m.orientation
        self.dr_R = quat_R(q.x, q.y, q.z, q.w)         # absolute orientation (drift-free)
        self.dr_yaw = yaw_of(q.x, q.y, q.z, q.w)

    def _leg_cb(self, m: Odometry):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        v = m.twist.twist.linear                       # world-frame body velocity
        if self.last_leg_t is not None:
            dt = t - self.last_leg_t
            if 0.0 < dt < 0.5:
                self.dr_pos += np.array([v.x, v.y, v.z]) * dt
                self.have_dr = True
        self.last_leg_t = t

    def _dr_pose(self):
        T = np.eye(4); T[:3, :3] = self.dr_R; T[:3, 3] = self.dr_pos
        return T, (self.dr_pos[0], self.dr_pos[1], self.dr_pos[2], self.dr_yaw)

    def _depth_cb(self, m: Image):
        if self.lmap is None or not self.have_dr:
            return
        img = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        T_odom_base, _ = self._dr_pose()
        self.lmap.add(img, T_odom_base @ self.T_base_cam)

    # ---- RViz debug clouds ----
    def _cloud(self, frame, X, Y, Z):
        m = np.isfinite(Z)
        pts = np.stack([X[m], Y[m], Z[m]], axis=1).astype(np.float32)
        h = Header(); h.frame_id = frame; h.stamp = self.get_clock().now().to_msg()
        return point_cloud2.create_cloud_xyz32(h, pts)

    def _publish_dem_cloud(self):
        GX, GY = np.meshgrid(self.dem.gx, self.dem.gy)
        self.dem_pub.publish(self._cloud(self.map_frame, GX, GY, self.dem.Z))

    def _publish_local_cloud(self):
        GX, GY = np.meshgrid(self.lmap.gx, self.lmap.gy)
        self.lmap_pub.publish(self._cloud("odom", GX, GY, self.lmap.mean_grid()))

    # ---- 2D pose helpers (map<-odom correction) ----
    @staticmethod
    def _compose(T, p):
        tx, ty, ty_yaw = T; ox, oy, oyaw = p
        c, s = np.cos(ty_yaw), np.sin(ty_yaw)
        return (tx + c*ox - s*oy, ty + s*ox + c*oy, ty_yaw + oyaw)

    @staticmethod
    def _correction(map_pose, odom_pose):
        mx, my, myaw = map_pose; ox, oy, oyaw = odom_pose
        tyaw = myaw - oyaw
        c, s = np.cos(tyaw), np.sin(tyaw)
        return np.array([mx - (c*ox - s*oy), my - (s*ox + c*oy), tyaw])

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        cyaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))
        _, (ox, oy, _, oyaw) = self._dr_pose()
        self.map_T_odom = self._correction((p.x, p.y, cyaw), (ox, oy, oyaw))
        self._reloc_xy = None; self._reloc_n = 0
        self.last_map_xy = np.array([p.x, p.y])
        self.last_dr_xy = np.array([ox, oy])
        self.last_fix_t = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().warn(
            f"localization RESET via /initialpose -> ({p.x:.1f}, {p.y:.1f}, "
            f"{np.rad2deg(cyaw):.0f}deg)")

    def _grids(self, scale):
        w = min(self.base_xy*scale, self.max_xy); st = self.base_step*scale
        dxs = np.arange(-w, w+1e-6, st); dys = dxs.copy()
        yw = min(self.base_yaw*scale, np.pi); yst = self.base_yaw_step*scale
        dyaws = np.arange(-yw, yw+1e-6, yst)
        return dxs, dys, dyaws

    # ---- match + publish ----
    def _match(self):
        if self.lmap is None or not self.have_dr:
            return
        self._publish_local_cloud()
        _, (ox, oy, oz, oyaw) = self._dr_pose()           # own smooth dead-reckoning
        obs = self.lmap.patch(ox, oy, oyaw, self.px, self.py)  # robot-centric terrain
        if np.isfinite(obs).mean() < self.min_fill:
            return
        # seed = map_T_odom applied to the smooth DR
        sx, sy, syaw = self._compose(self.map_T_odom, (ox, oy, oyaw))
        scale = 1 if self.fail_count < 2 else min(self.fail_count, 3)   # escalate (capped) to re-acquire
        dxs, dys, dyaws = self._grids(scale)

        best, cost_xy = matcher.search(self.dem, obs, self.px, self.py,
                                       sx, sy, syaw, dxs, dys, dyaws)
        if not matcher.confident(best, cost_xy):
            self.fail_count += 1
            self.get_logger().info(
                f"no confident match (scale x{scale}), escalating", throttle_duration_sec=4.0)
            return
        cost, dx, dy, dyaw, _, _ = best
        mx, my, myaw = sx+dx, sy+dy, syaw+dyaw

        # motion-consistency gate with re-localization hysteresis
        now = self.get_clock().now().nanoseconds * 1e-9
        relocalized = False
        if self.last_map_xy is not None:
            pred = self.last_map_xy + (np.array([ox, oy]) - self.last_dr_xy)
            jump = float(np.hypot(mx - pred[0], my - pred[1]))
            # tolerance grows with the gap since the last fix, capped at gate_max
            tol = min(self.gate_base + self.gate_growth * (now - self.last_fix_t), self.gate_max)
            if jump > tol:
                # far fix: cluster consecutive far-fixes, commit once enough agree
                if (self._reloc_xy is not None and
                        np.hypot(mx - self._reloc_xy[0], my - self._reloc_xy[1]) < self.reloc_cluster):
                    self._reloc_n += 1
                    self._reloc_xy = 0.5 * (self._reloc_xy + np.array([mx, my]))
                else:
                    self._reloc_xy = np.array([mx, my]); self._reloc_n = 1
                if self._reloc_n < self.reloc_min:
                    self.fail_count += 1
                    self.get_logger().info(
                        f"holding (far fix {jump:.1f}m>{tol:.1f}m, "
                        f"{self._reloc_n}/{self.reloc_min} to relocalize)",
                        throttle_duration_sec=3.0)
                    return
                relocalized = True
                self.get_logger().warn(
                    f"RE-LOCALIZED after {self._reloc_n} consistent far-fixes -> "
                    f"({mx:.1f},{my:.1f})")
        self._reloc_xy = None; self._reloc_n = 0
        self.last_map_xy = np.array([mx, my]); self.last_dr_xy = np.array([ox, oy])
        self.last_fix_t = now

        self.map_T_odom = self._correction((mx, my, myaw), (ox, oy, oyaw))
        self.fail_count = 0
        cov = matcher.covariance(cost_xy, dxs, dys, best)
        # an escalated/relocalized fix is less certain than the cost basin implies
        if relocalized or scale > 1:
            cov = cov.copy()
            cov[0, 0] = max(cov[0, 0], self.reloc_sigma**2)
            cov[1, 1] = max(cov[1, 1], self.reloc_sigma**2)
        # terrain-relative z: DEM ground height at the matched (x,y) + nominal stand height
        z_dem = float(self.dem.sample(np.array([mx]), np.array([my]))[0])
        if np.isfinite(z_dem):
            z_pin, sig_z = z_dem + self.base_height, self.base_z_sigma
        else:
            z_pin, sig_z = oz, 1e3        # off the DEM -> fall back to dead-reckoning
        self._publish(mx, my, z_pin, myaw, cov, sig_z)
        self.get_logger().info(
            f"FIX(x{scale}): map=({mx:.2f},{my:.2f},{np.rad2deg(myaw):.0f}deg) "
            f"z={z_pin:.2f} corr=({dx:+.2f},{dy:+.2f}) cost={cost:.4f} "
            f"sig=({np.sqrt(cov[0,0]):.2f},{np.sqrt(cov[1,1]):.2f})m")

    def _publish(self, x, y, z, yaw, cov2, sig_z=1e3):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.position.z = float(z)
        msg.pose.pose.orientation.z = float(np.sin(yaw/2))
        msg.pose.pose.orientation.w = float(np.cos(yaw/2))
        C = np.zeros((6, 6))
        C[0, 0] = cov2[0, 0]; C[1, 1] = cov2[1, 1]
        C[2, 2] = float(sig_z)**2         # z from DEM (terrain-relative height)
        C[3, 3] = C[4, 4] = 1e6           # roll/pitch unconstrained
        C[5, 5] = np.deg2rad(5.0)**2       # yaw
        msg.pose.covariance = C.flatten().tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TerrainMatcherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
