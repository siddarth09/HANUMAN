#!/usr/bin/env python3
"""Mock topics to exercise the dashboard without the full stack.

Drives ground truth in a slow arc; EKF tracks with noise; the terrain/GTSAM fix
tracks too, then takes a large jump after ~15 s to trigger the diverged warning.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist, PoseWithCovarianceStamped


def quat(yaw):
    return math.sin(yaw / 2), math.cos(yaw / 2)


class Mock(Node):
    def __init__(self):
        super().__init__("dashboard_mock")
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.cm_pub = self.create_publisher(OccupancyGrid, "/nav/global_costmap", latched)
        self.gp_pub = self.create_publisher(Path, "/nav/global_path", 1)
        self.mp_pub = self.create_publisher(Path, "/nav/mppi_path", 1)
        self.gt_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.ekf_pub = self.create_publisher(Odometry, "/odometry/filtered", 10)
        self.tm_pub = self.create_publisher(PoseWithCovarianceStamped, "/terrain_match/pose", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.t0 = self.now()
        self.cm_pub.publish(self._costmap())
        self.gp_pub.publish(self._path([(0, 0), (6, 2), (12, 1), (18, 6)]))
        self.create_timer(0.1, self._tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _costmap(self):
        W = H = 80
        res = 0.5
        grid = np.zeros((H, W), np.int8)
        yy, xx = np.mgrid[0:H, 0:W]
        for cx, cy, r in [(40, 55, 9), (20, 30, 7), (60, 40, 8)]:
            grid[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = 100
        grid = (grid + (np.random.default_rng(1).random((H, W)) * 25).astype(np.int8)).clip(0, 100)
        m = OccupancyGrid()
        m.header.frame_id = "map"
        m.info.resolution = res
        m.info.width, m.info.height = W, H
        m.info.origin.position.x = -8.0
        m.info.origin.position.y = -14.0
        m.info.origin.orientation.w = 1.0
        m.data = grid.astype(np.int8).flatten().tolist()
        return m

    def _path(self, pts):
        m = Path()
        m.header.frame_id = "map"
        for x, y in pts:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        return m

    def _odom(self, x, y, yaw, vx=0.0, wz=0.0):
        m = Odometry()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = float(x), float(y)
        qz, qw = quat(yaw)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = qz, qw
        m.twist.twist.linear.x = vx
        m.twist.twist.angular.z = wz
        return m

    def _tick(self):
        t = self.now() - self.t0
        gx, gy = 9 + 9 * math.cos(0.1 * t), 0 + 6 * math.sin(0.1 * t)
        gyaw = 0.1 * t + math.pi / 2
        self.gt_pub.publish(self._odom(gx, gy, gyaw))

        n = 0.05 * math.sin(t)
        self.ekf_pub.publish(self._odom(gx + n, gy - n, gyaw + 0.02, vx=0.3, wz=0.2))

        jump = 4.0 if t > 15.0 else 0.0          # simulate a GTSAM divergence
        tm = PoseWithCovarianceStamped()
        tm.header.frame_id = "map"
        tm.header.stamp = self.get_clock().now().to_msg()
        tm.pose.pose.position.x = gx + 0.1 + jump
        tm.pose.pose.position.y = gy - 0.1 + jump * 0.5
        qz, qw = quat(gyaw)
        tm.pose.pose.orientation.z, tm.pose.pose.orientation.w = qz, qw
        cov = 0.15 + (0.9 if jump else 0.0)
        tm.pose.covariance[0] = cov ** 2
        self.tm_pub.publish(tm)

        c = Twist()
        c.linear.x = 0.3 + 0.1 * math.sin(t)
        c.angular.z = 0.4 * math.sin(0.5 * t)
        self.cmd_pub.publish(c)

        self.mp_pub.publish(self._path(
            [(gx + 0.3 * i * math.cos(gyaw), gy + 0.3 * i * math.sin(gyaw)) for i in range(6)]))


def main(args=None):
    rclpy.init(args=args)
    node = Mock()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
