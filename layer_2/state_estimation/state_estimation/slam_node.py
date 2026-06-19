#!/usr/bin/env python3
"""GTSAM factor-graph SLAM: IMU preintegration + leg odom + terrain prior.

Subscribes: /imu_broadcaster/imu  (preintegrated between keyframes)
            /leg_odometry         (world-frame body velocity -> V prior)
            /ground_truth/odom    (validation only, never fused)
Publishes:  /slam/odometry        (optimized pose at the keyframe rate)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped

from state_estimation.factor_graph import ImuLegGraph


class SlamNode(Node):
    def __init__(self):
        super().__init__("slam_node")
        self.declare_parameter("gravity", 9.81)
        self.declare_parameter("keyframe_hz", 10.0)
        self.declare_parameter("leg_vel_sigma", 0.15)

        g = self.get_parameter("gravity").value
        self.graph = ImuLegGraph(
            gravity=g, leg_vel_sigma=self.get_parameter("leg_vel_sigma").value)

        self._last_imu_t = None
        self._leg_vel = None
        self._gt = None
        self._terrain_fix = None     # pending terrain prior fix

        be = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, "/imu_broadcaster/imu", self._imu_cb, be)
        self.create_subscription(Odometry, "/leg_odometry", self._leg_cb, be)
        self.create_subscription(Odometry, "/ground_truth/odom", self._gt_cb, be)
        self.create_subscription(PoseWithCovarianceStamped, "/terrain_match/pose",
                                 self._terrain_cb, 10)
        self.pub = self.create_publisher(Odometry, "/slam/odometry", 10)

        hz = self.get_parameter("keyframe_hz").value
        self.create_timer(1.0 / hz, self._keyframe)
        self.get_logger().info(
            f"SLAM node up - gravity={g}, keyframe@{hz:.0f}Hz, "
            f"IMU + leg-odom factor graph (Step 1)")

    # ── Sensor callbacks ────────────────────────────────────────────────────
    def _imu_cb(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_imu_t is not None:
            dt = t - self._last_imu_t
            if 0.0 < dt < 0.1:   # guard against bad/duplicate stamps
                a = [msg.linear_acceleration.x, msg.linear_acceleration.y,
                     msg.linear_acceleration.z]
                w = [msg.angular_velocity.x, msg.angular_velocity.y,
                     msg.angular_velocity.z]
                self.graph.integrate(a, w, dt)
        self._last_imu_t = t

    def _leg_cb(self, msg: Odometry):
        v = msg.twist.twist.linear
        self._leg_vel = np.array([v.x, v.y, v.z])

    def _gt_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self._gt = np.array([p.x, p.y, p.z])

    def _terrain_cb(self, msg: PoseWithCovarianceStamped):
        # stash x, y, yaw + sigmas; applied next keyframe
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = float(np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z)))
        C = np.array(msg.pose.covariance).reshape(6, 6)
        self._terrain_fix = (p.x, p.y, yaw,
                             float(np.sqrt(C[0, 0])), float(np.sqrt(C[1, 1])),
                             float(np.sqrt(C[5, 5])),
                             float(p.z), float(np.sqrt(C[2, 2])))   # DEM terrain-relative z

    # ── Keyframe / optimize ─────────────────────────────────────────────────
    def _keyframe(self):
        # Need some IMU accumulated, else the IMU factor is degenerate.
        if self._last_imu_t is None or self.graph.dt_since_keyframe() < 1e-3:
            return
        pose, vel = self.graph.add_keyframe(self._leg_vel)
        # apply a pending terrain prior as a unary prior
        if self._terrain_fix is not None:
            x, y, yaw, sx, sy, syaw, z, sz = self._terrain_fix
            self._terrain_fix = None
            if self.graph.add_terrain_prior(x, y, yaw, sx, sy, syaw, z=z, sig_z=sz):
                pose = self.graph.pose
                self.get_logger().info(
                    f"applied terrain prior -> pos=[{pose.x():.2f} {pose.y():.2f}]")
            else:
                self.get_logger().warn(
                    "terrain prior rejected by iSAM2 (kept dead-reckoning)",
                    throttle_duration_sec=5.0)
        self._publish(pose, vel)
        if self._gt is not None:
            err = float(np.linalg.norm(np.array(pose.translation()) - self._gt))
            self.get_logger().info(
                f"pos=[{pose.x():.2f} {pose.y():.2f} {pose.z():.2f}] "
                f"err_vs_gt={err:.3f} m", throttle_duration_sec=1.0)

    def _publish(self, pose, vel):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        t = pose.translation()
        q = pose.rotation().toQuaternion()
        msg.pose.pose.position.x = float(t[0])
        msg.pose.pose.position.y = float(t[1])
        msg.pose.pose.position.z = float(t[2])
        msg.pose.pose.orientation.w = q.w()
        msg.pose.pose.orientation.x = q.x()
        msg.pose.pose.orientation.y = q.y()
        msg.pose.pose.orientation.z = q.z()
        msg.twist.twist.linear.x = float(vel[0])
        msg.twist.twist.linear.y = float(vel[1])
        msg.twist.twist.linear.z = float(vel[2])
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SlamNode()
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
