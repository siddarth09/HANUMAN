# leg_odom_node.py
# ROS2 node: subscribes to /joint_states and /imu, publishes /leg_odometry

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped
import numpy as np
from scipy.spatial.transform import Rotation

from state_estimation.leg_odom import LegOdometry


class LegOdomNode(Node):
    def __init__(self):
        super().__init__('leg_odom_node')

        # ---- Parameters ----
        self.declare_parameter('urdf_path', '')
        self.declare_parameter('contact_effort_threshold', 5.0)
        self.declare_parameter('base_noise', 0.05)
        self.declare_parameter('imu_topic', '/imu_broadcaster/imu')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('output_topic', '/leg_odometry')

        urdf_path = self.get_parameter('urdf_path').value
        threshold = self.get_parameter('contact_effort_threshold').value
        self.base_noise = self.get_parameter('base_noise').value
        imu_topic = self.get_parameter('imu_topic').value
        js_topic = self.get_parameter('joint_states_topic').value
        out_topic = self.get_parameter('output_topic').value

        if not urdf_path:
            self.get_logger().fatal("urdf_path parameter is required!")
            return

        # ---- Leg odometry core ----
        self.leg_odom = LegOdometry(urdf_path, threshold)

        # ---- Latest IMU data ----
        self.R_body_to_world = np.eye(3)
        self.omega_body = np.zeros(3)
        self.imu_received = False

        # ---- Subscribers ----
        self.js_sub = self.create_subscription(
            JointState, js_topic, self.joint_state_cb, 10)
        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_cb, 10)

        # ---- Publisher ----
        self.odom_pub = self.create_publisher(Odometry, out_topic, 10)

        self.get_logger().info(f"Leg odom node started")
        self.get_logger().info(f"  URDF: {urdf_path}")
        self.get_logger().info(f"  IMU: {imu_topic}")
        self.get_logger().info(f"  Joint states: {js_topic}")
        self.get_logger().info(f"  Output: {out_topic}")

    def imu_cb(self, msg: Imu):
        """Cache latest IMU orientation and angular velocity."""
        q = msg.orientation
        # ROS quaternion is [x,y,z,w], scipy expects [x,y,z,w] ✓
        self.R_body_to_world = Rotation.from_quat(
            [q.x, q.y, q.z, q.w]).as_matrix()

        self.omega_body = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ])
        self.imu_received = True

    def joint_state_cb(self, msg: JointState):
        """Compute and publish leg odometry on each joint state message."""
        if not self.imu_received:
            return

        # Build mapping on first message
        if not self.leg_odom._js_mapping_built:
            self.leg_odom.build_joint_mapping(list(msg.name))
            self.get_logger().info("Joint mapping built from first message")

        position = np.array(msg.position)
        velocity = np.array(msg.velocity)
        effort = np.array(msg.effort)

        # Compute velocity
        v_world, confidence = self.leg_odom.compute_velocity(
            position, velocity, effort,
            self.R_body_to_world, self.omega_body
        )

        if v_world is None:
            return  # no contact, nothing to publish

        # ---- Build Odometry message ----
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # We only provide velocity (twist), not pose
        odom.twist.twist.linear.x = float(v_world[0])
        odom.twist.twist.linear.y = float(v_world[1])
        odom.twist.twist.linear.z = float(v_world[2])

        # Covariance: 6x6 matrix, row-major
        # Only fill linear velocity diagonal (indices 0,7,14)
        noise = self.base_noise / max(confidence, 0.1)
        cov = [0.0] * 36
        cov[0]  = noise  # vx
        cov[7]  = noise  # vy
        cov[14] = noise  # vz
        cov[21] = 1e6    # wx (not measured, huge covariance)
        cov[28] = 1e6    # wy
        cov[35] = 1e6    # wz
        odom.twist.covariance = cov

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = LegOdomNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()