#!/usr/bin/env python3
"""
HANUMAN — Odometry to TF broadcaster.

Subscribes to /ground_truth/odom and publishes odom → pelvis TF.
This provides the root fixed frame that RViz/Rerun need.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf')
      

        self.br = TransformBroadcaster(self)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.odom_cb, qos)

        self.get_logger().info('odom_to_tf: broadcasting odom → pelvis')

    def odom_cb(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'pelvis'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()