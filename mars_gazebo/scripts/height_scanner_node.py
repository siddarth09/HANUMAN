#!/usr/bin/env python3
"""
HANUMAN — Height Scanner Node (standalone, for testing the RL obs separately)

Reproduces the policy's training "terrain_scan" sensor: a body-centered 17x11 =
187-point grid (1.6m x 1.0m, 0.1m resolution), yaw-aligned, that reports the
VERTICAL height of the pelvis above the terrain at each grid cell
(height = pelvis_z - terrain_hit_z), exactly matching mjlab's
`height_scan` observation (scale 1/max_distance is applied downstream).

Implementation: loads the MuJoCo scene and ray-casts terrain only (mj_ray with a
geom-group mask = group 0), from the LIVE robot pose on /ground_truth/odom. This
avoids the robot self-occlusion that a rigid in-MJCF rangefinder grid suffers,
and keeps the geometry/flip handling inside MuJoCo so it is correct by
construction.

Grid order matches mjlab GridPattern `meshgrid(x, y, indexing="xy").flatten()`:
y is the outer loop (-0.5..0.5), x the inner loop (-0.8..0.8).

Publishes:  /height_scan   (sensor_msgs/LaserScan, 187 ranges = heights in m)

Test it on its own:
    ros2 run mars_gazebo height_scanner_node.py
    ros2 topic echo /height_scan --once      # flat ground -> ~0.75 everywhere
"""

import math

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

# Training grid (mjlab terrain_scan): size (length_x, width_y), resolution.
GRID_SIZE = (1.6, 1.0)
GRID_RES = 0.1
MAX_DISTANCE = 5.0          # mjlab RayCastSensorCfg.max_distance / miss value
TERRAIN_GEOM_GROUP = 0      # mjlab include_geom_groups=(0,) -> terrain only


def build_grid_offsets() -> np.ndarray:
    """187 (x, y) offsets in mjlab meshgrid(indexing='xy').flatten() order."""
    sx, sy = GRID_SIZE
    x = np.arange(-sx / 2, sx / 2 + GRID_RES * 0.5, GRID_RES)
    y = np.arange(-sy / 2, sy / 2 + GRID_RES * 0.5, GRID_RES)
    offsets = []
    for yy in y:               # outer: y
        for xx in x:           # inner: x
            offsets.append((float(xx), float(yy)))
    return np.array(offsets, dtype=np.float64)


class HeightScanner(Node):
    def __init__(self):
        super().__init__("height_scanner_node")

        default_scene = ""
        try:
            share = get_package_share_directory("mars_gazebo")
            default_scene = share + "/unitree_g1_mjcf/mars_nav_scene.xml"
        except Exception:
            pass

        self.declare_parameter("scene_path", default_scene)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("odom_topic", "/ground_truth/odom")

        scene_path = self.get_parameter("scene_path").value
        self.get_logger().info(f"Loading scene for raycasting: {scene_path}")
        self._model = mujoco.MjModel.from_xml_path(scene_path)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self._offsets = build_grid_offsets()
        self._n = self._offsets.shape[0]
        self.get_logger().info(f"Height scan grid: {self._n} rays "
                               f"({GRID_SIZE} m @ {GRID_RES} m)")

        # Ray-cast only terrain (geom group 0); robot geoms (groups 2/3) ignored.
        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[TERRAIN_GEOM_GROUP] = 1

        # Latest pelvis pose in world (from odom).
        self._px = self._py = 0.0
        self._pz = 0.8
        self._yaw = 0.0
        self._have_pose = False

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.sub_odom = self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._odom_cb, qos)
        self.pub = self.create_publisher(LaserScan, "/height_scan", 10)

        rate = float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"Height scanner ready — {rate:.0f} Hz, "
                               f"terrain-only raycast, publishing /height_scan")

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._px, self._py, self._pz = p.x, p.y, p.z
        # yaw from quaternion
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)
        self._have_pose = True

    def _tick(self):
        if not self._have_pose:
            return
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        ranges = [0.0] * self._n
        geomid = np.zeros(1, dtype=np.int32)
        for k in range(self._n):
            gx, gy = self._offsets[k]
            # yaw-align the grid, place at the pelvis world position
            ox = self._px + (c * gx - s * gy)
            oy = self._py + (s * gx + c * gy)
            pnt = np.array([ox, oy, self._pz], dtype=np.float64)
            dist = mujoco.mj_ray(self._model, self._data, pnt, vec,
                                 self._geomgroup, 1, -1, geomid)
            # dist (vertical) == pelvis_z - terrain_z; miss -> max distance
            h = dist if dist >= 0.0 else MAX_DISTANCE
            ranges[k] = float(np.clip(h, 0.0, MAX_DISTANCE))

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "pelvis"
        # Angle fields are unused (this is a 2D grid, not a sweep); ranges[] is
        # the payload, ordered to match the policy's height_scan obs.
        msg.angle_min = 0.0
        msg.angle_max = float(self._n - 1)
        msg.angle_increment = 1.0
        msg.range_min = 0.0
        msg.range_max = MAX_DISTANCE
        msg.ranges = ranges
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = HeightScanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
