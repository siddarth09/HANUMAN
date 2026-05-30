#!/usr/bin/env python3
"""
HANUMAN — Simulated 3D LiDAR (Livox MID360-style) → PointCloud2

A standalone node that simulates a 3D lidar by ray-casting an azimuth×elevation
dome inside the MuJoCo scene and publishing the hit points as a 3D PointCloud2,
suitable for downstream elevation mapping / SLAM.

Why a node (not a mujoco_ros2_control lidar): that bridge models a *2D*
LaserScan (a single ring of rangefinders). A 3D lidar needs many elevation rings
and a 3D point cloud, so we ray-cast directly here.

How it stays in sync: it loads the scene once and ray-casts against TERRAIN +
obstacles only (mj_ray geom-group mask = group 0), so the robot never occludes
itself. The lidar's live world pose is obtained by mirroring the robot state
(/ground_truth/odom for the base, /joint_states for the joints) into the model
and reading the mount site — no TF tree required, so it is easy to test alone.

Publishes:  /mid360/points   (sensor_msgs/PointCloud2, in the lidar frame)

Test it alone (with the sim running):
    ros2 run mars_gazebo lidar3d_node.py
    ros2 topic echo /mid360/points --once | head
    # or visualise in RViz2: add PointCloud2 on /mid360/points (Fixed Frame: mid360_frame)
"""

import math

import mujoco
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

ENV_GEOM_GROUP = 0  # terrain + boulders are group 0; robot is groups 2/3


class Lidar3D(Node):
    def __init__(self):
        super().__init__("lidar3d_node")

        default_scene = ""
        try:
            share = get_package_share_directory("mars_gazebo")
            default_scene = share + "/unitree_g1_mjcf/mars_nav_scene.xml"
        except Exception:
            pass

        # ── Params (MID360-like FOV, biased downward for ground coverage) ──
        self.declare_parameter("scene_path", default_scene)
        self.declare_parameter("mount_site", "mid360_lidar")
        self.declare_parameter("frame_id", "mid360_frame")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("num_azimuth", 180)        # horizontal rays
        self.declare_parameter("num_elevation", 32)       # vertical rings
        self.declare_parameter("elevation_min_deg", -80.0)  # downward
        self.declare_parameter("elevation_max_deg", 15.0)   # slightly up
        self.declare_parameter("range_max", 30.0)
        self.declare_parameter("odom_topic", "/ground_truth/odom")

        scene_path = self.get_parameter("scene_path").value
        self.get_logger().info(f"Loading scene for 3D raycasting: {scene_path}")
        self._model = mujoco.MjModel.from_xml_path(scene_path)
        self._data = mujoco.MjData(self._model)

        self._mount = self.get_parameter("mount_site").value
        self._site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, self._mount)
        if self._site_id < 0:
            raise RuntimeError(f"mount_site '{self._mount}' not found in model")

        # Map /joint_states names -> qpos addresses, for state mirroring.
        self._qadr = {}
        for j in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is not None:
                self._qadr[name] = int(self._model.jnt_qposadr[j])
        # The floating base free joint qpos address (base pose).
        self._base_adr = None
        for j in range(self._model.njnt):
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self._base_adr = int(self._model.jnt_qposadr[j])
                break

        self._geomgroup = np.zeros(6, dtype=np.uint8)
        self._geomgroup[ENV_GEOM_GROUP] = 1
        self._range_max = float(self.get_parameter("range_max").value)
        self._dirs = self._build_ray_dirs()   # (N, 3) unit dirs in lidar frame
        self.get_logger().info(
            f"3D lidar pattern: {self._dirs.shape[0]} rays "
            f"({self.get_parameter('num_azimuth').value} az × "
            f"{self.get_parameter('num_elevation').value} el)")

        self._have_state = False

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._odom_cb, qos)
        self.create_subscription(JointState, "/joint_states", self._joints_cb, qos)
        self.pub = self.create_publisher(PointCloud2, "/mid360/points", 5)

        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"3D lidar ready — {rate:.0f} Hz, frame '{self.get_parameter('frame_id').value}', "
            f"publishing /mid360/points")

    def _build_ray_dirs(self) -> np.ndarray:
        n_az = int(self.get_parameter("num_azimuth").value)
        n_el = int(self.get_parameter("num_elevation").value)
        el0 = math.radians(float(self.get_parameter("elevation_min_deg").value))
        el1 = math.radians(float(self.get_parameter("elevation_max_deg").value))
        az = np.linspace(-math.pi, math.pi, n_az, endpoint=False)
        el = np.linspace(el0, el1, n_el)
        dirs = []
        for e in el:
            ce, se = math.cos(e), math.sin(e)
            for a in az:
                dirs.append((ce * math.cos(a), ce * math.sin(a), se))
        return np.array(dirs, dtype=np.float64)

    def _odom_cb(self, msg: Odometry):
        if self._base_adr is None:
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        a = self._base_adr
        self._data.qpos[a:a + 3] = [p.x, p.y, p.z]
        self._data.qpos[a + 3:a + 7] = [q.w, q.x, q.y, q.z]  # MuJoCo wxyz
        self._have_state = True

    def _joints_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            adr = self._qadr.get(name)
            if adr is not None:
                self._data.qpos[adr] = pos
        self._have_state = True

    def _tick(self):
        if not self._have_state:
            return
        # Forward kinematics to place the lidar mount at the live robot pose.
        mujoco.mj_kinematics(self._model, self._data)
        origin = np.array(self._data.site_xpos[self._site_id], dtype=np.float64)
        R = np.array(self._data.site_xmat[self._site_id], dtype=np.float64).reshape(3, 3)

        dirs_world = self._dirs @ R.T          # rotate local dirs into world
        geomid = np.zeros(1, dtype=np.int32)
        points = []
        for i in range(self._dirs.shape[0]):
            vec = dirs_world[i]
            dist = mujoco.mj_ray(self._model, self._data, origin, vec,
                                 self._geomgroup, 1, -1, geomid)
            if 0.0 <= dist <= self._range_max:
                # report the hit in the lidar (sensor) frame: local_dir * dist
                points.append(self._dirs[i] * dist)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.get_parameter("frame_id").value
        cloud = point_cloud2.create_cloud_xyz32(
            header, np.asarray(points, dtype=np.float32))
        self.pub.publish(cloud)


def main():
    rclpy.init()
    node = Lidar3D()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
