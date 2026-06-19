#!/usr/bin/env python3
"""Global planner: HiRISE DEM -> geometric cost map -> A* path.

Publishes the global cost map (latched OccupancyGrid) and, on each "2D Goal Pose",
plans an A* path from the current pose to the goal.

Topics:
  pub  /nav/global_costmap  nav_msgs/OccupancyGrid   (latched)
  pub  /nav/global_path     nav_msgs/Path
  sub  /goal_pose           geometry_msgs/PoseStamped   (RViz 2D Goal Pose)
  sub  /terrain_match/pose  geometry_msgs/PoseWithCovarianceStamped (start = current pose)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

from navigation.costmap import build_costmap, to_occupancy
from navigation.planner import astar


class GlobalPlanner(Node):
    def __init__(self):
        super().__init__("global_planner")
        gp = self.declare_parameter
        gp("dem_cache", "/tmp/hanuman_dem.npz")
        gp("map_frame", "map")
        gp("slope_max_deg", 25.0)
        gp("rough_radius", 2)
        gp("rough_max", 0.15)
        gp("w_slope", 0.6)
        gp("w_rough", 0.4)
        gp("cost_penalty", 4.0)      # weight on high-cost cells
        gp("start_xy", [0.0, 0.0])   # fallback start if no pose yet
        P = lambda n: self.get_parameter(n).value

        self.map_frame = P("map_frame")
        self.cost_penalty = P("cost_penalty")

        # load DEM (cached by terrain_localization)
        cache = P("dem_cache")
        try:
            z = np.load(cache)
            self.gx, self.gy, self.Z = z["gx"], z["gy"], z["Z"]
        except FileNotFoundError:
            self.get_logger().error(
                f"DEM cache {cache} not found — run terrain_localization once to build it.")
            raise
        self.res = float(self.gx[1] - self.gx[0])

        # build cost map
        self.cost, self.lethal = build_costmap(
            self.Z, self.res, P("slope_max_deg"), int(P("rough_radius")),
            P("rough_max"), P("w_slope"), P("w_rough"))
        self.get_logger().info(
            f"cost map: {self.cost.shape} @ {self.res:.2f} m/px, "
            f"{100*self.lethal.mean():.1f}% lethal, "
            f"bounds x[{self.gx[0]:.1f},{self.gx[-1]:.1f}] y[{self.gy[0]:.1f},{self.gy[-1]:.1f}]")

        self.start_xy = np.array(P("start_xy"), float)

        # ROS I/O
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.costmap_pub = self.create_publisher(OccupancyGrid, "/nav/global_costmap", latched)
        self.path_pub = self.create_publisher(Path, "/nav/global_path", 1)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_cb, 1)
        self.create_subscription(Odometry, "/odometry/filtered", self._pose_cb, 10)

        self._publish_costmap()
        self.get_logger().info("global planner ready — set a goal in RViz (2D Goal Pose).")

    # ---- helpers ----
    def _world_to_cell(self, x, y):
        i = int(round((x - self.gx[0]) / self.res))
        j = int(round((y - self.gy[0]) / self.res))
        return i, j

    def _pose_cb(self, msg: Odometry):
        self.start_xy = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

    def _publish_costmap(self):
        occ = to_occupancy(self.cost, self.lethal)
        msg = OccupancyGrid()
        msg.header = Header(frame_id=self.map_frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = self.res
        msg.info.width = len(self.gx)
        msg.info.height = len(self.gy)
        msg.info.origin.position.x = float(self.gx[0] - self.res / 2)
        msg.info.origin.position.y = float(self.gy[0] - self.res / 2)
        msg.info.origin.orientation.w = 1.0
        msg.data = occ.flatten(order="C").tolist()   # row-major: data[j*W + i]
        self.costmap_pub.publish(msg)

    def _goal_cb(self, msg: PoseStamped):
        gx, gy = msg.pose.position.x, msg.pose.position.y
        start = self._world_to_cell(*self.start_xy)
        goal = self._world_to_cell(gx, gy)
        self.get_logger().info(
            f"planning {tuple(np.round(self.start_xy,2))} -> ({gx:.2f},{gy:.2f}) "
            f"cells {start} -> {goal}")
        cells = astar(self.cost, self.lethal, start, goal, self.cost_penalty)
        if cells is None:
            self.get_logger().warn("no path found (start/goal lethal or unreachable)")
            return
        self._publish_path(cells)

    def _publish_path(self, cells):
        path = Path()
        path.header = Header(frame_id=self.map_frame)
        path.header.stamp = self.get_clock().now().to_msg()
        for (i, j) in cells:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(self.gx[i])
            ps.pose.position.y = float(self.gy[j])
            zc = self.Z[j, i]
            ps.pose.position.z = float(zc + 0.05) if np.isfinite(zc) else 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.path_pub.publish(path)
        length = sum(np.hypot(self.gx[cells[k+1][0]] - self.gx[cells[k][0]],
                              self.gy[cells[k+1][1]] - self.gy[cells[k][1]])
                     for k in range(len(cells) - 1))
        self.get_logger().info(f"path: {len(cells)} waypoints, {length:.1f} m")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlanner()
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
