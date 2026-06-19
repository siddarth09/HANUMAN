#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Header

from navigation.mppi import MPPI


class MPPINode(Node):
    def __init__(self):
        super().__init__("mppi_local_planner")
        gp = self.declare_parameter
        gp("rate", 10.0)
        gp("lookahead", 1.5)          # carrot distance along the path (m)
        gp("goal_tol", 0.4)           # stop when within this of the final waypoint (m)
        gp("map_frame", "map")
        gp("odom_topic", "/odometry/filtered")
        # turn in place when heading error to the carrot is large, drive forward once aligned
        gp("align_thresh", 0.5)       # rad (~29 deg): above this, turn in place
        gp("turn_gain", 1.2)          # P-gain mapping heading error -> wz
        P = lambda n: self.get_parameter(n).value

        self.lookahead = P("lookahead")
        self.goal_tol = P("goal_tol")
        self.map_frame = P("map_frame")
        self.align_thresh = P("align_thresh")
        self.turn_gain = P("turn_gain")
        self.mppi = MPPI()

        # state
        self.cost_grid = None         # float (H,W); lethal/unknown -> mppi.lethal_cost
        self.origin = None            # (ox, oy)
        self.res = None
        self.path_xy = None           # (N, 2)
        self.pose = None              # (x, y, yaw)

        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/nav/global_costmap", self._costmap_cb, latched)
        self.create_subscription(Path, "/nav/global_path", self._path_cb, 1)
        self.create_subscription(Odometry, P("odom_topic"), self._odom_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.viz_pub = self.create_publisher(Path, "/nav/mppi_path", 1)

        self.create_timer(1.0 / P("rate"), self._control)
        self.get_logger().info("MPPI local planner ready — waiting for costmap + path + odom.")

    # ---- inputs ----
    def _costmap_cb(self, msg: OccupancyGrid):
        H, W = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(H, W)
        grid = arr.astype(float) / 100.0
        grid[arr < 0] = self.mppi.lethal_cost          # unknown / no DEM
        grid[arr >= 100] = self.mppi.lethal_cost       # lethal
        self.cost_grid = grid
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.res = msg.info.resolution

    def _path_cb(self, msg: Path):
        self.path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])
        self.mppi.reset()

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = float(np.arctan2(2 * (q.w * q.z + q.x * q.y),
                               1 - 2 * (q.y * q.y + q.z * q.z)))
        self.pose = (p.x, p.y, yaw)

    # ---- cost lookup over the global cost grid ----
    def _cost_query(self, xs, ys):
        ox, oy = self.origin
        H, W = self.cost_grid.shape
        i = np.floor((xs - ox) / self.res).astype(int)
        j = np.floor((ys - oy) / self.res).astype(int)
        inb = (i >= 0) & (i < W) & (j >= 0) & (j < H)
        out = np.full(xs.shape, self.mppi.lethal_cost)
        out[inb] = self.cost_grid[j[inb], i[inb]]
        return out

    def _carrot(self, x, y):
        """Pure-pursuit: closest point on the path, then advance by lookahead."""
        d = np.hypot(self.path_xy[:, 0] - x, self.path_xy[:, 1] - y)
        k = int(np.argmin(d))
        acc = 0.0
        while k < len(self.path_xy) - 1 and acc < self.lookahead:
            acc += np.hypot(*(self.path_xy[k + 1] - self.path_xy[k]))
            k += 1
        return self.path_xy[k]

    def _control(self):
        if self.cost_grid is None or self.path_xy is None or self.pose is None:
            return
        x, y, yaw = self.pose
        # goal reached?
        if np.hypot(*(self.path_xy[-1] - [x, y])) < self.goal_tol:
            self.cmd_pub.publish(Twist())          # stop
            self.mppi.reset()
            return
        carrot = self._carrot(x, y)
        yaw_err = np.arctan2(carrot[1] - y, carrot[0] - x) - yaw
        yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))   # wrap to [-pi,pi]

        t = Twist()
        if abs(yaw_err) > self.align_thresh:
            # turn in place toward the goal
            t.angular.z = float(np.clip(self.turn_gain * yaw_err,
                                        -self.mppi.wz_lim, self.mppi.wz_lim))
            self.mppi.reset()                       # drop stale forward momentum
            self.cmd_pub.publish(t)
            return
        cmd, traj = self.mppi.compute(x, y, yaw, carrot, self._cost_query)
        t.linear.x = float(cmd[0])                  # vy stays 0
        t.angular.z = float(cmd[2])
        self.cmd_pub.publish(t)
        self._publish_viz(traj)

    def _publish_viz(self, traj):
        path = Path()
        path.header = Header(frame_id=self.map_frame)
        path.header.stamp = self.get_clock().now().to_msg()
        for (px, py) in traj:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(px)
            ps.pose.position.y = float(py)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.viz_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = MPPINode()
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
