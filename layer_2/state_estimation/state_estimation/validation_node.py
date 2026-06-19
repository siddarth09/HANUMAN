import csv
import math
import os
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseWithCovarianceStamped


def _t(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class ValidationRecorder(Node):
    def __init__(self):
        super().__init__('validation_recorder')

        self.declare_parameter('ground_truth_topic', '/ground_truth/odom')
        self.declare_parameter('filtered_topic', '/odometry/filtered')
        self.declare_parameter('slam_topic', '/slam/odometry')
        self.declare_parameter('terrain_topic', '/terrain_match/pose')
        self.declare_parameter('leg_odom_topic', '/leg_odometry')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('left_ankle_joint', 'left_ankle_pitch_joint')
        self.declare_parameter('right_ankle_joint', 'right_ankle_pitch_joint')
        self.declare_parameter('left_wrench_topic',
                               '/left_foot_ft_broadcaster/wrench')
        self.declare_parameter('right_wrench_topic',
                               '/right_foot_ft_broadcaster/wrench')
        self.declare_parameter('force_contact_threshold', 30.0)  # N, foot Fz
        self.declare_parameter('contact_effort_threshold', 5.0)  # kept for reference
        self.declare_parameter('output_dir', 'output/')  
        self.declare_parameter('plot_on_exit', True)

        gp = lambda n: self.get_parameter(n).value
        self.left_joint = gp('left_ankle_joint')
        self.right_joint = gp('right_ankle_joint')
        self.effort_threshold = gp('contact_effort_threshold')
        self.force_threshold = gp('force_contact_threshold')
        self.output_dir = gp('output_dir') or os.path.join(
            '/tmp', 'se_validation', time.strftime('%Y%m%d_%H%M%S'))
        self.plot_on_exit = gp('plot_on_exit')
        os.makedirs(self.output_dir, exist_ok=True)

        # Buffers ---------------------------------------------------------
        self.gt = []        # t, x,y,z, qx,qy,qz,qw, vx,vy,vz
        self.ekf = []       # t, x,y,z, vx,vy,vz, qx,qy,qz,qw
        self.slam = []      # t, x,y,z, vx,vy,vz, qx,qy,qz,qw  (GTSAM SLAM)
        self.terrain = []   # t, x,y, yaw, sig_x, sig_y  (orbital-prior fixes)
        self.legodom = []   # t, vx,vy,vz, cov_vx
        self.joints = []    # t, left_effort, right_effort
        self.forces = []    # t, fz_left, fz_right  (foot ground-reaction force)
        self._li = None     # cached joint indices
        self._ri = None
        self.fz_left = 0.0
        self.fz_right = 0.0

        self.create_subscription(Odometry, gp('ground_truth_topic'), self.gt_cb, 50)
        self.create_subscription(Odometry, gp('filtered_topic'), self.ekf_cb, 50)
        self.create_subscription(Odometry, gp('slam_topic'), self.slam_cb, 50)
        self.create_subscription(PoseWithCovarianceStamped, gp('terrain_topic'),
                                 self.terrain_cb, 20)
        self.create_subscription(Odometry, gp('leg_odom_topic'), self.legodom_cb, 50)
        self.create_subscription(JointState, gp('joint_states_topic'), self.joints_cb, 50)
        self.create_subscription(WrenchStamped, gp('left_wrench_topic'), self.left_ft_cb, 50)
        self.create_subscription(WrenchStamped, gp('right_wrench_topic'), self.right_ft_cb, 50)

        self.get_logger().info(f"Validation recorder started. Output: {self.output_dir}")
        self.get_logger().info("Drive the robot, then Ctrl-C to write CSVs + plot.")

    # ---------------------------------------------------------------- callbacks
    def gt_cb(self, m: Odometry):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        v = m.twist.twist.linear
        self.gt.append([_t(m.header.stamp), p.x, p.y, p.z,
                        q.x, q.y, q.z, q.w, v.x, v.y, v.z])

    def ekf_cb(self, m: Odometry):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        v = m.twist.twist.linear
        self.ekf.append([_t(m.header.stamp), p.x, p.y, p.z,
                         v.x, v.y, v.z, q.x, q.y, q.z, q.w])

    def slam_cb(self, m: Odometry):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        v = m.twist.twist.linear
        self.slam.append([_t(m.header.stamp), p.x, p.y, p.z,
                          v.x, v.y, v.z, q.x, q.y, q.z, q.w])

    def terrain_cb(self, m: PoseWithCovarianceStamped):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        C = m.pose.covariance
        self.terrain.append([_t(m.header.stamp), p.x, p.y, yaw,
                             C[0]**0.5, C[7]**0.5])

    def legodom_cb(self, m: Odometry):
        v = m.twist.twist.linear
        self.legodom.append([_t(m.header.stamp), v.x, v.y, v.z,
                             m.twist.covariance[0]])

    def joints_cb(self, m: JointState):
        if not m.effort:
            return
        if self._li is None:
            names = list(m.name)
            self._li = names.index(self.left_joint) if self.left_joint in names else -1
            self._ri = names.index(self.right_joint) if self.right_joint in names else -1
            if self._li < 0 or self._ri < 0:
                self.get_logger().warn(
                    f"ankle joints not found in /joint_states (have {len(names)} names)")
        le = m.effort[self._li] if self._li is not None and self._li >= 0 else float('nan')
        re = m.effort[self._ri] if self._ri is not None and self._ri >= 0 else float('nan')
        self.joints.append([_t(m.header.stamp), le, re])

    def left_ft_cb(self, m: WrenchStamped):
        self.fz_left = m.wrench.force.z
        self.forces.append([_t(m.header.stamp), self.fz_left, self.fz_right])

    def right_ft_cb(self, m: WrenchStamped):
        self.fz_right = m.wrench.force.z
        self.forces.append([_t(m.header.stamp), self.fz_left, self.fz_right])

    # ------------------------------------------------------------------- output
    def _dump(self, name, header, rows):
        path = os.path.join(self.output_dir, name)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        self.get_logger().info(f"  wrote {len(rows):6d} rows -> {path}")
        return path

    def write_all(self):
        self.get_logger().info("Writing CSVs...")
        self._dump('gt.csv',
                   ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'vx', 'vy', 'vz'],
                   self.gt)
        self._dump('ekf.csv',
                   ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw'],
                   self.ekf)
        self._dump('slam.csv',
                   ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'qx', 'qy', 'qz', 'qw'],
                   self.slam)
        self._dump('terrain.csv', ['t', 'x', 'y', 'yaw', 'sig_x', 'sig_y'], self.terrain)
        self._dump('legodom.csv', ['t', 'vx', 'vy', 'vz', 'cov_vx'], self.legodom)
        self._dump('joints.csv', ['t', 'left_effort', 'right_effort'], self.joints)
        self._dump('forces.csv', ['t', 'fz_left', 'fz_right'], self.forces)
        # Record thresholds so the plotter can shade contact bands. Foot force is
        # the real contact signal; effort is kept only for reference.
        with open(os.path.join(self.output_dir, 'meta.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['force_contact_threshold', self.force_threshold])
            w.writerow(['contact_effort_threshold', self.effort_threshold])


def main(args=None):
    rclpy.init(args=args)
    node = ValidationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_all()
        out = node.output_dir
        plot = node.plot_on_exit
        node.destroy_node()
        rclpy.shutdown()
        if plot:
            try:
                from state_estimation.plot_validation import plot_run
                plot_run(out)
            except Exception as e:  # plotting must never crash the recorder
                print(f"[validation] plot failed ({e}); run manually:\n"
                      f"  python3 -m state_estimation.plot_validation {out}")


if __name__ == '__main__':
    main()
