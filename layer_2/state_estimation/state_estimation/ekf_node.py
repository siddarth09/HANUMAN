from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from state_estimation.ekf_core import ErrorStateEKF


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        p = self._declare_and_get_params()
        self.p = p

        # ---- Build the EKF core from config ----
        ekf_config = {
            'gravitational_acceleration': p['gravitational_acceleration'],
            'initial_state': p['initial_state'],
            'initial_estimate_covariance': p['initial_estimate_covariance'],
            'accel_noise_density': p['accel_noise_density'],
            'gyro_noise_density': p['gyro_noise_density'],
            'accel_bias_random_walk': p['accel_bias_random_walk'],
            'gyro_bias_random_walk': p['gyro_bias_random_walk'],
            'frequency': p['frequency'],
            'sensor_timeout': p['sensor_timeout'],
            'print_diagnostics': p['print_diagnostics'],
        }
        self.ekf = ErrorStateEKF(ekf_config)
        self.gravity = np.array([0.0, 0.0, -p['gravitational_acceleration']])

        # ---- Register update sensors (prediction-only components masked out) ----
        # imu0: only the orientation components (idx 3,4,5) are registerable;
        # angular velocity / accel drive prediction, not the update.
        self.imu_orient_mask = self._registerable_mask(p['imu0_config'])
        self.imu_has_orientation = any(self.imu_orient_mask)
        if self.imu_has_orientation:
            self.ekf.register_sensor(
                name='imu0',
                config=self.imu_orient_mask,
                noise=np.full(sum(self.imu_orient_mask), p['imu0_orientation_noise']),
                differential=p['imu0_differential'],
                relative=p['imu0_relative'],
                pose_rejection_threshold=p['imu0_pose_rejection_threshold'],
            )

        self.odom0_mask = self._registerable_mask(p['odom0_config'])
        self.odom0_active = [i for i, a in enumerate(self.odom0_mask) if a]
        if self.odom0_active:
            self.ekf.register_sensor(
                name='odom0',
                config=self.odom0_mask,
                noise=np.full(len(self.odom0_active), p['odom0_noise']),
                differential=p['odom0_differential'],
                relative=p['odom0_relative'],
                pose_rejection_threshold=p['odom0_pose_rejection_threshold'],
                twist_rejection_threshold=p['odom0_twist_rejection_threshold'],
            )

        # terrain0: absolute z-position update from the DEM terrain matcher.
        terrain_z_cfg = [False] * 15
        terrain_z_cfg[2] = True   # meas index 2 == position z
        self.ekf.register_sensor(
            name='terrain0',
            config=terrain_z_cfg,
            noise=np.array([p['terrain0_z_noise'] ** 2]),
            pose_rejection_threshold=p['terrain0_rejection_threshold'],
        )

        # ---- Runtime state ----
        self._last_imu_time = None
        self._last_imu_stamp = None
        self.omega_body = np.zeros(3)          # latest bias-corrected gyro (for twist output)
        self.gt_position = None                 # latest ground truth (validation)
        self._diag_count = 0

        # ---- Subscribers ----
        self.create_subscription(
            Imu, p['imu0'], self.imu_cb, p['imu0_queue_size'])
        if self.odom0_active:
            self.create_subscription(
                Odometry, p['odom0'], self.odom0_cb, p['odom0_queue_size'])
        self.create_subscription(
            PoseWithCovarianceStamped, p['terrain0'], self.terrain0_cb,
            p['terrain0_queue_size'])
        # manual localization reset: snap EKF x,y to the clicked pose
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initialpose_cb, 10)
        if p['ground_truth_enabled']:
            self.create_subscription(
                Odometry, p['ground_truth_topic'], self.ground_truth_cb, 10)

        # ---- Publisher + TF ----
        self.odom_pub = self.create_publisher(Odometry, p['output_topic'], 10)
        self.tf_broadcaster = TransformBroadcaster(self) if p['publish_tf'] else None

        # ---- Publish timer ----
        self.create_timer(1.0 / max(p['frequency'], 1.0), self.publish_cb)

        self.get_logger().info("EKF node started")
        self.get_logger().info(f"  IMU (predict):   {p['imu0']}")
        self.get_logger().info(f"  IMU orientation fused: {self.imu_has_orientation}")
        self.get_logger().info(f"  odom0 (update):  {p['odom0']} active idx={self.odom0_active}")
        self.get_logger().info(f"  output:          {p['output_topic']} @ {p['frequency']} Hz")
        self.get_logger().info(f"  frames:          {p['world_frame']} -> {p['base_link_frame']}")
        self.get_logger().info(f"  ground truth:    {p['ground_truth_topic']} "
                               f"(enabled={p['ground_truth_enabled']}, never fused)")

    # ------------------------------------------------------------------ params
    def _declare_and_get_params(self) -> dict:
        defaults = {
            'frequency': 50.0,
            'sensor_timeout': 0.1,
            'print_diagnostics': True,
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_link_frame': 'base_link',
            'world_frame': 'odom',
            'publish_tf': True,
            'transform_time_offset': 0.0,
            'initial_state': [0.0, 0.0, 0.75] + [0.0] * 12,
            'imu0': '/imu_broadcaster/imu',
            'imu0_config': [False, False, False, True, True, True,
                            False, False, False, True, True, True,
                            True, True, True],
            'imu0_remove_gravitational_acceleration': True,
            'imu0_queue_size': 10,
            'imu0_differential': False,
            'imu0_relative': False,
            'imu0_orientation_noise': 0.01,
            'imu0_pose_rejection_threshold': float('inf'),
            'odom0': '/leg_odometry',
            'odom0_config': [False] * 6 + [True, True, True] + [False] * 6,
            'odom0_queue_size': 10,
            'odom0_differential': False,
            'odom0_relative': False,
            'odom0_noise': 0.05,
            'odom0_pose_rejection_threshold': float('inf'),
            'odom0_twist_rejection_threshold': float('inf'),
            # terrain0: absolute z from the DEM terrain matcher (the only z observation)
            'terrain0': '/terrain_match/pose',
            'terrain0_queue_size': 10,
            'terrain0_z_noise': 0.15,                 # std (m); overridden per-msg by its cov
            'terrain0_rejection_threshold': 25.0,     # reject gross z outliers
            'gravitational_acceleration': 9.81,
            'accel_noise_density': 0.5,
            'gyro_noise_density': 0.01,
            'accel_bias_random_walk': 0.01,
            'gyro_bias_random_walk': 0.001,
            'initial_estimate_covariance': (
                [1e-2] * 3 + [1e-2] * 3 + [1e-3] * 3 + [1e-2] * 3 + [1e-3] * 3),
            'ground_truth_topic': '/ground_truth/odom',
            'ground_truth_enabled': True,
            'output_topic': '/odometry/filtered',
        }
        out = {}
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            out[name] = self.get_parameter(name).value
        return out

    # ----------------------------------------------------------------- helpers
    def _registerable_mask(self, config15: list) -> list:
        """Keep only the config entries that map to an error state in the EKF.

        Components that drive prediction (angular velocity, linear accel) or
        sit past the MEAS_TO_ERROR table are dropped so register_sensor never
        sees an index it cannot map.
        """
        table = ErrorStateEKF.MEAS_TO_ERROR
        mask = [False] * 15
        for i, active in enumerate(config15):
            if active and i < len(table) and table[i] != -1:
                mask[i] = True
        return mask

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def _build_R(self, active_indices, pose_cov, twist_cov, fallback) -> np.ndarray:
        """Per-measurement noise from message covariance, falling back to a default.

        active_indices use the 15-element measurement convention; pose_cov and
        twist_cov are the flattened 6x6 row-major covariances from the message.
        """
        # diagonal element of a flattened 6x6 for the i-th of x,y,z,r,p,yaw
        pose_diag = [pose_cov[k * 6 + k] for k in range(6)]
        twist_diag = [twist_cov[k * 6 + k] for k in range(6)]
        variances = []
        for idx in active_indices:
            if idx < 6:          # pose: x,y,z,roll,pitch,yaw
                v = pose_diag[idx]
            elif idx < 12:       # twist: vx,vy,vz,wx,wy,wz
                v = twist_diag[idx - 6]
            else:
                v = 0.0
            if not np.isfinite(v) or v <= 0.0:
                v = fallback ** 2
            variances.append(v)
        return np.diag(variances)

    # --------------------------------------------------------------- callbacks
    def imu_cb(self, msg: Imu):
        # ---- Prediction inputs ----
        accel = np.array([msg.linear_acceleration.x,
                          msg.linear_acceleration.y,
                          msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x,
                         msg.angular_velocity.y,
                         msg.angular_velocity.z])

        # ekf_core expects raw specific force; if the IMU already removed
        # gravity, add it back (in body frame) so the internal model holds.
        if self.p['imu0_remove_gravitational_acceleration']:
            R = self.ekf._get_rotation_matrix()
            accel = accel - R.T @ self.gravity

        t = self._stamp_to_sec(msg.header.stamp)
        if self._last_imu_time is not None:
            dt = t - self._last_imu_time
            self.ekf.predict(accel, gyro, dt)
        self._last_imu_time = t
        self._last_imu_stamp = msg.header.stamp
        self.omega_body = gyro - self.ekf.bias_gyro

        # ---- Absolute orientation update from the IMU ----
        if self.imu_has_orientation:
            q = msg.orientation
            euler = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')
            measurement = np.zeros(15)
            measurement[3:6] = euler

            cov = list(msg.orientation_covariance)  # 3x3 row-major, [r,p,yaw]
            active = self.ekf.sensors['imu0']['active_meas_indices']
            variances = []
            for idx in active:
                v = cov[(idx - 3) * 3 + (idx - 3)] if idx >= 3 else 0.0
                if not np.isfinite(v) or v <= 0.0:
                    v = self.p['imu0_orientation_noise']
                variances.append(v)
            self.ekf.sensors['imu0']['R'] = np.diag(variances)
            self.ekf.update('imu0', measurement, timestamp=t)

    def odom0_cb(self, msg: Odometry):
        # Build a full 15-element measurement; the EKF picks the active indices.
        measurement = np.zeros(15)
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        measurement[0:3] = [pos.x, pos.y, pos.z]
        measurement[3:6] = Rotation.from_quat(
            [q.x, q.y, q.z, q.w]).as_euler('xyz')
        measurement[6:9] = [lin.x, lin.y, lin.z]
        measurement[9:12] = [ang.x, ang.y, ang.z]

        active = self.ekf.sensors['odom0']['active_meas_indices']
        self.ekf.sensors['odom0']['R'] = self._build_R(
            active, list(msg.pose.covariance), list(msg.twist.covariance),
            self.p['odom0_noise'])

        t = self._stamp_to_sec(msg.header.stamp)
        self.ekf.update('odom0', measurement, timestamp=t)

    def terrain0_cb(self, msg: PoseWithCovarianceStamped):
        # Absolute z (DEM ground height + nominal stand height) from the terrain matcher.
        # Weight it by the matcher's published z variance (loose during a relocalization).
        C = np.array(msg.pose.covariance).reshape(6, 6)
        self.ekf.sensors['terrain0']['R'] = np.array([[max(C[2, 2], 1e-6)]])
        measurement = np.zeros(15)
        measurement[2] = msg.pose.pose.position.z
        self.ekf.update('terrain0', measurement,
                        timestamp=self._stamp_to_sec(msg.header.stamp))

    def initialpose_cb(self, msg: PoseWithCovarianceStamped):
        self.ekf.position[0] = msg.pose.pose.position.x
        self.ekf.position[1] = msg.pose.pose.position.y
        self.get_logger().warn(
            f"EKF position RESET via /initialpose -> "
            f"({self.ekf.position[0]:.1f}, {self.ekf.position[1]:.1f})")

    def ground_truth_cb(self, msg: Odometry):
        # Validation only — never fused into the filter.
        self.gt_position = np.array([msg.pose.pose.position.x,
                                     msg.pose.pose.position.y,
                                     msg.pose.pose.position.z])

    # ---------------------------------------------------------------- publish
    def publish_cb(self):
        if self._last_imu_stamp is None:
            return  # no data yet

        state = self.ekf.get_state()
        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.p['world_frame']
        odom.child_frame_id = self.p['base_link_frame']

        pos = state['position']
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])

        # ekf quaternion is [w,x,y,z] -> ROS [x,y,z,w]
        qw, qx, qy, qz = state['quaternion']
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        vel = state['velocity']
        odom.twist.twist.linear.x = float(vel[0])
        odom.twist.twist.linear.y = float(vel[1])
        odom.twist.twist.linear.z = float(vel[2])
        odom.twist.twist.angular.x = float(self.omega_body[0])
        odom.twist.twist.angular.y = float(self.omega_body[1])
        odom.twist.twist.angular.z = float(self.omega_body[2])

        # Covariance from the error-state P diagonal.
        # P order: [p(0:3), v(3:6), theta(6:9), b_a(9:12), b_g(12:15)]
        pd = state['P_diagonal']
        pose_cov = [0.0] * 36
        for k in range(3):
            pose_cov[k * 6 + k] = float(pd[k])          # x,y,z
            pose_cov[(k + 3) * 6 + (k + 3)] = float(pd[6 + k])  # roll,pitch,yaw
        twist_cov = [0.0] * 36
        for k in range(3):
            twist_cov[k * 6 + k] = float(pd[3 + k])     # vx,vy,vz
            twist_cov[(k + 3) * 6 + (k + 3)] = float(self.p['gyro_noise_density'] ** 2)
        odom.pose.covariance = pose_cov
        odom.twist.covariance = twist_cov

        self.odom_pub.publish(odom)

        # ---- TF ----
        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = self.p['world_frame']
            tf.child_frame_id = self.p['base_link_frame']
            tf.transform.translation.x = float(pos[0])
            tf.transform.translation.y = float(pos[1])
            tf.transform.translation.z = float(pos[2])
            tf.transform.rotation.x = float(qx)
            tf.transform.rotation.y = float(qy)
            tf.transform.rotation.z = float(qz)
            tf.transform.rotation.w = float(qw)
            self.tf_broadcaster.sendTransform(tf)

        # ---- Diagnostics vs ground truth ----
        if self.p['print_diagnostics'] and self.gt_position is not None:
            self._diag_count += 1
            if self._diag_count % int(max(self.p['frequency'], 1.0)) == 0:
                err = np.linalg.norm(pos - self.gt_position)
                self.get_logger().info(
                    f"pos={np.round(pos, 3)} vel={np.round(vel, 3)} "
                    f"|err_vs_gt|={err:.3f} m")


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
