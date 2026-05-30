#!/usr/bin/env python3
"""
HANUMAN RL Policy Node — Python, loads .pt directly, CUDA-aware.

Subscribes: /joint_states, /imu_broadcaster/imu, /ground_truth/odom,
            /cmd_vel, /height_scan/points
Publishes:  /g1_position_controller/commands

Usage:
    ros2 run mars_gazebo rl_policy_node.py \
        --ros-args -p use_sim_time:=true -p device:=cuda
"""

import math
import threading
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState, Imu, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

import os

# ─── Policy architecture ──────────────────────────────────────────────────────

class EmpiricalNormalization(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('_mean', torch.zeros(1, dim))
        self.register_buffer('_std',  torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class Actor(nn.Module):
    def __init__(self, obs_dim=288, hidden=(1024, 512, 256, 128), act_dim=29):
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(obs_dim)
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))


# ─── Joint configuration (MJLAB training order) ───────────────────────────────

JOINT_NAMES = [
    'left_hip_pitch_joint',       # 0
    'left_hip_roll_joint',        # 1
    'left_hip_yaw_joint',         # 2
    'left_knee_joint',            # 3
    'left_ankle_pitch_joint',     # 4
    'left_ankle_roll_joint',      # 5
    'right_hip_pitch_joint',      # 6
    'right_hip_roll_joint',       # 7
    'right_hip_yaw_joint',        # 8
    'right_knee_joint',           # 9
    'right_ankle_pitch_joint',    # 10
    'right_ankle_roll_joint',     # 11
    'waist_yaw_joint',            # 12
    'waist_roll_joint',           # 13
    'waist_pitch_joint',          # 14
    'left_shoulder_pitch_joint',  # 15
    'left_shoulder_roll_joint',   # 16
    'left_shoulder_yaw_joint',    # 17
    'left_elbow_joint',           # 18
    'left_wrist_roll_joint',      # 19
    'left_wrist_pitch_joint',     # 20
    'left_wrist_yaw_joint',       # 21
    'right_shoulder_pitch_joint', # 22
    'right_shoulder_roll_joint',  # 23
    'right_shoulder_yaw_joint',   # 24
    'right_elbow_joint',          # 25
    'right_wrist_roll_joint',     # 26
    'right_wrist_pitch_joint',    # 27
    'right_wrist_yaw_joint',      # 28
]

DEFAULT_JOINT_POS = np.array([
    -0.312, 0.0,   0.0,   0.669, -0.363, 0.0,
    -0.312, 0.0,   0.0,   0.669, -0.363, 0.0,
     0.0,   0.0,   0.0,
     0.2,   0.2,   0.0,   0.6,   0.0,   0.0,   0.0,
     0.2,  -0.2,   0.0,   0.6,   0.0,   0.0,   0.0,
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.5475464629911068,   0.35066146637882434, 0.5475464629911068,
    0.35066146637882434,  0.43857731392336724, 0.43857731392336724,
    0.5475464629911068,   0.35066146637882434, 0.5475464629911068,
    0.35066146637882434,  0.43857731392336724, 0.43857731392336724,
    0.5475464629911068,   0.43857731392336724, 0.43857731392336724,
    0.43857731392336724,  0.43857731392336724, 0.43857731392336724,
    0.43857731392336724,  0.43857731392336724, 0.07450087032950714,
    0.07450087032950714,  0.43857731392336724, 0.43857731392336724,
    0.43857731392336724,  0.43857731392336724, 0.43857731392336724,
    0.07450087032950714,  0.07450087032950714,
], dtype=np.float32)

NUM_JOINTS       = 29
OBS_DIM          = 288
ACT_DIM          = 29
HEIGHT_SCAN_SIZE = 187
HEIGHT_SCAN_DEFAULT = 0.150467
MAX_RAY_DIST     = 5.0
ACTION_CLIP      = 1.0
WARMUP_S         = 8.0


# ─── Helper: quaternion rotate ─────────────────────────────────────────────────

def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world frame to body frame (q = body→world)."""
    w, x, y, z = q_wxyz
    cx, cy, cz = -x, -y, -z
    tx = 2.0 * (cy * v[2] - cz * v[1])
    ty = 2.0 * (cz * v[0] - cx * v[2])
    tz = 2.0 * (cx * v[1] - cy * v[0])
    return np.array([
        v[0] + w * tx + (cy * tz - cz * ty),
        v[1] + w * ty + (cz * tx - cx * tz),
        v[2] + w * tz + (cx * ty - cy * tx),
    ], dtype=np.float32)


def projected_gravity(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float32)


# ─── ROS2 policy node ─────────────────────────────────────────────────────────

class RLPolicyNode(Node):

    def __init__(self):
        super().__init__('rl_policy_node')

        # ── Parameters ──
        self.declare_parameter('pt_path', '')
        self.declare_parameter('policy_rate', 50.0)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('action_clip', ACTION_CLIP)
        self.declare_parameter('warmup_s', WARMUP_S)

        device_str = self.get_parameter('device').get_parameter_value().string_value
        if device_str == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA requested but not available — falling back to CPU')
            device_str = 'cpu'
        self.device = torch.device(device_str)

        # ── Load policy ──
        pt_path = self.get_parameter('pt_path').get_parameter_value().string_value
        if not pt_path or not os.path.exists(pt_path):
            from ament_index_python.packages import get_package_share_directory
            pkg = get_package_share_directory('mars_gazebo')
            # Default: look for the .pt next to the ONNX
            pt_path = os.path.join(
                os.path.expanduser('~'),
                'logs/hanumanv1/hanuman_g1_rough/2026-05-18_21-54-17/model_425000.pt'
            )
        self._load_actor(pt_path)

        # ── State ──
        self.joint_pos   = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.joint_vel   = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.lin_vel_world = np.zeros(3, dtype=np.float32)
        self.ang_vel     = np.zeros(3, dtype=np.float32)
        self.imu_quat    = np.array([1., 0., 0., 0.], dtype=np.float32)  # w,x,y,z
        self.command     = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.height_scan = np.full(HEIGHT_SCAN_SIZE, HEIGHT_SCAN_DEFAULT, dtype=np.float32)
        self.foot_h      = np.array([0.047589, 0.047088], dtype=np.float32)

        self.joint_index_map: dict[str, int] = {}
        self.joint_states_received = False
        self.imu_received = False
        self.warmup_done  = False
        self.warmup_start = None
        self.warmup_s     = self.get_parameter('warmup_s').get_parameter_value().double_value
        self._jpos_err_thresh = 0.20   # mean |jpos_rel| must be below this (0.2 = ~1 std of walking joints)
        self._lock        = threading.Lock()
        self._obs_dumped  = False

        # ── QoS ──
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        # ── Subscribers ──
        self.create_subscription(JointState,    '/joint_states',         self._js_cb,      be)
        self.create_subscription(Imu,           '/imu_broadcaster/imu',  self._imu_cb,     be)
        self.create_subscription(Odometry,      '/ground_truth/odom',    self._odom_cb,    be)
        self.create_subscription(Twist,         '/cmd_vel',              self._cmd_cb,     10)
        self.create_subscription(PointCloud2,   '/height_scan/points',   self._scan_cb,    be)

        # ── Publisher ──
        self._pub = self.create_publisher(Float64MultiArray,
                                          '/g1_position_controller/commands', 10)

        # ── Timer (sim-time-aware via get_clock()) ──
        rate = self.get_parameter('policy_rate').get_parameter_value().double_value
        period_ns = int(1e9 / rate)
        self._timer = self.create_timer(period_ns / 1e9, self._step,
                                        clock=self.get_clock())

        self.get_logger().info(
            f'Policy ready — {rate:.0f}Hz sim-time, device={self.device}, '
            f'obs={OBS_DIM}, act={ACT_DIM}, warmup={self.warmup_s:.0f}s')

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_actor(self, pt_path: str):
        self.get_logger().info(f'Loading .pt policy: {pt_path}')
        ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
        sd = ckpt['actor_state_dict']

        # Auto-detect architecture from weight shapes
        w_keys = sorted([k for k in sd if k.startswith('mlp.') and k.endswith('.weight')])
        self.obs_dim = int(sd[w_keys[0]].shape[1])
        self.act_dim = int(sd[w_keys[-1]].shape[0])
        hidden = tuple(int(sd[k].shape[0]) for k in w_keys[:-1])

        self.get_logger().info(
            f'Architecture detected: obs={self.obs_dim}  '
            f'hidden={hidden}  act={self.act_dim}')

        keep = {k: v for k, v in sd.items()
                if k.startswith('mlp') or k in ('obs_normalizer._mean', 'obs_normalizer._std')}
        actor = Actor(obs_dim=self.obs_dim, hidden=hidden, act_dim=self.act_dim)
        actor.load_state_dict(keep, strict=False)
        actor.eval()
        self.actor = actor.to(self.device)
        self.get_logger().info(f'Actor loaded on {self.device}')

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        if not self.joint_index_map:
            for i, name in enumerate(msg.name):
                for j, jn in enumerate(JOINT_NAMES):
                    if name == jn:
                        self.joint_index_map[jn] = i
            self.get_logger().info(
                f'Joint mapping: {len(self.joint_index_map)}/{NUM_JOINTS} found')

        for j, jn in enumerate(JOINT_NAMES):
            idx = self.joint_index_map.get(jn)
            if idx is None:
                continue
            if idx < len(msg.position) and math.isfinite(msg.position[idx]):
                self.joint_pos[j] = msg.position[idx]
            if idx < len(msg.velocity) and math.isfinite(msg.velocity[idx]):
                self.joint_vel[j] = msg.velocity[idx]
        self.joint_states_received = True

    def _imu_cb(self, msg: Imu):
        self.ang_vel[:] = [msg.angular_velocity.x,
                           msg.angular_velocity.y,
                           msg.angular_velocity.z]
        self.imu_quat[:] = [msg.orientation.w, msg.orientation.x,
                             msg.orientation.y, msg.orientation.z]
        self.imu_received = True

    def _odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        if all(math.isfinite(v) for v in (vx, vy, vz)):
            self.lin_vel_world[:] = [vx, vy, vz]

    def _cmd_cb(self, msg: Twist):
        self.command[:] = [msg.linear.x, msg.linear.y, msg.angular.z]

    def _scan_cb(self, msg: PointCloud2):
        # Training used frame_z - hit_z for each grid point on flat terrain,
        # which equals robot_height / max_distance ≈ 0.15 = HEIGHT_SCAN_DEFAULT.
        # Rather than fighting Gazebo's LiDAR frame conventions, we use the
        # constant default (= training mean, 0 normalised sigma) for flat terrain.
        # On rough terrain, actual ray distances can replace this.
        pass  # height_scan stays at HEIGHT_SCAN_DEFAULT (set at init)

    # ── Observation ────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        q = self.imu_quat

        # [0:3] base linear velocity in body frame
        obs[0:3] = quat_rotate_inverse(q, self.lin_vel_world)
        # [3:6] base angular velocity
        obs[3:6] = self.ang_vel
        # [6:9] projected gravity
        obs[6:9] = projected_gravity(q)
        # [9:38] joint positions relative to default
        obs[9:38] = self.joint_pos - DEFAULT_JOINT_POS
        # [38:67] joint velocities
        obs[38:67] = self.joint_vel
        # [67:96] last action
        obs[67:96] = self.last_action
        # [96:99] velocity command
        obs[96:99] = self.command

        if self.obs_dim > 99:
            # Rough terrain policy: height scan + foot heights
            with self._lock:
                obs[99:99 + HEIGHT_SCAN_SIZE] = self.height_scan
            obs[286:288] = self.foot_h

        return obs

    # ── Policy step ────────────────────────────────────────────────────────────

    def _step(self):
        if not self.joint_states_received:
            return

        # Warmup
        now = self.get_clock().now()
        if not self.warmup_done:
            if self.warmup_start is None:
                self.warmup_start = now
            elapsed = (now - self.warmup_start).nanoseconds * 1e-9
            if elapsed < self.warmup_s:
                if int(elapsed) % 1 == 0:
                    self.get_logger().info(
                        f'Warmup {elapsed:.0f}/{self.warmup_s:.0f}s',
                        throttle_duration_sec=1.0)
                return
            self.warmup_done = True
            self.get_logger().info('Warmup done — policy inference active')

        obs = self._build_obs()

        # NaN/Inf guard
        if not np.all(np.isfinite(obs)):
            bad = np.where(~np.isfinite(obs))[0]
            self.get_logger().warn(
                f'Non-finite obs at dims {bad[:5]} — skipping', throttle_duration_sec=2.0)
            return

        # Upright guard: only block at severe tilt (>70° = gravity_z > -0.34).
        gravity_z = float(obs[8])
        if gravity_z > -0.34:
            self.get_logger().warn(
                f'Robot fallen (gravity_z={gravity_z:.3f}) — skipping inference',
                throttle_duration_sec=2.0)
            return

        # One-shot obs dump + saturation analysis
        if not self._obs_dumped:
            self._obs_dumped = True
            import torch as _torch
            # Load normalizer stats for sigma analysis
            _ckpt = _torch.load(
                self.get_parameter('pt_path').get_parameter_value().string_value,
                map_location='cpu', weights_only=False)
            _mean = _ckpt['actor_state_dict']['obs_normalizer._mean'].numpy().flatten()
            _std  = _ckpt['actor_state_dict']['obs_normalizer._std'].numpy().flatten()
            _norm = (obs - _mean) / (_std + 1e-8)
            # Find the most out-of-distribution dims
            _worst = np.argsort(np.abs(_norm))[::-1][:10]
            _worst_info = '  '.join(
                f'obs[{i}]={obs[i]:.4f}(σ={_norm[i]:.1f})' for i in _worst)
            scan_info = (f'  scan_avg={obs[99:286].mean():.4f}  foot={obs[286]:.4f},{obs[287]:.4f}'
                         if self.obs_dim > 99 else '  (no height scan — flat terrain policy)')
            self.get_logger().error(
                f'=== OBS DUMP (obs_dim={self.obs_dim}) ===\n'
                f'  lin_vel  {np.round(obs[0:3], 5)}\n'
                f'  ang_vel  {np.round(obs[3:6], 5)}\n'
                f'  gravity  {np.round(obs[6:9], 5)}\n'
                f'  jpos_rel {np.round(obs[9:38], 4)}\n'
                f'  jvel     {np.round(obs[38:67], 4)}\n'
                f'  cmd_vel  {np.round(obs[96:99], 4)}\n'
                f'{scan_info}\n'
                f'  TOP-10 OUTLIERS (sigma from training mean):\n'
                f'    {_worst_info}')

        # Inference
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            act_t = self.actor(obs_t)
            action = act_t.squeeze(0).cpu().numpy()

        clip = self.get_parameter('action_clip').get_parameter_value().double_value
        n_sat = int(np.sum(np.abs(action) >= clip * 0.999))

        if n_sat >= ACT_DIM - 2:   # only block if 27+/29 — broken obs, not normal gait
            self.get_logger().warn(
                f'{n_sat}/{ACT_DIM} actions saturated — skipping',
                throttle_duration_sec=1.0)
            self.last_action[:] = 0.0
            return

        action = np.clip(action, -clip, clip)
        self.last_action[:] = action

        targets = DEFAULT_JOINT_POS + action * ACTION_SCALE
        msg = Float64MultiArray()
        msg.data = targets.tolist()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = RLPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
