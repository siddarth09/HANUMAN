#!/usr/bin/env python3
import math
import os
import threading

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState, Imu, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory




class EmpiricalNormalization(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class Actor(nn.Module):
    def __init__(self, obs_dim=288, hidden=(1024, 512, 256, 128), act_dim=29):
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(obs_dim)
        layers, prev = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))


# ─── Joint config (MJLAB training order) ──────────────────────────────────────

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

DEFAULT_JOINT_POS = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.5475464629911068, 0.35066146637882434, 0.5475464629911068,
    0.35066146637882434, 0.43857731392336724, 0.43857731392336724,
    0.5475464629911068, 0.35066146637882434, 0.5475464629911068,
    0.35066146637882434, 0.43857731392336724, 0.43857731392336724,
    0.5475464629911068, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.07450087032950714,
    0.07450087032950714, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
    0.07450087032950714, 0.07450087032950714,
], dtype=np.float32)

NUM_JOINTS = 29
OBS_DIM = 288
ACT_DIM = 29
HEIGHT_SCAN_SIZE = 187
HEIGHT_SCAN_DEFAULT = 0.150467
MAX_RAY_DIST = 5.0


def quat_rotate_inverse(q_wxyz, v):
    """Rotate v from world into body frame (q = body→world)."""
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


def projected_gravity(q_wxyz):
    w, x, y, z = q_wxyz
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float32)


class RLPolicyNode(Node):
    def __init__(self):
        super().__init__("rl_policy_node")

        self.declare_parameter("model_path", "/home/sid/projects25/src/HANUMAN/mars_gazebo/policy/model_270000.pt")
        self.declare_parameter("policy_rate", 50.0)   # policy trained at 50 Hz
        self.declare_parameter("device", "cuda")        # "cpu" or "cuda"
        self.declare_parameter("action_clip", 1.0)
        self.declare_parameter("warmup_s", 5.0)
        # Source for base linear velocity (obs[0:3]). EKF closes the state-estimation
        # loop (no ground-truth cheat); set to /ground_truth/odom to A/B against truth.
        # Both publish WORLD-frame twist, rotated world->body by quat_rotate_inverse.
        self.declare_parameter("odom_topic", "/odometry/filtered")

        self.device = self._select_device(self.get_parameter("device").value)
        self._load_actor(self._resolve_model_path())

        # ── State ──
        self.joint_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.lin_vel_world = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.imu_quat = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.height_scan = np.full(HEIGHT_SCAN_SIZE, HEIGHT_SCAN_DEFAULT, dtype=np.float32)
        self.foot_h = np.array([0.047589, 0.047088], dtype=np.float32)

        self.joint_index_map = {}
        self.joint_states_received = False
        self.height_scan_received = False
        self.warmup_done = False
        self.warmup_start = None
        self.warmup_s = self.get_parameter("warmup_s").value
        self._last_warmup_log = -1
        self._lock = threading.Lock()

        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(JointState, "/joint_states", self._js_cb, be)
        self.create_subscription(Imu, "/imu_broadcaster/imu", self._imu_cb, be)
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._odom_cb, be)
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)
        self.create_subscription(LaserScan, "/height_scan", self._scan_cb, be)
        self._pub = self.create_publisher(
            Float64MultiArray, "/g1_position_controller/commands", 10)

        rate = self.get_parameter("policy_rate").value
        self.create_timer(1.0 / rate, self._step, clock=self.get_clock())
        self.get_logger().info(
            f"Policy ready — {rate:.0f} Hz, device={self.device}, "
            f"obs={OBS_DIM}, act={ACT_DIM}, warmup={self.warmup_s:.0f}s")

    # ── Setup helpers ──────────────────────────────────────────────────────

    def _select_device(self, requested: str) -> torch.device:
        if requested != "cuda":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            self.get_logger().warn("device=cuda but CUDA unavailable — using CPU")
            return torch.device("cpu")
        try:  
            _ = (torch.zeros(8, device="cuda") + 1).sum().item()
            torch.cuda.synchronize()
            self.get_logger().info(f"CUDA OK: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except Exception as e:
            self.get_logger().warn(
                f"CUDA present but unusable ({str(e)[:80]}) — using CPU. "
                f"Launch with a torch built for this GPU (e.g. "
                f"/home/sid/mujoco_env/bin/python) for GPU inference.")
            return torch.device("cpu")

    def _resolve_model_path(self) -> str:
        p = self.get_parameter("model_path").value
        if p and os.path.exists(p):
            return p
     
        share = get_package_share_directory("mars_gazebo")
        for cand in ("model_425000.pt", "hanuman_policy.pt"):
            fp = os.path.join(share, "policy", cand)
            if os.path.exists(fp):
                return fp
        return os.path.join(share, "policy", "hanuman_policy.pt")

    def _load_actor(self, path: str):
        self.get_logger().info(f"Loading policy: {path}")
        # Try TorchScript first; fall back to a raw rsl_rl checkpoint.
        try:
            m = torch.jit.load(path, map_location=self.device)
            m.eval()
            self.actor = m
            self.get_logger().info("Loaded as TorchScript module")
            return
        except (RuntimeError, ValueError):
            pass
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["actor_state_dict"]
        w_keys = sorted(k for k in sd if k.startswith("mlp.") and k.endswith(".weight"))
        obs_dim = int(sd[w_keys[0]].shape[1])
        act_dim = int(sd[w_keys[-1]].shape[0])
        hidden = tuple(int(sd[k].shape[0]) for k in w_keys[:-1])
        keep = {k: v for k, v in sd.items()
                if k.startswith("mlp")
                or k in ("obs_normalizer._mean", "obs_normalizer._std")}
        actor = Actor(obs_dim, hidden, act_dim)
        actor.load_state_dict(keep, strict=False)
        actor.eval()
        self.actor = actor.to(self.device)
        self.get_logger().info(
            f"Loaded rsl_rl checkpoint — obs={obs_dim} hidden={hidden} act={act_dim}")

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _js_cb(self, msg: JointState):
        if not self.joint_index_map:
            for i, name in enumerate(msg.name):
                if name in JOINT_NAMES:
                    self.joint_index_map[name] = i
            self.get_logger().info(
                f"Joint mapping: {len(self.joint_index_map)}/{NUM_JOINTS} found")
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
        self.ang_vel[:] = [msg.angular_velocity.x, msg.angular_velocity.y,
                           msg.angular_velocity.z]
        self.imu_quat[:] = [msg.orientation.w, msg.orientation.x,
                            msg.orientation.y, msg.orientation.z]

    def _odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear
        if all(math.isfinite(c) for c in (v.x, v.y, v.z)):
            self.lin_vel_world[:] = [v.x, v.y, v.z]

    def _cmd_cb(self, msg: Twist):
        self.command[:] = [msg.linear.x, msg.linear.y, msg.angular.z]

    def _scan_cb(self, msg: LaserScan):
        # height_scanner_node publishes 187 vertical heights (pelvis_z - terrain_z)
        # in mjlab grid order; obs = height / max_distance (miss -> max).
        n = min(len(msg.ranges), HEIGHT_SCAN_SIZE)
        r = np.asarray(msg.ranges[:n], dtype=np.float32)
        r = np.where(np.isfinite(r), np.clip(r, 0.0, MAX_RAY_DIST), MAX_RAY_DIST)
        with self._lock:
            self.height_scan[:n] = r / MAX_RAY_DIST
            if n < HEIGHT_SCAN_SIZE:
                self.height_scan[n:] = 1.0
        self.height_scan_received = True

    # ── Observation + step ───────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        q = self.imu_quat
        obs[0:3] = quat_rotate_inverse(q, self.lin_vel_world)
        obs[3:6] = self.ang_vel
        obs[6:9] = projected_gravity(q)
        obs[9:38] = self.joint_pos - DEFAULT_JOINT_POS
        obs[38:67] = self.joint_vel
        obs[67:96] = self.last_action
        obs[96:99] = self.command
        with self._lock:
            obs[99:99 + HEIGHT_SCAN_SIZE] = self.height_scan
        obs[286:288] = self.foot_h
        return obs

    def _step(self):
        if not self.joint_states_received:
            return

        now = self.get_clock().now()
        if not self.warmup_done:
            if self.warmup_start is None:
                self.warmup_start = now
            elapsed = (now - self.warmup_start).nanoseconds * 1e-9
            if elapsed < self.warmup_s:
                if int(elapsed) != self._last_warmup_log:
                    self._last_warmup_log = int(elapsed)
                    self.get_logger().info(
                        f"Warmup {elapsed:.0f}/{self.warmup_s:.0f}s — settling")
                return
            self.warmup_done = True
            if not self.height_scan_received:
                self.get_logger().warn(
                    "No /height_scan yet — run height_scanner_node.py "
                    "(obs[99:286] is using the flat-terrain default)")
            self.get_logger().info("Warmup done — policy inference active")

        obs = self._build_obs()
        if not np.all(np.isfinite(obs)):
            self.get_logger().warn("Non-finite obs — skipping",
                                   throttle_duration_sec=2.0)
            return

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action = self.actor(obs_t).squeeze(0).float().cpu().numpy()

        clip = self.get_parameter("action_clip").value
        n_sat = int(np.sum(np.abs(action) >= clip * 0.999))
        if n_sat > ACT_DIM // 2:
            self.get_logger().warn(
                f"{n_sat}/{ACT_DIM} actions saturated — resetting last_action",
                throttle_duration_sec=1.0)
            self.last_action[:] = 0.0
            return

        action = np.clip(action, -clip, clip)
        self.last_action[:] = action

        msg = Float64MultiArray()
        msg.data = (DEFAULT_JOINT_POS + action * ACTION_SCALE).tolist()
        self._pub.publish(msg)


def main():
    rclpy.init()
    node = RLPolicyNode()
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
