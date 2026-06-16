#!/usr/bin/env python3
"""
HANUMAN standalone MuJoCo policy runner — no ROS2, no ONNX.
Loads .pt checkpoint directly via PyTorch and runs in MuJoCo viewer.

Controls (keyboard in MuJoCo window):
  ↑ / ↓      — forward / backward
  ← / →      — yaw left / right
  Shift+← /→ — strafe left / right
  R          — reset simulation
  Esc / 0    — stop (zero velocity command)
"""

import sys
import time
import argparse
import threading
import numpy as np
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer

# ─── Paths ────────────────────────────────────────────────────────────────────
import os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR    = os.path.dirname(_SCRIPT_DIR)

DEFAULT_XML = os.path.join(_PKG_DIR, "unitree_g1_mjcf", "mars_scene.xml")
DEFAULT_PT  = os.path.join(os.path.expanduser("~"),
    "logs/hanumanv1/hanuman_g1_rough/2026-05-18_21-54-17/model_425000.pt")

# ─── Policy architecture ──────────────────────────────────────────────────────

class EmpiricalNormalization(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_std",  torch.ones(1, dim))

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


# ─── Joint configuration ──────────────────────────────────────────────────────

NUM_JOINTS = 29
OBS_DIM    = 288
ACT_DIM    = 29
POLICY_HZ  = 50
ACTION_CLIP = 1.0
HEIGHT_SCAN_DEFAULT = 0.150467

DEFAULT_JOINT_POS = np.array([
    -0.312, 0.0,  0.0,   0.669, -0.363, 0.0,   # left leg
    -0.312, 0.0,  0.0,   0.669, -0.363, 0.0,   # right leg
     0.0,   0.0,  0.0,                           # waist
     0.2,   0.2,  0.0,   0.6,                    # left arm
     0.0,   0.0,  0.0,                           # left wrist
     0.2,  -0.2,  0.0,   0.6,                    # right arm
     0.0,   0.0,  0.0,                           # right wrist
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.4386, 0.4386,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
], dtype=np.float32)

# Policy joint i → MuJoCo qpos index (skip freejoint and hand joints)
# freejoint: qpos[0:7], then leg/waist/left-arm/left-wrist at qpos[7:29],
# then 7 left-hand joints at qpos[29:36], then right-arm/wrist at qpos[36:43]
QPOS_IDX = np.array([
     7,  8,  9, 10, 11, 12,  # left leg
    13, 14, 15, 16, 17, 18,  # right leg
    19, 20, 21,              # waist
    22, 23, 24, 25, 26, 27, 28,  # left arm + wrist
    36, 37, 38, 39, 40, 41, 42,  # right arm + wrist
], dtype=int)

# Policy joint i → MuJoCo qvel index (freejoint uses qvel[0:6])
QVEL_IDX = np.array([
     6,  7,  8,  9, 10, 11,  # left leg
    12, 13, 14, 15, 16, 17,  # right leg
    18, 19, 20,              # waist
    21, 22, 23, 24, 25, 26, 27,  # left arm + wrist
    35, 36, 37, 38, 39, 40, 41,  # right arm + wrist (skip 7 left-hand dofs)
], dtype=int)

# Policy output i → MuJoCo actuator index (exact 1:1 match for first 29)
CTRL_IDX = np.arange(29, dtype=int)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate world-frame vec into body frame given body-to-world quaternion."""
    w, x, y, z = quat_wxyz.astype(float)
    # Using the sandwich product q* ⊗ v ⊗ q (conjugate = (w,-x,-y,-z))
    cx, cy, cz = -x, -y, -z
    tx = 2.0 * (cy * vec[2] - cz * vec[1])
    ty = 2.0 * (cz * vec[0] - cx * vec[2])
    tz = 2.0 * (cx * vec[1] - cy * vec[0])
    return np.array([
        vec[0] + w * tx + (cy * tz - cz * ty),
        vec[1] + w * ty + (cz * tx - cx * tz),
        vec[2] + w * tz + (cx * ty - cy * tx),
    ], dtype=np.float32)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project world gravity [0,0,-1] into body frame."""
    w, x, y, z = quat_wxyz
    return np.array([
        -2.0 * (x*z - w*y),
        -2.0 * (y*z + w*x),
        -(1.0 - 2.0 * (x*x + y*y)),
    ], dtype=np.float32)


def build_obs(data: mujoco.MjData, last_action: np.ndarray,
              command: np.ndarray) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    quat = data.qpos[3:7]                 # (w, x, y, z) from freejoint

    # [0:3] base linear velocity in body frame
    # freejoint qvel[0:3] is WORLD-frame linear velocity
    obs[0:3] = quat_rotate_inverse(quat, data.qvel[0:3])

    # [3:6] base angular velocity (freejoint qvel[3:6] is body-frame in MuJoCo)
    obs[3:6] = data.qvel[3:6]

    # [6:9] projected gravity
    obs[6:9] = projected_gravity(quat)

    # [9:38] joint positions relative to default
    obs[9:38] = data.qpos[QPOS_IDX] - DEFAULT_JOINT_POS

    # [38:67] joint velocities
    obs[38:67] = data.qvel[QVEL_IDX]

    # [67:96] last action
    obs[67:96] = last_action

    # [96:99] velocity command [vx, vy, yaw_rate]
    obs[96:99] = command

    # [99:286] height scan — default flat terrain
    obs[99:286] = HEIGHT_SCAN_DEFAULT

    # [286:288] foot heights — nominal standing height
    obs[286] = 0.047589
    obs[287] = 0.047088

    return obs


# ─── Policy loader ────────────────────────────────────────────────────────────

def load_actor(pt_path: str) -> Actor:
    print(f"Loading policy: {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    sd = ckpt["actor_state_dict"]
    keep = {k: v for k, v in sd.items()
            if k.startswith("mlp") or k in ("obs_normalizer._mean", "obs_normalizer._std")}

    actor = Actor()
    actor.load_state_dict(keep, strict=False)
    actor.eval()
    print(f"Actor loaded — obs={OBS_DIM}  act={ACT_DIM}")
    return actor


# ─── Simulation state ─────────────────────────────────────────────────────────

class SimState:
    def __init__(self):
        self.command   = np.zeros(3, dtype=np.float32)  # [vx, vy, yaw]
        self.do_reset  = False
        self.lock      = threading.Lock()

    def set_cmd(self, vx=0.0, vy=0.0, yaw=0.0):
        with self.lock:
            self.command[:] = [vx, vy, yaw]

    def get_cmd(self):
        with self.lock:
            return self.command.copy()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=DEFAULT_XML)
    parser.add_argument("--pt",  default=DEFAULT_PT)
    parser.add_argument("--cmd_vx",  type=float, default=0.0,
                        help="Forward velocity command (m/s)")
    parser.add_argument("--cmd_vy",  type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0,
                        help="Yaw rate (rad/s)")
    args = parser.parse_args()

    actor = load_actor(args.pt)
    state = SimState()
    state.set_cmd(args.cmd_vx, args.cmd_vy, args.cmd_yaw)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    # Initialise at default standing pose
    data.qpos[QPOS_IDX] = DEFAULT_JOINT_POS
    data.ctrl[CTRL_IDX] = DEFAULT_JOINT_POS
    mujoco.mj_forward(model, data)

    last_action  = np.zeros(ACT_DIM, dtype=np.float32)
    policy_dt    = 1.0 / POLICY_HZ
    sim_dt       = model.opt.timestep
    steps_per_policy = max(1, int(round(policy_dt / sim_dt)))
    step_count   = 0
    warmup_steps = int(3.0 / sim_dt)   # 3 s warmup before policy runs

    print(f"sim_dt={sim_dt*1000:.1f} ms  policy every {steps_per_policy} steps")
    print(f"Warmup: {warmup_steps} steps ({3.0:.0f} s)")
    print()
    # GLFW arrow key codes
    KEY_UP    = 265
    KEY_DOWN  = 264
    KEY_LEFT  = 263
    KEY_RIGHT = 262

    print("Controls: ↑/↓=forward/back  ←/→=yaw  Shift+←/→=strafe  0/Esc=stop  R=reset")

    def key_callback(keycode):
        STEP = 0.2
        YAW  = 0.3
        cmd  = state.get_cmd()
        if   keycode == KEY_UP:    cmd[0] += STEP
        elif keycode == KEY_DOWN:  cmd[0] -= STEP
        elif keycode == KEY_LEFT:  cmd[2] += YAW    # yaw left
        elif keycode == KEY_RIGHT: cmd[2] -= YAW    # yaw right
        elif keycode == ord('0') or keycode == 256:  # 256 = Esc
            cmd[:] = 0.0
        elif keycode == ord('R'):
            with state.lock:
                state.do_reset = True
            return
        # Shift+arrow for strafe (glfw shift modifier adds 256 offset in some builds)
        elif keycode == KEY_LEFT  + 256: cmd[1] += STEP
        elif keycode == KEY_RIGHT + 256: cmd[1] -= STEP
        else:
            return  # ignore other keys
        cmd = np.clip(cmd, [-1.5, -1.0, -1.0], [1.5, 1.0, 1.0])
        state.set_cmd(*cmd)
        print(f"cmd_vel: vx={cmd[0]:.2f}  vy={cmd[1]:.2f}  yaw={cmd[2]:.2f}")

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        viewer.cam.distance  = 3.0
        viewer.cam.elevation = -20

        t_next = time.perf_counter()
        while viewer.is_running():
            t0 = time.perf_counter()

            # Reset if requested
            with state.lock:
                if state.do_reset:
                    mujoco.mj_resetData(model, data)
                    data.qpos[QPOS_IDX] = DEFAULT_JOINT_POS
                    data.ctrl[CTRL_IDX] = DEFAULT_JOINT_POS
                    mujoco.mj_forward(model, data)
                    last_action[:] = 0.0
                    step_count = 0
                    state.do_reset = False
                    print("Reset.")

            # Policy step
            if step_count % steps_per_policy == 0:
                if step_count >= warmup_steps:
                    cmd = state.get_cmd()
                    obs = build_obs(data, last_action, cmd)

                    with torch.no_grad():
                        obs_t  = torch.from_numpy(obs).unsqueeze(0)
                        act_t  = actor(obs_t)
                        action = act_t.squeeze(0).numpy()

                    # Saturation check
                    n_sat = int(np.sum(np.abs(action) >= ACTION_CLIP * 0.999))
                    if n_sat > ACT_DIM // 2:
                        print(f"[WARN] {n_sat}/{ACT_DIM} actions saturated — "
                              "reset last_action")
                        last_action[:] = 0.0
                    else:
                        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
                        last_action[:] = action
                        targets = DEFAULT_JOINT_POS + action * ACTION_SCALE
                        data.ctrl[CTRL_IDX] = targets
                else:
                    # Warmup: hold default pose
                    data.ctrl[CTRL_IDX] = DEFAULT_JOINT_POS
                    remaining = (warmup_steps - step_count) * sim_dt
                    if step_count % (steps_per_policy * 50) == 0:
                        print(f"Warmup: {remaining:.1f} s remaining ...")

            mujoco.mj_step(model, data)
            step_count += 1
            viewer.sync()

            # Real-time pacing
            elapsed = time.perf_counter() - t0
            sleep_t = sim_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
