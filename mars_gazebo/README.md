# mars_gazebo — HANUMAN G1 on Mars (MuJoCo)

Drives the Unitree G1 in MuJoCo via `mujoco_ros2_control`, on real NASA Jezero
HiRISE terrain, with a TorchScript RL locomotion policy.

> The old `hanuman_mission_launcher` package is gone — everything runs from this
> package's `mujoco.launch.py` plus a couple of standalone helper nodes.

## Build

```bash
cd ~/projects25
colcon build --packages-select mars_gazebo
source install/setup.bash
```

This is a Python-only package (`ament_cmake` installing the node scripts) — no
C++ build, no LibTorch dependency. `torch` is needed only at runtime for the
policy node.

## Layout

- `unitree_g1_mjcf/g1_mars.xml` — G1 MJCF: 29 position actuators, IMU
  (pelvis/torso), `d435` camera, MID360 lidar ring, foot force/torque + touch.
- `unitree_g1_mjcf/mars_nav_scene.xml` — controller scene: 200 m Jezero terrain
  (`mars_nav_200`, real HiRISE orthophoto) + the G1, offset so it spawns on a flat
  spot. Regenerate with `scripts/make_mars_nav_scene.py <terrain_dir> g1_mars.xml
  mars_nav_scene.xml`. The in-sim 360-beam MID360 ring was removed (it cost ~99% of
  every sim step); use `lidar3d_node.py` for a 3D `/mid360/points` cloud instead.
- `unitree_g1/g1_mujoco.urdf` — `ros2_control` config; `mujoco_model` param
  points at `mars_nav_scene.xml`.
- `config/controller_mjcf.yaml` — controllers (joint group position, JSB, IMU,
  foot FT broadcasters).
- `policy/model_425000.pt` — raw rsl_rl checkpoint the policy node loads directly.
  `policy/hanuman_policy.pt` is the TorchScript fallback (used if the checkpoint
  is absent).

## Run the RL policy (step by step)

Open four terminals; `source install/setup.bash` in each.

**1. Simulator + controllers** (MuJoCo, robot, controllers spawn automatically):
```bash
ros2 launch mars_gazebo mujoco.launch.py
```
Gives you `/joint_states`, `/imu_broadcaster/imu`, `/ground_truth/odom`, and the
`g1_position_controller` listening on `/g1_position_controller/commands`.

**2. Height scanner** (the policy's 187-pt terrain obs — required):
```bash
ros2 run mars_gazebo height_scanner_node.py
# check: ros2 topic echo /height_scan --once    (flat ground ≈ 0.74 everywhere)
```

**3. Policy node** (50 Hz; waits 5 s warmup after first `/joint_states`):

Loads the raw `model_425000.pt` directly (CPU or GPU):
```bash
# CPU (works with system python3):
ros2 run mars_gazebo rl_policy_node.py --ros-args -p use_sim_time:=true

# sm_120 GPU — launch with a CUDA-capable torch venv (ROS already sourced):
/home/sid/mujoco_env/bin/python \
    install/mars_gazebo/lib/mars_gazebo/rl_policy_node.py \
    --ros-args -p use_sim_time:=true -p device:=cuda
```
Prints "Policy ready — 50 Hz …". CUDA ≈ CPU speed here (~73 vs ~82 µs);
inference is never the bottleneck, so CPU is the safe default.

**4. Send a velocity command** (vx, vy, yaw-rate):
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.5}, angular: {z: 0.0}}' -r 10
```

### Use a different / freshly trained policy
Point the node at any rsl_rl checkpoint or TorchScript `.pt`:
```bash
ros2 run mars_gazebo rl_policy_node.py \
    --ros-args -p use_sim_time:=true -p model_path:=/abs/path/policy.pt
```
To produce a TorchScript `.pt` from a checkpoint (e.g. for other LibTorch
consumers):
```bash
python3 scripts/export_policy_torchscript.py \
    --ckpt policy/model_425000.pt --out policy/hanuman_policy.pt
```

## Optional — 3D LiDAR for elevation mapping

Not needed for the policy. Publishes a `sensor_msgs/PointCloud2` on
`/mid360/points` (simulated MID360 dome, terrain + obstacles):
```bash
ros2 run mars_gazebo lidar3d_node.py
# RViz2: PointCloud2 on /mid360/points, Fixed Frame: mid360_frame
```

## Key topics

| Topic | Type | From → To |
|---|---|---|
| `/joint_states` | JointState | sim → policy |
| `/imu_broadcaster/imu` | Imu | sim → policy |
| `/ground_truth/odom` | Odometry | sim → policy / scanners |
| `/height_scan` | LaserScan (187) | height_scanner → policy (NOT for RViz — packed heights, not a real scan) |
| `/height_scan/cloud` | PointCloud2 | height_scanner → RViz (the 187 grid hits; Fixed Frame `pelvis`) |
| `/cmd_vel` | Twist | you → policy |
| `/g1_position_controller/commands` | Float64MultiArray | policy → controller |
| `/mid360/points` | PointCloud2 | lidar3d → elevation mapping |
| `/left_foot_ft_broadcaster/wrench`, `/right_…` | WrenchStamped | sim → you |

## Viewing sensors in RViz2
The scanner/lidar topics are **Best Effort** — in each RViz display set
*Reliability Policy → Best Effort* or you'll see nothing. Set *Fixed Frame* to a
frame that's in the TF tree (e.g. `pelvis`).
- Height-scan grid: PointCloud2 on **`/height_scan/cloud`** (Fixed Frame `pelvis`).
  Do NOT use `/height_scan` (LaserScan) — it's packed heights for the policy, not
  real scan geometry.
- 3D lidar: PointCloud2 on **`/mid360/points`** (Fixed Frame `mid360_frame`).

## Known gaps

- `foot_height` (obs[286:288]) is left at the nominal standing default — the
  per-foot ring scan isn't wired yet.
- The policy's height scan is yaw-aligned at the live pose; it matches training
  closely while the base is roughly level.
