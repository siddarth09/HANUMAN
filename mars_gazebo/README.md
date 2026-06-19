# mars_gazebo — HANUMAN G1 on Mars (MuJoCo deployment sim)

Runs the Unitree G1 in MuJoCo through `mujoco_ros2_control`, on real NASA Jezero
HiRISE terrain, driven by the RL locomotion policy. This is the deployment-side
simulation that exercises the full ROS 2 stack (state estimation, localization,
navigation) against the same nodes that would run on hardware.

## Build

```bash
cd ~/projects25
colcon build --packages-select mars_gazebo
source install/setup.bash
```

Python-only package (node scripts installed via `ament_cmake`); no C++ build.
`torch` is needed only at runtime for the policy node. Requires the NASA-JSC
`mujoco_ros2_control` fork (MuJoCo 3.8, camera support) sourced in the overlay.

## Layout

- `unitree_g1_mjcf/g1_mars.xml` — G1 MJCF: 43 `<position>` actuators (29 body +
  14 dexterous hand), per-group armature, capsule feet (7 capsules per foot,
  matched to MuJoCo Menagerie), pelvis/torso IMU sites, `d435` depth camera,
  `mid360_lidar` site, foot force/torque + touch sensors, downward range site.
- `unitree_g1_mjcf/mars_nav_scene.xml` — deployment scene: the 200 m Jezero tile
  (`mars_nav_200`, real HiRISE orthophoto) + the G1. The terrain body is offset
  to `pos="0 72 -3.577"` so the robot spawns on a flat patch in the tile interior,
  surrounded by terrain on all sides. Regenerate with
  `scripts/make_mars_nav_scene.py`. The in-sim 360-beam MID360 ring was removed
  (it cost ~99% of every step); use `lidar3d_node.py` for a 3D cloud instead.
- `unitree_g1/g1_mujoco.urdf` — `ros2_control` config; `mujoco_model` points at
  `mars_nav_scene.xml`.
- `config/controller_mjcf.yaml` — controllers: `joint_state_broadcaster`,
  `g1_position_controller` (JointGroupPositionController), `imu_sensor_broadcaster`,
  left/right foot `force_torque_sensor_broadcaster`.
- `policy/` — RL checkpoints: `model_270000.pt` (the node default), plus
  `model_220000.pt`, `model_425000.pt`, a TorchScript `hanuman_policy.pt`, and an
  ONNX export.

## Run the RL policy

`source install/setup.bash` in each terminal.

1. Simulator + controllers (MuJoCo, robot, controllers spawn automatically):
   ```bash
   ros2 launch mars_gazebo mujoco.launch.py
   ```
   Provides `/joint_states`, `/imu_broadcaster/imu`, `/ground_truth/odom`, the foot
   FT broadcasters, and `g1_position_controller` on `/g1_position_controller/commands`.

2. Policy (height scanner + policy node; scanner comes up first, policy 3 s later):
   ```bash
   ros2 launch mars_gazebo policy.launch.py
   ```
   Loads `policy/model_270000.pt` and runs at 50 Hz after a 5 s warmup. Override
   the checkpoint or device:
   ```bash
   ros2 run mars_gazebo rl_policy_node.py --ros-args -p use_sim_time:=true \
       -p model_path:=/abs/path/policy.pt -p device:=cuda
   ```

3. Command a velocity (vx, vy, yaw-rate):
   ```bash
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}' -r 10
   ```
   In the full stack, navigation publishes `/cmd_vel` instead.

## 3D lidar (for elevation mapping / localization)

```bash
ros2 run mars_gazebo lidar3d_node.py
# RViz2: PointCloud2 on /mid360/points, Fixed Frame mid360_frame
```
Simulated MID360 dome (terrain + obstacles only); consumed by Layer 2's terrain matcher.

## Key topics

| Topic | Type | From -> To |
|---|---|---|
| `/joint_states` | JointState | sim -> policy / leg odom |
| `/imu_broadcaster/imu` | Imu | sim -> policy / EKF |
| `/ground_truth/odom` | Odometry | sim -> scanners / validation (never fused) |
| `/left_foot_ft_broadcaster/wrench`, `/right_...` | WrenchStamped | sim -> leg odom contact |
| `/height_scan` | LaserScan (187) | height_scanner -> policy (packed heights, not real scan) |
| `/height_scan/cloud` | PointCloud2 | height_scanner -> RViz (Fixed Frame `pelvis`) |
| `/cmd_vel` | Twist | operator / navigation -> policy |
| `/g1_position_controller/commands` | Float64MultiArray | policy -> controller |
| `/mid360/points` | PointCloud2 | lidar3d -> terrain localization |

## Viewing sensors in RViz2

Scanner/lidar topics are Best Effort — set each display's Reliability Policy to
Best Effort, and Fixed Frame to a frame in the TF tree (e.g. `pelvis`).
- Height-scan grid: PointCloud2 on `/height_scan/cloud` (Fixed Frame `pelvis`).
  Do not use `/height_scan` (LaserScan) — it is packed heights for the policy.
- 3D lidar: PointCloud2 on `/mid360/points` (Fixed Frame `mid360_frame`).

## Scripts

| Script | Purpose |
|---|---|
| `rl_policy_node.py` | Loads an rsl_rl checkpoint, runs the 288-obs / 29-act policy at 50 Hz |
| `height_scanner_node.py` | Ray-casts the scene terrain into the policy's 187-pt height scan |
| `lidar3d_node.py` | Simulated MID360 3D point cloud |
| `make_mars_nav_scene.py` | Regenerates `mars_nav_scene.xml` from a terrain dir + the G1 MJCF |
| `teleop.py` | Keyboard `/cmd_vel` teleop |
| `export_policy_torchscript.py` / `export_policy_onnx.py` | Export a checkpoint for other consumers |

## Notes

- `make_mars_nav_scene.py` generates `mars_nav_scene.xml` — re-running it overwrites
  the hand-tuned terrain offset; the interior flat-patch spawn must be re-applied
  (or baked into the generator).
- `foot_height` (obs[286:288]) uses the nominal standing default; the per-foot ring
  scan is not wired into the policy obs yet.
- The Gazebo path (`gazebo.launch.py`, `gz_bridge.yaml`) is kept but not the primary
  flow; run it outside pixi (its `ros-gz` deps conflict with the ROS env's mutex).

## Part of [HANUMAN](https://github.com/siddarth09/HANUMAN)

Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain.
