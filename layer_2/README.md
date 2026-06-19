# HANUMAN — Layer 2: State Estimation & Localization

Where-am-I for the G1 on Mars, from scratch. Two ROS 2 (Jazzy) packages:

- **`state_estimation`** — proprioceptive odometry: leg odometry, an error-state EKF, and a GTSAM factor-graph SLAM backend.
- **`terrain_localization`** — absolute "Mars-GPS": match an onboard elevation map (3D lidar / depth) against the orbital HiRISE DEM.

No GPS, no motion capture — only IMU, leg encoders, foot force/torque, a depth camera, and a 3D lidar.

---

## Packages & nodes

### `state_estimation`
| Node | Role | Publishes |
|---|---|---|
| `leg_odom_node` | Body velocity from foot FK + contact (foot Fz > 30 N) | `/leg_odometry` |
| `ekf_node` | Error-state EKF: IMU predict + leg-velocity update + terrain-z update | `/odometry/filtered` |
| `slam_node` | GTSAM iSAM2: IMU preintegration + leg-velocity + terrain priors | `/slam/odometry` |
| `validation_node` | Logs GT vs EKF vs SLAM vs terrain to CSV for analysis | `output/*.csv` |

The EKF and SLAM are **independent parallel estimators** (neither consumes the other's output) — they share inputs (IMU, leg-odom, terrain fix). The EKF is the high-rate pose used by navigation; SLAM is the absolute-pose backend.

### `terrain_localization`
| Node | Role | Publishes |
|---|---|---|
| `terrain_matcher_node` | Accumulate a local elevation map (lidar/depth) -> match against the HiRISE DEM -> absolute pose | `/terrain_match/pose`, `/terrain/{dem,local}_cloud` |

Supporting modules: `dem.py` (DEM ray-cast + cache), `local_map.py` (elevation grid; `add` for depth, `add_cloud` for lidar), `matcher.py` (search + confidence + covariance), `calib.py` (base->camera / base->lidar extrinsics).

---

## Run

```bash
# 1. proprioceptive estimation (leg odom + EKF + GTSAM)
ros2 launch state_estimation state_estimation.launch.py

# 2. terrain-relative localization (+ 3D lidar node + map->odom TF + RViz)
ros2 launch terrain_localization terrain_localization.launch.py rviz:=true
#   lidar:=false  to use only the depth camera (narrower, more ambiguous)
```

Requires the sim (`mars_gazebo`) running so the sensors publish.

### Validate against ground truth

```bash
ros2 run state_estimation validation_node      # drive the robot, Ctrl-C to dump CSVs
ros2 run state_estimation plot_validation      # or analyse output/*.csv
```

---

## Key topics

| Topic | Type | Meaning |
|---|---|---|
| `/leg_odometry` | `nav_msgs/Odometry` | world-frame body velocity (twist) |
| `/odometry/filtered` | `nav_msgs/Odometry` | EKF pose + velocity (the nav pose) |
| `/slam/odometry` | `nav_msgs/Odometry` | GTSAM optimized pose |
| `/terrain_match/pose` | `geometry_msgs/PoseWithCovarianceStamped` | absolute HiRISE fix (+covariance) |
| `/mid360/points` | `sensor_msgs/PointCloud2` | 3D lidar (from `mars_gazebo/lidar3d_node`) |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | manual localization reset (matcher + EKF re-anchor) |

---

## How it works

- **Leg odometry:** forward-kinematic body velocity from the stance foot, gated on foot force (effort is useless for contact in sim). World-frame velocity -> EKF/SLAM.
- **EKF (`ekf_core`):** IMU drives predict; leg-odom velocity is the twist update; the terrain matcher's DEM height is an absolute-z update. Position x/y is otherwise dead-reckoned (low-drift but unbounded). Impact/speed clamps keep it bounded on falls.
- **GTSAM SLAM (`factor_graph`):** IMU `CombinedImuFactor` + leg-velocity prior per keyframe + a unary terrain prior (x,y,yaw,z) when the matcher is confident.
- **Terrain-relative localization:** the matcher dead-reckons a seed (from the EKF pose), accumulates a robot-centric elevation map from the **wide 3D lidar** (or depth), and searches (x,y,yaw) for the best match against the HiRISE DEM. A confidence gate + re-localization hysteresis + honest covariance keep false matches from poisoning the graph. Lidar's wide scan is far less ambiguous than the depth cone.

---

## Notes & known limits

- Terrain matching on gentle, self-similar Jezero relief is the hard part; the lidar path produces sub-meter fixes when the local map is sharp, but accuracy degrades during fast motion. Bounded but ~1–2 m on a short traverse.
- The matcher seeds from `/odometry/filtered`, so **run `state_estimation` before/with `terrain_localization`.**
- `map ≈ odom` at spawn (identity static TF); a TF tree / loop closure is future work.

---

## Part of [HANUMAN](https://github.com/siddarth09/HANUMAN) — Layer 2 of 3.
