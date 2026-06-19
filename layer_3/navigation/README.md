# HANUMAN — Layer 3: Navigation

Autonomous navigation for the G1 on the HiRISE Jezero map. A ROS 2 (Jazzy) package:
global cost map -> A* global planner -> MPPI local planner -> `/cmd_vel`, plus a
custom operator console.

---

## Pipeline

```
HiRISE DEM ── geometric cost (slope + roughness) ── A* ── global path (map frame)
                                                              |
local pose (EKF) ───────────────────────────────────────────── MPPI ── /cmd_vel ── RL policy
```

- **Global cost map** — slope + roughness from the orbital DEM; steep / no-data cells are lethal.
- **A\* global planner** — 8-connected, cost-weighted; replans on each operator goal.
- **MPPI local planner** — samples body-velocity rollouts, scores them against a carrot on the
  global path plus terrain cost, and publishes `/cmd_vel`. Shaped for human-like motion:
  capped speed (0.4 m/s), no lateral crab, and turn-in-place when the heading error is large.
- **Operator console** — a custom-painted Qt window: HiRISE basemap, cost overlay, live path,
  robot estimates, and localization-health readouts, with click-to-goal and click-to-re-anchor.

---

## Nodes

| Executable | Role | Topics |
|---|---|---|
| `global_planner_node` | DEM -> cost map -> A* path | pub `/nav/global_costmap`, `/nav/global_path`; sub `/goal_pose`, `/odometry/filtered` |
| `mppi_node` | Local planner -> velocity command | pub `/cmd_vel`, `/nav/mppi_path`; sub `/nav/global_costmap`, `/nav/global_path`, `/odometry/filtered` |
| `dashboard_qt` | PyQt5 operator console | pub `/goal_pose`, `/initialpose`; sub all of the above + pose estimates |
| `dashboard_mock` | Synthetic publishers to exercise the console without the full stack | mock topics |

Libraries: `costmap.py` (DEM -> cost), `planner.py` (A*), `mppi.py` (vectorized MPPI controller).

---

## Run

```bash
# planner + MPPI; add the Qt console and/or RViz
ros2 launch navigation navigation.launch.py dashboard:=true
#   mppi:=false   plan-only (no /cmd_vel)
#   rviz:=true    open the shared RViz config

# standalone console + mock data (no sim needed)
ros2 run navigation dashboard_qt
ros2 run navigation dashboard_mock
```

Full autonomous loop:

```bash
ros2 launch mars_gazebo mujoco.launch.py                     # sim + controllers
ros2 launch state_estimation state_estimation.launch.py      # /odometry/filtered
ros2 launch terrain_localization terrain_localization.launch.py
ros2 launch navigation navigation.launch.py dashboard:=true  # planner + MPPI + console
ros2 launch mars_gazebo policy.launch.py                     # walks /cmd_vel
```

Set a goal with the console's "2D Nav Goal" tool (or RViz "2D Goal Pose"); use "2D Pose
Estimate" / "Re-anchor" to correct localization when the absolute estimate jumps.

---

## Topics

| Topic | Type | Meaning |
|---|---|---|
| `/nav/global_costmap` | `nav_msgs/OccupancyGrid` (latched) | DEM traversability cost |
| `/nav/global_path` | `nav_msgs/Path` | A* global route |
| `/nav/mppi_path` | `nav_msgs/Path` | MPPI rollout |
| `/cmd_vel` | `geometry_msgs/Twist` | velocity command to the RL policy |
| `/goal_pose` | `geometry_msgs/PoseStamped` | operator destination |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | operator localization reset |

The planner and console use the EKF pose (`/odometry/filtered`) as the robot pose, since it is
currently more reliable than the terrain-corrected SLAM pose on short traverses.

---

## Cost model

Geometric v1: `cost = w_slope * (slope / slope_max) + w_rough * (roughness / rough_max)`, with
steep (>= `slope_max`) and no-data cells marked lethal. Tunable via node parameters
(`slope_max_deg`, `rough_max`, `w_slope`, `w_rough`, `cost_penalty`). A learned, self-supervised
foothold cost is a planned extension.

---

## Known limits / next steps

- MPPI scores against the global cost map; fusing the live local elevation map and the GTSAM
  pose into MPPI is the next increment.
- The footprint safety circle is drawn but not yet enforced as costmap inflation.
- Navigation quality is bounded by locomotion — the robot follows commands only as well as it walks.

---

## Part of [HANUMAN](https://github.com/siddarth09/HANUMAN) — Layer 3 of 3.
