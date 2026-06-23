# HANUMAN

### Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain

A full autonomy stack for deploying a bipedal humanoid (Unitree G1) on unstructured terrain — from Earth wilderness to the Martian surface. Learned locomotion, GPS-denied state estimation and localization, and DEM-based navigation, all exercised in a MuJoCo simulation built on **real NASA Jezero Crater HiRISE terrain**.

https://github.com/user-attachments/assets/9b7e004e-dade-4a36-8365-b67483db4bc4
---

## The Problem

Mars has no flat floors, no GPS, and a 4–24 minute communication delay to Earth. A humanoid operating there must walk over rocks it has never seen, localize itself without external infrastructure, and make every time-critical decision autonomously.

HANUMAN solves this as three cooperating layers, plus the simulation and terrain tooling that make it reproducible:

```
Layer 3: Navigation         ~1 Hz    DEM cost map -> A* global plan -> MPPI local plan -> operator console
Layer 2: State Est. + Loc. ~10 Hz    Leg odometry + error-state EKF + GTSAM SLAM; terrain-relative "Mars-GPS"
Layer 1: RL Locomotion      50 Hz    PPO policy: proprioception + heightmap -> joint targets, under Mars gravity
```

Each layer talks to the next through a minimal interface — waypoints, velocity commands, heightmaps — and can be built, tested, and swapped independently.

---

## What's Inside

### Layer 1 — Terrain-Adaptive Locomotion (`layer_1/`)
A PPO policy trained with [mjlab](https://github.com/mujocolab/mjlab) + RSL-RL on MuJoCo Warp, under **Mars gravity (−3.72 m/s²)**. The actor observes proprioception + a 187-point body heightmap + per-foot terrain scan (288 dims) and outputs 29 joint-position targets at 50 Hz. It learns safe foothold selection implicitly by training on a mix of **real Jezero HiRISE DEM crops** and procedural slopes/rough/dunes, with domain randomization over friction, CoM, encoder bias, and external pushes. Two registered tasks: `Hanuman-Mars-v0` (synthetic mix) and `Hanuman-MarsRealistic-v0` (real Jezero DEM).

### Layer 2 — GPS-Denied State Estimation & Localization (`layer_2/`)
- **`state_estimation`** — leg odometry (forward kinematics on contact feet), an **error-state EKF** (IMU predict + leg-velocity update, the deployment replacement for the policy's privileged base velocity), and a **SLAM** backend (IMU preintegration + leg velocity + terrain priors). Includes a validation node that logs EKF/SLAM vs ground truth to CSV.
- **`terrain_localization`** — the "Mars-GPS": matches a robot-centric elevation map (from the MID360 lidar or D435 depth) against the orbital HiRISE DEM to produce a **drift-free absolute pose fix**, consumed by GTSAM as a unary prior. EKF-seeded search + confidence gating + re-localization hysteresis handle Jezero's self-similar relief.

### Layer 3 — Autonomous Navigation (`layer_3/`)
A **`navigation`** package: a DEM cost map (slope + roughness) feeds an 8-connected **A\*** global planner; an **MPPI** local planner samples body-velocity rollouts, scores them against the global path and terrain, and emits speed-capped `/cmd_vel` shaped for human-like motion (no crab-walking, turn-in-place for large heading errors). A **Mission operator console** overlays the HiRISE basemap, cost map, live plan, pose estimates, localization health, and a first-person D435 viewport, with click-to-goal and click-to-re-anchor.

### Simulation — `mars_gazebo/`
The deployment-side MuJoCo simulation via [`mujoco_ros2_control`](https://github.com/UM-ARM-Lab/mujoco_ros2_control), running the same ROS 2 nodes that would run on hardware. The G1 MJCF carries IMU, foot force/torque, a D435 depth camera, and a MID360 lidar; the scene is a 200 m Jezero tile draped with the real HiRISE orthophoto. Ships the trained policy node, a height-scanner that reproduces the training heightmap, a lidar ray-caster, and TorchScript/ONNX export tools.

### Terrain tooling — `mars_terrain_exporter/`
A CLI that turns **real NASA Mars 2020 TRN HiRISE DTMs** (1 m/pixel, Jezero region) into MuJoCo `hfield` models — heightfield PNG + includable MJCF + standalone viewer scene + metadata. This is how all the terrain in the project is generated.

---

## Repository Layout

```
HANUMAN/
├── layer_1/                 RL locomotion (mjlab task: env, rewards, curriculum, PPO cfg)
│   └── system1/             Registered tasks: Hanuman-Mars-v0, Hanuman-MarsRealistic-v0
├── layer_2/
│   ├── state_estimation/    ROS 2 pkg: leg odometry + error-state EKF + GTSAM SLAM
│   └── terrain_localization/ROS 2 pkg: terrain-relative localization (HiRISE DEM matching)
├── layer_3/
│   └── navigation/          ROS 2 pkg: DEM cost map + A* + MPPI + PyQt5 console
├── mars_gazebo/             ROS 2 pkg: MuJoCo deployment sim, controllers, RL policy nodes
├── mars_terrain_exporter/   ROS 2 pkg / CLI: HiRISE DTM -> MuJoCo hfield terrain
├── scripts/                 link_hanuman_tasks.sh (registers Layer 1 tasks into mjlab)
├── pixi.toml                Reproducible envs: `default` (RL) + `ros` (ROS 2 Jazzy)
└── LICENSE                  MIT
```

The project spans two toolchains: a **CUDA RL stack** (Python 3.12 + torch + mjlab) for Layer 1 training, and a **ROS 2 Jazzy stack** for Layers 2–3 and the simulation. `pixi` keeps them in separate environments so they don't collide.

---

## Installation

### Option A — with pixi (recommended)

[pixi](https://pixi.sh) reproduces both toolchains from `pixi.toml` + `pixi.lock`.

```bash
git clone https://github.com/siddarth09/HANUMAN && cd HANUMAN

# RL / simulation training stack (default environment).
# Installs Python 3.12, torch (cu130), mujoco, mujoco-warp, rsl-rl, mjlab, etc.,
# then links the HANUMAN tasks into mjlab on activation.
pixi install

# ROS 2 Jazzy environment (separate, conda-based via RoboStack) and build the
# mujoco_ros2_control workspace. Builds from source — see the note in pixi.toml.
pixi run -e ros build-ros
```

> **Note:** `mujoco_ros2_control` is built from source (RoboStack's binary is MuJoCo 3.5 / no camera support, incompatible with this project's 3.8). The Gazebo path is intentionally left out of the pixi `ros` env; if you want it, run that path outside pixi.

### Option B — without pixi (manual)

**Layer 1 — RL stack** (Python 3.12, NVIDIA GPU):

```bash
git clone https://github.com/siddarth09/HANUMAN && cd HANUMAN
python3.12 -m venv .venv && source .venv/bin/activate

# torch must come from the cu130 wheel index (Blackwell / sm_120 support)
pip install torch==2.11.0 --extra-index-url https://download.pytorch.org/whl/cu130
pip install "mujoco>=3.8,<3.9" "mujoco-warp>=3.8,<3.9" "warp-lang>=1.13" \
            "rsl-rl-lib>=5.2" wandb tyro
pip install "git+https://github.com/mujocolab/mjlab.git@5a433e83"

# Register the HANUMAN tasks into the installed mjlab (idempotent helper)
./scripts/link_hanuman_tasks.sh
python -c "import mjlab.tasks.hanuman; print('tasks OK')"
```

**Layers 2–3 + simulation — ROS 2 stack** (Ubuntu 24.04 + [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation.html)):

```bash
# ROS 2 Jazzy + control stack
sudo apt install ros-jazzy-ros-base ros-jazzy-ros2-control \
                 ros-jazzy-ros2-controllers ros-jazzy-robot-state-publisher
# Python deps used by the nodes
pip install gtsam pyqt5 numpy scipy
# mujoco_ros2_control must be built from source against MuJoCo 3.8 (see pixi.toml note)

# Build the workspace (run from a colcon workspace whose src/ contains HANUMAN)
colcon build --symlink-install
source install/setup.bash
```

---

## Running

### Train / play the locomotion policy (Layer 1)

```bash
# with pixi
pixi run train          # python -m mjlab.scripts.train Hanuman-MarsRealistic-v0 ...
pixi run play           # python -m mjlab.scripts.play Hanuman-MarsRealistic-v0 --num_envs 1

# without pixi (256 envs fits 8 GB VRAM)
python -m mjlab.scripts.train Hanuman-Mars-v0 --env.scene.num-envs 256 --agent.max-iterations 300000
python -m mjlab.scripts.play  Hanuman-Mars-v0 --checkpoint_file logs/rsl_rl/hanuman_g1_mars/<run>/model_<iter>.pt
```

See [`layer_1/README.md`](layer_1/README.md) for the observation/reward/curriculum details and checkpoint-resume flags.

### Run the full autonomy stack in simulation (Layers 1–3)

Each line is a separate terminal (all in the `ros` environment — prefix with `pixi run -e ros` if using pixi, or `source install/setup.bash` otherwise):

```bash
# 1. MuJoCo simulator + controllers + height scanner
ros2 launch mars_gazebo mujoco.launch.py

# 2. State estimation (leg odometry + EKF + GTSAM SLAM)
ros2 launch state_estimation state_estimation.launch.py

# 3. Terrain-relative localization (lidar-based "Mars-GPS")
ros2 launch terrain_localization terrain_localization.launch.py lidar:=true rviz:=true

# 4. Navigation (A* global + MPPI local + PyQt5 operator console)
ros2 launch navigation navigation.launch.py dashboard:=true

# 5. RL policy (start after the simulator is fully up)
ros2 launch mars_gazebo policy.launch.py
```

Then set a goal in the console (or RViz "2D Nav Goal"); the planner publishes `/cmd_vel` and the policy walks the robot there.

### Generate Mars terrain

```bash
mars_terrain_exporter list                                   # available HiRISE sites
mars_terrain_exporter site jezero_c --size 150 --output-dir ./models
python3 -m mujoco.viewer --mjcf=models/jezero_c/scene.xml    # preview
```

Per-package details live in each directory's `README.md`.

---

## Platform

| | |
|---|---|
| Robot | Unitree G1 (29 body DoF; 43 actuators with dexterous hands) |
| Gravity | −3.72 m/s² (Mars) |
| RL sim | MuJoCo Warp, dt 0.005, decimation 4, 256 parallel envs |
| Deployment sim | MuJoCo 3.8 via `mujoco_ros2_control` (ROS 2 Jazzy) |
| Terrain | Real NASA Jezero Crater HiRISE DTM (1 m/px) + procedural scaffold |
| Sensors | IMU, foot force/torque, D435 depth, MID360 lidar |

---

## Starting Point

HANUMAN builds on [Bheema](https://github.com/siddarth09/Bheema) — a bipedal locomotion controller for the Unitree G1 with both a classical MPC pipeline and a trained RL policy. HANUMAN extends the flat-terrain baseline to rough Mars terrain and adds perception, GPS-denied localization, and autonomous navigation.

## References

- Miki et al., *Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild*, Science Robotics 2022
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, CoRL 2022
- Dellaert & Kaess, *Factor Graphs for Robot Perception*, Foundations and Trends in Robotics 2017
- Williams et al., *Model Predictive Path Integral Control (MPPI)*, ICRA 2017

## License

[MIT](LICENSE) © 2026 Siddarth Dayasagar

Terrain derived from NASA/JPL-Caltech/University of Arizona HiRISE products (Mars 2020 TRN). Built on [mjlab](https://github.com/mujocolab/mjlab), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), [MuJoCo](https://mujoco.org), [GTSAM](https://gtsam.org), and [ROS 2](https://ros.org).
