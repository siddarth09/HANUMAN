# HANUMAN — Layer 1: Rough Terrain Locomotion

RL locomotion policy for the Unitree G1 on Mars-like unstructured terrain.

Trained with PPO using [mjlab](https://github.com/mujocolab/mjlab) + RSL-RL + MuJoCo Warp, under Mars gravity (−3.72 m/s²).

---

## Setup

```bash
# Clone and install mjlab
git clone https://github.com/mujocolab/mjlab && cd mjlab
pip install -e ".[all]"

# Link HANUMAN's system1 env into the installed mjlab task registry.
# From the HANUMAN repo root, either run the helper (also run automatically
# under pixi):
./scripts/link_hanuman_tasks.sh

# ...or do it by hand (resolves the installed mjlab tasks dir for you):
MJLAB_TASKS=$(python -c 'import os,mjlab.tasks;print(os.path.dirname(mjlab.tasks.__file__))')
ln -sfn "$(pwd)/layer_1/system1" "$MJLAB_TASKS/hanuman"
grep -q "from . import hanuman" "$MJLAB_TASKS/__init__.py" || echo "from . import hanuman" >> "$MJLAB_TASKS/__init__.py"

python -c "import mjlab.tasks.hanuman; print('OK')"
```

Task ids: `Hanuman-Mars-v0` (synthetic rough/slope mix) and `Hanuman-MarsRealistic-v0` (real Jezero DEM) · experiment name: `hanuman_g1_mars` (logs land in `logs/rsl_rl/hanuman_g1_mars/`).

---

## Train

```bash
# 256 envs fits 8GB VRAM (nconmax=300 per env)
python -m mjlab.scripts.train Hanuman-Mars-v0 \
    --env.scene.num-envs 256 \
    --agent.max-iterations 300000
```

The RL config caps `max-iterations` at 150k by default — pass it explicitly for longer runs.

### Resume from a checkpoint

```bash
python -m mjlab.scripts.train Hanuman-Mars-v0 \
    --env.scene.num-envs 256 \
    --agent.resume True \
    --agent.load-run <run-name> \
    --agent.load-checkpoint model_<iter>.pt \
    --agent.max-iterations 400000
```

`load-run` is resolved inside `logs/rsl_rl/hanuman_g1_mars/`. The policy weights, optimizer state, and iteration count are restored, so `max-iterations` must exceed the checkpoint iter.

> **Note on resume vs config changes:** changing reward weights or the terrain mix does *not* change the network (obs=288, act=29), so checkpoints stay loadable. But the command curriculum is keyed on the global step (`iter x 24`), so resuming at iter 270k starts past every stage -> full command speed immediately (the slow-start stages only apply to a fresh run). Terrain levels reset to `max_init_terrain_level=5` on any new run.

---

## Play

```bash
python -m mjlab.scripts.play Hanuman-Mars-v0 \
    --checkpoint_file logs/rsl_rl/hanuman_g1_mars/<run>/model_270000.pt
```

---

## Environment

| | |
|---|---|
| Robot | Unitree G1 (29-DoF) |
| Gravity | −3.72 m/s² (Mars) |
| Sim | MuJoCo Warp, dt=0.005, decimation=4, episode 30 s |
| Parallel envs | 256 (default), `nconmax=300` per env |

### Terrain mix

The bulk is now the **real Jezero HiRISE DEM** (`mars_dem`), which crops difficulty-graded windows from the orbital elevation map; procedural terrains remain only as a clean difficulty scaffold.

| Terrain | Proportion | Purpose |
|---|---|---|
| `mars_dem` (real Jezero DEM windows) | 45% | Real Mars relief, curriculum-graded crater floor -> wall |
| `hf_pyramid_slope` | 20% | Crater walls / hillsides (≤31°), controllable slope curriculum |
| `hf_pyramid_slope_inv` | 15% | Descending into craters |
| `random_rough` | 10% | Regolith roughness (≤10 cm) |
| `perlin_noise` | 10% | Rolling dunes (relief ≤0.8 m) |

### Observation space (actor: 288 dims)

| Component | Dim | Source |
|---|---|---|
| Base linear velocity | 3 | IMU |
| Base angular velocity | 3 | IMU |
| Projected gravity | 3 | IMU orientation |
| Joint positions (rel) | 29 | Encoders |
| Joint velocities | 29 | Encoders |
| Previous action | 29 | Internal |
| Velocity command | 3 | Operator |
| Body heightmap scan | 187 | Raycast grid (1.6 × 1.0 m @ 0.1 m) |
| Foot height | 2 | Per-foot terrain height |

Critic gets 298 dims (actor + privileged foot air-time, contact, contact forces).

### Reward function (14 terms)

| Term | Weight | Purpose |
|---|---|---|
| `track_linear_velocity` | +2.0 | XY velocity tracking (primary task) |
| `track_angular_velocity` | +2.0 | Yaw-rate tracking |
| `upright` | +0.8 | Torso aligned to the **terrain normal** (via `terrain_scan`), not world up |
| `pose` | +1.0 | Speed-dependent posture regulation |
| `air_time` | +0.5 | Proper swing duration (sized so stepping beats a flat shuffle) |
| `foot_clearance` | −0.5 | Clear rocks during swing (velocity-weighted; kept light so it doesn't suppress the swing) |
| `foot_swing_height` | −3.0 | Reach the target swing peak (dominant foot-lift signal) |
| `foot_slip` | −0.1 | No sliding on tilted surfaces |
| `soft_landing` | −1e-5 | Gentle touchdown |
| `body_ang_vel` | −0.05 | Minimize wobble |
| `angular_momentum` | −0.02 | Clean motion |
| `dof_pos_limits` | −1.0 | Respect joint limits |
| `action_rate_l2` | −0.1 | Smooth actions |
| `self_collisions` | −1.0 | No limb-to-limb contact |

### Domain randomization

| Parameter | Range |
|---|---|
| Foot friction | 0.3 – 0.8 (Mars regolith) |
| Base CoM offset | ±2.5 cm XY, ±3 cm Z |
| Encoder bias | ±0.015 rad |
| External pushes | ±0.5 m/s (x,y), every 1–3 s |
| Terrain patch | Randomized each episode |

### Curriculum

- **Command (step-based):** vx ±0.4 -> ±0.6 (8k) -> ±0.8 (18k, held); yaw-rate ramps to ±0.6. Capped at 0.8 m/s — full 1.2 m/s on 31° Mars slopes was infeasible and drove a shuffle gait + terrain-curriculum regression.
- **Terrain (performance-gated):** `terrain_levels_vel` promotes envs that traverse far enough, demotes those that fall.

> **Read training by `track_linear_velocity` (>0.5) + a visual rollout + per-terrain levels — never mean reward.** A rising scalar reward with low track-velocity = the robot has found a stationary "squat" optimum. If that happens with `mars_dem` at 45%, cut it back toward 0.30 and/or restore easy terrain.

---

## File structure

```
layer_1/system1/
├── __init__.py           Task registration with mjlab
├── env_cfg.py            Environment definition (terrain, obs, rewards, curriculum)
├── rl_cfg.py             PPO hyperparameters (experiment_name = hanuman_g1_mars)
└── mars_dem_terrain.py   Real Jezero DEM -> difficulty-graded heightfield sub-terrain
```

---

## Part of [HANUMAN](https://github.com/siddarth09/HANUMAN)

Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain.
