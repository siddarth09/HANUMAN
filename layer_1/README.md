# HANUMAN — Layer 1: Rough Terrain Locomotion

RL locomotion policy for the Unitree G1 on Mars-like unstructured terrain.

Trained with PPO using [mjlab](https://github.com/mujocolab/mjlab) + RSL-RL + MuJoCo Warp.

---

## Setup

```bash
# Clone and install mjlab
git clone https://github.com/mujocolab/mjlab && cd mjlab
pip install -e ".[all]"

# Symlink HANUMAN into mjlab's task registry
ln -s /path/to/HANUMAN/layer_1 ~/mjlab/src/mjlab/tasks/hanuman
echo "from . import hanuman" >> ~/mjlab/src/mjlab/tasks/__init__.py

# Verify
python -c "from mjlab.tasks import hanuman; print('OK')"
```

---

## Train

```bash
# Default (256 envs fits 8GB VRAM)
python -m mjlab.scripts.train Hanuman-Velocity-Rough-G1 \
    --env.scene.num-envs 256 \
    --agent.max-iterations 150000 \
    --log-root logs/hanuman

# With W&B logging
python -m mjlab.scripts.train Hanuman-Velocity-Rough-G1 \
    --env.scene.num-envs 256 \
    --agent.max-iterations 150000 \
    --agent.logger wandb \
    --log-root logs/hanuman

# Resume from checkpoint
python -m mjlab.scripts.train Hanuman-Velocity-Rough-G1 \
    --env.scene.num-envs 256 \
    --agent.max-iterations 150000 \
    --agent.resume True \
    --agent.load-run <TIMESTAMP_FOLDER> \
    --agent.load-checkpoint <MODEL_FILE.pt> \
    --log-root logs/hanuman
```

Training takes ~32 hours on an RTX 5060 (8GB) with 256 envs.

> **Note:** `height field collision overflow` warnings are normal on rough terrain. MuJoCo caps contacts per geom and continues — the training is not affected.

---

## Play

```bash
# Evaluate with viewer
python -m mjlab.scripts.play Hanuman-Velocity-Rough-G1 \
    --checkpoint_file logs/hanuman/hanuman_g1_rough/<run>/model_150000.pt

# Record video
python -m mjlab.scripts.play Hanuman-Velocity-Rough-G1 \
    --checkpoint_file logs/hanuman/hanuman_g1_rough/<run>/model_150000.pt \
    --record_video --video_length 300
```

---

## Environment Details

### Terrain Mix (Mars-relevant)

| Terrain | Proportion | Purpose |
|---|---|---|
| Random rough | 25% | Regolith, small rocks |
| Discrete obstacles | 15% | Boulders, rock outcrops |
| Pyramid slopes | 15% | Crater walls, hillsides |
| Inverted slopes | 10% | Descending into craters |
| Stepping stones | 10% | Isolated footholds |
| Pyramid stairs | 10% | Layered rock formations |
| Flat | 10% | Dome interiors, crater floors |
| Perlin noise | 5% | Organic terrain variation |

### Observation Space (288 dims — actor)

| Component | Dim | Source |
|---|---|---|
| Base linear velocity | 3 | IMU |
| Base angular velocity | 3 | IMU |
| Projected gravity | 3 | IMU orientation |
| Joint positions (relative) | 29 | Encoders |
| Joint velocities | 29 | Encoders |
| Previous action | 29 | Internal |
| Velocity command | 3 | Operator |
| Body heightmap scan | 187 | Raycast (1.6m × 1.0m grid) |
| Foot height scan | 2 | Per-foot terrain height |

Critic receives 298 dims (actor obs + foot air time, contact, contact forces).

### Reward Function (14 terms)

| Term | Weight | Purpose |
|---|---|---|
| track_linear_velocity | +2.0 | XY velocity tracking |
| track_angular_velocity | +2.0 | Yaw rate tracking |
| upright | +1.0 | Terrain-normal aligned (not world up) |
| pose | +1.0 | Speed-dependent posture regulation |
| air_time | +0.5 | Proper swing duration |
| foot_clearance | -2.5 | Clear rocks during swing |
| foot_swing_height | -0.25 | Don't kick too high |
| foot_slip | -0.3 | No sliding on tilted surfaces |
| soft_landing | -0.001 | Gentle foot placement |
| body_ang_vel | -0.05 | Minimize wobble |
| angular_momentum | -0.02 | Clean motion |
| dof_pos_limits | -1.0 | Respect joint limits |
| action_rate_l2 | -0.1 | Smooth actions |
| self_collisions | -1.5 | No limb-to-limb contact |

### Domain Randomization

| Parameter | Range |
|---|---|
| Foot friction | 0.3 – 1.2 |
| Base CoM offset | ±2.5cm XY, ±3cm Z |
| Encoder bias | ±0.015 rad |
| External pushes | ±0.5 m/s, every 1-3s |
| Terrain patch | Randomized each episode |

### Curriculum

Velocity commands expand during training:
- Start: vx ∈ [-1.0, 1.0], ω ∈ [-0.5, 0.5]
- 120K steps: vx ∈ [-1.5, 2.0], ω ∈ [-0.7, 0.7]
- 240K steps: vx ∈ [-2.0, 3.0]

Terrain difficulty increases as the policy masters easier levels.

---

## File Structure

```
layer_1/
├── __init__.py     Task registration with mjlab
├── env_cfg.py      Complete environment definition
└── rl_cfg.py       PPO hyperparameters
```

---

## Part of [HANUMAN](https://github.com/siddarth09/HANUMAN)

Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain.