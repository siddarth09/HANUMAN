# HANUMAN

### Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain

A proposed four-layer autonomy stack for deploying bipedal humanoid robots on unstructured terrain — from Earth wilderness to the Martian surface.

---

## The Problem

Mars has no flat floors, no GPS, and a 4-24 minute communication delay to Earth. A humanoid operating there must walk over rocks it has never seen, localize itself without external infrastructure, and make every decision autonomously. No human in the loop for anything time-critical.

This requires solving four problems simultaneously: walking on rough terrain, perceiving the ground ahead, knowing where you are, and coordinating with other agents to explore efficiently.

---

## Proposed Architecture

```
Layer 4: Mission Coordination       0.1 Hz     Drone + humanoid task assignment
Layer 3: Navigation & SLAM            1 Hz     GPS-denied localization, path planning
Layer 2: Terrain Perception          10 Hz     Depth camera → local elevation map
Layer 1: RL Locomotion               50 Hz     Learned joint-level balance + stepping
```

Each layer communicates downward through a minimal interface — waypoints, heightmaps, velocity commands. Each can be built, tested, and swapped independently.

---

## Layer 1: Terrain-Adaptive Locomotion

PPO-trained policy on massively parallel simulation. Observes proprioception + local heightmap, outputs joint position targets. Domain randomization covers friction, mass, external pushes, and critically **gravity (3.0–10.0 m/s²)** so the same policy works on Earth (9.81) and Mars (3.72) without retraining.

## Layer 2: Perception-Driven Foothold Selection

Depth camera produces a local elevation map fed directly into the RL observation space. No explicit footstep planner — the policy learns what terrain is safe through training on diverse terrain curriculum (flat, slopes, rubble, steps, loose regolith).

## Layer 3: GPS-Denied Navigation

Factor graph SLAM using GTSAM. Fuses IMU preintegration, visual odometry, loop closures, and orbital terrain priors. Outputs a traversability costmap for A* path planning. Replans when the robot encounters obstacles not visible from the global map.

## Layer 4: Multi-Agent Exploration

Aerial scout (drone) provides terrain reconnaissance over ROS2. Shared elevation map fuses aerial and ground observations. Mission planner assigns science targets and coordinates approach routes.

---

## Roadmap

| Phase | Timeline | Goal | Status |
|---|---|---|---|
| Foundation | Month 1–2 | Rough terrain RL, Mars gravity curriculum | Planned |
| Perception | Month 3–4 | Depth camera, heightmap-conditioned policy | Planned |
| Navigation | Month 5–6 | GTSAM SLAM, autonomous 100m+ traverses | Planned |
| Multi-Agent | Month 7–8 | Drone coordination, exploration demo | Stretch |

---

## Starting Point

HANUMAN builds on [Bheema](https://github.com/siddarth09/Bheema) — a working bipedal locomotion controller for the Unitree G1 with both a classical MPC pipeline and a trained RL policy. The flat-terrain locomotion (Layer 1 baseline) is already demonstrated. This project extends it to rough terrain, adds perception, navigation, and multi-agent coordination.

---

## References

- Miki et al., *Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild*, Science Robotics 2022
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, CoRL 2022
- Dellaert & Kaess, *Factor Graphs for Robot Perception*, Foundations and Trends in Robotics 2017
- Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3 Through Convex MPC*, IROS 2018