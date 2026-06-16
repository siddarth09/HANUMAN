# HANUMAN

### Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain

A proposed four-layer autonomy stack for deploying bipedal humanoid robots on unstructured terrain — from Earth wilderness to the Martian surface.

https://github.com/user-attachments/assets/156a253e-15a4-4766-9701-f271a3bcd135

---

## The Problem

Mars has no flat floors, no GPS, and a 4-24 minute communication delay to Earth. A humanoid operating there must walk over rocks it has never seen, localise itself without external infrastructure, and make every decision autonomously. No human in the loop for anything time-critical.

This requires solving four problems simultaneously: walking over rough terrain, perceiving the ground ahead, knowing your position, and coordinating with other agents to explore efficiently.

---

## Proposed Architecture

```
Layer 3: Mission Coordination       0.1 Hz     Drone + humanoid task assignment
Layer 2: Navigation & SLAM            1 Hz     GPS-denied localisation, path planning
Layer 1: RL Locomotion + Perception  50 Hz     Learned balance, stepping, terrain reading
```

Each layer communicates downward through a minimal interface — waypoints, heightmaps, velocity commands. Each can be built, tested, and swapped independently.

## Layer 1: Terrain-Adaptive Locomotion + Perception

PPO-trained policy on massively parallel simulation. Observes proprioception 
+ 187-dim body heightmap + per-foot terrain scan, outputs joint position 
targets. The policy implicitly learns safe foothold selection through 
training on 8 Mars-relevant terrain types. Domain randomization covers 
friction, mass, external pushes, and gravity (3.0–10.0 m/s²).

## Layer 2: GPS-Denied Navigation

Factor graph SLAM using GTSAM. Depth camera provides visual features for 
odometry and mapping (not for foothold selection — Layer 1 handles that). 
Fuses IMU preintegration, visual odometry, loop closures, and orbital 
terrain priors. Outputs velocity waypoints to Layer 1.

## Layer 3: Multi-Agent Exploration

Aerial scout provides terrain reconnaissance over ROS2. Shared elevation 
map fuses aerial and ground observations. Mission planner assigns science 
targets and coordinates approach routes.
---

| Phase | Timeline | Goal | Status |
|---|---|---|---|
| Locomotion | Month 1–2 | Rough terrain RL with heightmap + Mars gravity | In Progress |
| Navigation | Month 3–5 | GTSAM SLAM, depth camera, autonomous traverses | Planned |
| Multi-Agent | Month 6–8 | Drone coordination, exploration demo | Stretch |
---

## Starting Point

HANUMAN builds on [Bheema](https://github.com/siddarth09/Bheema) — a working bipedal locomotion controller for the Unitree G1 with both a classical MPC pipeline and a trained RL policy. The flat-terrain locomotion (Layer 1 baseline) is already demonstrated. This project extends it to rough terrain, adds perception, navigation, and multi-agent coordination.

---

## References

- Miki et al., *Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild*, Science Robotics 2022
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, CoRL 2022
- Dellaert & Kaess, *Factor Graphs for Robot Perception*, Foundations and Trends in Robotics 2017
- Di Carlo et al., *Dynamic Locomotion in the MIT Cheetah 3 Through Convex MPC*, IROS 2018
