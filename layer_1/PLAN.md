# Plan: hanumanv2 (Mars-slope RL) + System0/System1 hierarchy

## Status (done)
- Restructured: `system1/` = RL (env_cfg, rl_cfg, mars_dem_terrain, __init__); `system0/` = MPC
  (copied from BHEEMA, Phase 2). Added `layer_1/__init__.py` shim importing `system1` so the mjlab
  symlink (`hanuman -> layer_1`) keeps `Hanuman-Mars-v0` registered. Verified
  `import mjlab.tasks.hanuman` + a 3-iteration training smoke test (checkpoints written, curriculum
  tracks hf_pyramid_slope / _inv / mars_dem).
- hanumanv2 slope change: capped synthetic pyramid slopes 0.8→0.5 (~27°) and bumped slope weight, so
  the curriculum gives achievable graded climbing (38° pyramids stalled the top rows). Real steep
  crater walls still come from `mars_dem`.
- **Mars gravity**: `MujocoCfg(gravity=(0,0,-3.72))` for train + play (was Earth -9.81). Physics only;
  `projected_gravity` obs / `upright` reward are orientation-only (normalized down-vector) so they're
  unaffected. Lower gravity = more torque margin per unit weight → should aid slope climbing. NOTE:
  this makes the policy gravity-specific — deploy on a Mars-gravity sim (e.g. mars_terrain.xml uses
  -3.72; mars_nav_scene.xml currently uses -9.81 and would need changing to match).

## Train hanumanv2
```
python -m mjlab.scripts.train Hanuman-Mars-v0 --env.scene.num-envs 256 \
    --agent.max-iterations 150000 --log-root logs/hanumanv2
python -m mjlab.scripts.play Hanuman-Mars-v0 \
    --checkpoint_file logs/hanumanv2/hanuman_g1_rough/<run>/model_*.pt   # eval on slopes
```

## Further slope levers if capping isn't enough
- `mars_dem` `vertical_exaggeration` >1.0 (1 m/px DEM relief is gentle at 8 m patches).
- Ensure `terrain_levels_vel` promotion credits slow uphill (don't punish climbing slowly).
- Re-check `base_too_low` (0.30 terrain-rel) / `bad_orientation` (70°) don't fire on legit climbs.



## Context
G1 humanoid locomotion. Two controllers exist: **BHEEMA** centroidal MPC (separate package, leave
untouched) and the **HANUMAN RL** policy (`layer_1`). Target end-state is hierarchical:
**System1 = RL high-level terrain-aware planner**, **System0 = MPC reflexive executor**.

**This weekend's focus (Phase 1): retrain the RL policy → `hanumanv2` so the G1 can climb the real
Mars terrain / slopes**, which the current policy does poorly. The `HfDemTerrainCfg` (real Jezero DEM)
is already wired into `layer_1/env_cfg.py`'s `MARS_TERRAINS_CFG`. Phases 2–3 (MPC in ROS, hybrid) come
after and are scoped here but not the weekend priority.

## Folder restructure (layer_1)
```
layer_1/
  PLAN.md                ← this plan
  system1/               ← RL ("System 1"): env_cfg.py, rl_cfg.py, __init__.py, mars_dem_terrain.py
  system0/               ← MPC ("System 0"): user copies BHEEMA mpc files here (no edits to BHEEMA pkg)
```
- Move the existing RL files into `system1/`. **Care:** the mjlab task is registered via the symlink
  `~/mjlab/src/mjlab/tasks/hanuman -> layer_1` and `__init__.py`. Moving files changes import paths
  (`from .mars_dem_terrain import ...`, the symlink target, and `from mjlab.tasks.hanuman ...`). Repoint
  the symlink to `layer_1/system1` (or keep `__init__.py` at `layer_1/` re-exporting from `system1`)
  and verify `python -c "from mjlab.tasks import hanuman"` still works **before** training.
- `system0/` is created empty now; user drops MPC files in. The ROS adapter + executor (Phase 2) live
  in HANUMAN, not BHEEMA.

## PHASE 1 — hanumanv2: climb real Mars terrain (THE WEEKEND TASK)
Goal: a low-level joint policy that reliably walks up Mars slopes / the real DEM.

Key files: `layer_1/system1/env_cfg.py` (`MARS_TERRAINS_CFG`, rewards, curriculum, terminations),
`rl_cfg.py` (PPO), `mars_dem_terrain.py` (DEM sub-terrain).

Planned, iterative changes (train → eval → adjust):
1. **Terrain emphasis for slopes**: in `MARS_TERRAINS_CFG`, raise the weight/difficulty of climbable
   terrain — `hf_pyramid_slope` / `hf_pyramid_slope_inv` and the real-DEM `mars_dem` (steep
   crater-wall windows live at high `difficulty`). Confirm the curriculum reaches the steep rows
   (`num_rows`, `max_init_terrain_level`, `difficulty_range`). Consider `vertical_exaggeration` on the
   DEM if the 1 m/px relief is too gentle at 8 m patches.
2. **Reward review for climbing** (current weights in `env_cfg.py`): ensure forward-velocity tracking
   is achievable uphill; check that penalties don't suppress the larger hip/knee motions climbing
   needs (`pose`, `action_rate_l2`, `foot_clearance`); `upright` is already terrain-normal aware (keep).
   Re-check `base_too_low` termination (0.30 terrain-relative) and `bad_orientation` (70°) don't fire
   on legit steep climbs.
3. **Command curriculum**: keep slow→fast `command_vel` stages; verify slope difficulty isn't gated
   out by velocity-based `terrain_levels_vel` promotion (robot must get credit for climbing slowly).
4. **Train** (per `layer_1/README.md`): `python -m mjlab.scripts.train Hanuman-Mars-v0
   --env.scene.num-envs <fit VRAM> --log-root logs/hanumanv2` (W&B optional). New run id `hanumanv2`.
5. **Evaluate slope climbing**: `mjlab.scripts.play` on the DEM/slope scene; record video; measure
   uphill success/fall rate vs the old checkpoint. Iterate on (1)-(3).

## PHASE 2 — System0: MPC executor in ROS (after the weekend)
- Copy BHEEMA MPC modules into `layer_1/system0/` (user). **No edits to the BHEEMA package.**
- New `system0/ros_state_adapter.py` (in HANUMAN): fill BHEEMA's Pinocchio state contract from
  `/odometry/filtered` + `/joint_states`. Conventions: `z -= 0.793` (BODY_OFFSET), quat `[x,y,z,w]`
  direct, `base_vel = R.T @ twist.linear` (world→body), `base_ang_vel = twist.angular` (already body).
  **Adapter unit test in HANUMAN** vs the known MuJoCo-sync values.
- New `system0/mpc_executor_node.py`: two timers (MPC ~42 Hz solve, leg torque 200 Hz interp), publish
  12 leg torques to a new `g1_leg_effort_controller`.
- Interface change (verify against the mujoco_ros2_control fork first): 12 leg joints
  `position`→`effort` in `g1_mujoco.urdf`; leg actuators `<position>`→`<motor>` in `g1_mars.xml`;
  add effort controller + split upper-body position controller in `controller_mjcf.yaml`.
- Stage validation: teleop `/cmd_vel` → MPC; off `/ground_truth/odom` first, then the EKF.

## PHASE 3 — Hierarchical RL-plans / MPC-executes
- New `system1` high-level policy whose **action = MPC command** (`vx,vy,yaw_rate` → +height/pitch →
  +footsteps), trained on the Mars DEM. Publishes `/cmd_vel` (later a `hanuman_msgs/LocomotionCommand`).
- Hard problem: MPC can't run inside thousands of GPU envs → train against a fast surrogate of the MPC
  closed loop, deploy with the real MPC. (Decision deferred; does not block Phase 1.)
- `rl_policy_node.py` stays as the joint-level baseline.

## Risks
- Phase 1 is empirical RL tuning — expect several train/eval iterations; slope climbing may need
  reward + curriculum changes, not just terrain weights.
- Folder move can break the mjlab symlink/import + task registration — verify import before training.
- (Phase 2) effort-interface mechanics in the mujoco_ros2_control fork; frames/0.793 offset; EKF
  quality feeding MPC.
- (Phase 3) MPC-in-training-loop infeasibility → surrogate needed.

## Verification
- Phase 1: training launches and runs; `play` on slope/DEM scene shows the G1 climbing grades the old
  policy failed; lower fall rate / higher uphill progress. `from mjlab.tasks import hanuman` imports
  after the folder move.
- Phase 2/3: as in the per-phase notes (teleop→MPC tracking off EKF; then RL planner closes the loop).
