"""
HANUMAN — Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain
=====================================================================================


  python -m mjlab.scripts.train Hanuman-Mars-v0 \
      --env.scene.num-envs 256 \
      --agent.resume True \
      --agent.load-run <run-name> \
      --agent.load-checkpoint model_<iter>.pt \
      --agent.max-iterations 400000

"""

import math
from dataclasses import replace
from pathlib import Path

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import (
    flat,
    random_rough,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    discrete_obstacles,
    stepping_stones,
    pyramid_stairs,
    perlin_noise,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .mars_dem_terrain import mars_dem


# Real HiRISE DEM exported by mars_terrain_exporter (Jezero Crater center).
# 200x200 px @ 1.0 m/px, elevation span 31.45 m.
# Resolved relative to the repo root so the task is portable:
# <repo>/layer_1/system1/env_cfg.py -> parents[2] == <repo root>.
_REPO_ROOT = Path(__file__).resolve().parents[2]
MARS_DEM_FILE = str(
    _REPO_ROOT
    / "mars_gazebo"
    / "unitree_g1_mjcf"
    / "mars_nav_200"
    / "mars_nav_200.png"
)


# =============================================================================
# MARS TERRAIN CONFIGURATION
# =============================================================================
# Proportions reflect Mars surface: mostly rocky rubble and slopes,
# occasional flat patches, rare staircase-like rock formations.
# Curriculum enabled — starts easy, progresses to harder terrain.

MARS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    curriculum=True,
    sub_terrains={
        # Real Jezero DEM windows are now the bulk of training (difficulty-graded
        # crops: easy rows = crater floor, hard rows = crater wall).
        "mars_dem": mars_dem(
            proportion=0.45,
            dem_file=MARS_DEM_FILE,
            elevation_range_m=31.45,
            dem_resolution_m=1.0,
            vertical_exaggeration=1.0,
        ),
        # Procedural terrain kept only as curriculum scaffolding — a clean
        # difficulty axis the DEM crops don't guarantee on their own.
        "perlin_noise": perlin_noise(  # rolling dunes, bumped relief
            proportion=0.10,
            height_range=(0.0, 0.8),
        ),
        "random_rough": random_rough(
            proportion=0.10,
            noise_range=(0.02, 0.10),  # up to 10 cm
            noise_step=0.04,
        ),
        # crater walls / hillsides, ~31 deg max — controllable slope curriculum
        "hf_pyramid_slope": hf_pyramid_slope(
            proportion=0.20,
            slope_range=(0.0, 0.6),
        ),
        # descending into craters
        "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
            proportion=0.15,
            slope_range=(0.0, 0.5),
        ),
    },
    add_lights=True,
)


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def hanuman_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create HANUMAN G1 rough terrain velocity configuration."""

    # ─── Sensors ───────────────────────────────────────────
    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="pelvis", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
        max_distance=5.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),  # Terrain only
        debug_vis=True,
    )

    site_names = ("left_foot", "right_foot")
    foot_height_scan = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=s, entity="robot") for s in site_names
        ),
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=True,
        viz=TerrainHeightSensorCfg.VizCfg(
            show_rays=True,
            hit_color=(1.0, 0.0, 1.0, 0.8),
            hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
        ),
    )

    # Foot-ground contact sensor
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # Self-collision sensor
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # ─── Observations ──────────────────────────────────────
    # Actor: 271 dims (proprioception + body heightmap + foot scan)
    # Critic: 271 + privileged info (clean heightmap, foot height, contact)

    actor_terms = {
        # IMU linear velocity in body frame (3)
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        # IMU angular velocity in body frame (3)
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        # Projected gravity — which way is "down" relative to body (3)
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        # Joint positions relative to default standing pose (29)
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        # Joint velocities (29)
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        # Previous action for smoothness (29)
        "actions": ObservationTermCfg(func=mdp.last_action),
        # Velocity command [vx, vy, yaw_rate] (3)
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        # Body-centered terrain heightmap — the "planning" scan (160)
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=1 / terrain_scan.max_distance,
        ),
        # Per-foot terrain height — the "reactive" scan (12)
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"sensor_name": "foot_height_scan"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
    }
   
    critic_terms = {
        **actor_terms,
      
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            scale=1 / terrain_scan.max_distance,
        ),
        # Override foot_height without noise
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"sensor_name": "foot_height_scan"},
        ),
        # Privileged: foot air time (2)
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
        ),
        # Privileged: binary foot contact (2)
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        # Privileged: contact force magnitudes (6)
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,  # Noise active for actor
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,  # No noise for critic 
        ),
    }

    # ─── Actions ───────────────────────────────────────────

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=G1_ACTION_SCALE,
            use_default_offset=True,
        ),
    }

    # ─── Commands ──────────────────────────────────────────

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.3,
            rel_forward_envs=0.2,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.5, 0.5),  
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
            viz=UniformVelocityCommandCfg.VizCfg(z_offset=1.15),
        ),
    }

    # ─── Events (Domain Randomization) ─────────────────────

    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in range(1, 8)
    )

    events = {
        # Reset base position with terrain-aware height
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.01, 0.05),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        # Reset joints with small random offset
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        # Random pushes during walking (more aggressive for rough terrain)
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        # Randomize foot friction (Mars regolith varies widely)
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
                "operation": "abs",
                "ranges": (0.3, 0.8),  # Wider range for Mars regolith
                "shared_random": True,
            },
        ),
        # Encoder bias (sensor imperfections)
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        # Base CoM offset (payload/model uncertainty)
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
        # Randomize terrain on reset (different terrain patch each episode)
        "randomize_terrain": EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        ),
    }

   
    rewards = {
        # === TASK REWARDS ===
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_linear_velocity,
            weight=2.0,  
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_angular_velocity,
            weight=2.0,  
            params={"command_name": "twist", "std": math.sqrt(0.5)},
        ),
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=0.8, 
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "terrain_sensor_names": ("terrain_scan",),  # Mars: terrain-normal aware
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0, 
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "command_name": "twist",
                "std_standing": {".*": 0.05},
                "std_walking": {
                    r".*hip_pitch.*": 0.3,
                    r".*hip_roll.*": 0.15,
                    r".*hip_yaw.*": 0.15,
                    r".*knee.*": 0.35,
                    r".*ankle_pitch.*": 0.25,
                    r".*ankle_roll.*": 0.1,
                    r".*waist_yaw.*": 0.2,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.1,
                    r".*shoulder_pitch.*": 0.15,
                    r".*shoulder_roll.*": 0.15,
                    r".*shoulder_yaw.*": 0.1,
                    r".*elbow.*": 0.15,
                    r".*wrist.*": 0.3,
                },
                "std_running": {
                    r".*hip_pitch.*": 0.5,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.2,
                    r".*knee.*": 0.6,
                    r".*ankle_pitch.*": 0.35,
                    r".*ankle_roll.*": 0.15,
                    r".*waist_yaw.*": 0.3,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.2,
                    r".*shoulder_pitch.*": 0.5,
                    r".*shoulder_roll.*": 0.2,
                    r".*shoulder_yaw.*": 0.15,
                    r".*elbow.*": 0.35,
                    r".*wrist.*": 0.3,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        ),
    
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            # Reward a genuine swing phase, sized to compete with the velocity
            # tracking terms so the gait steps instead of settling into a shuffle.
            weight=0.5,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "twist",
                "command_threshold": 0.5,
            },
        ),

      
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            # Penalizes |foot_height - target| weighted by foot speed. Kept light
            # so it shapes clearance without penalizing the swing motion itself;
            # feet_swing_height owns the foot-lift target.
            weight=-0.5,
            params={
                "target_height": 0.1,
                "height_sensor_name": "foot_height_scan",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
            },
        ),
        "foot_swing_height": RewardTermCfg(
            func=mdp.feet_swing_height,
            # Penalizes (peak_swing_height/target - 1)^2 at landing — the dominant
            # foot-lift signal that drives the foot to the target swing clearance.
            weight=-3.0,
            params={
                "sensor_name": "feet_ground_contact",
                "height_sensor_name": "foot_height_scan",
                "target_height": 0.1,
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.1,  
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
                "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.05,
            },
        ),

        # === STABILITY ===
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
        ),
        "angular_momentum": RewardTermCfg(
            func=mdp.angular_momentum_penalty,
            weight=-0.02,  
            params={"sensor_name": "robot/root_angmom"},
        ),

        # === SAFETY ===
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.1,
        ),
        
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,  
            params={
                "sensor_name": self_collision_cfg.name,
                "force_threshold": 10.0,
            },
        ),
    }
    # ─── Terminations ──────────────────────────────────────

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "out_of_terrain_bounds": TerminationTermCfg(
            func=mdp.out_of_terrain_bounds,
            time_out=True,
        ),
        "base_too_low": TerminationTermCfg(
            func=mdp.root_height_below_minimum_terrain_relative,
            params={"minimum_height": 0.30},
        ),
    }

    # ─── Curriculum ────────────────────────────────────────
    # Start with easy terrain and slow commands, progress to harder

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                # Ramp commands over the first ~30k iters (step = iter * 24).
                # Command speed capped at 0.8 m/s: full-speed commands on the steep
                # (~31 deg) Mars slopes are infeasible and push the policy toward a
                # flat-footed shuffle. Learn the gait at a walkable pace first.
                "velocity_stages": [
                    {"step": 0, "lin_vel_x": (-0.4, 0.4), "ang_vel_z": (-0.3, 0.3)},
                    {"step": 8000 * 24, "lin_vel_x": (-0.6, 0.6), "ang_vel_z": (-0.5, 0.5)},
                    {"step": 18000 * 24, "lin_vel_x": (-0.8, 0.8), "ang_vel_z": (-0.6, 0.6)},
                    {"step": 30000 * 24, "lin_vel_x": (-0.8, 0.8)},
                ],
            },
        ),
    
    }

    # ─── Metrics ───────────────────────────────────────────

    metrics = {
        "mean_action_acc": MetricsTermCfg(
            func=mdp.mean_action_acc,
        ),
    }

    # ─── Assemble ──────────────────────────────────────────

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={"robot": get_g1_robot_cfg()},
            terrain=TerrainEntityCfg(
                terrain_type="generator",
                terrain_generator=replace(MARS_TERRAINS_CFG),
                max_init_terrain_level=5,
            ),
            sensors=(terrain_scan, foot_height_scan, feet_ground_cfg, self_collision_cfg),
            
            num_envs=256,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="torso_link",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            # nconmax is per-env (total buffer = nconmax * num_envs). Keep margin
            # over the DEM's peak contact count — dropped contacts blow up the sim.
            nconmax=300,
            njmax=1500,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500,
                gravity=(0.0, 0.0, -3.72),  # Mars
            ),
        ),
        decimation=4,
        episode_length_s=30.0,
    )

    # ─── Play mode overrides ───────────────────────────────

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.sim.nconmax = 400  # single env at play time, can afford a bigger buffer
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

        # Mars-feasible command range for play
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg