"""
HANUMAN — Humanoid Autonomous Navigation on Unstructured Martian And Natural Terrain

Registers:
  - Hanuman-Velocity-Rough-G1       (training)
  - Hanuman-Velocity-Rough-G1-Play  (evaluation)
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import hanuman_g1_rough_env_cfg
from .rl_cfg import hanuman_g1_ppo_runner_cfg

# Training task
register_mjlab_task(
    task_id="Hanuman-Mars-v0",
    env_cfg=hanuman_g1_rough_env_cfg(),
    play_env_cfg=hanuman_g1_rough_env_cfg(play=True),
    rl_cfg=hanuman_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)