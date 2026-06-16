"""HANUMAN — G1 velocity locomotion tasks on Martian terrain.

Registers:
  - Hanuman-Mars-v0           synthetic rough/slope mix
  - Hanuman-MarsRealistic-v0  Jezero DEM terrain (env_cfg_mars.py)
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import hanuman_g1_rough_env_cfg
from .env_cfg_mars import hanuman_g1_mars_env_cfg
from .rl_cfg import hanuman_g1_ppo_runner_cfg

register_mjlab_task(
    task_id="Hanuman-Mars-v0",
    env_cfg=hanuman_g1_rough_env_cfg(),
    play_env_cfg=hanuman_g1_rough_env_cfg(play=True),
    rl_cfg=hanuman_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Hanuman-MarsRealistic-v0",
    env_cfg=hanuman_g1_mars_env_cfg(),
    play_env_cfg=hanuman_g1_mars_env_cfg(play=True),
    rl_cfg=hanuman_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)