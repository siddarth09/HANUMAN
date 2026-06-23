"""RL configuration for HANUMAN G1 rough terrain task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)



def hanuman_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for HANUMAN rough terrain."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(1024,512,256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(1024,512,256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            # Was 1.0e-5 == rsl_rl's HARDCODED adaptive-LR floor (ppo.py:
            # lr = max(1e-5, lr/1.5)). Starting AT the floor meant the scheduler
            # could never cut LR when KL spiked -> the 2026-06-20 run diverged at
            # iter ~373k with no way to self-rescue. Start at 1e-4 so adaptive has
            # ~10x downward headroom while staying gentle for a fine-tune resume.
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        # Clamp policy outputs to +/-10 (applied pre-env-step in the vecenv
        # wrapper). Healthy actions sit near +/-3 at init_std=1, so this never
        # touches normal gait, but it bounds the unbounded action_rate_l2 penalty
        # (sum of squared action deltas) that ran to -1.7M and exploded the value
        # loss to 1e14 once the policy std blew up. Caps the divergence fuel.
        clip_actions=10.0,
        experiment_name="hanuman_g1_mars",
        save_interval=10000,
        num_steps_per_env=24,
        max_iterations=150_000,
    )