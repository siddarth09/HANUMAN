#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn


class EmpiricalNormalization(nn.Module):
    """Mirrors rsl_rl.networks.EmpiricalNormalization for inference only.

    At export time we bake the frozen mean/std from training into the graph.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, input_dim))
        self.register_buffer("_var", torch.ones(1, input_dim))
        self.register_buffer("_std", torch.ones(1, input_dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class ActorForExport(nn.Module):
    """Wraps obs_normalizer + MLP actor for clean ONNX export.

    The MLP architecture is reconstructed from checkpoint weight shapes:
        288 → 1024 → ELU → 512 → ELU → 256 → ELU → 128 → ELU → 29
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()

        # Observation normalizer
        self.obs_normalizer = EmpiricalNormalization(obs_dim)

        # Build MLP: Linear → ELU → Linear → ELU → ... → Linear
        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ELU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = self.obs_normalizer(obs)
        return self.mlp(normalized)


def extract_architecture(actor_state_dict: dict) -> tuple[int, int, list[int]]:
    """Infer obs_dim, action_dim, and hidden_dims from weight shapes."""
    mlp_weights = {
        k: v for k, v in actor_state_dict.items() if k.startswith("mlp.") and "weight" in k
    }
    # Sort by layer index: mlp.0.weight, mlp.2.weight, ...
    sorted_keys = sorted(mlp_weights.keys(), key=lambda k: int(k.split(".")[1]))

    obs_dim = mlp_weights[sorted_keys[0]].shape[1]
    action_dim = mlp_weights[sorted_keys[-1]].shape[0]
    hidden_dims = [mlp_weights[k].shape[0] for k in sorted_keys[:-1]]

    return obs_dim, action_dim, hidden_dims


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HANUMAN policy to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model_*.pt checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .onnx path (default: same dir as checkpoint)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    actor_sd = ckpt["actor_state_dict"]

    # Infer architecture from weights
    obs_dim, action_dim, hidden_dims = extract_architecture(actor_sd)
    print(f"Architecture: obs={obs_dim} → {' → '.join(map(str, hidden_dims))} → {action_dim}")
    print(f"Iteration: {ckpt.get('iter', '?')}")

    # Build model and load weights
    model = ActorForExport(obs_dim, action_dim, hidden_dims)
    actor_sd.pop("distribution.std_param", None)
    model.load_state_dict(actor_sd, strict=True)
    model.eval()

    # Verify forward pass
    dummy_obs = torch.randn(1, obs_dim)
    with torch.no_grad():
        dummy_actions = model(dummy_obs)
    print(f"Test forward pass: obs {list(dummy_obs.shape)} → actions {list(dummy_actions.shape)}")

    # Export to ONNX
    output_path = Path(args.output) if args.output else ckpt_path.parent / "hanuman_policy.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_obs,
        str(output_path),
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "batch"},
            "actions": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Quick sanity check with onnxruntime if available
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output_path))
        ort_out = sess.run(None, {"observations": dummy_obs.numpy()})
        diff = abs(ort_out[0] - dummy_actions.numpy()).max()
        print(f"ONNX vs PyTorch max diff: {diff:.6e} {'✓' if diff < 1e-5 else '✗ WARNING'}")
    except ImportError:
        print("onnxruntime not installed — skipping verification (pip install onnxruntime)")


if __name__ == "__main__":
    main()