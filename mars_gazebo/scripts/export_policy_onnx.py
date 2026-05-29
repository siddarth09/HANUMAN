#!/usr/bin/env python3
"""
Convert rsl_rl actor checkpoint (.pt) → ONNX for HANUMAN deployment.

Architecture recovered from actor_state_dict keys:
  obs_normalizer: EmpiricalNormalization(288)
  mlp: Linear(288,1024) ELU → Linear(1024,512) ELU →
       Linear(512,256)  ELU → Linear(256,128)  ELU → Linear(128,29)

Usage:
    python3 export_policy_onnx.py \
        --pt  /path/to/model_425000.pt \
        --out /path/to/hanuman_policy.onnx
"""

import argparse
import sys
import torch
import torch.nn as nn

OBS_DIM = 288
ACT_DIM = 29
HIDDEN  = [1024, 512, 256, 128]


class EmpiricalNormalization(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_std",  torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.obs_normalizer = EmpiricalNormalization(OBS_DIM)
        layers = []
        prev = OBS_DIM
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ACT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.obs_normalizer(obs))


def convert(pt_path: str, onnx_path: str) -> None:
    print(f"Loading checkpoint: {pt_path}")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    if "actor_state_dict" not in ckpt:
        sys.exit(f"Expected 'actor_state_dict' key, got: {list(ckpt.keys())}")

    actor_sd = ckpt["actor_state_dict"]

    # Strip distribution.* and unused normalizer bookkeeping buffers
    # Only _mean and _std are needed for inference; _var and count are training-only
    keep = {"obs_normalizer._mean", "obs_normalizer._std"}
    filtered = {k: v for k, v in actor_sd.items()
                if k.startswith("mlp") or k in keep}

    actor = Actor()
    missing, unexpected = actor.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(f"  missing keys : {missing}")
        print(f"  unexpected   : {unexpected}")
        sys.exit("State dict mismatch — check architecture.")

    actor.eval()

    dummy = torch.zeros(1, OBS_DIM)
    with torch.no_grad():
        out = actor(dummy)
    print(f"Forward check OK — output shape: {tuple(out.shape)}")

    torch.onnx.export(
        actor,
        dummy,
        onnx_path,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={"observations": {0: "batch_size"},
                      "actions":      {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Saved ONNX: {onnx_path}")

    # Quick verify with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        obs_np = np.zeros((1, OBS_DIM), dtype=np.float32)
        actions = sess.run(["actions"], {"observations": obs_np})[0]
        print(f"OnnxRuntime verify OK — action shape: {actions.shape}")
    except ImportError:
        print("onnxruntime not installed — skipping runtime verify")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out", required=True, help="Output .onnx path")
    args = parser.parse_args()
    convert(args.pt, args.out)
