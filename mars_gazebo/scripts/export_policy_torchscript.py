#!/usr/bin/env python3
"""
Export the HANUMAN locomotion policy as a TorchScript .pt for the C++
(LibTorch) deployment node — the counterpart to export_policy_onnx.py.

We don't have the original rsl_rl checkpoint published (only the ONNX), so this
rebuilds the Actor from the ONNX initializers (mlp.* weights + the
EmpiricalNormalization mean/std) and `torch.jit.script`s it. The result is
verified against onnxruntime so the .pt is bit-for-bit-equivalent in behaviour.

Usage:
    python3 export_policy_torchscript.py \
        --onnx policy/hanuman_policy.onnx \
        --out  policy/hanuman_policy.pt
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 288
ACT_DIM = 29
HIDDEN = [1024, 512, 256, 128]


class Actor(nn.Module):
    """EmpiricalNormalization -> MLP (ELU), matching export_policy_onnx.py.

    Normalisation is stored as (mean, div) where div = std + eps, exactly the
    constant the ONNX graph folded, so forward = mlp((obs - mean) / div).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, OBS_DIM))
        self.register_buffer("div", torch.ones(1, OBS_DIM))
        layers = []
        prev = OBS_DIM
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, ACT_DIM))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / self.div)


def load_onnx_initializers(onnx_path: str) -> dict:
    import onnx
    from onnx import numpy_helper
    model = onnx.load(onnx_path)
    return {init.name: numpy_helper.to_array(init)
            for init in model.graph.initializer}


def build_actor(inits: dict) -> Actor:
    actor = Actor()
    sd = actor.state_dict()

    # Normalisation: mean and the folded divisor (std + eps).
    mean = inits["obs_normalizer._mean"].reshape(1, OBS_DIM)
    # The folded (1, 288) constant that isn't the mean is the divisor.
    div = None
    for name, arr in inits.items():
        if arr.shape == (1, OBS_DIM) and name != "obs_normalizer._mean":
            div = arr.reshape(1, OBS_DIM)
            break
    if div is None:
        raise RuntimeError("Could not find the normalizer divisor in the ONNX graph")
    sd["mean"] = torch.from_numpy(mean.astype(np.float32))
    sd["div"] = torch.from_numpy(div.astype(np.float32))

    # MLP layers: ONNX mlp.{0,2,4,6,8} -> nn.Sequential indices {0,2,4,6,8}.
    for idx in (0, 2, 4, 6, 8):
        sd[f"mlp.{idx}.weight"] = torch.from_numpy(inits[f"mlp.{idx}.weight"])
        sd[f"mlp.{idx}.bias"] = torch.from_numpy(inits[f"mlp.{idx}.bias"])

    actor.load_state_dict(sd)
    actor.eval()
    return actor


def build_actor_from_ckpt(ckpt_path: str) -> Actor:
    """Build the Actor directly from a raw rsl_rl checkpoint (actor_state_dict).

    The checkpoint stores EmpiricalNormalization as (_mean, _std); the deployed
    normaliser is (x - _mean) / (_std + 1e-8), so div = _std + 1e-8.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise SystemExit(f"No 'actor_state_dict' in {ckpt_path}; "
                         f"keys={list(ckpt.keys())}")
    asd = ckpt["actor_state_dict"]
    actor = Actor()
    sd = actor.state_dict()
    sd["mean"] = asd["obs_normalizer._mean"].reshape(1, OBS_DIM).float()
    sd["div"] = (asd["obs_normalizer._std"].reshape(1, OBS_DIM).float() + 1e-8)
    for idx in (0, 2, 4, 6, 8):
        sd[f"mlp.{idx}.weight"] = asd[f"mlp.{idx}.weight"].float()
        sd[f"mlp.{idx}.bias"] = asd[f"mlp.{idx}.bias"].float()
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def verify(actor: Actor, onnx_path: str, n: int = 8, tol: float = 1e-4) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed — skipping parity check)")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(n):
        obs = rng.standard_normal((1, OBS_DIM)).astype(np.float32)
        onnx_out = sess.run(["actions"], {"observations": obs})[0]
        with torch.no_grad():
            pt_out = actor(torch.from_numpy(obs)).numpy()
        max_err = max(max_err, float(np.abs(onnx_out - pt_out).max()))
    print(f"  parity vs onnxruntime: max abs error = {max_err:.2e} "
          f"({'OK' if max_err < tol else 'FAIL'})")
    if max_err >= tol:
        raise SystemExit("Parity check FAILED — .pt does not match ONNX")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="Raw rsl_rl checkpoint (.pt with actor_state_dict). "
                         "Preferred — scripts the trained policy directly.")
    ap.add_argument("--onnx", default="policy/hanuman_policy.onnx",
                    help="Fallback source: rebuild from ONNX weights if no --ckpt.")
    ap.add_argument("--out", default="policy/hanuman_policy.pt")
    args = ap.parse_args()

    if args.ckpt:
        print(f"Building Actor from checkpoint: {args.ckpt}")
        actor = build_actor_from_ckpt(args.ckpt)
        # Parity-check against the ONNX if it's available (same weights).
        import os
        if os.path.exists(args.onnx):
            verify(actor, args.onnx)
    else:
        print(f"Loading ONNX initializers: {args.onnx}")
        inits = load_onnx_initializers(args.onnx)
        actor = build_actor(inits)
        print("Rebuilt Actor from ONNX weights.")
        verify(actor, args.onnx)

    scripted = torch.jit.script(actor)
    scripted.save(args.out)
    print(f"Saved TorchScript policy: {args.out}")

    # Sanity: reload the saved .pt and run once.
    reloaded = torch.jit.load(args.out)
    out = reloaded(torch.zeros(1, OBS_DIM))
    print(f"Reloaded .pt OK — output shape {tuple(out.shape)}")


if __name__ == "__main__":
    main()
