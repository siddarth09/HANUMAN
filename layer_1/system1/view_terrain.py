"""Preview the EXACT training terrain (MARS_TERRAINS_CFG) before launching a run.

Builds the same curriculum grid mjlab will train on — including the custom
`mars_dem` sub-terrain — so you can eyeball every terrain type / difficulty level.

Curriculum grid: rows = difficulty (easy -> hard), columns = terrain type.

Interactive (needs a display):
    python -m mjlab.tasks.hanuman.system1.view_terrain

Headless render to an image (no display):
    MUJOCO_GL=egl python -m mjlab.tasks.hanuman.system1.view_terrain --save terrain.png
"""
from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.terrains.terrain_generator import TerrainGenerator

from .env_cfg import MARS_TERRAINS_CFG


def build_model():
    spec = mujoco.MjSpec()
    gen = TerrainGenerator(MARS_TERRAINS_CFG, device="cpu")
    gen.compile(spec)
    return spec.compile(), gen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=None,
                    help="Render a top-down PNG instead of opening the interactive viewer")
    args = ap.parse_args()

    model, gen = build_model()
    rows, cols = gen.terrain_origins.shape[:2]
    print(f"Terrain grid: {rows} rows (difficulty, easy->hard) x {cols} cols (types)")
    print("Columns:", list(MARS_TERRAINS_CFG.sub_terrains.keys()))
    print(f"Patch size: {MARS_TERRAINS_CFG.size} m | border: {MARS_TERRAINS_CFG.border_width} m")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if args.save:
        from PIL import Image
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.elevation = -75.0  # near top-down
        cam.distance = max(MARS_TERRAINS_CFG.size) * max(rows, cols) * 1.3
        with mujoco.Renderer(model, height=1080, width=1920) as r:
            r.update_scene(data, cam)
            img = r.render()
        Image.fromarray(img).save(args.save)
        print(f"saved {args.save}")
    else:
        mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
