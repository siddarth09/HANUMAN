"""Realistic-Mars velocity env for the G1 (Hanuman-MarsRealistic-v0).

Reuses the sensor/observation/action/reward/DR stack from ``env_cfg.py`` and
swaps in a terrain set built around the Jezero DEM: the orbital DEM carries
procedural regolith micro-roughness (see ``mars_dem_terrain.py``) so the surface
is textured Mars rather than a smooth ramp. Discrete rocks are left to the nav
layer to route around, not stepped on (rock_count_max=0).
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.terrains.config import flat, hf_pyramid_slope, perlin_noise, random_rough
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from .env_cfg import MARS_DEM_FILE, hanuman_g1_rough_env_cfg
from .mars_dem_terrain import mars_dem

# Difficulty is driven by the terrain-level curriculum, so proportions set
# exposure, not hardness. Bulk on traversable regolith; DEM + slopes are the tail.
MARS_REALISTIC_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    curriculum=True,
    sub_terrains={
        "flat": flat(proportion=0.05),
        "random_rough": random_rough(
            proportion=0.25,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
        ),
        "mars_dem": mars_dem(
            proportion=0.40,
            dem_file=MARS_DEM_FILE,
            elevation_range_m=31.45,
            dem_resolution_m=1.0,
            vertical_exaggeration=1.0,
            regolith_amplitude_m=0.05,
            rock_count_max=0,  # rocks are nav-layer obstacles, not terrain to step on
            rock_height_range_m=(0.04, 0.18),
            rock_radius_range_m=(0.08, 0.30),
        ),
        "hf_pyramid_slope": hf_pyramid_slope(
            proportion=0.15,
            slope_range=(0.0, 0.5),
        ),
        "perlin_noise": perlin_noise(
            proportion=0.15,
            height_range=(0.0, 0.4),
        ),
    },
    add_lights=True,
)


def hanuman_g1_mars_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Base G1 rough-terrain config with the realistic Mars terrain swapped in."""
    cfg = hanuman_g1_rough_env_cfg(play=play)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_generator = replace(MARS_REALISTIC_TERRAINS_CFG)

    if play:
        # Re-apply the base play() generator trims to the swapped generator.
        gen = cfg.scene.terrain.terrain_generator
        gen.curriculum = False
        gen.num_rows = 5
        gen.num_cols = 5
        gen.border_width = 10.0

    return cfg
