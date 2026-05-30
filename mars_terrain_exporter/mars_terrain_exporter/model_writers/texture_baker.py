# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# The lunar/Mars terrain pipelines this is modelled on drape a real orthophoto
# over the DTM so a camera sees trackable surface features. We don't have an
# orthoimage for these HiRISE tiles, so we *bake* a plausible Mars regolith
# albedo texture (hillshade + multi-octave mottling + scattered rocks) from the
# elevation data. This gives visual odometry / SLAM dense, non-repetitive
# features instead of a flat colour, and places 3D boulders as nav landmarks.


"""Bake a draped surface texture + scatter boulders for VO / navigation scenes."""

from __future__ import annotations

import numpy as np

from ..raster_processors.procedural_terrain import _upsample

# Mars regolith albedo endpoints (sRGB 0..1): shadowed dust -> bright crest.
_MARS_DARK = np.array([0.32, 0.16, 0.09], dtype=np.float32)
_MARS_LIGHT = np.array([0.78, 0.52, 0.33], dtype=np.float32)


def _value_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Single octave of bilinearly-upsampled value noise in [0, 1]."""
    coarse = rng.random((max(2, cells), max(2, cells))).astype(np.float32)
    n = _upsample(coarse, (size, size))
    lo, hi = float(n.min()), float(n.max())
    return (n - lo) / (hi - lo + 1e-9)


def _fbm(size: int, rng: np.random.Generator,
         octaves: int = 5, base_cells: int = 4) -> np.ndarray:
    """Fractional Brownian motion (summed value-noise octaves) in [0, 1]."""
    out = np.zeros((size, size), np.float32)
    amp, total, cells = 1.0, 0.0, base_cells
    for _ in range(octaves):
        out += amp * _value_noise(size, cells, rng)
        total += amp
        amp *= 0.5
        cells *= 2
    out /= total
    return out


def _hillshade(elev: np.ndarray, resolution_m: float,
               size: int, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    """Relief shading in [0, 1] at texture resolution."""
    h = _upsample(elev.astype(np.float32), (size, size))
    texel_m = (elev.shape[1] * resolution_m) / size
    gy, gx = np.gradient(h, max(texel_m, 1e-3))
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0.0, 1.0)


def bake_surface_texture(
    elevations: np.ndarray,
    resolution_m: float,
    size_px: int = 2048,
    n_rocks: int = 1200,
    seed: int | None = None,
) -> np.ndarray:
    """Bake an RGB (uint8) Mars surface albedo to drape over the heightfield.

    Combines relief shading (macro features), multi-octave albedo mottling
    (mid-scale features) and scattered rock specks (high-frequency corners) so
    a downward/forward camera always has trackable texture for VO/SLAM.
    """
    rng = np.random.default_rng(seed)
    size = int(size_px)

    # Albedo: blend dark<->light by low/mid-frequency mottling.
    mottle = 0.6 * _fbm(size, rng, octaves=5, base_cells=4) \
        + 0.4 * _fbm(size, rng, octaves=4, base_cells=16)
    mottle = np.clip(mottle, 0.0, 1.0)
    albedo = (_MARS_DARK[None, None, :]
              + (_MARS_LIGHT - _MARS_DARK)[None, None, :] * mottle[..., None])

    # Fine grain (per-texel) keeps high-frequency detail under close-up cameras.
    grain = (rng.random((size, size, 1)).astype(np.float32) - 0.5) * 0.08
    albedo = np.clip(albedo + grain, 0.0, 1.0)

    # Relief shading so slopes/craters read as visual structure.
    hs = _hillshade(elevations, resolution_m, size)
    albedo *= (0.55 + 0.55 * hs)[..., None]

    # Scattered rocks: small dark/bright discs -> strong corner features.
    # Drawn in local windows (cheap) rather than full-image masks.
    rmax = max(3, size // 256)
    for _ in range(int(n_rocks)):
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        r = int(rng.integers(2, rmax))
        x0, x1 = max(0, cx - r), min(size, cx + r + 1)
        y0, y1 = max(0, cy - r), min(size, cy + r + 1)
        ly, lx = np.ogrid[y0:y1, x0:x1]
        mask = (lx - cx) ** 2 + (ly - cy) ** 2 <= r * r
        shade = rng.uniform(0.45, 0.7) if rng.random() < 0.65 else rng.uniform(1.2, 1.5)
        albedo[y0:y1, x0:x1][mask] *= shade

    return (np.clip(albedo, 0.0, 1.0) * 255).astype(np.uint8)


def scatter_boulders(
    elevations: np.ndarray,
    resolution_m: float,
    count: int = 25,
    seed: int | None = None,
) -> list[dict]:
    """Place 3D boulders on the surface as navigation landmarks / obstacles.

    Returns a list of dicts with ``pos`` (x, y, z, centred on the hfield frame),
    ``size`` (ellipsoid radii) and ``quat`` for each boulder. Boulders are
    half-buried so they sit on the terrain rather than float.
    """
    rng = np.random.default_rng(None if seed is None else seed + 9973)
    rows, cols = elevations.shape
    size_x = cols * resolution_m
    size_y = rows * resolution_m
    boulders: list[dict] = []
    for _ in range(int(count)):
        c = rng.integers(0, cols)
        r = rng.integers(0, rows)
        # hfield local frame is centred: x spans [-size_x/2, +size_x/2].
        x = (c / cols - 0.5) * size_x
        y = (0.5 - r / rows) * size_y
        z_terrain = float(elevations[r, c])
        rad = rng.uniform(0.15, 0.6)
        sx, sy, sz = rad * rng.uniform(0.8, 1.3, size=3)
        boulders.append({
            "pos": (x, y, z_terrain + sz * 0.35),  # ~1/3 buried
            "size": (sx, sy, sz),
            "quat": tuple(rng.uniform(-1, 1, size=4)),
        })
    return boulders
