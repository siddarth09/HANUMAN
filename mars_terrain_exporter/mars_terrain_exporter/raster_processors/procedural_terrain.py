# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# Procedural heightfield generation — the synthetic counterpart to
# dem_processor.py. Instead of cropping a real Mars DTM, this builds rough /
# hilly / sloped terrain in the spirit of the IsaacLab / legged_gym terrain
# primitives (random_uniform, pyramid_slope, pyramid_slope_inv), so a humanoid
# can be trained / tested on non-flat ground.


"""Procedural terrain generators producing MuJoCo-ready heightfield arrays.

All generators return a ``float32`` array of elevations in meters, normalized
so the minimum sits at ``0`` (the convention ``MJCFModelWriter`` expects).
"""

from __future__ import annotations

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover - PIL is a declared dependency
    _HAS_PIL = False


def _upsample(coarse: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a coarse grid up to ``shape`` (rows, cols)."""
    rows, cols = shape
    if _HAS_PIL:
        img = Image.fromarray(coarse.astype(np.float32), mode="F")
        img = img.resize((cols, rows), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)
    # Dependency-light fallback: separable linear interpolation.
    cr, cc = coarse.shape
    ys = np.linspace(0, cr - 1, rows)
    xs = np.linspace(0, cc - 1, cols)
    tmp = np.empty((cr, cols), np.float32)
    for i in range(cr):
        tmp[i] = np.interp(xs, np.arange(cc), coarse[i])
    out = np.empty((rows, cols), np.float32)
    for j in range(cols):
        out[:, j] = np.interp(ys, np.arange(cr), tmp[:, j])
    return out


def _zero_floor(field: np.ndarray) -> np.ndarray:
    field = field.astype(np.float32)
    field -= float(field.min())
    return field


def random_rough(
    rows: int,
    cols: int,
    resolution_m: float,
    noise_range: tuple[float, float] = (0.02, 0.10),
    noise_step: float = 0.02,
    downsampled_scale_m: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """Random uniform roughness (IsaacLab ``random_uniform_terrain``).

    Heights are drawn from ``arange(min, max, step)`` on a coarse grid spaced
    ``downsampled_scale_m`` apart, then bilinearly upsampled to full resolution.
    """
    rng = np.random.default_rng(seed)
    lo, hi = noise_range
    levels = np.arange(lo, hi + 1e-9, max(noise_step, 1e-3), dtype=np.float32)
    if levels.size == 0:
        levels = np.array([lo], dtype=np.float32)
    ds_r = max(2, int(rows * resolution_m / downsampled_scale_m))
    ds_c = max(2, int(cols * resolution_m / downsampled_scale_m))
    coarse = rng.choice(levels, size=(ds_r, ds_c)).astype(np.float32)
    return _zero_floor(_upsample(coarse, (rows, cols)))


def pyramid_slope(
    rows: int,
    cols: int,
    resolution_m: float,
    slope: float = 0.4,
    platform_m: float = 1.0,
    inverted: bool = False,
) -> np.ndarray:
    """Square pyramid slope (IsaacLab ``pyramid_sloped_terrain``).

    ``slope`` is the tangent of the incline (0.8 ~= 38 deg). The terrain rises
    linearly from the edges to a flat ``platform_m`` plateau at the center;
    ``inverted=True`` carves a pit instead (descending into a crater).
    """
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    # Chebyshev distance from center, in meters -> square (not round) pyramid.
    dist = np.maximum(np.abs(xx - cx), np.abs(yy - cy)) * resolution_m
    half = min(rows, cols) * resolution_m / 2.0
    edge = max(half - platform_m / 2.0, 1e-3)
    height = slope * np.clip(edge - dist, 0.0, edge)
    if inverted:
        height = float(height.max()) - height
    return _zero_floor(height)


def fractal_hills(
    rows: int,
    cols: int,
    resolution_m: float,
    amplitude_m: float = 1.5,
    base_wavelength_m: float = 18.0,
    octaves: int = 4,
    persistence: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Smooth rolling hills via summed value-noise octaves.

    ``amplitude_m`` is the approximate peak-to-trough height; smaller
    ``base_wavelength_m`` gives tighter hills.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((rows, cols), np.float32)
    amp = 1.0
    total = 0.0
    wavelength = base_wavelength_m
    for _ in range(max(1, octaves)):
        cells_r = max(2, int(rows * resolution_m / wavelength) + 1)
        cells_c = max(2, int(cols * resolution_m / wavelength) + 1)
        coarse = rng.standard_normal((cells_r, cells_c)).astype(np.float32)
        field += amp * _upsample(coarse, (rows, cols))
        total += amp
        amp *= persistence
        wavelength *= 0.5
    field = _zero_floor(field)
    peak = float(field.max())
    if peak > 1e-6:
        field *= amplitude_m / peak
    return field


def mix_grid(
    rows: int,
    cols: int,
    resolution_m: float,
    tile_m: float = 8.0,
    max_slope: float = 0.6,
    seed: int | None = None,
) -> np.ndarray:
    """legged_gym-style terrain curriculum: a grid of mixed sub-terrain tiles.

    Each ``tile_m`` x ``tile_m`` cell is independently one of {rough, pyramid
    slope, inverted slope, rolling hill, flat}. Slopes/hills are zero at their
    tile edges, so neighbouring tiles stay roughly continuous at the seams.
    """
    rng = np.random.default_rng(seed)
    field = np.zeros((rows, cols), np.float32)
    tile_px = max(4, int(round(tile_m / resolution_m)))
    kinds = ["rough", "slope", "slope_inv", "hills", "flat"]
    probs = [0.25, 0.20, 0.20, 0.25, 0.10]

    for r0 in range(0, rows, tile_px):
        for c0 in range(0, cols, tile_px):
            r1, c1 = min(r0 + tile_px, rows), min(c0 + tile_px, cols)
            tr, tc = r1 - r0, c1 - c0
            if tr < 2 or tc < 2:
                continue
            sub_seed = int(rng.integers(0, 2**31 - 1))
            kind = kinds[rng.choice(len(kinds), p=probs)]
            if kind == "rough":
                tile = random_rough(tr, tc, resolution_m,
                                    noise_range=(0.02, 0.12), seed=sub_seed)
            elif kind == "slope":
                tile = pyramid_slope(tr, tc, resolution_m,
                                     slope=rng.uniform(0.1, max_slope))
            elif kind == "slope_inv":
                tile = pyramid_slope(tr, tc, resolution_m,
                                     slope=rng.uniform(0.1, max_slope),
                                     inverted=True)
            elif kind == "hills":
                tile = fractal_hills(tr, tc, resolution_m,
                                     amplitude_m=rng.uniform(0.3, 1.0),
                                     base_wavelength_m=tile_m,
                                     octaves=3, seed=sub_seed)
            else:
                tile = np.zeros((tr, tc), np.float32)
            field[r0:r1, c0:c1] = tile
    return _zero_floor(field)


# Dispatch table for the CLI ``procedural`` subcommand: type -> (fn, fixed kwargs).
GENERATORS = {
    "rough": (random_rough, {}),
    "slope": (pyramid_slope, {}),
    "slope_inv": (pyramid_slope, {"inverted": True}),
    "hills": (fractal_hills, {}),
    "mix": (mix_grid, {}),
}

TERRAIN_TYPES = tuple(GENERATORS)


def generate(
    terrain_type: str,
    size_m: float,
    resolution_m: float,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Generate an elevation array for ``terrain_type`` over a square patch.

    ``kwargs`` are forwarded to the chosen generator (e.g. ``slope``,
    ``amplitude_m``, ``tile_m``, ``max_slope``); unsupported keys are dropped.
    """
    import inspect

    if terrain_type not in GENERATORS:
        raise ValueError(
            f"unknown terrain type {terrain_type!r}; "
            f"choose from {sorted(GENERATORS)}"
        )
    n = max(2, int(round(size_m / resolution_m)))
    fn, fixed = GENERATORS[terrain_type]
    params = inspect.signature(fn).parameters
    call_kwargs = dict(fixed)
    if "seed" in params:
        call_kwargs.setdefault("seed", seed)
    for key, value in kwargs.items():
        if key in params and value is not None:
            call_kwargs[key] = value
    return fn(n, n, resolution_m, **call_kwargs)
