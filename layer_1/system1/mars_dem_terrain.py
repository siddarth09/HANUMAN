"""Real Mars DEM heightfield sub-terrain for mjlab.

Loads a 16-bit grayscale heightfield exported by ``mars_terrain_exporter``
(real NASA HiRISE DTM, e.g. ``mars_nav_200.png``) and serves it to mjlab's
``TerrainGeneratorCfg`` as a procedural sub-terrain.

Because mjlab forces every sub-terrain to the generator's patch ``size``
(see ``TerrainGenerator.__init__``), a single small grid cell can only hold a
``size`` × ``size`` window of the full DEM. We therefore crop a window from the
DEM each time the generator asks for a patch, and use ``difficulty`` to bias the
crop toward rougher (steeper) regions so the curriculum still works: easy rows
get flat crater-floor windows, hard rows get crater-wall slopes.

Drop the result into ``MARS_TERRAINS_CFG.sub_terrains`` via the ``mars_dem``
factory in ``env_cfg.py``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

from mjlab.terrains.heightfield_terrains import (
    _compute_flat_patches,
    _fractal_perlin_noise_2d,
    color_by_height,
)
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg,
    TerrainGeometry,
    TerrainOutput,
)

# Cache decoded DEMs by path so we don't re-read the PNG for every one of the
# num_rows × num_cols patches the generator builds.
_DEM_CACHE: dict[str, np.ndarray] = {}


def _load_dem(path: str, elevation_range_m: float, vertical_exaggeration: float) -> np.ndarray:
    """Load a 16-bit DEM PNG as physical heights in meters.

    The exporter writes elevation normalized to the 16-bit range, with the true
    span recorded as ``elevation_range_m`` (the hfield ``size[2]`` in the scene
    XML). Physical height = (pixel / 65535) * elevation_range_m.
    """
    key = f"{path}|{elevation_range_m}|{vertical_exaggeration}"
    cached = _DEM_CACHE.get(key)
    if cached is not None:
        return cached
    raw = np.asarray(Image.open(path)).astype(np.float64)
    if raw.ndim == 3:  # RGB(A) — collapse to luminance just in case.
        raw = raw[..., 0]
    heights = (raw / 65535.0) * elevation_range_m * vertical_exaggeration
    _DEM_CACHE[key] = heights
    return heights


@dataclass(kw_only=True)
class HfDemTerrainCfg(SubTerrainCfg):
    """A heightfield patch cropped from a real Mars DEM."""

    dem_file: str
    """Absolute path to the 16-bit DEM PNG (e.g. mars_nav_200.png)."""
    elevation_range_m: float
    """Physical vertical span of the DEM, in meters (hfield size[2] in the scene
    XML / ``elevation_range_m`` in the exporter's metadata.yaml)."""
    dem_resolution_m: float = 1.0
    """DEM horizontal resolution, in meters per pixel (metadata ``resolution_m``)."""
    horizontal_scale: float = 0.1
    """Output heightfield grid resolution along x and y, in meters per cell."""
    vertical_scale: float = 0.005
    """Height quantization step, in meters per integer unit of the noise array."""
    base_thickness_ratio: float = 1.0
    """Ratio of the heightfield base thickness to its maximum surface height."""
    vertical_exaggeration: float = 1.0
    """Multiplier on DEM relief. >1 amplifies slopes (the source DTM is coarse at
    1 m/px, so mild exaggeration can make local topography more pronounced)."""
    window_candidates: int = 48
    """Number of random candidate crops sampled per patch. The one whose relief
    matches the requested ``difficulty`` percentile is selected."""
    flatten_to_local: bool = True
    """Subtract the window minimum so each patch starts near z=0 (keeps patches
    co-planar across the grid instead of inheriting the DEM's absolute datum)."""

    # Procedural surface texture layered on the DEM macro-relief (a 1 m/px orbital
    # DEM is otherwise a smooth ramp). Set rock_count_max=0 and
    # regolith_amplitude_m=0 for the bare DEM.
    regolith_amplitude_m: float = 0.05
    """Max amplitude (m) of fractal regolith noise at difficulty=1."""
    regolith_feature_m: float = 1.0
    """Characteristic wavelength (m) of the base regolith octave."""
    rock_count_max: int = 0
    """Scattered rocks per patch at difficulty=1. Off by default — rocks are
    nav-layer obstacles to route around, not terrain to step on."""
    rock_height_range_m: tuple[float, float] = (0.04, 0.18)
    """Min/max rock height (m), scaled by difficulty."""
    rock_radius_range_m: tuple[float, float] = (0.08, 0.30)
    """Min/max rock footprint radius (m)."""
    roughness_floor: float = 0.1
    """Minimum roughness fraction so even easy rows have some texture."""
    max_slope_deg: float = 25.0
    """Reject DEM windows whose 95th-pct local grade exceeds this — the orbital
    DEM has near-vertical crater walls a velocity walker can't climb. ~89 disables."""

    def function(
        self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
    ) -> TerrainOutput:
        body = spec.body("terrain")

        dem = _load_dem(self.dem_file, self.elevation_range_m, self.vertical_exaggeration)
        dem_rows, dem_cols = dem.shape

        # Window size in native DEM pixels for one size×size patch.
        win_r = max(2, int(round(self.size[0] / self.dem_resolution_m)))
        win_c = max(2, int(round(self.size[1] / self.dem_resolution_m)))
        if win_r > dem_rows or win_c > dem_cols:
            raise ValueError(
                f"Patch size {self.size} m at {self.dem_resolution_m} m/px needs a "
                f"{win_r}x{win_c} px window but DEM is only {dem_rows}x{dem_cols} px."
            )

        # Sample candidate crops; for each compute relief (max-min) and a robust
        # local steepness (95th-pct gradient magnitude = rise/run = tan(angle)).
        max_r = dem_rows - win_r
        max_c = dem_cols - win_c
        r0s = rng.integers(0, max_r + 1, size=self.window_candidates)
        c0s = rng.integers(0, max_c + 1, size=self.window_candidates)
        reliefs = np.empty(self.window_candidates)
        steepness = np.empty(self.window_candidates)
        for i, (r0, c0) in enumerate(zip(r0s, c0s)):
            w = dem[r0 : r0 + win_r, c0 : c0 + win_c]
            reliefs[i] = w.max() - w.min()
            gy, gx = np.gradient(w, self.dem_resolution_m)
            steepness[i] = float(np.percentile(np.hypot(gx, gy), 95))

        # Steepness cap: drop windows steeper than max_slope_deg (unclimbable
        # crater walls) so the curriculum only ever serves climbable DEM. Fall
        # back to the flattest candidate if every crop exceeds the cap.
        max_grade = np.tan(np.radians(self.max_slope_deg))
        climbable = np.where(steepness <= max_grade)[0]
        if climbable.size == 0:
            climbable = np.array([int(np.argmin(steepness))])

        # Rank the climbable windows by relief and pick the difficulty percentile
        # so curriculum rows still go flat -> steep within the climbable set.
        order = climbable[np.argsort(reliefs[climbable])]
        pick = int(round(np.clip(difficulty, 0.0, 1.0) * (len(order) - 1)))
        sel = order[pick]
        window = dem[r0s[sel] : r0s[sel] + win_r, c0s[sel] : c0s[sel] + win_c]

        if self.flatten_to_local:
            window = window - window.min()

        # Resample native window to the output grid resolution (bilinear).
        width_pixels = int(self.size[0] / self.horizontal_scale)
        length_pixels = int(self.size[1] / self.horizontal_scale)
        zoom = (width_pixels / window.shape[0], length_pixels / window.shape[1])
        window_hi = ndimage.zoom(window, zoom, order=1)

        # --- Overlay procedural Mars micro-roughness on the DEM macro-slope ---
        # The DEM gives the large-scale crater-wall/slope shape; here we add the
        # cm-scale regolith texture and scattered rocks that an orbital DEM can't
        # capture, so the surface looks/behaves like real Mars terrain rather
        # than a smooth ramp. Magnitude scales with curriculum difficulty.
        gx, gy = window_hi.shape
        rough = max(self.roughness_floor, float(np.clip(difficulty, 0.0, 1.0)))
        if self.regolith_amplitude_m > 0.0:
            scale = max(2.0, self.size[0] / self.regolith_feature_m)
            regolith = _fractal_perlin_noise_2d(gx, gy, rng, octaves=4, scale=scale)
            window_hi = window_hi + rough * self.regolith_amplitude_m * regolith
        n_rocks = int(round(rough * self.rock_count_max))
        if n_rocks > 0:
            ii, jj = np.mgrid[0:gx, 0:gy]
            for _ in range(n_rocks):
                ci = int(rng.integers(0, gx))
                cj = int(rng.integers(0, gy))
                r_px = max(1.0, rng.uniform(*self.rock_radius_range_m) / self.horizontal_scale)
                h_m = rough * rng.uniform(*self.rock_height_range_m)
                d2 = ((ii - ci) ** 2 + (jj - cj) ** 2) / (r_px * r_px)
                window_hi = window_hi + h_m * np.maximum(0.0, 1.0 - d2)

        noise = np.rint(window_hi / self.vertical_scale).astype(np.int16)

        elevation_min = int(np.min(noise))
        elevation_max = int(np.max(noise))
        elevation_range = elevation_max - elevation_min if elevation_max != elevation_min else 1

        max_physical_height = elevation_range * self.vertical_scale
        base_thickness = max_physical_height * self.base_thickness_ratio
        normalized_elevation = (noise - elevation_min) / elevation_range

        unique_id = uuid.uuid4().hex
        field = spec.add_hfield(
            name=f"hfield_{unique_id}",
            size=[self.size[0] / 2, self.size[1] / 2, max_physical_height, base_thickness],
            nrow=noise.shape[0],
            ncol=noise.shape[1],
            userdata=normalized_elevation.flatten().astype(np.float32).tolist(),
        )

        material_name = color_by_height(spec, noise, unique_id, normalized_elevation)

        hfield_geom = body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname=field.name,
            pos=[self.size[0] / 2, self.size[1] / 2, 0],
            material=material_name,
        )

        # Spawn the robot on the surface at the patch center.
        cx = normalized_elevation.shape[0] // 2
        cy = normalized_elevation.shape[1] // 2
        spawn_height = float(normalized_elevation[cx, cy]) * max_physical_height
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_height])

        flat_patches = _compute_flat_patches(
            noise,
            self.vertical_scale,
            self.horizontal_scale,
            0,
            self.flat_patch_sampling,
            rng,
        )

        geom = TerrainGeometry(geom=hfield_geom, hfield=field)
        return TerrainOutput(origin=origin, geometries=[geom], flat_patches=flat_patches)


def mars_dem(proportion: float, dem_file: str, **overrides) -> HfDemTerrainCfg:
    """Factory for a real-Mars-DEM sub-terrain, mirroring mjlab's terrain presets.

    Args:
      proportion: Robot-spawning weight for this terrain column.
      dem_file: Absolute path to the exporter's 16-bit DEM PNG.
      **overrides: Any HfDemTerrainCfg field (elevation_range_m, dem_resolution_m,
        vertical_exaggeration, ...).
    """
    return HfDemTerrainCfg(proportion=proportion, dem_file=dem_file, **overrides)
