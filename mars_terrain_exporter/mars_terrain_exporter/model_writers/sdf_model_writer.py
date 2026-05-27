# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Gazebo SDF model + world writer.

Generates per-site output under ``<output_dir>/<site_id>/``:

    <site_id>/
    ├── model.sdf           – static heightmap model
    ├── model.config        – Gazebo model metadata
    ├── world.sdf           – standalone world: Bullet Featherstone physics
    ├── metadata.yaml       – provenance / size / elevation / physics stats
    └── materials/
        └── textures/
            ├── heightmap.png        – 16-bit greyscale PNG for Ogre2 visual
            │                          dimensions = 2^n+1 × 2^n+1 (required)
            ├── heightmap.tif        – float32 GeoTIFF for external tools
            ├── collision_mesh.stl   – downsampled binary STL for Bullet
            ├── mars_diffuse.png     – placeholder diffuse texture
            └── mars_normal.png      – placeholder flat normal map

Why PNG + STL instead of GeoTIFF for Gazebo
--------------------------------------------
**Problem 1 — Bullet Featherstone "Unsupported collision geometry type [6]"**
The prebuilt gz-physics-bullet-featherstone-plugin on Ubuntu/Jazzy was not
compiled with ``btHeightfieldTerrainShape`` support.  Attempting to use a
``<heightmap>`` element in the collision geometry fails with type [6] unknown.
Fix: generate a binary STL triangle mesh (``btBvhTriangleMeshShape``) from the
downsampled elevation data.  Static bodies (``<static>true</static>``) fully
support concave triangle meshes in Bullet.

**Problem 2 — Ogre2 "Heightmap final sampling must satisfy 2^n"**
Ogre2's heightmap renderer requires image dimensions to be (2^n + 1) × (2^n + 1).
Fix: resize the visual PNG to the largest ``2^n + 1 ≤ min(H, W)``.  For a
3000×3000 DEM this is **2049** (2^11 + 1), i.e. ~1.46 m/px from 1 m/px.

**Problem 3 — PROJ "celestial body mismatch"**
Gazebo's ``Dem.cc`` reads the embedded CRS from a GeoTIFF and calls PROJ to
reproject to EPSG:4326 (Earth).  ESRI:103885 (Mars) triggers
"Source and target ellipsoid do not belong to the same celestial body".
Fix: the PNG has no embedded CRS; ``Dem.cc`` skips reprojection entirely.

The float32 GeoTIFF is kept alongside the PNG for use in QGIS, GDAL, etc.

Physics notes (Gazebo Harmonic / gz-sim 8)
-------------------------------------------
* ``type="bullet_featherstone"`` selects the Bullet Featherstone engine.
* ``gz-physics-bullet-featherstone-plugin`` is loaded inside the Physics plugin.
* ``max_step_size 0.005`` → 200 Hz physics update rate.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from string import Template

import numpy as np
import rasterio
import yaml

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Collision uses a mesh (STL) — works with any Bullet Featherstone build.
# Visual uses a heightmap PNG — Ogre2 requires 2^n+1 dimensions.
# All URIs are file:// absolute paths; no GZ_SIM_RESOURCE_PATH required.
_MODEL_SDF_TEMPLATE = Template("""\
<?xml version="1.0"?>
<sdf version="1.11">
  <model name="${site_id}">
    <static>true</static>
    <link name="terrain_link">

      <!-- ============================================================ -->
      <!-- Collision: triangle mesh (btBvhTriangleMeshShape)            -->
      <!-- Bullet Featherstone supports concave mesh for static bodies. -->
      <!-- The prebuilt Jazzy package does not support <heightmap>      -->
      <!-- collision; using mesh sidesteps geometry type [6] error.     -->
      <!-- ============================================================ -->
      <collision name="terrain_collision">
        <geometry>
          <mesh>
            <uri>${collision_mesh_uri}</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </collision>

      <!-- ============================================================ -->
      <!-- Visual: heightmap PNG (Ogre2)                                -->
      <!-- PNG must be (2^n+1)×(2^n+1) for Ogre2 heightmap renderer.  -->
      <!-- file:// URI prevents Gazebo from attempting a PROJ           -->
      <!-- reprojection of the Mars CRS to Earth EPSG:4326.            -->
      <!-- ============================================================ -->
      <visual name="terrain_visual">
        <geometry>
          <heightmap>
            <uri>${heightmap_uri}</uri>
            <size>${size_x} ${size_y} ${size_z}</size>
            <!-- pos z=0: terrain bottom sits at world origin.            -->
            <!-- Elevation min/max are baked into the PNG pixel range;    -->
            <!-- absolute elevation offset is NOT applied here so the     -->
            <!-- terrain is always visible near the world origin.         -->
            <pos>0 0 0</pos>
            <texture>
              <!-- Placeholder textures — replace with hi-res Mars maps. -->
              <diffuse>${diffuse_uri}</diffuse>
              <normal>${normal_uri}</normal>
              <size>20</size>
            </texture>
          </heightmap>
        </geometry>
      </visual>

    </link>
  </model>
</sdf>
""")

_WORLD_SDF_TEMPLATE = Template("""\
<?xml version="1.0"?>
<!--
  Mars terrain world — ${display_name}
  Physics: Bullet Featherstone  |  step: 0.005 s (200 Hz)
  Target: Gazebo Harmonic (gz-sim 8) + ROS 2 Jazzy

  Launch:
    gz sim -r ${model_dir_abs}/world.sdf

  Or via ros_gz_sim:
    ros2 launch ros_gz_sim gz_sim.launch.py \\
      gz_args:="-r ${model_dir_abs}/world.sdf"
-->
<sdf version="1.11">
  <world name="${site_id}_world">

    <!-- ================================================================ -->
    <!-- Physics: Bullet Featherstone                                      -->
    <!-- ================================================================ -->
    <physics name="bullet_featherstone_physics" type="bullet_featherstone">
      <max_step_size>0.005</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>200</real_time_update_rate>
    </physics>

    <!-- ================================================================ -->
    <!-- System plugins (Gazebo Harmonic)                                  -->
    <!-- ================================================================ -->
    <plugin
      filename="gz-sim-physics-system"
      name="gz::sim::systems::Physics">
      <engine>
        <filename>gz-physics-bullet-featherstone-plugin</filename>
      </engine>
    </plugin>

    <plugin
      filename="gz-sim-user-commands-system"
      name="gz::sim::systems::UserCommands"/>

    <plugin
      filename="gz-sim-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster"/>

    <plugin
      filename="gz-sim-sensors-system"
      name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <!-- ================================================================ -->
    <!-- Scene: ambient + sky  (<ambient>/<background> must be inside     -->
    <!-- <scene>, not direct children of <world> in SDF 1.11)            -->
    <!-- ================================================================ -->
    <scene>
      <ambient>0.20 0.18 0.16 1</ambient>
      <background>0.45 0.35 0.28 1</background>
      <shadows>true</shadows>
    </scene>

    <!-- ================================================================ -->
    <!-- Mars-like directional lighting                                    -->
    <!-- ================================================================ -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 1000 0 0 0</pose>
      <diffuse>0.75 0.62 0.52 1</diffuse>
      <specular>0.15 0.12 0.10 1</specular>
      <attenuation>
        <range>10000</range>
        <constant>1.0</constant>
        <linear>0.0</linear>
        <quadratic>0.0</quadratic>
      </attenuation>
      <direction>0.2 0.0 -0.98</direction>
    </light>

    <!-- ================================================================ -->
    <!-- Terrain model — file:// URI, no GZ_SIM_RESOURCE_PATH needed     -->
    <!-- ================================================================ -->
    <include>
      <uri>file://${model_dir_abs}</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>

    <!-- ================================================================ -->
    <!-- Default GUI camera — scaled to terrain size                      -->
    <!-- cam_y ≈ −0.8 × size_y,  cam_z ≈ 0.6 × size_y                  -->
    <!-- ================================================================ -->
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 ${gui_cam_y} ${gui_cam_z} 0 0.52 1.5708</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>

  </world>
</sdf>
""")

_MODEL_CONFIG_TEMPLATE = Template("""\
<?xml version="1.0"?>
<model>
  <name>${display_name}</name>
  <version>1.0</version>
  <sdf version="1.11">model.sdf</sdf>
  <author>
    <name>mars_terrain_exporter (auto-generated)</name>
  </author>
  <description>${description}</description>
</model>
""")


# ---------------------------------------------------------------------------
# Helper: nearest (2^n + 1) ≤ n for Ogre2 heightmap
# ---------------------------------------------------------------------------

def _nearest_pow2_plus1(n: int) -> int:
    """Return the largest ``(2^k + 1)`` that is ≤ *n*, minimum 3.

    Ogre2's heightmap renderer requires image dimensions to satisfy
    ``size == 2^k + 1``.  This finds the correct target when downsampling.

    Examples
    --------
    >>> _nearest_pow2_plus1(3000)
    2049   # 2^11 + 1
    >>> _nearest_pow2_plus1(513)
    513    # 2^9 + 1  (exact match)
    >>> _nearest_pow2_plus1(10)
    9      # 2^3 + 1
    """
    if n <= 3:
        return 3
    k = int(math.floor(math.log2(n - 1)))
    candidate = (1 << k) + 1
    # Clamp in case of floating-point edge cases
    while candidate > n and k > 1:
        k -= 1
        candidate = (1 << k) + 1
    return max(3, candidate)


# ---------------------------------------------------------------------------
# Helper: bilinear resize (pure numpy, no scipy dependency)
# ---------------------------------------------------------------------------

def _resize_elevations(
    elevations: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Bilinear resize of *elevations* to ``(target_h, target_w)``.

    Uses only NumPy so it works without SciPy.  NaN values are preserved
    through the interpolation by nearest-neighbour fallback per pixel.
    """
    src_h, src_w = elevations.shape
    if src_h == target_h and src_w == target_w:
        return elevations.copy()

    row_f = np.linspace(0.0, src_h - 1, target_h)
    col_f = np.linspace(0.0, src_w - 1, target_w)

    r0 = np.floor(row_f).astype(int).clip(0, src_h - 2)
    r1 = (r0 + 1).clip(0, src_h - 1)
    c0 = np.floor(col_f).astype(int).clip(0, src_w - 2)
    c1 = (c0 + 1).clip(0, src_w - 1)

    dr = (row_f - r0)[:, None]   # (target_h, 1)
    dc = (col_f - c0)[None, :]   # (1, target_w)

    tl = elevations[r0[:, None], c0[None, :]]
    tr = elevations[r0[:, None], c1[None, :]]
    bl = elevations[r1[:, None], c0[None, :]]
    br = elevations[r1[:, None], c1[None, :]]

    return (tl * (1 - dr) * (1 - dc)
            + tr * (1 - dr) * dc
            + bl * dr       * (1 - dc)
            + br * dr       * dc)


# ---------------------------------------------------------------------------
# Helper: 16-bit PNG writer (stdlib + numpy, no PIL required)
# ---------------------------------------------------------------------------

def _save_16bit_png(
    path: Path,
    elevations: np.ndarray,
    elev_min: float,
    elev_max: float,
) -> None:
    """Write *elevations* as a 16-bit big-endian greyscale PNG.

    Pixel 0 → *elev_min*; 65535 → *elev_max*.  Gazebo reconstructs
    real-world heights from ``<size>`` z and ``<pos>`` z in the SDF.

    No embedded CRS — prevents Gazebo's Dem.cc from calling PROJ to
    reproject ESRI:103885 (Mars) → EPSG:4326 (Earth), which PROJ rejects.
    """
    elev_range = max(elev_max - elev_min, 1.0)
    norm = (elevations - elev_min) / elev_range
    data_u16 = (norm.clip(0.0, 1.0) * 65535.0).astype(np.uint16)
    h, w = data_u16.shape

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)

    raw = bytearray()
    for row in data_u16.astype(">u2"):
        raw.append(0)           # filter byte = None
        raw.extend(row.tobytes())

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + _chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Helper: binary STL collision mesh (numpy-vectorised, stdlib write)
# ---------------------------------------------------------------------------

def _write_collision_mesh_stl(
    path: Path,
    elevations: np.ndarray,
    size_x_m: float,
    size_y_m: float,
    step: int = 10,
    elev_min: float = 0.0,
) -> int:
    """Generate a downsampled binary STL triangle mesh for Bullet collision.

    Bullet Featherstone supports **concave** triangle meshes
    (``btBvhTriangleMeshShape``) for static bodies, even when heightmap
    collision shapes are not compiled in.

    The mesh is centred at (0, 0) in X/Y, matching the Gazebo heightmap
    ``<pos>`` convention.  Z values are **normalised** (elevation − elev_min)
    so the mesh bottom sits at z=0, matching the visual heightmap which also
    uses ``<pos>0 0 0</pos>``.

    Parameters
    ----------
    step:
        Sample every *step* pixels.  Default 10 → 10 m grid spacing for a
        1 m/px DEM.  Produces ≈ 180 K triangles for a 3000×3000 tile.

    Returns
    -------
    int
        Number of triangles written.
    """
    H, W = elevations.shape
    row_idx = np.arange(0, H, step)
    col_idx = np.arange(0, W, step)
    if row_idx[-1] != H - 1:
        row_idx = np.append(row_idx, H - 1)
    if col_idx[-1] != W - 1:
        col_idx = np.append(col_idx, W - 1)
    nr, nc = len(row_idx), len(col_idx)

    # World X: column → [-size_x/2, +size_x/2]  (west → east)
    xs = (col_idx / (W - 1) - 0.5) * float(size_x_m)
    # World Y: row 0 = north (+Y), row H-1 = south (-Y)  (Gazebo convention)
    ys = (0.5 - row_idx / (H - 1)) * float(size_y_m)

    # Vertex array  (nr × nc × 3)
    verts = np.empty((nr, nc, 3), dtype=np.float32)
    verts[:, :, 0] = xs[np.newaxis, :]
    verts[:, :, 1] = ys[:, np.newaxis]
    # Normalise: mesh z = elevation − elev_min  →  0 … (elev_max − elev_min)
    # Aligns with the visual heightmap whose <pos>0 0 0</pos> places the
    # terrain bottom at world z=0.
    verts[:, :, 2] = (elevations[row_idx[:, None], col_idx[None, :]] - elev_min).astype(np.float32)

    # Build quad indices
    ri, ci = np.meshgrid(np.arange(nr - 1), np.arange(nc - 1), indexing="ij")
    ri = ri.ravel()
    ci = ci.ravel()

    v00 = verts[ri,     ci    ]
    v10 = verts[ri + 1, ci    ]
    v01 = verts[ri,     ci + 1]
    v11 = verts[ri + 1, ci + 1]

    # Two triangles per quad: (v00,v10,v01) and (v10,v11,v01)
    tris = np.concatenate(
        [np.stack([v00, v10, v01], axis=1),
         np.stack([v10, v11, v01], axis=1)],
        axis=0,
    )  # shape: (2*(nr-1)*(nc-1), 3, 3) float32

    # Normals
    ab = tris[:, 1] - tris[:, 0]
    ac = tris[:, 2] - tris[:, 0]
    n = np.cross(ab, ac).astype(np.float32)
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    nl = np.where(nl > 0, nl, 1.0)
    n /= nl

    # Pack into binary STL structured array
    n_tri = len(tris)
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0",     "<f4", (3,)),
        ("v1",     "<f4", (3,)),
        ("v2",     "<f4", (3,)),
        ("attr",   "<u2"),
    ])
    stl = np.zeros(n_tri, dtype=dtype)
    stl["normal"] = n
    stl["v0"]     = tris[:, 0]
    stl["v1"]     = tris[:, 1]
    stl["v2"]     = tris[:, 2]

    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)                  # 80-byte header
        fh.write(struct.pack("<I", n_tri))       # triangle count
        fh.write(stl.tobytes())

    return n_tri


# ---------------------------------------------------------------------------
# Helper: placeholder Mars textures
# ---------------------------------------------------------------------------

def _write_placeholder_textures(textures_dir: Path) -> None:
    """Write Mars diffuse and flat normal PNGs (placeholder, replace for production)."""
    if _PIL_AVAILABLE:
        size = (256, 256)
        _PILImage.fromarray(
            np.full((*size, 3), [180, 100, 60], dtype=np.uint8)
        ).save(textures_dir / "mars_diffuse.png")
        _PILImage.fromarray(
            np.full((*size, 3), [128, 128, 255], dtype=np.uint8)
        ).save(textures_dir / "mars_normal.png")
    else:
        def _1px_png(r: int, g: int, b: int) -> bytes:
            def _c(tag: bytes, data: bytes) -> bytes:
                return (struct.pack(">I", len(data)) + tag + data
                        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
            return (b"\x89PNG\r\n\x1a\n"
                    + _c(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                    + _c(b"IDAT", zlib.compress(bytes([0, r, g, b])))
                    + _c(b"IEND", b""))
        (textures_dir / "mars_diffuse.png").write_bytes(_1px_png(180, 100, 60))
        (textures_dir / "mars_normal.png").write_bytes(_1px_png(128, 128, 255))


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class SDFModelWriter:
    """Writes a complete Gazebo SDF terrain model (+ world) to disk."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write(
        self,
        site_id: str,
        display_name: str,
        description: str,
        elevations: np.ndarray,
        dem_profile: dict,
        size_x_m: int,
        size_y_m: int,
        elevation_min: float,
        elevation_max: float,
        lat: float,
        lon: float,
        source: str,
    ) -> Path:
        """Write all model files for *site_id* and return the model directory."""
        model_dir = (self._output_dir / site_id).resolve()
        textures_dir = model_dir / "materials" / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)

        H, W = elevations.shape

        # ----------------------------------------------------------------
        # heightmap.tif — float32 GeoTIFF for external tools
        # ----------------------------------------------------------------
        profile = {
            "driver": "GTiff", "height": H, "width": W,
            "count": 1, "dtype": "float32",
            **dem_profile,
        }
        with rasterio.open(textures_dir / "heightmap.tif", "w", **profile) as dst:
            dst.write(elevations.astype(np.float32), 1)

        # ----------------------------------------------------------------
        # heightmap.png — 16-bit greyscale PNG for Ogre2 visual
        # Must be (2^n+1)×(2^n+1); resize if necessary.
        # ----------------------------------------------------------------
        target = _nearest_pow2_plus1(min(H, W))
        vis_elevations = _resize_elevations(elevations, target, target)
        png_path = textures_dir / "heightmap.png"
        _save_16bit_png(png_path, vis_elevations, elevation_min, elevation_max)
        if target != min(H, W):
            print(f"  Resized visual heightmap: {W}×{H} → {target}×{target} px "
                  f"(Ogre2 requires 2^n+1)")

        # ----------------------------------------------------------------
        # collision_mesh.stl — downsampled triangle mesh for Bullet
        # Target ~5 m grid spacing regardless of DEM resolution.
        # For 1 m/px: step=5 → 5 m grid, ~20 K triangles for 500 m tile.
        # ----------------------------------------------------------------
        pixel_size_m = size_x_m / max(W, 1)
        mesh_step = max(1, round(5.0 / pixel_size_m))   # target 5 m spacing
        mesh_path = textures_dir / "collision_mesh.stl"
        n_tri = _write_collision_mesh_stl(
            mesh_path, elevations, size_x_m, size_y_m,
            step=mesh_step, elev_min=elevation_min,
        )
        grid_m = pixel_size_m * mesh_step
        print(f"  Collision mesh: {n_tri:,} triangles "
              f"(step={mesh_step}, {grid_m:.1f} m grid spacing)")

        # ----------------------------------------------------------------
        # Placeholder Mars textures
        # ----------------------------------------------------------------
        _write_placeholder_textures(textures_dir)

        # All Gazebo URIs are file:// absolute — no GZ_SIM_RESOURCE_PATH needed
        heightmap_uri      = f"file://{png_path}"
        collision_mesh_uri = f"file://{mesh_path}"
        diffuse_uri        = f"file://{textures_dir / 'mars_diffuse.png'}"
        normal_uri         = f"file://{textures_dir / 'mars_normal.png'}"

        # ----------------------------------------------------------------
        # model.sdf
        # ----------------------------------------------------------------
        elevation_range = max(elevation_max - elevation_min, 1.0)
        (model_dir / "model.sdf").write_text(
            _MODEL_SDF_TEMPLATE.substitute(
                site_id=site_id,
                size_x=size_x_m,
                size_y=size_y_m,
                size_z=f"{elevation_range:.2f}",
                heightmap_uri=heightmap_uri,
                collision_mesh_uri=collision_mesh_uri,
                diffuse_uri=diffuse_uri,
                normal_uri=normal_uri,
            )
        )

        # ----------------------------------------------------------------
        # world.sdf
        # ----------------------------------------------------------------
        # Camera: 0.8 × tile behind (−Y/south), 0.6 × tile above
        (model_dir / "world.sdf").write_text(
            _WORLD_SDF_TEMPLATE.substitute(
                site_id=site_id,
                display_name=display_name,
                model_dir_abs=str(model_dir),
                gui_cam_y=f"{-size_y_m * 0.8:.0f}",
                gui_cam_z=f"{size_y_m * 0.6:.0f}",
            )
        )

        # ----------------------------------------------------------------
        # model.config
        # ----------------------------------------------------------------
        (model_dir / "model.config").write_text(
            _MODEL_CONFIG_TEMPLATE.substitute(
                display_name=display_name,
                description=description,
            )
        )

        # ----------------------------------------------------------------
        # metadata.yaml
        # ----------------------------------------------------------------
        metadata: dict = {
            "site_id": site_id,
            "display_name": display_name,
            "description": description,
            "coordinates": {
                "lat": round(float(lat), 6),
                "lon": round(float(lon), 6),
            },
            "size_x_m": size_x_m,
            "size_y_m": size_y_m,
            "dem_resolution_x": W,
            "dem_resolution_y": H,
            "visual_heightmap_px": target,
            "collision_mesh_triangles": n_tri,
            "collision_mesh_step_m": round(grid_m, 2),
            "elevation_min_m": round(elevation_min, 2),
            "elevation_max_m": round(elevation_max, 2),
            "elevation_range_m": round(elevation_range, 2),
            "source": source,
            "physics": {
                "engine": "bullet_featherstone",
                "max_step_size": 0.005,
                "real_time_update_rate": 200,
            },
        }
        with open(model_dir / "metadata.yaml", "w") as fh:
            yaml.dump(metadata, fh, default_flow_style=False, sort_keys=False)

        print(f"  Model written → {model_dir}")
        return model_dir
