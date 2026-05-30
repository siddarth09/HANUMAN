# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# This is the MuJoCo analog of artemis_mission_simulator's sdf_model_writer:
# instead of a Gazebo SDF <heightmap> from a GeoTIFF, it writes a MuJoCo
# <hfield> backed by a 16-bit grayscale PNG, plus an includable terrain model,
# a standalone viewer scene, and a metadata YAML.


"""MuJoCo MJCF model writer — heightfield PNG, terrain model, scene, metadata."""

import struct
import zlib
from pathlib import Path
from string import Template

import numpy as np
import yaml

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover - PIL is a declared dependency
    _HAS_PIL = False

# Mars surface gravity (m/s^2).
_MARS_GRAVITY = 3.72

# Includable terrain model: the asset + geom only (analog of model.sdf).
_MODEL_XML_TEMPLATE = Template("""\
<mujoco model="${site_id}">
  <!-- HANUMAN Mars terrain (${source})
       Area: ${size_x}m x ${size_y}m, resolution ${resolution}m/px
       Elevation range: ${elev_range}m  -->
  <asset>
    <hfield name="${site_id}_hfield" file="${png_file}"
            size="${rx} ${ry} ${elev_range} ${base}"/>
${material_block}  </asset>
  <worldbody>
    <geom name="${site_id}_terrain" type="hfield" hfield="${site_id}_hfield"
          pos="0 0 0" material="${site_id}_mat"
          contype="1" conaffinity="1" condim="6"
          friction="1.0 0.005 0.0001"/>
${boulder_geoms}  </worldbody>
</mujoco>
""")

# Flat-colour material (default) — fast, but poor visual features for VO.
_FLAT_MATERIAL = Template("""\
    <texture name="${site_id}_tex" type="2d" builtin="flat"
             rgb1="0.72 0.45 0.2" rgb2="0.6 0.35 0.15" width="512" height="512"/>
    <material name="${site_id}_mat" texture="${site_id}_tex"
              texrepeat="${texrepeat_x} ${texrepeat_y}"
              specular="0.05" roughness="0.95" rgba="0.72 0.45 0.2 1.0"/>
""")

# Draped baked orthophoto-style material — feature-rich, for visual odometry.
_BAKED_MATERIAL = Template("""\
    <texture name="${site_id}_tex" type="2d" file="${albedo_file}"/>
    <material name="${site_id}_mat" texture="${site_id}_tex"
              texrepeat="1 1" texuniform="false"
              specular="0.02" roughness="0.98"/>
    <material name="${site_id}_rock" rgba="0.4 0.27 0.18 1"
              specular="0.05" roughness="0.95"/>
""")

# Standalone viewer scene: Mars sky/lights/gravity + include the terrain model.
_SCENE_XML_TEMPLATE = Template("""\
<mujoco model="${site_id}_scene">
  <!-- HANUMAN Mars deployment scene — load this in the MuJoCo viewer or point
       mujoco_ros2_control's `mujoco_model` param at it.
       Uncomment the robot include below to spawn the G1. -->
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -${gravity}" timestep="0.002">
    <flag warmstart="enable"/>
  </option>
  <statistic center="0 0 ${stat_center_z}" extent="${extent}"/>
  <visual>
    <headlight diffuse="0.8 0.6 0.4" ambient="0.3 0.2 0.15" specular="0.1 0.1 0.1"/>
    <rgba haze="0.6 0.35 0.15 1"/>
    <map znear="0.01" zfar="500"/>
    <quality shadowsize="4096"/>
    <global azimuth="140" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.75 0.55 0.35" rgb2="0.35 0.15 0.05" width="512" height="3072"/>
  </asset>
  <worldbody>
    <light name="sun" pos="${light_x} ${light_y} 50" dir="-0.3 -0.3 -1"
           diffuse="1.2 0.9 0.6" specular="0.3 0.2 0.1" castshadow="true"/>
    <light name="fill" pos="0 0 30" dir="0 0 -1"
           diffuse="0.3 0.2 0.15" specular="0 0 0" castshadow="false"/>
    <site name="spawn" pos="0 0 ${spawn_z}" size="0.1" rgba="0 1 0 0.5"/>
  </worldbody>

  <include file="model.xml"/>
  <!-- Spawn the G1 here once wired through mujoco_ros2_control:
  <include file="g1_mars.xml"/>
  -->
</mujoco>
""")


def _save_heightfield_png(hf: np.ndarray, path: Path) -> None:
    """Save a float heightfield as a 16-bit grayscale PNG.

    MuJoCo downconverts >8-bit PNGs to 8 bits on load, but 16-bit keeps the
    source asset faithful and matches the existing HANUMAN prototype.
    """
    h_min, h_max = float(hf.min()), float(hf.max())
    if h_max - h_min < 1e-6:
        norm = np.zeros_like(hf, dtype=np.uint16)
    else:
        norm = ((hf - h_min) / (h_max - h_min) * 65535).astype(np.uint16)

    if _HAS_PIL:
        Image.fromarray(norm, mode="I;16").save(path)
        return

    # Manual 16-bit grayscale PNG fallback.
    rows, cols = norm.shape

    def _chunk(ct: bytes, data: bytes) -> bytes:
        c = ct + data
        return (struct.pack(">I", len(data)) + c
                + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF))

    raw = b"".join(
        b"\x00" + b"".join(struct.pack(">H", norm[r, c]) for c in range(cols))
        for r in range(rows)
    )
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", cols, rows, 16, 0, 0, 0, 0)))
        f.write(_chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(_chunk(b"IEND", b""))


class MJCFModelWriter:
    """Writes a complete MuJoCo terrain model (PNG + MJCF + metadata) to disk."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write(
        self,
        site_id: str,
        display_name: str,
        description: str,
        elevations: np.ndarray,
        resolution_m: float,
        elevation_min: float,
        elevation_max: float,
        lat: float,
        lon: float,
        source: str,
        bake_texture: bool = False,
        texture_size: int = 2048,
        n_rocks: int = 1200,
        n_boulders: int = 0,
        seed: int | None = None,
        albedo_rgb: "np.ndarray | None" = None,
    ) -> Path:
        """Write all model files (PNG, model.xml, scene.xml, metadata.yaml).

        Texture/material modes:
          * ``albedo_rgb`` given  → drape that RGB image (e.g. a real HiRISE
            orthophoto) over the hfield. Best surface features for VO/SLAM.
          * ``bake_texture``      → bake a synthetic Mars albedo (relief shading
            + regolith mottling + ``n_rocks`` rock specks).
          * otherwise             → flat colour (fast, poor VO features).
        ``n_boulders`` adds 3D rocks on the surface as nav landmarks / obstacles.
        """
        model_dir = self._output_dir / site_id
        model_dir.mkdir(parents=True, exist_ok=True)

        rows, cols = elevations.shape
        size_x = cols * resolution_m
        size_y = rows * resolution_m
        elev_range = max(float(elevations.max()), 1e-3)
        base = max(0.01, elev_range * 0.1)

        png_name = f"{site_id}.png"
        _save_heightfield_png(elevations, model_dir / png_name)

        # ---- material: flat colour (default) or draped albedo image --------
        textured = bake_texture or albedo_rgb is not None
        if textured:
            from .texture_baker import bake_surface_texture, scatter_boulders
            if not _HAS_PIL:
                raise RuntimeError("textured terrain requires Pillow (PIL)")
            if albedo_rgb is not None:
                albedo = albedo_rgb            # real orthophoto (already RGB uint8)
            else:
                albedo = bake_surface_texture(
                    elevations, resolution_m, size_px=texture_size,
                    n_rocks=n_rocks, seed=seed,
                )
            albedo_name = f"{site_id}_albedo.png"
            Image.fromarray(albedo, mode="RGB").save(model_dir / albedo_name)
            material_block = _BAKED_MATERIAL.substitute(
                site_id=site_id, albedo_file=albedo_name,
            )
            boulder_geoms = self._boulder_geoms(
                site_id, scatter_boulders(elevations, resolution_m,
                                          count=n_boulders, seed=seed),
            ) if n_boulders > 0 else ""
        else:
            material_block = _FLAT_MATERIAL.substitute(
                site_id=site_id,
                texrepeat_x=max(1, int(size_x / 4)),
                texrepeat_y=max(1, int(size_y / 4)),
            )
            boulder_geoms = ""

        model_xml = _MODEL_XML_TEMPLATE.substitute(
            site_id=site_id,
            source=source,
            size_x=f"{size_x:.1f}",
            size_y=f"{size_y:.1f}",
            resolution=f"{resolution_m:.3f}",
            elev_range=f"{elev_range:.4f}",
            png_file=png_name,
            rx=f"{size_x / 2:.3f}",
            ry=f"{size_y / 2:.3f}",
            base=f"{base:.4f}",
            material_block=material_block,
            boulder_geoms=boulder_geoms,
        )
        (model_dir / "model.xml").write_text(model_xml)

        scene_xml = _SCENE_XML_TEMPLATE.substitute(
            site_id=site_id,
            gravity=f"{_MARS_GRAVITY:.2f}",
            stat_center_z=f"{elev_range / 2:.2f}",
            extent=f"{max(size_x, size_y):.1f}",
            light_x=f"{size_x / 2:.0f}",
            light_y=f"{size_y / 2:.0f}",
            spawn_z=f"{elev_range + 1:.2f}",
        )
        (model_dir / "scene.xml").write_text(scene_xml)

        metadata = {
            "site_id": site_id,
            "display_name": display_name,
            "description": description,
            "coordinates": {"lat": float(lat), "lon": float(lon)},
            "size_x_m": round(size_x, 2),
            "size_y_m": round(size_y, 2),
            "resolution_m": round(resolution_m, 4),
            "resolution_x": int(cols),
            "resolution_y": int(rows),
            "elevation_min_m": round(elevation_min, 2),
            "elevation_max_m": round(elevation_max, 2),
            "elevation_range_m": round(elev_range, 2),
            "source": source,
            "textured": bool(textured),
            "n_boulders": int(n_boulders) if textured else 0,
        }
        with open(model_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        print(f"  Model written to: {model_dir}")
        return model_dir

    @staticmethod
    def _boulder_geoms(site_id: str, boulders: list[dict]) -> str:
        """Render scattered boulders as static ellipsoid geoms for model.xml."""
        lines = []
        for i, b in enumerate(boulders):
            px, py, pz = b["pos"]
            sx, sy, sz = b["size"]
            qw, qx, qy, qz = b["quat"]
            lines.append(
                f'    <geom name="{site_id}_rock{i}" type="ellipsoid" '
                f'material="{site_id}_rock"\n'
                f'          pos="{px:.3f} {py:.3f} {pz:.3f}" '
                f'size="{sx:.3f} {sy:.3f} {sz:.3f}" '
                f'quat="{qw:.3f} {qx:.3f} {qy:.3f} {qz:.3f}"\n'
                f'          contype="1" conaffinity="1" condim="3" '
                f'friction="1.0 0.005 0.0001"/>'
            )
        return "\n".join(lines) + ("\n" if lines else "")
