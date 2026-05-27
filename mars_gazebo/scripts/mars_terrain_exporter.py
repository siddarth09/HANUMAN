#!/usr/bin/env python3
"""
HANUMAN — Mars Terrain Exporter (Real HiRISE Data)

Downloads real Mars terrain from NASA/USGS HiRISE DTMs and exports to:
  1. MuJoCo heightfield (16-bit PNG)
  2. OBJ mesh (optional, for Gazebo)
  3. MuJoCo scene XML

Data: NASA Mars 2020 HiRISE DTMs + Human Exploration Zone archive.
Requirements: pip install rasterio numpy Pillow

Usage:
    python3 mars_terrain_exporter.py --list-sites
    python3 mars_terrain_exporter.py --site jezero_c --size 150 --output-dir mars_terrain/
    python3 mars_terrain_exporter.py --site jezero_c --size 150 --export-obj --output-dir mars_terrain/
"""

import argparse
import json
import os
import struct
import sys
import zlib
from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    print("ERROR: rasterio required. Install: pip install rasterio")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ═══════════════════════════════════════════════════════════════════
# Verified Mars DTM URLs (direct S3 links, no auth required)
# ═══════════════════════════════════════════════════════════════════

MARS_SITES = {
    # Mars 2020 TRN HiRISE DTMs (Jezero Crater, 1m/pixel)
    "jezero_c": {
        "name": "Jezero Crater Center",
        "url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_C_Edited_affine_1m_Eqc_latTs0_lon0.tif",
    },
    "jezero_n": {
        "name": "Jezero Crater North",
        "url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_N_Edited_affine_1m_Eqc_latTs0_lon0.tif",
    },
    "jezero_e": {
        "name": "Jezero Crater East",
        "url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_E_Edited_affine_1m_Eqc_latTs0_lon0.tif",
    },
    "jezero_w": {
        "name": "Jezero Crater West",
        "url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_W_Edited_affine_1m_Eqc_latTs0_lon0.tif",
    },
    "jezero_dl": {
        "name": "Jezero Crater Delta",
        "url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_DL_Edited_affine_1m_Eqc_latTs0_lon0.tif",
    },
    "jezero_cr_north": {
        "name": "Jezero Crater Rim North",
        "url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_CR_NORTH_Edited_affine_1m_Eqc_latTs0_lon0.tif",
    },
    "jezero_cr_south": {
        "name": "Jezero Crater Rim South",
        "url": "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_CR_SOUTH_Edited_affine_1m_Eqc_latTs0_lon0.tif",
    },
}


def list_sites():
    print("\n  HANUMAN — Available Mars Terrain Sites")
    print("  " + "=" * 50)
    for key, site in MARS_SITES.items():
        print(f"  {key:<22s} {site['name']}")
    print(f"\n  Usage: python3 mars_terrain_exporter.py --site jezero_c --size 150\n")


def download_region(url, size_m=100.0, resolution=0.1, output_dir="."):
    """Stream a crop from the COG DTM."""
    print(f"  Opening: {url.split('/')[-1]}")

    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
    )

    with env:
        with rasterio.open(url) as src:
            print(f"  Full size: {src.width} x {src.height} px")
            print(f"  CRS: {src.crs}")
            print(f"  Bounds: {src.bounds}")

            # Detect CRS type — projected = meters, geographic = degrees
            if src.crs and src.crs.is_projected:
                native_res = abs(src.res[0])  # already meters
                print(f"  Projected CRS — resolution: {native_res:.2f} m/px")
            else:
                mars_r = 3389500.0
                deg2m = mars_r * np.pi / 180.0
                lat_c = (src.bounds.bottom + src.bounds.top) / 2
                native_res = abs(src.res[0]) * deg2m * np.cos(np.radians(lat_c))
                print(f"  Geographic CRS — resolution: ~{native_res:.2f} m/px")
            # Crop from center
            
            cx, cy = src.width // 2, src.height // 2
            half_px = int(size_m / (2 * native_res))
            c0 = max(0, cx - half_px)
            r0 = max(0, cy - half_px)
            w = min(2 * half_px, src.width - c0)
            h = min(2 * half_px, src.height - r0)

            print(f"  Reading {w}x{h} px window...")
            elev = src.read(1, window=Window(c0, r0, w, h)).astype(np.float32)

            # Handle nodata
            nd = src.nodata
            if nd is not None:
                mask = (elev == nd) | np.isnan(elev)
            else:
                mask = np.isnan(elev)

            if mask.any():
                print(f"  Filling {mask.sum()} nodata pixels...")
                # Simple fill: replace with median
                valid_median = np.median(elev[~mask]) if (~mask).any() else 0
                elev[mask] = valid_median

            # Downsample if target resolution is coarser than native
            ds = max(1, int(resolution / native_res))
            if ds > 1:
                elev = elev[::ds, ::ds]
                effective_res = native_res * ds
                print(f"  Downsampled {ds}x → ~{effective_res:.2f} m/px")
            else:
                effective_res = native_res

            # Normalize to start at z=0
            e_min, e_max = float(elev.min()), float(elev.max())
            elev -= e_min

            print(f"  Elevation: {e_min:.1f}m to {e_max:.1f}m (range: {e_max - e_min:.1f}m)")
            print(f"  Output: {elev.shape[1]}x{elev.shape[0]} px")

            meta = {
                "source": url.split('/')[-1],
                "size_m": size_m,
                "resolution_m": effective_res,
                "elevation_min": e_min,
                "elevation_max": e_max,
                "elevation_range": e_max - e_min,
                "shape": list(elev.shape),
            }
            return elev, effective_res, meta


def save_heightfield_png(hf, path):
    """Save as 16-bit grayscale PNG."""
    h_min, h_max = hf.min(), hf.max()
    if h_max - h_min < 1e-6:
        norm = np.zeros_like(hf, dtype=np.uint16)
    else:
        norm = ((hf - h_min) / (h_max - h_min) * 65535).astype(np.uint16)

    if HAS_PIL:
        Image.fromarray(norm, mode='I;16').save(path)
    else:
        # Manual 16-bit PNG
        rows, cols = norm.shape
        def chunk(ct, d):
            c = ct + d
            return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
        raw = b''.join(b'\x00' + b''.join(struct.pack('>H', norm[r, c]) for c in range(cols)) for r in range(rows))
        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', cols, rows, 16, 0, 0, 0, 0)))
            f.write(chunk(b'IDAT', zlib.compress(raw, 9)))
            f.write(chunk(b'IEND', b''))

    print(f"  Saved: {path}")


def save_obj(hf, res, path):
    """Export as OBJ mesh."""
    rows, cols = hf.shape
    # Downsample if too large
    max_v = 250000
    if rows * cols > max_v:
        skip = int(np.ceil(np.sqrt(rows * cols / max_v)))
        hf = hf[::skip, ::skip]
        res *= skip
        rows, cols = hf.shape

    with open(path, 'w') as f:
        f.write(f"# HANUMAN Mars Terrain {rows}x{cols} res={res:.3f}m\n")
        for r in range(rows):
            for c in range(cols):
                f.write(f"v {c*res:.4f} {r*res:.4f} {hf[r,c]:.4f}\n")
        for r in range(rows - 1):
            for c in range(cols - 1):
                v = r * cols + c + 1
                f.write(f"f {v} {v+1} {v+cols+1}\nf {v} {v+cols+1} {v+cols}\n")

    print(f"  Saved: {path}")


def save_scene_xml(output_dir, hf_file, meta):
    """Generate MuJoCo scene XML."""
    s = meta["shape"]
    r = meta["resolution_m"]
    er = meta["elevation_range"]
    tx, ty = s[1] * r, s[0] * r

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="mars_hirise_scene">
  <!--
    HANUMAN — Real Mars Terrain (NASA HiRISE)
    Source: {meta['source']}
    Area: {tx:.1f}m x {ty:.1f}m, Resolution: {r:.3f}m/px
    Elevation range: {er:.1f}m
  -->
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -3.72" timestep="0.005">
    <flag warmstart="enable"/>
  </option>
  <statistic center="{tx/2:.1f} {ty/2:.1f} {er/2:.1f}" extent="{max(tx,ty):.1f}"/>
  <visual>
    <headlight diffuse="0.8 0.6 0.4" ambient="0.3 0.2 0.15" specular="0.1 0.1 0.1"/>
    <rgba fog="0.8 0.6 0.4 1.0"/>
    <map znear="0.01" zfar="500"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <hfield name="mars_terrain" file="{hf_file}"
            size="{tx/2:.3f} {ty/2:.3f} {er:.4f} 0.01"/>
    <texture name="mars_tex" type="2d" builtin="flat"
             rgb1="0.72 0.45 0.2" rgb2="0.6 0.35 0.15" width="512" height="512"/>
    <material name="mars_mat" texture="mars_tex"
              texrepeat="{max(1,int(tx/4))} {max(1,int(ty/4))}"
              specular="0.05" roughness="0.95" rgba="0.72 0.45 0.2 1.0"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.85 0.55 0.3" rgb2="0.4 0.2 0.1" width="512" height="512"/>
  </asset>
  <worldbody>
    <light name="sun" pos="{tx/2:.0f} {ty/2:.0f} 50" dir="-0.3 -0.3 -1"
           diffuse="1.2 0.9 0.6" specular="0.3 0.2 0.1" castshadow="true"/>
    <light name="fill" pos="0 0 30" dir="0 0 -1"
           diffuse="0.3 0.2 0.15" specular="0 0 0" castshadow="false"/>
    <geom name="mars_terrain" type="hfield" hfield="mars_terrain"
          pos="{tx/2:.3f} {ty/2:.3f} 0" material="mars_mat"
          contype="1" conaffinity="1" friction="1.0 0.005 0.001"/>
    <site name="spawn" pos="{tx/2:.1f} {ty/2:.1f} {er+1:.1f}"
          size="0.1" rgba="0 1 0 0.5"/>
    <!-- <include file="g1_with_hands.xml"/> -->
  </worldbody>
</mujoco>
"""
    p = os.path.join(output_dir, "mars_hirise_scene.xml")
    with open(p, 'w') as f:
        f.write(xml)
    print(f"  Saved: {p}")


def main():
    parser = argparse.ArgumentParser(description="HANUMAN Mars Terrain Exporter")
    parser.add_argument("--list-sites", action="store_true")
    parser.add_argument("--site", choices=list(MARS_SITES.keys()))
    parser.add_argument("--url", type=str, help="Direct GeoTIFF URL")
    parser.add_argument("--size", type=float, default=100.0, help="Crop size in meters")
    parser.add_argument("--resolution", type=float, default=0.1, help="Target m/px")
    parser.add_argument("--output-dir", default="mars_terrain")
    parser.add_argument("--export-obj", action="store_true")
    args = parser.parse_args()

    if args.list_sites:
        list_sites()
        return

    url = args.url or (MARS_SITES[args.site]["url"] if args.site else None)
    if not url:
        print("ERROR: --site or --url required")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    name = MARS_SITES.get(args.site, {}).get("name", "Custom")
    print(f"\n  Mars Terrain Export: {name}")
    print(f"  Size: {args.size}m, Resolution: {args.resolution}m/px\n")

    hf, res, meta = download_region(url, args.size, args.resolution, args.output_dir)

    png = os.path.join(args.output_dir, "mars_heightfield.png")
    save_heightfield_png(hf, png)

    if args.export_obj:
        save_obj(hf, res, os.path.join(args.output_dir, "mars_terrain.obj"))

    save_scene_xml(args.output_dir, "mars_heightfield.png", meta)

    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Done! Files in {args.output_dir}/\n")

if __name__ == "__main__":
    main()