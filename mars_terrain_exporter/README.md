# mars_terrain_exporter

A CLI tool and Python pipeline for generating simulation-ready Mars terrain
models from NASA HiRISE DTMs for Gazebo Harmonic + ROS 2 Jazzy.

## Overview

`mars_terrain_exporter` downloads 1 m/pixel HiRISE Digital Terrain Models
(DTMs) from the ASC-PDS Services S3 bucket, processes the elevation data, and
outputs complete Gazebo-ready SDF model directories — including a **Bullet
Featherstone physics world** (`world.sdf`) tuned for legged-robot simulation.

### Data source

| Product | CRS | Resolution | Host |
|---------|-----|-----------|------|
| HiRISE DTMs mosaicked with MOLA topography (delta-geoid corrected) | ESRI:103885 (Mars Equidistant Cylindrical, metres) | 1 m/px | ASC-PDS Services (AWS S3, public) |

Because ESRI:103885 is a **projected CRS already in metres**, `src.res` gives
`(x_res_m, y_res_m)` directly — no degree-to-metre conversion is performed
anywhere in this package.

### Pre-built sites

| Name | Site code | Description |
|------|-----------|-------------|
| `jezero_c`  | `jezero_c`  | Jezero Crater — central region. Perseverance landing and primary operations area. |
| `jezero_dl` | `jezero_dl` | Jezero Crater — delta/fan lobe. Ancient river delta fan north-west of landing site. |

## Physics configuration

Every exported site includes a `world.sdf` configured for:

| Parameter | Value |
|-----------|-------|
| Physics engine | `bullet_featherstone` |
| `max_step_size` | **0.005 s** (200 Hz) |
| `real_time_update_rate` | 200 |
| Gazebo system plugins | Physics, UserCommands, SceneBroadcaster, Sensors (ogre2) |

## Usage

### Single site — full DTM tile

```bash
mars_terrain_exporter site jezero_c --output-dir ./models
```

### Single site — custom ROI bounding-box crop

```bash
mars_terrain_exporter site jezero_dl \
  --lat 18.45 --lon 77.40 --width 3 --height 3 \
  --output-dir ./models
```

### Batch mode

```bash
mars_terrain_exporter batch \
  --config config/jezero_sites.yaml --output-dir ./models
```

### Config file format

```yaml
sites:
  - site: jezero_c
  - site: jezero_dl
    roi:
      use_full: false
      bounding_box:
        lat: 18.45
        lon: 77.40
        width_km: 5.0
        height_km: 5.0
```

### Launch in Gazebo Harmonic

```bash
# Source your ROS 2 Jazzy + Gazebo Harmonic workspace, then:
gz sim -r models/jezero_c/world.sdf
```

Or via `ros_gz_sim`:

```bash
ros2 launch ros_gz_sim gz_sim.launch.py \
  gz_args:="-r $(ros2 pkg prefix mars_terrain_exporter)/share/mars_terrain_exporter/models/jezero_c/world.sdf"
```

## Output structure

```
models/<site_name>/
├── model.sdf          # Gazebo heightmap model (static)
├── model.config       # Gazebo model metadata
├── world.sdf          # Standalone world — Bullet Featherstone, 0.005 s step
├── metadata.yaml      # Provenance, size, elevation stats, physics config
└── materials/
    └── textures/
        └── heightmap.tif   # float32 elevation GeoTIFF
```

## How it works

The pipeline has three stages:

1. **Download** — Fetches the HiRISE GeoTIFF from AWS S3 with local caching
   (`WORKSPACE_DIR/.hirise_dem_cache/`).
2. **DEM Processing** (`raster_processors/dem_processor.py`) — Reads the
   projected GeoTIFF, optionally crops to a geographic bounding box (using
   pyproj for ESRI:103885 → geographic conversion, with a Mars-radius fallback),
   and extracts float64 elevation data.
3. **Model Writing** (`model_writers/sdf_model_writer.py`) — Generates
   `heightmap.tif`, `model.sdf`, `world.sdf`, `model.config`, and
   `metadata.yaml`.

## Package structure

```
mars_terrain_exporter/
├── cli.py
├── mars_terrain_exporter.py
├── raster_processors/
│   └── dem_processor.py
├── model_writers/
│   └── sdf_model_writer.py
└── utils/
    ├── types.py
    ├── site_catalog.py
    ├── raster_utils.py
    └── file_downloader.py
```

## Build & install (ROS 2 Jazzy / ament_cmake)

```bash
cd ~/projects25/src/HANUMAN
colcon build --packages-select mars_terrain_exporter
source install/setup.bash
```

## Running tests

```bash
colcon test --packages-select mars_terrain_exporter
# or directly:
cd mars_terrain_exporter
python -m pytest test/ -v
```

## Data citation

Elevation data from NASA Mars 2020 terrain products hosted by the USGS
Astrogeology Science Center (ASC) via the Planetary Data System (PDS).
HiRISE instrument: University of Arizona.  MOLA instrument: NASA/GSFC.
