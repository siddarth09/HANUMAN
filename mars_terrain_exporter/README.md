# mars_terrain_exporter

Generate **MuJoCo** terrain models from real **NASA Mars 2020 TRN HiRISE**
Digital Terrain Models (1 m/pixel, Jezero Crater region).

This is the MuJoCo port of
[`artemis_mission_simulator/lunar_terrain_exporter`](https://github.com/jasmeet0915/artemis_mission_simulator):
same layered pipeline (`cli → exporter → dem_processor / model_writer /
file_downloader / site_catalog`), but it targets Mars instead of the Moon and
emits a MuJoCo `<hfield>` model instead of a Gazebo SDF `<heightmap>`.

## Pipeline

```
download (cached) → DEMProcessor.extract_from_raw → MJCFModelWriter.write
```

Each export writes `<output-dir>/<site>/`:

| File           | Purpose                                                        |
|----------------|----------------------------------------------------------------|
| `<site>.png`   | 16-bit grayscale heightfield (MuJoCo `hfield` asset)           |
| `model.xml`    | Includable terrain MJCF (`<asset><hfield>` + `<geom>`)         |
| `scene.xml`    | Standalone viewer scene: Mars gravity/sky/lights + `include`   |
| `metadata.yaml`| Coordinates, extents, resolution, elevation range, source      |

`scene.xml` has a commented `<include file="g1_mars.xml"/>` to spawn the G1 once
wired through `mujoco_ros2_control`.

## Usage

```bash
# List sites
mars_terrain_exporter list

# 150 m patch from the raster center at 0.1 m/px
mars_terrain_exporter site jezero_c --size 150 --output-dir ./models

# Geographic crop around a lat/lon
mars_terrain_exporter site jezero_c --lat 18.44 --lon 77.45 \
    --width 0.2 --height 0.2 --output-dir ./models

# Batch from config
mars_terrain_exporter batch --config config/hanuman_sites.yaml --output-dir ./models
```

Open the result:

```bash
python3 -m mujoco.viewer --mjcf=models/jezero_c/scene.xml
```

## Data source

NASA/USGS Mars 2020 Terrain-Relative Navigation HiRISE DTMs, served from the
public ASC PDS S3 mosaic. DEMs are cached under `~/.dem_cache` (override with
`WORKSPACE_DIR`).
