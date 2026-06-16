# Copyright 2026 HANUMAN — MIT License.

"""Tests for the MJCF model writer (offline, synthetic heightfield)."""

import numpy as np
import pytest
import yaml

from mars_terrain_exporter.model_writers.mjcf_model_writer import MJCFModelWriter


def _make_elevations(rows=32, cols=48):
    """A smooth non-flat heightfield normalized so min == 0."""
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float32)
    hf = 2.0 * np.sin(x / 6.0) + 1.5 * np.cos(y / 5.0)
    return (hf - hf.min()).astype(np.float32)


def _write(tmp_path, res=0.1):
    elev = _make_elevations()
    writer = MJCFModelWriter(tmp_path)
    model_dir = writer.write(
        site_id="jezero_c", display_name="Jezero C", description="test",
        elevations=elev, resolution_m=res,
        elevation_min=-1200.0, elevation_max=-1196.5,
        lat=18.44, lon=77.45, source="nasa_hirise_mars2020",
    )
    return model_dir, elev


class TestOutputs:
    def test_all_files_written(self, tmp_path):
        model_dir, _ = _write(tmp_path)
        for name in ["jezero_c.png", "model.xml", "scene.xml", "metadata.yaml"]:
            assert (model_dir / name).exists(), name

    def test_metadata_contents(self, tmp_path):
        model_dir, elev = _write(tmp_path, res=0.1)
        meta = yaml.safe_load((model_dir / "metadata.yaml").read_text())
        assert meta["site_id"] == "jezero_c"
        assert meta["resolution_x"] == elev.shape[1]
        assert meta["resolution_y"] == elev.shape[0]
        assert meta["size_x_m"] == pytest.approx(elev.shape[1] * 0.1, abs=0.01)
        assert meta["source"] == "nasa_hirise_mars2020"

    def test_hfield_size_matches_extent(self, tmp_path):
        model_dir, elev = _write(tmp_path, res=0.1)
        model_xml = (model_dir / "model.xml").read_text()
        # half-extents: rx = cols*res/2, ry = rows*res/2
        rx = elev.shape[1] * 0.1 / 2
        ry = elev.shape[0] * 0.1 / 2
        assert f"{rx:.3f} {ry:.3f}" in model_xml


class TestMujocoLoads:
    def test_scene_loads_in_mujoco(self, tmp_path):
        mujoco = pytest.importorskip("mujoco")
        model_dir, _ = _write(tmp_path)
        model = mujoco.MjModel.from_xml_path(str(model_dir / "scene.xml"))
        assert model.nhfield == 1
        assert model.ngeom >= 1
