# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.

"""Tests for the VO/navigation texture baking + boulder scattering."""

import numpy as np
import pytest

from mars_terrain_exporter.model_writers import texture_baker as tb
from mars_terrain_exporter.model_writers.mjcf_model_writer import MJCFModelWriter


def _demo_elev(n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    return ((np.sin(xx / 6.0) + np.cos(yy / 5.0)) + 2).astype(np.float32)


def test_bake_texture_shape_and_dtype():
    elev = _demo_elev()
    tex = tb.bake_surface_texture(elev, 1.0, size_px=128, n_rocks=50, seed=1)
    assert tex.shape == (128, 128, 3)
    assert tex.dtype == np.uint8
    # not a flat colour: rocks + mottling give real variation
    assert tex.std() > 5


def test_scatter_boulders_count_and_on_surface():
    elev = _demo_elev()
    bs = tb.scatter_boulders(elev, 1.0, count=12, seed=3)
    assert len(bs) == 12
    for b in bs:
        assert len(b["pos"]) == 3 and len(b["size"]) == 3 and len(b["quat"]) == 4
        # boulder sits on the hfield footprint
        assert abs(b["pos"][0]) <= elev.shape[1] / 2
        assert abs(b["pos"][1]) <= elev.shape[0] / 2


def test_writer_bakes_albedo_and_boulders(tmp_path):
    elev = _demo_elev()
    writer = MJCFModelWriter(tmp_path)
    out = writer.write(
        site_id="t", display_name="T", description="d", elevations=elev,
        resolution_m=1.0, elevation_min=0.0, elevation_max=float(elev.max()),
        lat=0.0, lon=0.0, source="procedural_test",
        bake_texture=True, texture_size=128, n_rocks=30, n_boulders=8, seed=5,
    )
    assert (out / "t_albedo.png").exists()
    model = (out / "model.xml").read_text()
    assert 't_albedo.png' in model
    assert model.count('type="ellipsoid"') == 8  # boulders present


def test_writer_flat_default_has_no_albedo(tmp_path):
    elev = _demo_elev()
    writer = MJCFModelWriter(tmp_path)
    out = writer.write(
        site_id="f", display_name="F", description="d", elevations=elev,
        resolution_m=1.0, elevation_min=0.0, elevation_max=float(elev.max()),
        lat=0.0, lon=0.0, source="procedural_test",
    )
    assert not (out / "f_albedo.png").exists()
    assert 'builtin="flat"' in (out / "model.xml").read_text()
