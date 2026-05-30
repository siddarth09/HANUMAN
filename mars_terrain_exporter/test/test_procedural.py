# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.

"""Tests for the procedural terrain generators."""

import numpy as np
import pytest

from mars_terrain_exporter.raster_processors import procedural_terrain as pt


@pytest.mark.parametrize("terrain_type", pt.TERRAIN_TYPES)
def test_generate_shape_and_floor(terrain_type):
    """Every generator returns a square, zero-floored, non-NaN array."""
    size_m, res = 8.0, 0.1
    elev = pt.generate(terrain_type, size_m, res, seed=1)
    n = int(round(size_m / res))
    assert elev.shape == (n, n)
    assert elev.dtype == np.float32
    assert np.isfinite(elev).all()
    assert float(elev.min()) == pytest.approx(0.0, abs=1e-5)


def test_rough_is_not_flat():
    elev = pt.random_rough(80, 80, 0.1, noise_range=(0.02, 0.12), seed=2)
    assert elev.max() > 0.05


def test_slope_peaks_at_center():
    elev = pt.pyramid_slope(80, 80, 0.1, slope=0.5)
    cy, cx = elev.shape[0] // 2, elev.shape[1] // 2
    assert elev[cy, cx] == pytest.approx(elev.max())
    # edges are the low ground
    assert elev[0, 0] < elev[cy, cx]


def test_slope_inverted_is_pit():
    elev = pt.pyramid_slope(80, 80, 0.1, slope=0.5, inverted=True)
    cy, cx = elev.shape[0] // 2, elev.shape[1] // 2
    # center is the lowest point of a pit (zero-floored)
    assert elev[cy, cx] == pytest.approx(0.0, abs=1e-5)
    assert elev[0, 0] > 0.0


def test_seed_is_reproducible():
    a = pt.generate("mix", 16.0, 0.1, seed=42)
    b = pt.generate("mix", 16.0, 0.1, seed=42)
    assert np.array_equal(a, b)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        pt.generate("does_not_exist", 8.0, 0.1)
