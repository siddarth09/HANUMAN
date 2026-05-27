# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DEM processor for NASA HiRISE DTMs in ESRI:103885 (Mars Equidistant Cylindrical).

Key difference from the lunar processor
----------------------------------------
The lunar PGDA-78 DEMs use EPSG:3031 (polar stereographic, units = metres) but
the resolution is stored in degree-equivalent grid spacings in some older files,
requiring a separate degree→metre conversion.

The HiRISE DTMs used here are already in ESRI:103885 — a *projected* CRS whose
axes are in metres — so ``src.res`` gives ``(x_res_m, y_res_m)`` directly and
no degree-to-metre conversion is needed anywhere in this module.

ROI bounding-box crops
-----------------------
When a geographic (lat/lon) bounding box is requested, the center coordinates
are transformed to the projected CRS using pyproj.  ESRI:103885 is Mars 2000
Equidistant Cylindrical; pyproj resolves it via the ESRI authority.  A
Mars-radius fallback is provided for environments where pyproj's ESRI DB is not
installed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from ..utils.types import ROI

# Semi-major axis of the Mars 2000 sphere used in ESRI:103885
# (equidistant cylindrical Eqc_latTs0_lon0).  This must match the sphere
# radius baked into the projection so that the fallback formula
#   x = R * lon_rad,  y = R * lat_rad
# agrees with pyproj to within one pixel.
# Do NOT use the IAU volumetric mean (3 389 500 m) here — it disagrees by
# ~6.5 km at 77° E and will place the crop window outside the file bounds.
_MARS_RADIUS_M: float = 3_396_190.0


class DEMProcessor:
    """Extracts elevation data from HiRISE DTMs (ESRI:103885 projected CRS)."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_elevations(
        raw: np.ndarray,
        nodata: float | None,
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> np.ndarray:
        """Convert raw pixel values → elevation metres; nodata → NaN."""
        result = raw.astype(np.float64) * scale + offset
        if nodata is not None:
            mask = np.isclose(raw.astype(np.float64), float(nodata))
            result[mask] = np.nan
        return result

    @staticmethod
    def _projected_to_geographic(
        crs,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Convert projected (x, y) → (lon, lat) using pyproj.

        Falls back to a simple equidistant-cylindrical approximation using the
        Mars mean radius when pyproj cannot resolve the CRS's geodetic base.
        """
        try:
            from pyproj import CRS as ProjCRS, Transformer

            proj_crs = ProjCRS.from_user_input(crs)
            geo_crs = proj_crs.geodetic_crs
            to_geo = Transformer.from_crs(proj_crs, geo_crs, always_xy=True)
            lon, lat = to_geo.transform(x, y)
            return float(lon), float(lat)
        except Exception:
            # Equidistant cylindrical approximation (lat_ts = 0°)
            lon = math.degrees(x / _MARS_RADIUS_M)
            lat = math.degrees(y / _MARS_RADIUS_M)
            return lon, lat

    @staticmethod
    def _geographic_to_projected(
        crs,
        lon: float,
        lat: float,
    ) -> tuple[float, float]:
        """Convert geographic (lon, lat) → projected (x, y) using pyproj.

        Falls back to the equidistant-cylindrical approximation when pyproj
        cannot resolve the CRS.
        """
        try:
            from pyproj import CRS as ProjCRS, Transformer

            proj_crs = ProjCRS.from_user_input(crs)
            geo_crs = proj_crs.geodetic_crs
            to_proj = Transformer.from_crs(geo_crs, proj_crs, always_xy=True)
            x, y = to_proj.transform(lon, lat)
            return float(x), float(y)
        except Exception:
            x = math.radians(lon) * _MARS_RADIUS_M
            y = math.radians(lat) * _MARS_RADIUS_M
            return x, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def extract_from_raw(
        dem_path: Path,
        roi: ROI,
    ) -> tuple[np.ndarray, float, float, dict, dict]:
        """Extract elevation data from a HiRISE DTM.

        Parameters
        ----------
        dem_path:
            Path to the GeoTIFF DTM file.
        roi:
            Region-of-interest: ``use_full=True`` reads the entire tile;
            ``use_full=False`` crops to the geographic bounding box.

        Returns
        -------
        elevations : np.ndarray, shape (H, W), float64
            Elevation values in metres (NaN where nodata).
        elev_min : float
            Minimum elevation (metres).
        elev_max : float
            Maximum elevation (metres).
        bounds : dict
            ``center_lat``, ``center_lon`` (degrees),
            ``width_km``, ``height_km`` (kilometres).
        dem_profile : dict
            ``crs`` and ``transform`` of the extracted window.
        """
        with rasterio.open(dem_path) as src:
            crs = src.crs

            if roi.use_full:
                raw = src.read(1)
                out_transform = src.transform
                rb = src.bounds
                x_min, y_min = rb.left, rb.bottom
                x_max, y_max = rb.right, rb.top

            else:
                bb = roi.bounding_box
                # CRS is projected (metres) — convert lat/lon BB centre to
                # projected coords, then add half-extents in metres.
                x_c, y_c = DEMProcessor._geographic_to_projected(
                    crs, bb.lon, bb.lat
                )
                half_w = bb.width_km * 500.0   # km → m, then /2
                half_h = bb.height_km * 500.0
                x_min = x_c - half_w
                x_max = x_c + half_w
                y_min = y_c - half_h
                y_max = y_c + half_h

                # ----------------------------------------------------------
                # Guard: verify the requested window intersects the file.
                # rasterio silently returns an empty array for out-of-bounds
                # windows; we raise a helpful error instead.
                # ----------------------------------------------------------
                rb = src.bounds
                x_min_clamp = max(x_min, rb.left)
                x_max_clamp = min(x_max, rb.right)
                y_min_clamp = max(y_min, rb.bottom)
                y_max_clamp = min(y_max, rb.top)

                if x_min_clamp >= x_max_clamp or y_min_clamp >= y_max_clamp:
                    raise ValueError(
                        f"Requested ROI (lat={bb.lat}°, lon={bb.lon}°) maps to "
                        f"projected box [{x_min:.0f}, {y_min:.0f}] – "
                        f"[{x_max:.0f}, {y_max:.0f}] m, which does not overlap "
                        f"the DEM file extent "
                        f"[{rb.left:.0f}, {rb.bottom:.0f}] – "
                        f"[{rb.right:.0f}, {rb.top:.0f}] m.\n"
                        f"Hint: use --lat/--lon in Mars planetocentric degrees "
                        f"(0–360° E longitude) matching the HiRISE product header."
                    )

                # Clamp to file extent (warn if partially outside)
                if (x_min < rb.left or x_max > rb.right
                        or y_min < rb.bottom or y_max > rb.top):
                    print(
                        f"  Warning: requested ROI extends beyond DEM bounds "
                        f"— clamping to file extent."
                    )
                    x_min, x_max = x_min_clamp, x_max_clamp
                    y_min, y_max = y_min_clamp, y_max_clamp

                window = from_bounds(x_min, y_min, x_max, y_max, src.transform)
                raw = src.read(1, window=window)
                out_transform = src.window_transform(window)

            nodata = src.nodata
            scale = src.scales[0] if src.scales else 1.0
            offset = src.offsets[0] if src.offsets else 0.0

        # ----------------------------------------------------------------
        # Elevation array
        # ----------------------------------------------------------------
        elevations = DEMProcessor._read_elevations(raw, nodata, scale, offset)
        elev_min = float(np.nanmin(elevations))
        elev_max = float(np.nanmax(elevations))

        # ----------------------------------------------------------------
        # Bounds — width/height are already in metres (projected CRS).
        # No degree-to-metre conversion is needed.
        # ----------------------------------------------------------------
        width_m = float(x_max - x_min)
        height_m = float(y_max - y_min)

        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        center_lon, center_lat = DEMProcessor._projected_to_geographic(
            crs, center_x, center_y
        )

        bounds = {
            "center_lat": center_lat,
            "center_lon": center_lon,
            "width_km": width_m / 1000.0,
            "height_km": height_m / 1000.0,
        }
        dem_profile = {"crs": crs, "transform": out_transform}

        return elevations, elev_min, elev_max, bounds, dem_profile
