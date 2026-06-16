# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# Adapted from jasmeet0915/artemis_mission_simulator (Apache-2.0). The lunar
# south-pole polar-stereographic handling is replaced with the equirectangular
# (Eqc) projection used by the Mars 2020 TRN HiRISE DTMs.


"""Heightmap extraction from Mars 2020 HiRISE GeoTIFF DTMs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window

from ..utils.types import ROI

# Mean Mars radius (m), used to convert degrees → meters for geographic CRS.
_MARS_RADIUS_M = 3389500.0


class DEMProcessor:
    """Extracts elevation data from Mars HiRISE GeoTIFF DTMs.

    Supports full-raster reads and center / geographic bounding-box crops, with
    optional downsampling to a target resolution and nodata filling.
    """

    @staticmethod
    def _native_resolution(src) -> float:
        """Return ground resolution in meters/pixel for projected or geographic CRS."""
        if src.crs and src.crs.is_projected:
            return abs(src.res[0])
        deg2m = _MARS_RADIUS_M * np.pi / 180.0
        lat_c = (src.bounds.bottom + src.bounds.top) / 2.0
        return abs(src.res[0]) * deg2m * float(np.cos(np.radians(lat_c)))

    @staticmethod
    def _crop_center(src, roi: ROI) -> tuple[int, int, int, int]:
        """Compute a (col_off, row_off, width, height) window for the ROI.

        If ``roi.bounding_box`` is set the window is centered on the geographic
        coordinate (transformed via pyproj); otherwise it is centered on the
        raster center. The window spans ``size_m`` (or the bounding box size).
        """
        native_res = DEMProcessor._native_resolution(src)

        if roi.bounding_box is not None:
            bb = roi.bounding_box
            geographic_crs = CRS(src.crs).geodetic_crs
            to_projected = Transformer.from_crs(
                geographic_crs, src.crs, always_xy=True,
            )
            x, y = to_projected.transform(bb.lon, bb.lat)
            row, col = src.index(x, y)
            half_w = int((bb.width_km * 1000.0) / (2.0 * native_res))
            half_h = int((bb.height_km * 1000.0) / (2.0 * native_res))
            cx, cy = int(col), int(row)
        else:
            cx, cy = src.width // 2, src.height // 2
            half_w = half_h = int(roi.size_m / (2.0 * native_res))

        c0 = max(0, cx - half_w)
        r0 = max(0, cy - half_h)
        w = min(2 * half_w, src.width - c0)
        h = min(2 * half_h, src.height - r0)
        return c0, r0, w, h

    @staticmethod
    def extract_from_raw(
        dem_path: Path,
        roi: ROI,
    ) -> tuple[np.ndarray, float, float, dict, dict]:
        """Extract elevation data from a DEM (full tile or crop).

        Returns:
            (elevations, elev_min, elev_max, bounds, meta)

            *elevations*: float32 array of elevation in meters, normalized so the
            minimum is 0 (matching the MuJoCo hfield convention).
            *elev_min*/*elev_max*: original (pre-normalization) elevation range.
            *bounds*: dict with ``center_lat``, ``center_lon``, ``width_km``,
            ``height_km``.
            *meta*: dict with ``resolution_m`` (effective), ``source``, ``shape``.
        """
        with rasterio.open(dem_path) as src:
            native_res = DEMProcessor._native_resolution(src)

            if roi.use_full:
                c0, r0 = 0, 0
                w, h = src.width, src.height
            else:
                c0, r0, w, h = DEMProcessor._crop_center(src, roi)

            elev = src.read(1, window=Window(c0, r0, w, h)).astype(np.float32)

            # ---- nodata fill (median of valid pixels) ------------------
            nd = src.nodata
            mask = np.isnan(elev)
            if nd is not None:
                mask |= np.isclose(elev, float(nd))
            if mask.any():
                valid = elev[~mask]
                fill = float(np.median(valid)) if valid.size else 0.0
                elev[mask] = fill

            # ---- geographic center of the read window ------------------
            cx_px = c0 + w / 2.0
            cy_px = r0 + h / 2.0
            x_c, y_c = src.xy(cy_px, cx_px)  # (row, col) → (x, y)
            if src.crs and src.crs.is_projected:
                to_geographic = Transformer.from_crs(
                    src.crs, CRS(src.crs).geodetic_crs, always_xy=True,
                )
                center_lon, center_lat = to_geographic.transform(x_c, y_c)
            else:
                center_lon, center_lat = x_c, y_c

        # ---- downsample to target resolution ---------------------------
        ds = max(1, int(round(roi.resolution_m / native_res)))
        if ds > 1:
            elev = elev[::ds, ::ds]
        effective_res = native_res * ds

        # ---- normalize so terrain floor sits at z = 0 ------------------
        elev_min = float(elev.min())
        elev_max = float(elev.max())
        elev = elev - elev_min

        rows, cols = elev.shape
        bounds = {
            "center_lat": float(center_lat),
            "center_lon": float(center_lon),
            "width_km": cols * effective_res / 1000.0,
            "height_km": rows * effective_res / 1000.0,
        }
        meta = {
            "resolution_m": effective_res,
            "shape": [rows, cols],
            "source": Path(dem_path).name,
        }
        return elev, elev_min, elev_max, bounds, meta
