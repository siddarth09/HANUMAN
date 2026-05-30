# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# Fetches the real NASA Mars 2020 TRN HiRISE *orthophoto* mosaic and crops the
# window matching a DTM ROI, so it can be draped over the heightfield as a true
# surface texture (instead of the procedurally baked albedo). This is what gives
# visual odometry / SLAM real, photo-accurate features.


"""Windowed reader for the Jezero HiRISE 25 cm orthophoto mosaic."""

from __future__ import annotations

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.windows import Window

from .types import BoundingBox

# Public ASC PDS S3 mosaic — the Jezero HiRISE orthophoto (25 cm/px, grayscale,
# equirectangular). One mosaic covers every jezero_* DTM tile.
ORTHO_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com"
    "/mosaic/mars2020_trn/HiRISE"
    "/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif"
)

# Mars regolith tint applied to the grayscale orthophoto (the mosaic is single
# band; multiply to get a plausible reddish-brown RGB albedo).
_MARS_TINT = np.array([1.05, 0.72, 0.48], dtype=np.float32)


def fetch_window(
    bbox: BoundingBox,
    ortho_url: str = ORTHO_URL,
    tint: bool = True,
) -> tuple[np.ndarray, float]:
    """Read the orthophoto crop covering ``bbox`` (a geographic BoundingBox).

    Uses a windowed ``/vsicurl`` read so only the crop's byte range is fetched,
    not the multi-GB mosaic. Returns ``(rgb_uint8, resolution_m)`` where
    ``rgb_uint8`` is an ``(H, W, 3)`` Mars-tinted texture.
    """
    vsipath = ortho_url if ortho_url.startswith("/vsicurl/") else f"/vsicurl/{ortho_url}"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_USE_HEAD="NO"):
        with rasterio.open(vsipath) as src:
            res = abs(src.res[0])
            to_proj = Transformer.from_crs(
                CRS(src.crs).geodetic_crs, src.crs, always_xy=True,
            )
            x, y = to_proj.transform(bbox.lon, bbox.lat)
            row, col = src.index(x, y)
            half_w = int((bbox.width_km * 1000.0) / (2.0 * res))
            half_h = int((bbox.height_km * 1000.0) / (2.0 * res))
            c0 = max(0, int(col) - half_w)
            r0 = max(0, int(row) - half_h)
            w = min(2 * half_w, src.width - c0)
            h = min(2 * half_h, src.height - r0)
            gray = src.read(1, window=Window(c0, r0, w, h))

    g = gray.astype(np.float32)
    # Stretch the (often low-contrast) mosaic to use the full 0..1 range.
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    if tint:
        rgb = np.clip(g[..., None] * _MARS_TINT[None, None, :], 0.0, 1.0)
    else:
        rgb = np.repeat(g[..., None], 3, axis=2)
    return (rgb * 255).astype(np.uint8), float(res)
