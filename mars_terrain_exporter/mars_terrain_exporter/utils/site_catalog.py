# Copyright 2026 HANUMAN Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mars HiRISE DTM site catalog — Jezero Crater (Mars 2020 / Perseverance).

Data source: ASC-PDS Services (Amazon S3), mars2020_trn mosaic.
CRS: ESRI:103885 — Mars 2000 Equidistant Cylindrical (projected, metres).
Resolution: 1 m/pixel.

Each entry stores the full DEM URL because HiRISE product paths are unique
per scene (no shared URL pattern like the lunar PGDA-78 archive).
"""

from typing import TypedDict


class CatalogEntry(TypedDict):
    site_code: str
    site_name: str
    description: str
    dem_url: str


_CATALOG: list[CatalogEntry] = [
    {
        "site_code": "jezero_c",
        "site_name": "jezero_c",
        "description": (
            "Jezero Crater — central region (C tile). "
            "Mars 2020 / Perseverance primary landing and operations area. "
            "HiRISE DTM mosaicked with MOLA topography, delta geoid corrected, "
            "1 m/px, ESRI:103885."
        ),
        "dem_url": (
            "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn"
            "/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_C_Edited_affine_1m"
            "_Eqc_latTs0_lon0.tif"
        ),
    },
    {
        "site_code": "jezero_dl",
        "site_name": "jezero_dl",
        "description": (
            "Jezero Crater — delta/fan lobe (DL tile). "
            "Ancient river delta fan visible to the north-west of the landing site. "
            "HiRISE DTM mosaicked with MOLA topography, delta geoid corrected, "
            "1 m/px, ESRI:103885."
        ),
        "dem_url": (
            "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn"
            "/HiRISE/DTM_MOLAtopography_DeltaGeoid_Jezero_DL_Edited_affine_1m"
            "_Eqc_latTs0_lon0.tif"
        ),
    },
]

_BY_NAME: dict[str, CatalogEntry] = {e["site_name"]: e for e in _CATALOG}
_BY_CODE: dict[str, CatalogEntry] = {e["site_code"]: e for e in _CATALOG}


def list_sites() -> list[CatalogEntry]:
    """Return a copy of all entries in the catalog."""
    return list(_CATALOG)


def get_site(identifier: str) -> CatalogEntry:
    """Look up a site by name or site_code.

    Raises
    ------
    KeyError
        If *identifier* does not match any catalog entry.
    """
    if identifier in _BY_NAME:
        return _BY_NAME[identifier]
    if identifier in _BY_CODE:
        return _BY_CODE[identifier]

    available = sorted(_BY_NAME.keys())
    raise KeyError(
        f"Site {identifier!r} not found in Mars catalog. "
        f"Available sites: {available}"
    )
