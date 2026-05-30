# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# Adapted from jasmeet0915/artemis_mission_simulator (Apache-2.0) — the lunar
# PGDA-78 catalog is replaced with NASA Mars 2020 TRN HiRISE DTM sites.


"""Mars terrain site catalog — NASA Mars 2020 TRN HiRISE DTMs.

Each entry stores the site code, site name, description and the DEM filename
on the ASC PDS public S3 mosaic. URL construction is handled by
:class:`~.types.MarsSite`.

Source: NASA/USGS Mars 2020 Terrain-Relative Navigation HiRISE Digital Terrain
Models (1 m/pixel, equirectangular), Jezero Crater region.
"""

from typing import TypedDict


class CatalogEntry(TypedDict):
    """Metadata for a single Mars HiRISE DTM site."""

    site_code: str
    site_name: str
    description: str
    dem_filename: str


_CATALOG: list[CatalogEntry] = [
    {"site_code": "Jezero_C", "site_name": "jezero_c",
        "description": "Jezero Crater center (Mars 2020 landing ellipse)",
        "dem_filename": "DTM_MOLAtopography_DeltaGeoid_Jezero_C_Edited_affine_1m_Eqc_latTs0_lon0.tif"},
    {"site_code": "Jezero_N", "site_name": "jezero_n",
        "description": "Jezero Crater north",
        "dem_filename": "DTM_MOLAtopography_DeltaGeoid_Jezero_N_Edited_affine_1m_Eqc_latTs0_lon0.tif"},
    {"site_code": "Jezero_E", "site_name": "jezero_e",
        "description": "Jezero Crater east",
        "dem_filename": "DTM_MOLAtopography_DeltaGeoid_Jezero_E_Edited_affine_1m_Eqc_latTs0_lon0.tif"},
    {"site_code": "Jezero_W", "site_name": "jezero_w",
        "description": "Jezero Crater west",
        "dem_filename": "DTM_MOLAtopography_DeltaGeoid_Jezero_W_Edited_affine_1m_Eqc_latTs0_lon0.tif"},
    {"site_code": "Jezero_DL", "site_name": "jezero_dl",
        "description": "Jezero Crater delta",
        "dem_filename": "DTM_MOLAtopography_DeltaGeoid_Jezero_DL_Edited_affine_1m_Eqc_latTs0_lon0.tif"},
    {"site_code": "Jezero_CR_NORTH", "site_name": "jezero_cr_north",
        "description": "Jezero Crater rim north",
        "dem_filename": "DTM_MOLAtopography_DeltaGeoid_Jezero_CR_NORTH_Edited_affine_1m_Eqc_latTs0_lon0.tif"},
    {"site_code": "Jezero_CR_SOUTH", "site_name": "jezero_cr_south",
        "description": "Jezero Crater rim south",
        "dem_filename": "DTM_MOLAtopography_DeltaGeoid_Jezero_CR_SOUTH_Edited_affine_1m_Eqc_latTs0_lon0.tif"},
]

# Build lookup indices
_BY_NAME: dict[str, CatalogEntry] = {e["site_name"]: e for e in _CATALOG}
_BY_CODE: dict[str, CatalogEntry] = {e["site_code"]: e for e in _CATALOG}


def list_sites() -> list[CatalogEntry]:
    """Return all catalog entries in insertion order."""
    return list(_CATALOG)


def get_site(identifier: str) -> CatalogEntry:
    """Look up a site by name **or** site code.

    Raises :exc:`KeyError` if no matching entry is found.
    """
    if identifier in _BY_NAME:
        return _BY_NAME[identifier]
    if identifier in _BY_CODE:
        return _BY_CODE[identifier]
    available_names = sorted(_BY_NAME.keys())
    available_codes = sorted(_BY_CODE.keys())
    raise KeyError(
        f"Site {identifier!r} not found in catalog. "
        f"Available names: {available_names}  "
        f"Available codes: {available_codes}"
    )
