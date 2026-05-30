# Copyright 2026 HANUMAN
#
# Licensed under the MIT License.
#
# Adapted from jasmeet0915/artemis_mission_simulator (Apache-2.0).


"""Data types for Mars terrain generation configuration."""

import re
from dataclasses import dataclass, field

_VALID_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

# Public ASC PDS S3 mosaic holding the Mars 2020 TRN HiRISE DTMs.
_BASE_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com"
    "/mosaic/mars2020_trn/HiRISE"
)


@dataclass
class BoundingBox:
    """Geographic bounding box defined by center and dimensions.

    Unlike the lunar south-pole exporter this was adapted from, there is no
    latitude constraint — Mars sites of interest (e.g. Jezero ~18.4 N) sit
    near the equator.
    """

    lat: float
    lon: float
    width_km: float = 1.0
    height_km: float = 1.0

    def validate(self) -> None:
        """Validate bounding box values. Raises ValueError on invalid data."""
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat must be in [-90, 90] (got: {self.lat})")
        if not -180.0 <= self.lon <= 360.0:
            raise ValueError(f"lon must be in [-180, 360] (got: {self.lon})")
        if self.width_km <= 0:
            raise ValueError(f"width_km must be > 0 (got: {self.width_km})")
        if self.height_km <= 0:
            raise ValueError(f"height_km must be > 0 (got: {self.height_km})")


@dataclass
class ROI:
    """Defines how much of a DEM to use.

    Three modes, in order of precedence:
      * ``bounding_box`` set        → crop around a geographic lat/lon center.
      * ``use_full`` True           → read the entire raster.
      * neither                     → crop ``size_m`` x ``size_m`` from the
                                       raster center (default; full HiRISE DTMs
                                       are very large).
    """

    use_full: bool = False
    bounding_box: BoundingBox | None = None
    size_m: float = 100.0
    resolution_m: float = 0.1

    def validate(self) -> None:
        """Validate ROI configuration. Raises ValueError on invalid data."""
        if self.bounding_box is not None:
            self.bounding_box.validate()
        if self.size_m <= 0:
            raise ValueError(f"size_m must be > 0 (got: {self.size_m})")
        if self.resolution_m <= 0:
            raise ValueError(
                f"resolution_m must be > 0 (got: {self.resolution_m})"
            )


@dataclass
class MarsSite:
    """A single Mars terrain site with its DEM source and region of interest.

    The DEM URL is derived from the catalog entry's *dem_filename*.
    """

    site_code: str
    name: str
    dem_filename: str
    description: str = ""
    roi: ROI = field(default_factory=ROI)

    @classmethod
    def from_catalog(cls, identifier: str, roi: ROI | None = None) -> "MarsSite":
        """Build a MarsSite by looking up *identifier* (name or code) in the catalog."""
        from .site_catalog import get_site
        entry = get_site(identifier)
        site = cls(
            site_code=entry["site_code"],
            name=entry["site_name"],
            dem_filename=entry["dem_filename"],
            description=entry["description"],
            roi=roi or ROI(),
        )
        site.validate()
        return site

    @property
    def dem_url(self) -> str:
        """DEM (surface elevation) GeoTIFF URL."""
        return f"{_BASE_URL}/{self.dem_filename}"

    def validate(self) -> None:
        """Validate configuration values. Raises ValueError on invalid data."""
        if not self.name or not _VALID_NAME_RE.match(self.name):
            raise ValueError(
                f"name must be non-empty and contain only alphanumeric, "
                f"hyphens, or underscores (got: {self.name!r})"
            )
        if not self.site_code:
            raise ValueError("site_code must be a non-empty string")
        if not self.dem_filename:
            raise ValueError("dem_filename must be a non-empty string")
        self.roi.validate()
